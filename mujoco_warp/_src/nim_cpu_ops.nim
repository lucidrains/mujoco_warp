# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# CPU fast paths for the dense solver kernels.  Warp runs `wp.tile_*` kernels
# with a single lane on the CPU, so the dense BLAS-like kernels dominate a CPU
# step.  These equivalents are plain scalar loops spread across a small
# persistent worker pool.  All buffers are raw `wp.array` addresses passed from
# Python as uint64.

import std/atomics
import std/os
import std/cpuinfo
import std/locks
import std/math
import std/strutils
import std/typedthreads

import nimpy

const
  Satisfied = 0'i32  # types.ConstraintState.SATISFIED
  Quadratic = 1'i32  # types.ConstraintState.QUADRATIC
  LinearNeg = 2'i32  # types.ConstraintState.LINEARNEG
  LinearPos = 3'i32  # types.ConstraintState.LINEARPOS
  MaxPoolThreads = 64
  JRowMax = 256  # stack row buffer bound for the Newton H build
  AlphaNoiseEps = 8.0'f32 * 1.1920929e-07'f32
  MjMaxVal = 1.0e10'f32
  MjMinMu = 1.0e-5'f32
  OverflowIterations = 512'i32  # types.OverflowType.ITERATIONS (1 << 9)
  OverflowLsIterations = 1024'i32  # types.OverflowType.LS_ITERATIONS (1 << 10)

type
  Job = proc(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.}

  Pool = object
    lock: Lock
    work, done: Cond
    job: Job
    arg: pointer
    nthreads, generation, remaining: int
    nextTask: Atomic[int]
    stop: bool

  Worker = object
    pool: ptr Pool
    tid: int

var
  pool: ptr Pool = nil
  workers: seq[Worker]
  threads: seq[Thread[ptr Worker]]
  numThreads = 0

func toPtr[T](address: uint64): ptr UncheckedArray[T] {.inline.} =
  cast[ptr UncheckedArray[T]](address)

proc claimTask(): int {.inline, gcsafe.} =
  pool.nextTask.fetchAdd(1, moRelaxed)

proc scratch(buf: var seq[float32], n: int): ptr UncheckedArray[float32] {.inline.} =
  if buf.len < n:
    buf.setLen(n)
  result = cast[ptr UncheckedArray[float32]](addr buf[0])
  zeroMem(result, n * sizeof(float32))

proc scratchNoZero(buf: var seq[float32], n: int): ptr UncheckedArray[float32] {.inline.} =
  if buf.len < n:
    buf.setLen(n)
  result = cast[ptr UncheckedArray[float32]](addr buf[0])

func at(p: ptr UncheckedArray[float32], i: int): ptr UncheckedArray[float32] {.inline.} =
  cast[ptr UncheckedArray[float32]](addr p[i])

proc workerLoop(w: ptr Worker) {.thread.} =
  let p = w.pool
  var seen = 0
  while true:
    acquire(p.lock)
    while p.generation == seen and not p.stop:
      wait(p.work, p.lock)
    if p.stop:
      release(p.lock)
      return
    seen = p.generation
    let (job, arg, nthreads) = (p.job, p.arg, p.nthreads)
    release(p.lock)

    job(arg, w.tid, nthreads)

    acquire(p.lock)
    dec p.remaining
    if p.remaining == 0:
      signal(p.done)
    release(p.lock)

proc initPool() =
  pool = create(Pool)
  initLock(pool.lock)
  initCond(pool.work)
  initCond(pool.done)
  pool.nextTask.store(0, moRelaxed)
  numThreads = max(1, min(MaxPoolThreads, countProcessors()))
  # MUJOCO_WARP_NIM_THREADS caps the pool (debugging/testing); default = all cores
  let envThreads = getEnv("MUJOCO_WARP_NIM_THREADS")
  if envThreads.len > 0:
    try:
      let n = parseInt(envThreads)
      if n >= 1:
        numThreads = min(MaxPoolThreads, n)
    except ValueError:
      discard
  workers.setLen(numThreads)
  threads.setLen(numThreads)  # POSIX createThread passes `addr`, so the handles must outlive the workers
  for tid in 0 ..< numThreads:
    workers[tid] = Worker(pool: pool, tid: tid)
    createThread(threads[tid], workerLoop, addr workers[tid])

proc runJob(job: Job, arg: pointer, nTasks = MaxPoolThreads) =
  if nTasks <= 0:
    return
  if pool == nil:
    initPool()
  if nTasks == 1 or numThreads <= 1:
    pool.nextTask.store(0, moRelaxed)
    job(arg, 0, 1)
    return
  acquire(pool.lock)
  pool.job = job
  pool.arg = arg
  pool.nthreads = numThreads
  pool.remaining = numThreads
  pool.nextTask.store(0, moRelaxed)
  inc pool.generation
  broadcast(pool.work)
  while pool.remaining > 0:
    wait(pool.done, pool.lock)
  release(pool.lock)

func choleskyFactorize(a: ptr UncheckedArray[float32], n, stride: int) {.inline.} =
  ## In-place upper Cholesky factorization A = UᵀU using row-oriented unit-stride inner loops.
  ## Row pointers are hoisted: with a single float pointer gcc must assume the two rows
  ## alias and cannot keep them in registers (~40% of the factorization).
  for i in 0 ..< n:
    let pi = cast[ptr UncheckedArray[float32]](addr a[i * stride])
    for k in 0 ..< i:
      let pk = cast[ptr UncheckedArray[float32]](addr a[k * stride])
      let uki = pk[i]
      for j in i ..< n:
        pi[j] = pi[j] - uki * pk[j]
    let uii = sqrt(pi[i])
    pi[i] = uii
    let inv = 1.0'f32 / uii
    for j in (i + 1) ..< n:
      pi[j] = pi[j] * inv

func choleskySolve(a: ptr UncheckedArray[float32], n, stride: int, x: ptr UncheckedArray[float32]) {.inline.} =
  ## In-place solve UᵀUx = b for upper triangular U using unit-stride inner loops.
  for k in 0 ..< n:
    let pk = cast[ptr UncheckedArray[float32]](addr a[k * stride])
    let xk = x[k] / pk[k]
    x[k] = xk
    for i in (k + 1) ..< n:
      x[i] = x[i] - pk[i] * xk
  for i in countdown(n - 1, 0):
    let iRow = i * stride
    var s = x[i]
    for k in (i + 1) ..< n:
      s -= a[iRow + k] * x[k]
    x[i] = s / a[iRow + i]

# Tree LDLᵀ for the upper-triangle H layout used by the Newton worker:
# H = U D Uᵀ, U[i, d] lives at a[i * stride + d] for ancestor i.
func treeFactorize(a: ptr UncheckedArray[float32], n, stride: int,
                   parent, descPtr, descIdx: ptr UncheckedArray[int32]) {.inline.} =
  for d in countdown(n - 1, 0):
    let dRow = d * stride
    var diag = a[dRow + d]
    for t in descPtr[d] ..< descPtr[d + 1]:
      let q = int(descIdx[t])
      let u = a[dRow + q]
      diag -= u * u * a[q * stride + q]
    a[dRow + d] = diag
    var i = int(parent[d])
    while i >= 0:
      let iRow = i * stride
      var u = a[iRow + d]
      for t in descPtr[d] ..< descPtr[d + 1]:
        let q = int(descIdx[t])
        u -= a[iRow + q] * a[q * stride + q] * a[dRow + q]
      a[iRow + d] = u / diag
      i = int(parent[i])

func treeSolve(a: ptr UncheckedArray[float32], n, stride: int,
               parent, descPtr, descIdx: ptr UncheckedArray[int32],
               b, x, work: ptr UncheckedArray[float32]) {.inline.} =
  for i in countdown(n - 1, 0):
    var s = b[i]
    for t in descPtr[i] ..< descPtr[i + 1]:
      let q = int(descIdx[t])
      s -= a[i * stride + q] * work[q]
    work[i] = s
  for i in 0 ..< n:
    work[i] = work[i] / a[i * stride + i]
  for d in 0 ..< n:
    var v = work[d]
    var i = int(parent[d])
    while i >= 0:
      v -= a[i * stride + d] * x[i]
      i = int(parent[i])
    x[d] = v

func denseRowsAreChains(j: ptr UncheckedArray[float32], jBase, nv, nvPad, nefc: int,
                        anc: ptr UncheckedArray[int8], depth: ptr UncheckedArray[int32]): bool =
  ## True when every active row's nonzero dofs lie on one root-to-leaf dof chain.
  for r in 0 ..< nefc:
    var deepest = -1
    var maxDepth = -1
    for i in 0 ..< nv:
      if j[jBase + r * nvPad + i] != 0.0'f32 and int(depth[i]) > maxDepth:
        maxDepth = int(depth[i])
        deepest = i
    if deepest < 0:
      continue
    for i in 0 ..< nv:
      if i != deepest and j[jBase + r * nvPad + i] != 0.0'f32 and anc[deepest * nv + i] == 0:
        return false
  true

# H = M + JᵀDJ, dense per world, upper triangle only (Cholesky reads upper).

type
  JtdaJArgs = object
    nworld, nvPad, njmax, nC: int
    mColind, mHinitI, nefc: ptr UncheckedArray[int32]
    M, J, D, H: ptr UncheckedArray[float32]
    state: ptr UncheckedArray[int32]
    done: ptr UncheckedArray[bool]

proc jtdaJWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr JtdaJArgs](arg)
  while true:
    let w = claimTask()
    if w >= a.nworld:
      break
    if a.done[w]:
      continue

    let hOff = w * a.nvPad * a.nvPad
    zeroMem(at(a.H, hOff), a.nvPad * a.nvPad * sizeof(float32))

    # densify the upper triangle of M: CSR (row=mHinitI, col=mColind) -> (col, row)
    let mOff = w * a.nC
    for e in 0 ..< a.nC:
      let col = int(a.mColind[e])
      let row = int(a.mHinitI[e])
      a.H[hOff + col * a.nvPad + row] += a.M[mOff + e]

    let n = min(int(a.nefc[w]), a.njmax)
    let jOff = w * a.njmax * a.nvPad
    let dOff = w * a.njmax
    for k in 0 ..< n:
      if a.state[dOff + k] != Quadratic:
        continue
      let dk = a.D[dOff + k]
      if dk == 0.0'f32:
        continue
      let jk = jOff + k * a.nvPad
      for i in 0 ..< a.nvPad:
        let ji = dk * a.J[jk + i]
        if ji == 0.0'f32:
          continue
        let hRow = hOff + i * a.nvPad
        for jj in i ..< a.nvPad:
          a.H[hRow + jj] += ji * a.J[jk + jj]

proc jtdaJDense*(
  nworld, nvPad, njmax, nC: int,
  mColind, mHinitI, nefc: uint64,
  M, J, D: uint64,
  state, done, H: uint64,
): void {.exportpy: "jtda_j_dense".} =
  var args = JtdaJArgs(
    nworld: nworld, nvPad: nvPad, njmax: njmax, nC: nC,
    mColind: toPtr[int32](mColind), mHinitI: toPtr[int32](mHinitI), nefc: toPtr[int32](nefc),
    M: toPtr[float32](M), J: toPtr[float32](J), D: toPtr[float32](D),
    state: toPtr[int32](state), done: toPtr[bool](done), H: toPtr[float32](H),
  )
  runJob(jtdaJWorker, addr args, nworld)

# Cholesky factorize ctx.h (nv x nv upper) and solve for the Newton search direction.

var hCopyBuf {.threadvar.}: seq[float32]
var jSparseBuf {.threadvar.}: seq[float32]

type
  CholeskySolveArgs = object
    nworld, nv, nvPad: int
    skipNoflip, skipUnchanged, hasHfactor: bool
    grad, H, hfactor, search, searchDot, newtonDecrement: ptr UncheckedArray[float32]
    changed, quadChanged: ptr UncheckedArray[int32]
    done: ptr UncheckedArray[bool]

proc choleskySolveWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr CholeskySolveArgs](arg)
  let hc = if a.hasHfactor: nil else: scratchNoZero(hCopyBuf, a.nv * a.nvPad)
  while true:
    let w = claimTask()
    if w >= a.nworld:
      break
    if a.done[w]:
      continue
    if a.skipNoflip and a.changed[w] == 0:
      continue

    let hOff = w * a.nvPad * a.nvPad
    let factorPtr =
      if a.hasHfactor:
        at(a.hfactor, hOff)
      else:
        hc
    let needFactor = (not a.hasHfactor) or (not a.skipUnchanged) or (a.quadChanged == nil) or (a.quadChanged[w] > 0)
    if needFactor:
      for i in 0 ..< a.nv:
        let rowOff = i * a.nvPad + i
        copyMem(addr factorPtr[rowOff], addr a.H[hOff + rowOff], (a.nv - i) * sizeof(float32))
      choleskyFactorize(factorPtr, a.nv, a.nvPad)

    let gradOff = w * a.nvPad
    let xw = at(a.search, w * a.nv)
    for i in 0 ..< a.nv:
      xw[i] = a.grad[gradOff + i]
    choleskySolve(factorPtr, a.nv, a.nvPad, xw)

    var sd = 0.0'f32
    var nd = 0.0'f32
    for i in 0 ..< a.nv:
      let xi = xw[i]
      sd += xi * xi
      nd += a.grad[gradOff + i] * xi
      xw[i] = -xi
    a.searchDot[w] = sd
    a.newtonDecrement[w] = nd

proc choleskySolveDense*(
  nworld, nv, nvPad, skipNoflip: int,
  grad, H, changed, done, search, searchDot, newtonDecrement: uint64,
  hfactor: uint64 = 0,
  skipUnchanged: int = 0,
  quadChanged: uint64 = 0,
): void {.exportpy: "cholesky_solve_dense".} =
  var args = CholeskySolveArgs(
    nworld: nworld, nv: nv, nvPad: nvPad,
    skipNoflip: skipNoflip != 0,
    skipUnchanged: skipUnchanged != 0,
    hasHfactor: hfactor != 0,
    grad: toPtr[float32](grad), H: toPtr[float32](H), hfactor: toPtr[float32](hfactor),
    changed: toPtr[int32](changed), quadChanged: toPtr[int32](quadChanged), done: toPtr[bool](done),
    search: toPtr[float32](search), searchDot: toPtr[float32](searchDot),
    newtonDecrement: toPtr[float32](newtonDecrement),
  )
  runJob(choleskySolveWorker, addr args, nworld)

# Dense block Cholesky factorize+solve for the smooth inertia tiles.

var blkBuf {.threadvar.}: seq[float32]

type
  BlockCholeskySolveArgs = object
    nworld, ntiles, size, area, nC, nY, nL: int
    blkAdr, elemid, qldBlockAdr: ptr UncheckedArray[int32]
    M, Y, X, L: ptr UncheckedArray[float32]

proc blockCholeskySolveWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr BlockCholeskySolveArgs](arg)
  let bx = scratchNoZero(blkBuf, a.area)
  let nTasks = a.nworld * a.ntiles
  while true:
    let task = claimTask()
    if task >= nTasks:
      break
    let w = task div a.ntiles
    let blk = task mod a.ntiles
    let start = int(a.blkAdr[blk])

    let mOff = w * a.nC
    let ebase = blk * a.area
    for s in 0 ..< a.area:
      let idx = int(a.elemid[ebase + s])
      bx[s] = if idx < a.nC: a.M[mOff + idx] else: 0.0'f32

    choleskyFactorize(bx, a.size, a.size)

    let lOff = w * a.nL + int(a.qldBlockAdr[start])
    copyMem(addr a.L[lOff], addr bx[0], a.area * sizeof(float32))

    let xyOff = w * a.nY + start
    let xw = at(a.X, xyOff)
    for i in 0 ..< a.size:
      xw[i] = a.Y[xyOff + i]
    choleskySolve(bx, a.size, a.size, xw)

proc blockCholeskyFactorizeSolve*(
  nworld, ntiles, size, area, nC, nv, qldLen: int,
  blkAdr, elemid, qldBlockAdr: uint64,
  M, Y, X, L: uint64,
): void {.exportpy: "block_cholesky_factorize_solve".} =
  var args = BlockCholeskySolveArgs(
    nworld: nworld, ntiles: ntiles, size: size, area: area, nC: nC, nY: nv, nL: qldLen,
    blkAdr: toPtr[int32](blkAdr), elemid: toPtr[int32](elemid), qldBlockAdr: toPtr[int32](qldBlockAdr),
    M: toPtr[float32](M), Y: toPtr[float32](Y), X: toPtr[float32](X), L: toPtr[float32](L),
  )
  runJob(blockCholeskySolveWorker, addr args, nworld * ntiles)

# Contact Jacobian rows for the dense (non-flex) path.
# One thread per world walks the contact rows and writes J rows plus J @ qvel.

var jacProjBuf {.threadvar.}: seq[float32]

type
  ContactJacArgs = object
    nworld, nv, nvPad, njmax, nbody, nEfcAddr: int
    elliptic: bool
    bodyIsdofancestor, bodyRootid, geomBodyid: ptr UncheckedArray[int32]
    ne, nf, nl, nefc: ptr UncheckedArray[int32]
    qvel, subtreeCom, cdof: ptr UncheckedArray[float32]
    efcAddress, efcId, contactDim, contactGeom: ptr UncheckedArray[int32]
    contactPos, contactFrame, contactFriction: ptr UncheckedArray[float32]
    J, jqvel: ptr UncheckedArray[float32]

proc contactJacWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr ContactJacArgs](arg)
  let nv = a.nv
  let nvPad = a.nvPad
  let proj = scratch(jacProjBuf, nvPad * 6)
  let jp0 = at(proj, 0 * nvPad)
  let jp1 = at(proj, 1 * nvPad)
  let jp2 = at(proj, 2 * nvPad)
  let jr0 = at(proj, 3 * nvPad)
  let jr1 = at(proj, 4 * nvPad)
  let jr2 = at(proj, 5 * nvPad)

  while true:
    let w = claimTask()
    if w >= a.nworld:
      break
    let estart = int(a.ne[w]) + int(a.nf[w]) + int(a.nl[w])
    let eend = min(int(a.nefc[w]), a.njmax)
    var prevConid = -1
    var condim = 0

    for efcid in estart ..< eend:
      let conid = int(a.efcId[w * a.njmax + efcid])

      if conid != prevConid:
        prevConid = conid
        condim = int(a.contactDim[conid])

        let b1 = int(a.geomBodyid[int(a.contactGeom[conid * 2])])
        let b2 = int(a.geomBodyid[int(a.contactGeom[conid * 2 + 1])])
        let r1 = int(a.bodyRootid[b1]) * 3
        let r2 = int(a.bodyRootid[b2]) * 3
        let pOff = conid * 3
        let comOff = w * a.nbody * 3
        let off1x = a.contactPos[pOff + 0] - a.subtreeCom[comOff + r1 + 0]
        let off1y = a.contactPos[pOff + 1] - a.subtreeCom[comOff + r1 + 1]
        let off1z = a.contactPos[pOff + 2] - a.subtreeCom[comOff + r1 + 2]
        let off2x = a.contactPos[pOff + 0] - a.subtreeCom[comOff + r2 + 0]
        let off2y = a.contactPos[pOff + 1] - a.subtreeCom[comOff + r2 + 1]
        let off2z = a.contactPos[pOff + 2] - a.subtreeCom[comOff + r2 + 2]

        let f0x = a.contactFrame[conid * 9 + 0]
        let f0y = a.contactFrame[conid * 9 + 1]
        let f0z = a.contactFrame[conid * 9 + 2]
        let f1x = a.contactFrame[conid * 9 + 3]
        let f1y = a.contactFrame[conid * 9 + 4]
        let f1z = a.contactFrame[conid * 9 + 5]
        let f2x = a.contactFrame[conid * 9 + 6]
        let f2y = a.contactFrame[conid * 9 + 7]
        let f2z = a.contactFrame[conid * 9 + 8]
        let needRot = condim > 3

        for i in 0 ..< nv:
          let aff1 = a.bodyIsdofancestor[b1 * nvPad + i] != 0
          let aff2 = a.bodyIsdofancestor[b2 * nvPad + i] != 0
          if not aff1 and not aff2:
            jp0[i] = 0.0'f32
            jp1[i] = 0.0'f32
            jp2[i] = 0.0'f32
            if needRot:
              jr0[i] = 0.0'f32
              jr1[i] = 0.0'f32
              jr2[i] = 0.0'f32
            continue
          let cOff = (w * nv + i) * 6
          let ax = a.cdof[cOff + 0]
          let ay = a.cdof[cOff + 1]
          let az = a.cdof[cOff + 2]
          let lx = a.cdof[cOff + 3]
          let ly = a.cdof[cOff + 4]
          let lz = a.cdof[cOff + 5]
          var jpx = 0.0'f32
          var jpy = 0.0'f32
          var jpz = 0.0'f32
          var jrx = 0.0'f32
          var jry = 0.0'f32
          var jrz = 0.0'f32
          if aff2:
            jpx += lx + (ay * off2z - az * off2y)
            jpy += ly + (az * off2x - ax * off2z)
            jpz += lz + (ax * off2y - ay * off2x)
            jrx += ax
            jry += ay
            jrz += az
          if aff1:
            jpx -= lx + (ay * off1z - az * off1y)
            jpy -= ly + (az * off1x - ax * off1z)
            jpz -= lz + (ax * off1y - ay * off1x)
            jrx -= ax
            jry -= ay
            jrz -= az
          jp0[i] = jpx * f0x + jpy * f0y + jpz * f0z
          jp1[i] = jpx * f1x + jpy * f1y + jpz * f1z
          jp2[i] = jpx * f2x + jpy * f2y + jpz * f2z
          if needRot:
            jr0[i] = jrx * f0x + jry * f0y + jrz * f0z
            jr1[i] = jrx * f1x + jry * f1y + jrz * f1z
            jr2[i] = jrx * f2x + jry * f2y + jrz * f2z

      let eaddr = int(a.efcAddress[conid * a.nEfcAddr])
      let dimid = efcid - eaddr
      let jOff = (w * a.njmax + efcid) * nvPad
      let qvOff = w * nv

      var jq = 0.0'f32
      if a.elliptic:
        let src =
          case dimid
          of 0: jp0
          of 1: jp1
          of 2: jp2
          of 3: jr0
          of 4: jr1
          else: jr2
        for i in 0 ..< nv:
          let jij = src[i]
          a.J[jOff + i] = jij
          jq += jij * a.qvel[qvOff + i]
      else:
        if condim <= 1:
          for i in 0 ..< nv:
            let jij = jp0[i]
            a.J[jOff + i] = jij
            jq += jij * a.qvel[qvOff + i]
        else:
          let dimid2 = dimid div 2 + 1
          let frii = a.contactFriction[conid * 5 + dimid2 - 1]
          let sign = frii * (1.0'f32 - 2.0'f32 * float32(dimid and 1))
          let tang =
            case dimid2
            of 1: jp1
            of 2: jp2
            of 3: jr0
            of 4: jr1
            else: jr2
          for i in 0 ..< nv:
            let jij = jp0[i] + sign * tang[i]
            a.J[jOff + i] = jij
            jq += jij * a.qvel[qvOff + i]
      if nv < nvPad:
        zeroMem(addr a.J[jOff + nv], (nvPad - nv) * sizeof(float32))

      a.jqvel[w * a.njmax + efcid] = jq

proc contactJacDense*(
  nworld, nv, nvPad, njmax, nbody, nEfcAddr, elliptic: int,
  bodyIsdofancestor, bodyRootid, geomBodyid, ne, nf, nl, nefc, qvel, subtreeCom, cdof: uint64,
  efcAddress, efcId, contactDim, contactGeom, contactPos, contactFrame, contactFriction: uint64,
  J, jqvel: uint64,
): void {.exportpy: "contact_jac_dense".} =
  var args = ContactJacArgs(
    nworld: nworld, nv: nv, nvPad: nvPad, njmax: njmax, nbody: nbody, nEfcAddr: nEfcAddr,
    elliptic: elliptic != 0,
    bodyIsdofancestor: toPtr[int32](bodyIsdofancestor), bodyRootid: toPtr[int32](bodyRootid),
    geomBodyid: toPtr[int32](geomBodyid), ne: toPtr[int32](ne), nf: toPtr[int32](nf), nl: toPtr[int32](nl),
    nefc: toPtr[int32](nefc), qvel: toPtr[float32](qvel), subtreeCom: toPtr[float32](subtreeCom),
    cdof: toPtr[float32](cdof), efcAddress: toPtr[int32](efcAddress), efcId: toPtr[int32](efcId),
    contactDim: toPtr[int32](contactDim), contactGeom: toPtr[int32](contactGeom),
    contactPos: toPtr[float32](contactPos), contactFrame: toPtr[float32](contactFrame),
    contactFriction: toPtr[float32](contactFriction), J: toPtr[float32](J), jqvel: toPtr[float32](jqvel),
  )
  runJob(contactJacWorker, addr args, nworld)

# Iterative linesearch for dense pyramidal constraints.

type
  Vec3f = array[3, float32]

func safeDiv(x, y: float32): float32 {.inline.} =
  x / (if y != 0.0'f32: y else: 1.0e-15'f32)

func inBracket(x, y: Vec3f): bool {.inline.} =
  (x[1] < y[1] and y[1] < 0.0'f32) or (x[1] > y[1] and y[1] > 0.0'f32)

func evalFrictionlossPt(x, f, rf, jv, d: float32): Vec3f {.inline.} =
  if (-rf < x) and (x < rf):
    let jvD = jv * d
    [0.5'f32 * d * x * x, jvD * x, jv * jvD]
  elif x <= -rf:
    [f * (-0.5'f32 * rf - x), -f * jv, 0.0'f32]
  else:
    [f * (-0.5'f32 * rf + x), f * jv, 0.0'f32]

func evalFrictionlossCost(x, f, rf, d: float32): float32 {.inline.} =
  if (-rf < x) and (x < rf):
    0.5'f32 * d * x * x
  elif x <= -rf:
    f * (-0.5'f32 * rf - x)
  else:
    f * (-0.5'f32 * rf + x)

type
  LinesearchIterativeArgs = object
    nworld, nv, nvPad, njmax, lsIterations: int
    nTol, nLsTol, nMeaninertia: int
    qfrcSmoothStride, qaccStride, maStride: int
    fuseJv, incremental: bool
    optTolerance, optLsTolerance, statMeaninertia: ptr UncheckedArray[float32]
    ne, nf, nefc: ptr UncheckedArray[int32]
    qfrcSmooth, efcJ, efcD, efcFrictionloss: ptr UncheckedArray[float32]
    searchUnchanged, done, lsExhausted: ptr UncheckedArray[bool]
    jaref, search, searchDot, mv, jv: ptr UncheckedArray[float32]
    qacc, efcMa, improvement, alpha: ptr UncheckedArray[float32]
    overflow: ptr UncheckedArray[int32]

proc linesearchPyramidalWorld(
  w, jBase, nv, nvPad, njmax, lsIterations: int,
  nTol, nLsTol, nMeaninertia, qfrcSmoothStride, qaccStride, maStride: int,
  fuseJv, incremental: bool,
  optTolerance, optLsTolerance, statMeaninertia: ptr UncheckedArray[float32],
  neArr, nfArr, nefcArr: ptr UncheckedArray[int32],
  qfrcSmooth, efcJ, efcD, efcFrictionloss: ptr UncheckedArray[float32],
  searchUnchanged, lsExhausted: ptr UncheckedArray[bool],
  jaref, search, searchDot, mv, jv: ptr UncheckedArray[float32],
  qacc, efcMa, improvement, alpha: ptr UncheckedArray[float32],
  overflow: ptr UncheckedArray[int32],
) {.inline.} =
  if true:
    let ne = int(neArr[w])
    let nf = int(nfArr[w])
    let nefc = min(njmax, int(nefcArr[w]))
    let efcOff = w * njmax
    let sOff = w * nv
    let mvOff = w * nv
    let maOff = w * maStride
    let qsOff = w * qfrcSmoothStride
    let qaOff = w * qaccStride

    if fuseJv:
      let recomputeJv = if incremental: not searchUnchanged[w] else: true
      if recomputeJv:
        let jWorldOff = jBase
        for efcid in 0 ..< nefc:
          let jRow = jWorldOff + efcid * nvPad
          var jvVal = 0.0'f32
          for i in 0 ..< nv:
            jvVal += efcJ[jRow + i] * search[sOff + i]
          jv[efcOff + efcid] = jvVal

    let tolerance = optTolerance[w mod nTol]
    let lsTolerance = optLsTolerance[w mod nLsTol]
    let snorm = sqrt(searchDot[w])
    let meaninertia = statMeaninertia[w mod nMeaninertia]
    let scale = meaninertia * float32(nv)
    let gtol = max(tolerance * lsTolerance * snorm * scale, 1.0e-6'f32)

    var gauss1 = 0.0'f32
    var gauss2 = 0.0'f32
    for i in 0 ..< nv:
      let s = search[sOff + i]
      gauss1 += s * (efcMa[maOff + i] - qfrcSmooth[qsOff + i])
      gauss2 += 0.5'f32 * s * mv[mvOff + i]

    for efcid in 0 ..< ne:
      let jvVal = jv[efcOff + efcid]
      let jvD = jvVal * efcD[efcOff + efcid]
      gauss1 += jvD * jaref[efcOff + efcid]
      gauss2 += 0.5'f32 * jvVal * jvD

    var rows0 = 0.0'f32
    var rows1 = 0.0'f32
    var rows2 = 0.0'f32
    let nenf = min(ne + nf, nefc)
    for efcid in ne ..< nenf:
      let d = efcD[efcOff + efcid]
      let f = efcFrictionloss[efcOff + efcid]
      let jr = jaref[efcOff + efcid]
      let jvVal = jv[efcOff + efcid]
      let rf = safeDiv(f, d)
      let pt = evalFrictionlossPt(jr, f, rf, jvVal, d)
      rows0 += pt[0]
      rows1 += pt[1]
      rows2 += pt[2]

    for efcid in nenf ..< nefc:
      let jr = jaref[efcOff + efcid]
      if jr < 0.0'f32:
        let d = efcD[efcOff + efcid]
        let jvVal = jv[efcOff + efcid]
        let jvD = jvVal * d
        rows0 += 0.5'f32 * d * jr * jr
        rows1 += jvD * jr
        rows2 += jvVal * jvD

    let p01 = gauss1 + rows1
    let p02 = 2.0'f32 * gauss2 + rows2
    let p0Delta: Vec3f = [0.0'f32, p01, p02]

    let q1Abs = sqrt(2.0'f32 * max(rows0, 0.0'f32) * max(rows2, 0.0'f32)) + abs(gauss1)
    let gtolAccept = max(gtol, AlphaNoiseEps * q1Abs)
    let noiseFloor =
      if incremental:
        AlphaNoiseEps * max(1.0'f32, safeDiv(q1Abs, p02))
      else:
        0.0'f32

    let loAlphaIn = -safeDiv(p01, p02)
    var loIn0 = loAlphaIn * (loAlphaIn * gauss2 + gauss1)
    var loIn1 = 2.0'f32 * loAlphaIn * gauss2 + gauss1
    var loIn2 = 2.0'f32 * gauss2

    for efcid in ne ..< nenf:
      let d = efcD[efcOff + efcid]
      let f = efcFrictionloss[efcOff + efcid]
      let jr = jaref[efcOff + efcid]
      let jvVal = jv[efcOff + efcid]
      let rf = safeDiv(f, d)
      let x = jr + loAlphaIn * jvVal
      let pt = evalFrictionlossPt(x, f, rf, jvVal, d)
      let c0 = evalFrictionlossCost(jr, f, rf, d)
      loIn0 += pt[0] - c0
      loIn1 += pt[1]
      loIn2 += pt[2]

    for efcid in nenf ..< nefc:
      let d = efcD[efcOff + efcid]
      let jr = jaref[efcOff + efcid]
      let jvVal = jv[efcOff + efcid]
      let x = jr + loAlphaIn * jvVal
      let quad0 = 0.5'f32 * d * jr * jr
      let cost0 = if jr < 0.0'f32: quad0 else: 0.0'f32
      if x < 0.0'f32:
        let offset = quad0 - cost0
        let jvD = jvVal * d
        let hess = jvVal * jvD
        let alphaH = loAlphaIn * hess
        loIn0 += loAlphaIn * (jvD * jr + 0.5'f32 * alphaH) + offset
        loIn1 += jvD * jr + alphaH
        loIn2 += hess
      else:
        loIn0 -= cost0

    let loIn: Vec3f = [loIn0, loIn1, loIn2]
    let initialConverged = abs(loIn[1]) < gtolAccept and loIn[0] < 0.0'f32
    var lsConverged = initialConverged
    var alphaVal = 0.0'f32
    var improvementVal = 0.0'f32

    if not initialConverged:
      let loLess = loIn[1] < p01
      var lo = if loLess: loIn else: p0Delta
      var loAlpha = if loLess: loAlphaIn else: 0.0'f32
      var hi = if loLess: p0Delta else: loIn
      var hiAlpha = if loLess: 0.0'f32 else: loAlphaIn

      for iter in 0 ..< lsIterations:
        let loNextAlpha = loAlpha - safeDiv(lo[1], lo[2])
        let hiNextAlpha = hiAlpha - safeDiv(hi[1], hi[2])
        let midAlpha = 0.5'f32 * (loAlpha + hiAlpha)

        let gHess = 2.0'f32 * gauss2
        let loAq2 = loNextAlpha * gauss2
        let hiAq2 = hiNextAlpha * gauss2
        let midAq2 = midAlpha * gauss2
        var loNext: Vec3f = [loNextAlpha * loAq2 + loNextAlpha * gauss1, 2.0'f32 * loAq2 + gauss1, gHess]
        var hiNext: Vec3f = [hiNextAlpha * hiAq2 + hiNextAlpha * gauss1, 2.0'f32 * hiAq2 + gauss1, gHess]
        var mid: Vec3f = [midAlpha * midAq2 + midAlpha * gauss1, 2.0'f32 * midAq2 + gauss1, gHess]

        for efcid in ne ..< nenf:
          let d = efcD[efcOff + efcid]
          let f = efcFrictionloss[efcOff + efcid]
          let jr = jaref[efcOff + efcid]
          let jvVal = jv[efcOff + efcid]
          let rf = safeDiv(f, d)
          let c0 = evalFrictionlossCost(jr, f, rf, d)
          let pLo = evalFrictionlossPt(jr + loNextAlpha * jvVal, f, rf, jvVal, d)
          let pHi = evalFrictionlossPt(jr + hiNextAlpha * jvVal, f, rf, jvVal, d)
          let pMid = evalFrictionlossPt(jr + midAlpha * jvVal, f, rf, jvVal, d)
          loNext[0] += pLo[0] - c0
          loNext[1] += pLo[1]
          loNext[2] += pLo[2]
          hiNext[0] += pHi[0] - c0
          hiNext[1] += pHi[1]
          hiNext[2] += pHi[2]
          mid[0] += pMid[0] - c0
          mid[1] += pMid[1]
          mid[2] += pMid[2]

        for efcid in nenf ..< nefc:
          let d = efcD[efcOff + efcid]
          let jr = jaref[efcOff + efcid]
          let jvVal = jv[efcOff + efcid]
          let xLo = jr + loNextAlpha * jvVal
          let xHi = jr + hiNextAlpha * jvVal
          let xMid = jr + midAlpha * jvVal
          let quad0 = 0.5'f32 * d * jr * jr
          let cost0 = if jr < 0.0'f32: quad0 else: 0.0'f32
          let offset = quad0 - cost0
          let jvD = jvVal * d
          let grad0 = jvD * jr
          let hess = jvVal * jvD
          if xLo < 0.0'f32:
            let ah = loNextAlpha * hess
            loNext[0] += loNextAlpha * (grad0 + 0.5'f32 * ah) + offset
            loNext[1] += grad0 + ah
            loNext[2] += hess
          else:
            loNext[0] -= cost0
          if xHi < 0.0'f32:
            let ah = hiNextAlpha * hess
            hiNext[0] += hiNextAlpha * (grad0 + 0.5'f32 * ah) + offset
            hiNext[1] += grad0 + ah
            hiNext[2] += hess
          else:
            hiNext[0] -= cost0
          if xMid < 0.0'f32:
            let ah = midAlpha * hess
            mid[0] += midAlpha * (grad0 + 0.5'f32 * ah) + offset
            mid[1] += grad0 + ah
            mid[2] += hess
          else:
            mid[0] -= cost0

        let convLo = abs(loNext[1]) < gtolAccept and loNext[0] < 0.0'f32
        let convHi = abs(hiNext[1]) < gtolAccept and hiNext[0] < 0.0'f32
        let convMid = abs(mid[1]) < gtolAccept and mid[0] < 0.0'f32
        let converged = convLo or convHi or convMid
        var swapLo = false
        var swapHi = false
        if converged:
          var convPt: Vec3f = [MjMaxVal, 0.0'f32, 0.0'f32]
          var convAlpha = 0.0'f32
          if convLo and loNext[0] < convPt[0]:
            convPt = loNext
            convAlpha = loNextAlpha
          if convHi and hiNext[0] < convPt[0]:
            convPt = hiNext
            convAlpha = hiNextAlpha
          if convMid and mid[0] < convPt[0]:
            convPt = mid
            convAlpha = midAlpha
          lo = convPt
          loAlpha = convAlpha
          hi = convPt
          hiAlpha = convAlpha
        else:
          let swapLoLoNext = inBracket(lo, loNext)
          if swapLoLoNext:
            lo = loNext
            loAlpha = loNextAlpha
          let swapLoMid = inBracket(lo, mid)
          if swapLoMid:
            lo = mid
            loAlpha = midAlpha
          let swapLoHiNext = inBracket(lo, hiNext)
          if swapLoHiNext:
            lo = hiNext
            loAlpha = hiNextAlpha
          swapLo = swapLoLoNext or swapLoMid or swapLoHiNext

          let swapHiHiNext = inBracket(hi, hiNext) or (hi[1] < 0.0'f32 and hiNext[1] > 0.0'f32)
          if swapHiHiNext:
            hi = hiNext
            hiAlpha = hiNextAlpha
          let swapHiMid = inBracket(hi, mid)
          if swapHiMid:
            hi = mid
            hiAlpha = midAlpha
          let swapHiLoNext = inBracket(hi, loNext)
          if swapHiLoNext:
            hi = loNext
            hiAlpha = loNextAlpha
          swapHi = swapHiHiNext or swapHiMid or swapHiLoNext

        let lsDone =
          converged or
          (not swapLo and not swapHi) or
          (lo[0] < 0.0'f32 and lo[1] < 0.0'f32 and lo[1] > -gtol) or
          (hi[0] < 0.0'f32 and hi[1] > 0.0'f32 and hi[1] < gtol)

        let improved = lo[0] < 0.0'f32 or hi[0] < 0.0'f32
        if improved:
          let loBetter = lo[0] < hi[0]
          alphaVal = if loBetter: loAlpha else: hiAlpha
          improvementVal = -(if loBetter: lo[0] else: hi[0])

        if lsDone:
          lsConverged = true
          break
    else:
      alphaVal = loAlphaIn
      improvementVal = -loIn[0]

    for i in 0 ..< nv:
      qacc[qaOff + i] += alphaVal * search[sOff + i]
      efcMa[maOff + i] += alphaVal * mv[mvOff + i]

    for efcid in 0 ..< nefc:
      jaref[efcOff + efcid] += alphaVal * jv[efcOff + efcid]

    improvement[w] = improvementVal
    alpha[w] = alphaVal
    if incremental:
      lsExhausted[w] = abs(alphaVal) < noiseFloor
    if not lsConverged:
      overflow[w] = overflow[w] or OverflowLsIterations

proc linesearchIterativeWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr LinesearchIterativeArgs](arg)
  while true:
    let w = claimTask()
    if w >= a.nworld:
      break
    if a.done[w]:
      continue
    linesearchPyramidalWorld(
      w, w * a.njmax * a.nvPad, a.nv, a.nvPad, a.njmax, a.lsIterations,
      a.nTol, a.nLsTol, a.nMeaninertia, a.qfrcSmoothStride, a.qaccStride, a.maStride,
      a.fuseJv, a.incremental,
      a.optTolerance, a.optLsTolerance, a.statMeaninertia,
      a.ne, a.nf, a.nefc,
      a.qfrcSmooth, a.efcJ, a.efcD, a.efcFrictionloss,
      a.searchUnchanged, a.lsExhausted,
      a.jaref, a.search, a.searchDot, a.mv, a.jv,
      a.qacc, a.efcMa, a.improvement, a.alpha,
      a.overflow,
    )

proc linesearchIterativeDense*(
  nworld, nv, nvPad, njmax, lsIterations, fuseJv, incremental: int,
  nTol, nLsTol, nMeaninertia, qfrcSmoothStride, qaccStride, maStride: int,
  optTolerance, optLsTolerance, statMeaninertia: uint64,
  ne, nf, nefc, qfrcSmooth, efcJ, efcD, efcFrictionloss: uint64,
  searchUnchanged, jaref, search, searchDot, mv, jv, done: uint64,
  qacc, efcMa, overflow, improvement, alpha, lsExhausted: uint64,
): void {.exportpy: "linesearch_iterative_dense".} =
  var args = LinesearchIterativeArgs(
    nworld: nworld, nv: nv, nvPad: nvPad, njmax: njmax, lsIterations: lsIterations,
    nTol: nTol, nLsTol: nLsTol, nMeaninertia: nMeaninertia,
    qfrcSmoothStride: qfrcSmoothStride, qaccStride: qaccStride, maStride: maStride,
    fuseJv: fuseJv != 0, incremental: incremental != 0,
    optTolerance: toPtr[float32](optTolerance), optLsTolerance: toPtr[float32](optLsTolerance),
    statMeaninertia: toPtr[float32](statMeaninertia),
    ne: toPtr[int32](ne), nf: toPtr[int32](nf), nefc: toPtr[int32](nefc),
    qfrcSmooth: toPtr[float32](qfrcSmooth), efcJ: toPtr[float32](efcJ),
    efcD: toPtr[float32](efcD), efcFrictionloss: toPtr[float32](efcFrictionloss),
    searchUnchanged: toPtr[bool](searchUnchanged), done: toPtr[bool](done),
    lsExhausted: toPtr[bool](lsExhausted),
    jaref: toPtr[float32](jaref), search: toPtr[float32](search),
    searchDot: toPtr[float32](searchDot), mv: toPtr[float32](mv), jv: toPtr[float32](jv),
    qacc: toPtr[float32](qacc), efcMa: toPtr[float32](efcMa),
    overflow: toPtr[int32](overflow), improvement: toPtr[float32](improvement),
    alpha: toPtr[float32](alpha),
  )
  runJob(linesearchIterativeWorker, addr args, nworld)

# Fused full-solve Newton worker for dense pyramidal constraints.

type
  NewtonSolveArgs = object
    nworld, nv, nvPad, njmax, nC, optIterations, lsIterations: int
    nTol, nLsTol, nMeaninertia: int
    qfrcSmoothStride, qaccSmoothStride, qaccWarmstartStride, qaccStride, qfrcConstraintStride, maStride: int
    warmstart, hasHfactor, tree: bool
    optTolerance, optLsTolerance, statMeaninertia: ptr UncheckedArray[float32]
    mColind, mHinitI, mMulmRowadr, mMulmCol, mMulmMadr: ptr UncheckedArray[int32]
    parent, descPtr, descIdx: ptr UncheckedArray[int32]
    anc: ptr UncheckedArray[int8]
    depth: ptr UncheckedArray[int32]
    sparse: bool
    jRowadr, jRownnz, jColind: ptr UncheckedArray[int32]
    jNnz: int
    ne, nf, nefc, solverNiter, overflow: ptr UncheckedArray[int32]
    M, qfrcSmooth, qaccSmooth, qaccWarmstart, qacc, qfrcConstraint: ptr UncheckedArray[float32]
    efcJ, efcD, efcAref, efcFrictionloss, efcForce, efcMa: ptr UncheckedArray[float32]
    efcState, quadChangedIds, quadChangedCount, stateChangedCount: ptr UncheckedArray[int32]
    jaref, search, searchDot, mv, jv, grad, gradDot, newtonDecrement, gradScale: ptr UncheckedArray[float32]
    improvement, alpha, H, hfactor: ptr UncheckedArray[float32]
    searchUnchanged, done, lsExhausted: ptr UncheckedArray[bool]

func evalPyramidalRow(efcid, ne, nenf: int, jaref, d, f: float32): (float32, int32) {.inline.} =
  if efcid < ne:
    (-d * jaref, Quadratic)
  elif efcid < nenf:
    let rf = safeDiv(f, d)
    if jaref <= -rf:
      (f, LinearNeg)
    elif jaref >= rf:
      (-f, LinearPos)
    else:
      (-d * jaref, Quadratic)
  else:
    if jaref >= 0.0'f32:
      (0.0'f32, Satisfied)
    else:
      (-d * jaref, Quadratic)

proc newtonSolveWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr NewtonSolveArgs](arg)
  let nv = a.nv
  let nvPad = a.nvPad
  let njmax = a.njmax
  let hScratch = scratchNoZero(hCopyBuf, 2 * nv * nvPad)
  let hLocal = hScratch
  let factorPtr = cast[ptr UncheckedArray[float32]](addr hScratch[nv * nvPad])

  while true:
    let w = claimTask()
    if w >= a.nworld:
      break

    var jrow {.noinit.}: array[JRowMax, float32]
    var treeWork {.noinit.}: array[JRowMax, float32]
    let treeWorkPtr = cast[ptr UncheckedArray[float32]](addr treeWork[0])

    let ne = int(a.ne[w])
    let nf = int(a.nf[w])
    let nefc = min(njmax, int(a.nefc[w]))
    let nenf = min(ne + nf, nefc)

    let qaOff = w * a.qaccStride
    let qasOff = w * a.qaccSmoothStride
    let qawOff = w * a.qaccWarmstartStride
    let qsOff = w * a.qfrcSmoothStride
    let qcOff = w * a.qfrcConstraintStride
    let maOff = w * a.maStride
    let mOff = w * a.nC
    let efcOff = w * njmax
    var jWorldOff = w * njmax * nvPad
    let gradOff = w * nvPad
    let sOff = w * nv
    let mvOff = w * nv
    var jPtr = a.efcJ
    if a.sparse and nefc > 0:
      let jd = scratchNoZero(jSparseBuf, nefc * nvPad)
      let jW = w * a.jNnz
      for r in 0 ..< nefc:
        let row = r * nvPad
        zeroMem(addr jd[row], nvPad * sizeof(float32))
        let radr = int(a.jRowadr[w * njmax + r])
        let rn = int(a.jRownnz[w * njmax + r])
        for k in 0 ..< rn:
          jd[row + int(a.jColind[jW + radr + k])] = a.efcJ[jW + radr + k]
      jPtr = jd
      jWorldOff = 0
    let useTree = a.tree and nv <= JRowMax and a.parent != nil and a.anc != nil and a.depth != nil and
      denseRowsAreChains(jPtr, jWorldOff, nv, nvPad, nefc, a.anc, a.depth)

    if a.warmstart and int(a.nefc[w]) > 0:
      copyMem(addr a.qacc[qaOff], addr a.qaccWarmstart[qawOff], nv * sizeof(float32))
    else:
      copyMem(addr a.qacc[qaOff], addr a.qaccSmooth[qasOff], nv * sizeof(float32))

    a.solverNiter[w] = 0
    a.done[w] = false
    a.searchDot[w] = 0.0'f32
    a.searchUnchanged[w] = false
    a.lsExhausted[w] = false

    # Jaref = J @ qacc - aref
    for efcid in 0 ..< nefc:
      let jRow = jWorldOff + efcid * nvPad
      var jr = 0.0'f32
      for i in 0 ..< nv:
        jr += jPtr[jRow + i] * a.qacc[qaOff + i]
      a.jaref[efcOff + efcid] = jr - a.efcAref[efcOff + efcid]

    # Ma = M @ qacc
    for i in 0 ..< nv:
      var acc = 0.0'f32
      let rStart = int(a.mMulmRowadr[i])
      let rEnd = int(a.mMulmRowadr[i + 1])
      for k in rStart ..< rEnd:
        acc += a.M[mOff + int(a.mMulmMadr[k])] * a.qacc[qaOff + int(a.mMulmCol[k])]
      a.efcMa[maOff + i] = acc

    # Initial constraint evaluation & qfrc_constraint = J^T @ force
    zeroMem(addr a.qfrcConstraint[qcOff], nv * sizeof(float32))
    for efcid in 0 ..< nefc:
      let d = a.efcD[efcOff + efcid]
      let f = if efcid >= ne and efcid < nenf: a.efcFrictionloss[efcOff + efcid] else: 0.0'f32
      let (force, state) = evalPyramidalRow(efcid, ne, nenf, a.jaref[efcOff + efcid], d, f)
      a.efcForce[efcOff + efcid] = force
      a.efcState[efcOff + efcid] = state
      if force != 0.0'f32:
        let jRow = jWorldOff + efcid * nvPad
        for i in 0 ..< nv:
          a.qfrcConstraint[qcOff + i] += force * jPtr[jRow + i]

    # Initial gradient & H = M + J^T D_quad J in thread-local L1 buffer
    var gDot = 0.0'f32
    for i in 0 ..< nv:
      let g = a.efcMa[maOff + i] - a.qfrcSmooth[qsOff + i] - a.qfrcConstraint[qcOff + i]
      a.grad[gradOff + i] = g
      gDot += g * g
    a.gradDot[w] = gDot
    a.gradScale[w] = 1.0'f32

    zeroMem(hLocal, nv * nvPad * sizeof(float32))
    for e in 0 ..< a.nC:
      let col = int(a.mColind[e])
      let row = int(a.mHinitI[e])
      hLocal[col * nvPad + row] += a.M[mOff + e]

    for efcid in 0 ..< nefc:
      if a.efcState[efcOff + efcid] != Quadratic:
        continue
      let dk = a.efcD[efcOff + efcid]
      if dk == 0.0'f32:
        continue
      let jk = jWorldOff + efcid * nvPad
      if nvPad <= JRowMax:
        # copy the row locally: with raw float pointers gcc must assume the
        # hLocal stores alias the J row and cannot vectorize the update
        for q in 0 ..< nv:
          jrow[q] = jPtr[jk + q]
        for i in 0 ..< nv:
          let ji = dk * jrow[i]
          if ji == 0.0'f32:
            continue
          let hRow = i * nvPad
          for jj in i ..< nv:
            hLocal[hRow + jj] += ji * jrow[jj]
      else:
        for i in 0 ..< nv:
          let ji = dk * jPtr[jk + i]
          if ji == 0.0'f32:
            continue
          let hRow = i * nvPad
          for jj in i ..< nv:
            hLocal[hRow + jj] += ji * jPtr[jk + jj]

    for i in 0 ..< nv:
      let rowOff = i * nvPad + i
      copyMem(addr factorPtr[rowOff], addr hLocal[rowOff], (nv - i) * sizeof(float32))
    if useTree:
      treeFactorize(factorPtr, nv, nvPad, a.parent, a.descPtr, a.descIdx)
    else:
      choleskyFactorize(factorPtr, nv, nvPad)

    let xw = at(a.search, sOff)
    for i in 0 ..< nv:
      xw[i] = a.grad[gradOff + i]
    if useTree:
      treeSolve(factorPtr, nv, nvPad, a.parent, a.descPtr, a.descIdx, xw, xw, treeWorkPtr)
    else:
      choleskySolve(factorPtr, nv, nvPad, xw)

    var sd = 0.0'f32
    var nd = 0.0'f32
    for i in 0 ..< nv:
      let xi = xw[i]
      sd += xi * xi
      nd += a.grad[gradOff + i] * xi
      xw[i] = -xi
    a.searchDot[w] = sd
    a.newtonDecrement[w] = nd

    let tolerance = a.optTolerance[w mod a.nTol]
    let meaninertia = a.statMeaninertia[w mod a.nMeaninertia]
    let invScale = 1.0'f32 / (meaninertia * float32(nv))

    for iter in 0 ..< a.optIterations:
      if a.done[w]:
        break

      if not a.searchUnchanged[w]:
        for i in 0 ..< nv:
          var acc = 0.0'f32
          let rStart = int(a.mMulmRowadr[i])
          let rEnd = int(a.mMulmRowadr[i + 1])
          for k in rStart ..< rEnd:
            acc += a.M[mOff + int(a.mMulmMadr[k])] * a.search[sOff + int(a.mMulmCol[k])]
          a.mv[mvOff + i] = acc
        for efcid in 0 ..< nefc:
          let jRow = jWorldOff + efcid * nvPad
          var jvVal = 0.0'f32
          for i in 0 ..< nv:
            jvVal += jPtr[jRow + i] * a.search[sOff + i]
          a.jv[efcOff + efcid] = jvVal

      linesearchPyramidalWorld(
        w, jWorldOff, nv, nvPad, njmax, a.lsIterations,
        a.nTol, a.nLsTol, a.nMeaninertia, a.qfrcSmoothStride, a.qaccStride, a.maStride,
        false, true,
        a.optTolerance, a.optLsTolerance, a.statMeaninertia,
        a.ne, a.nf, a.nefc,
        a.qfrcSmooth, jPtr, a.efcD, a.efcFrictionloss,
        a.searchUnchanged, a.lsExhausted,
        a.jaref, a.search, a.searchDot, a.mv, a.jv,
        a.qacc, a.efcMa, a.improvement, a.alpha,
        a.overflow,
      )

      var quadChanged = 0
      var stateChanged = if a.lsExhausted[w]: 1'i32 else: 0'i32
      for efcid in 0 ..< nefc:
        let oldState = a.efcState[efcOff + efcid]
        let d = a.efcD[efcOff + efcid]
        let f = if efcid >= ne and efcid < nenf: a.efcFrictionloss[efcOff + efcid] else: 0.0'f32
        let (force, newState) = evalPyramidalRow(efcid, ne, nenf, a.jaref[efcOff + efcid], d, f)
        a.efcForce[efcOff + efcid] = force
        a.efcState[efcOff + efcid] = newState
        if (oldState == Quadratic) != (newState == Quadratic):
          a.quadChangedIds[efcOff + quadChanged] = int32(efcid)
          inc quadChanged
        if oldState != newState:
          inc stateChanged

      a.quadChangedCount[w] = int32(quadChanged)
      a.stateChangedCount[w] = stateChanged

      if stateChanged == 0:
        a.searchUnchanged[w] = true
        let sigma = a.gradScale[w]
        let newSigma = sigma - a.alpha[w]
        let ratio = if sigma != 0.0'f32: newSigma / sigma else: 0.0'f32
        let ratioSq = ratio * ratio
        a.gradDot[w] *= ratioSq
        a.newtonDecrement[w] *= ratioSq
        a.gradScale[w] = newSigma
      else:
        a.searchUnchanged[w] = false
        a.gradScale[w] = 1.0'f32
        zeroMem(addr a.qfrcConstraint[qcOff], nv * sizeof(float32))
        for efcid in 0 ..< nefc:
          let force = a.efcForce[efcOff + efcid]
          if force != 0.0'f32:
            let jRow = jWorldOff + efcid * nvPad
            for i in 0 ..< nv:
              a.qfrcConstraint[qcOff + i] += force * jPtr[jRow + i]

        var gDotIter = 0.0'f32
        for i in 0 ..< nv:
          let g = a.efcMa[maOff + i] - a.qfrcSmooth[qsOff + i] - a.qfrcConstraint[qcOff + i]
          a.grad[gradOff + i] = g
          gDotIter += g * g
        a.gradDot[w] = gDotIter

        if quadChanged > 0:
          for cIdx in 0 ..< quadChanged:
            let efcid = int(a.quadChangedIds[efcOff + cIdx])
            let d = a.efcD[efcOff + efcid]
            let dk = if a.efcState[efcOff + efcid] == Quadratic: d else: -d
            if dk == 0.0'f32:
              continue
            let jk = jWorldOff + efcid * nvPad
            if nvPad <= JRowMax:
              for q in 0 ..< nv:
                jrow[q] = jPtr[jk + q]
              for i in 0 ..< nv:
                let ji = dk * jrow[i]
                if ji == 0.0'f32:
                  continue
                let hRow = i * nvPad
                for jj in i ..< nv:
                  hLocal[hRow + jj] += ji * jrow[jj]
            else:
              for i in 0 ..< nv:
                let ji = dk * jPtr[jk + i]
                if ji == 0.0'f32:
                  continue
                let hRow = i * nvPad
                for jj in i ..< nv:
                  hLocal[hRow + jj] += ji * jPtr[jk + jj]
          for i in 0 ..< nv:
            let rowOff = i * nvPad + i
            copyMem(addr factorPtr[rowOff], addr hLocal[rowOff], (nv - i) * sizeof(float32))
          if useTree:
            treeFactorize(factorPtr, nv, nvPad, a.parent, a.descPtr, a.descIdx)
          else:
            choleskyFactorize(factorPtr, nv, nvPad)

        for i in 0 ..< nv:
          xw[i] = a.grad[gradOff + i]
        if useTree:
          treeSolve(factorPtr, nv, nvPad, a.parent, a.descPtr, a.descIdx, xw, xw, treeWorkPtr)
        else:
          choleskySolve(factorPtr, nv, nvPad, xw)

        var sdIter = 0.0'f32
        var ndIter = 0.0'f32
        for i in 0 ..< nv:
          let xi = xw[i]
          sdIter += xi * xi
          ndIter += a.grad[gradOff + i] * xi
          xw[i] = -xi
        a.searchDot[w] = sdIter
        a.newtonDecrement[w] = ndIter

      a.solverNiter[w] += 1
      let impr = a.improvement[w] * invScale
      let gradNorm = sqrt(a.gradDot[w]) * invScale
      let modelImpr = 0.5'f32 * a.newtonDecrement[w] * invScale
      let isDone =
        (a.alpha[w] == 0.0'f32) or
        (impr > 0.0'f32 and impr < tolerance) or
        (gradNorm < tolerance) or
        (modelImpr < tolerance)
      if isDone or int(a.solverNiter[w]) == a.optIterations:
        if not isDone and int(a.solverNiter[w]) == a.optIterations:
          a.overflow[w] = a.overflow[w] or OverflowIterations
        a.done[w] = true
        break

    let gs = a.gradScale[w]
    for i in 0 ..< nv:
      a.qfrcConstraint[qcOff + i] = a.efcMa[maOff + i] - a.qfrcSmooth[qsOff + i] - gs * a.grad[gradOff + i]

proc newtonSolveDense*(
  nworld, nv, nvPad, njmax, nC, optIterations, lsIterations, warmstart: int,
  nTol, nLsTol, nMeaninertia: int,
  qfrcSmoothStride, qaccSmoothStride, qaccWarmstartStride, qaccStride, qfrcConstraintStride, maStride: int,
  optTolerance, optLsTolerance, statMeaninertia: uint64,
  mColind, mHinitI, mMulmRowadr, mMulmCol, mMulmMadr: uint64,
  ne, nf, nefc, solverNiter, overflow: uint64,
  M, qfrcSmooth, qaccSmooth, qaccWarmstart, qacc, qfrcConstraint: uint64,
  efcJ, efcD, efcAref, efcFrictionloss, efcForce, efcMa, efcState: uint64,
  quadChangedIds, quadChangedCount, stateChangedCount: uint64,
  jaref, search, searchDot, mv, jv, grad, gradDot, newtonDecrement, gradScale: uint64,
  improvement, alpha, H, hfactor: uint64,
  searchUnchanged, done, lsExhausted: uint64,
  tree: int, parent, descPtr, descIdx, anc, depth: uint64,
  sparse: int, jRowadr, jRownnz, jColind: uint64, jNnz: int,
): void {.exportpy: "newton_solve_dense".} =
  var args = NewtonSolveArgs(
    nworld: nworld, nv: nv, nvPad: nvPad, njmax: njmax, nC: nC,
    optIterations: optIterations, lsIterations: lsIterations,
    nTol: nTol, nLsTol: nLsTol, nMeaninertia: nMeaninertia,
    qfrcSmoothStride: qfrcSmoothStride, qaccSmoothStride: qaccSmoothStride,
    qaccWarmstartStride: qaccWarmstartStride, qaccStride: qaccStride,
    qfrcConstraintStride: qfrcConstraintStride, maStride: maStride,
    warmstart: warmstart != 0, hasHfactor: hfactor != 0,
    optTolerance: toPtr[float32](optTolerance), optLsTolerance: toPtr[float32](optLsTolerance),
    statMeaninertia: toPtr[float32](statMeaninertia),
    mColind: toPtr[int32](mColind), mHinitI: toPtr[int32](mHinitI),
    mMulmRowadr: toPtr[int32](mMulmRowadr), mMulmCol: toPtr[int32](mMulmCol),
    mMulmMadr: toPtr[int32](mMulmMadr),
    ne: toPtr[int32](ne), nf: toPtr[int32](nf), nefc: toPtr[int32](nefc),
    solverNiter: toPtr[int32](solverNiter), overflow: toPtr[int32](overflow),
    M: toPtr[float32](M), qfrcSmooth: toPtr[float32](qfrcSmooth),
    qaccSmooth: toPtr[float32](qaccSmooth), qaccWarmstart: toPtr[float32](qaccWarmstart),
    qacc: toPtr[float32](qacc), qfrcConstraint: toPtr[float32](qfrcConstraint),
    efcJ: toPtr[float32](efcJ), efcD: toPtr[float32](efcD), efcAref: toPtr[float32](efcAref),
    efcFrictionloss: toPtr[float32](efcFrictionloss), efcForce: toPtr[float32](efcForce),
    efcMa: toPtr[float32](efcMa), efcState: toPtr[int32](efcState),
    quadChangedIds: toPtr[int32](quadChangedIds), quadChangedCount: toPtr[int32](quadChangedCount),
    stateChangedCount: toPtr[int32](stateChangedCount),
    jaref: toPtr[float32](jaref), search: toPtr[float32](search),
    searchDot: toPtr[float32](searchDot), mv: toPtr[float32](mv), jv: toPtr[float32](jv),
    grad: toPtr[float32](grad), gradDot: toPtr[float32](gradDot),
    newtonDecrement: toPtr[float32](newtonDecrement), gradScale: toPtr[float32](gradScale),
    improvement: toPtr[float32](improvement), alpha: toPtr[float32](alpha),
    H: toPtr[float32](H), hfactor: toPtr[float32](hfactor),
    searchUnchanged: toPtr[bool](searchUnchanged), done: toPtr[bool](done),
    lsExhausted: toPtr[bool](lsExhausted),
    tree: tree != 0, parent: toPtr[int32](parent),
    descPtr: toPtr[int32](descPtr), descIdx: toPtr[int32](descIdx),
    anc: toPtr[int8](anc), depth: toPtr[int32](depth),
    sparse: sparse != 0,
    jRowadr: toPtr[int32](jRowadr), jRownnz: toPtr[int32](jRownnz),
    jColind: toPtr[int32](jColind), jNnz: jNnz,
  )
  runJob(newtonSolveWorker, addr args, if nworld * nv >= 256: nworld else: 1)

# ==============================================================================
# Step pipeline fast paths: gravity, broadphase, contact rows, kinematics.
#
# Warp's CPU backend runs each kernel on a single lane, so these scalar loops
# spread over the worker pool are a large win.  Quaternions are w-first
# (index 0 = real part, matching `_src/math.py`), matrices are column-major
# with memory order `m[c * 3 + r]`.

const
  BpFlagsCap = 1 shl 24  # store one pass flag per (world, pair); above this, re-filter
  EfcUpdateChunk = 32
  MjMinVal = 1.0e-15'f32
  MjMinImp = 0.0001'f32
  MjMaxImp = 0.9999'f32
  DisableRefsafe = 4096'i32
  ContactConstraint = 1'i32  # types.ContactType.CONSTRAINT
  ConstraintContactFrictionless = 5'i32
  ConstraintContactPyramidal = 6'i32
  GeomTypeMesh = 7'i32
  JointTypeFree = 0'i32
  JointTypeBall = 1'i32
  JointTypeSlide = 2'i32
  JointTypeHinge = 3'i32
  SleepStatic = -1'i32
  SleepAsleep = 0'i32

type
  Quatf = array[4, float32]

func mulQuat(u, v: Quatf): Quatf {.inline.} =
  Quatf([
    u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
    u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
    u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
    u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
  ])

func rotVecQuat(vec: Vec3f, quat: Quatf): Vec3f {.inline.} =
  # mirrors math.rot_vec_quat operation-for-operation (2 * (dot * u), then k * vec, then 2 * s * cross)
  let s = quat[0]
  let u = [quat[1], quat[2], quat[3]]
  let du = u[0] * vec[0] + u[1] * vec[1] + u[2] * vec[2]
  let uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2]
  let k0 = s * s - uu
  let cx = u[1] * vec[2] - u[2] * vec[1]
  let cy = u[2] * vec[0] - u[0] * vec[2]
  let cz = u[0] * vec[1] - u[1] * vec[0]
  let twoS = 2.0'f32 * s
  Vec3f([
    2.0'f32 * (du * u[0]) + k0 * vec[0] + twoS * cx,
    2.0'f32 * (du * u[1]) + k0 * vec[1] + twoS * cy,
    2.0'f32 * (du * u[2]) + k0 * vec[2] + twoS * cz,
  ])

proc fma32(a, b, c: float32): float32 {.importc: "fmaf", header: "<math.h>".}

func vadd(a, b: Vec3f): Vec3f {.inline.} =
  Vec3f([a[0] + b[0], a[1] + b[1], a[2] + b[2]])

func vsub(a, b: Vec3f): Vec3f {.inline.} =
  Vec3f([a[0] - b[0], a[1] - b[1], a[2] - b[2]])

func axisAngleToQuat(axis: Vec3f, angle: float32): Quatf {.inline.} =
  let s = sin(angle * 0.5'f32)
  let c = cos(angle * 0.5'f32)
  Quatf([c, axis[0] * s, axis[1] * s, axis[2] * s])

func normQuat(q: Quatf): Quatf {.inline.} =
  let l = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
  if l > 0.0'f32:
    let inv = 1.0'f32 / l
    return Quatf([q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv])
  Quatf([0.0'f32, 0.0'f32, 0.0'f32, 1.0'f32])

func quatToMat(q: Quatf): array[9, float32] {.inline.} =
  let q00 = q[0] * q[0]
  let q01 = q[0] * q[1]
  let q02 = q[0] * q[2]
  let q03 = q[0] * q[3]
  let q11 = q[1] * q[1]
  let q12 = q[1] * q[2]
  let q13 = q[1] * q[3]
  let q22 = q[2] * q[2]
  let q23 = q[2] * q[3]
  let q33 = q[3] * q[3]
  [
    q00 + q11 - q22 - q33,
    2.0'f32 * (q12 - q03),
    2.0'f32 * (q13 + q02),
    2.0'f32 * (q12 + q03),
    q00 - q11 + q22 - q33,
    2.0'f32 * (q23 - q01),
    2.0'f32 * (q13 - q02),
    2.0'f32 * (q23 + q01),
    q00 - q11 - q22 + q33,
  ]

# ------------------------------------------------------------------------------
# Gravity / gravcomp forces (mirror of passive._gravity_force).

type
  GravityForceArgs = object
    nworld, nbody, nv, qfrcStride: int
    gravityWorlds: int
    gravcompWorlds, gravcompStride: int
    massWorlds, massStride: int
    xiposStride, comStride, cdofStride, isdofStride: int
    gravity, mass, gravcomp: ptr UncheckedArray[float32]
    xipos, subtreeCom, cdof, qfrcGravcomp: ptr UncheckedArray[float32]
    bodyRootid, bodyIsdofancestor: ptr UncheckedArray[int32]

proc gravityForceWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr GravityForceArgs](arg)
  while true:
    let w = claimTask()
    if w >= a.nworld:
      break
    let gOff = (w mod a.gravityWorlds) * 3
    let gx = a.gravity[gOff]
    let gy = a.gravity[gOff + 1]
    let gz = a.gravity[gOff + 2]
    let gcOff = (w mod a.gravcompWorlds) * a.gravcompStride
    let mOff = (w mod a.massWorlds) * a.massStride
    let xiOff = w * a.xiposStride
    let comOff = w * a.comStride
    let cdofOff = w * a.cdofStride
    let qOff = w * a.qfrcStride

    for b in 1 ..< a.nbody:
      let gravcomp = a.gravcomp[gcOff + b]
      if gravcomp == 0.0'f32:
        continue
      let mass = a.mass[mOff + b] * gravcomp
      let fx = -gx * mass
      let fy = -gy * mass
      let fz = -gz * mass
      let root = int(a.bodyRootid[b])
      let ox = a.xipos[xiOff + b * 3] - a.subtreeCom[comOff + root * 3]
      let oy = a.xipos[xiOff + b * 3 + 1] - a.subtreeCom[comOff + root * 3 + 1]
      let oz = a.xipos[xiOff + b * 3 + 2] - a.subtreeCom[comOff + root * 3 + 2]
      let isdofOff = b * a.isdofStride
      for d in 0 ..< a.nv:
        if a.bodyIsdofancestor[isdofOff + d] == 0:
          continue
        let c = cdofOff + d * 6
        let ax = a.cdof[c]
        let ay = a.cdof[c + 1]
        let az = a.cdof[c + 2]
        let jx = a.cdof[c + 3] + (ay * oz - az * oy)
        let jy = a.cdof[c + 4] + (az * ox - ax * oz)
        let jz = a.cdof[c + 5] + (ax * oy - ay * ox)
        a.qfrcGravcomp[qOff + d] += jx * fx + jy * fy + jz * fz

proc gravityForceDense*(
  nworld, nbody, nv, qfrcStride: int,
  gravityWorlds, gravcompWorlds, gravcompStride, massWorlds, massStride: int,
  xiposStride, comStride, cdofStride, isdofStride: int,
  gravity, mass, gravcomp: uint64,
  xipos, subtreeCom, cdof, qfrcGravcomp: uint64,
  bodyRootid, bodyIsdofancestor: uint64,
): void {.exportpy: "gravity_force_dense".} =
  var args = GravityForceArgs(
    nworld: nworld, nbody: nbody, nv: nv, qfrcStride: qfrcStride,
    gravityWorlds: gravityWorlds,
    gravcompWorlds: gravcompWorlds, gravcompStride: gravcompStride,
    massWorlds: massWorlds, massStride: massStride,
    xiposStride: xiposStride, comStride: comStride, cdofStride: cdofStride, isdofStride: isdofStride,
    gravity: toPtr[float32](gravity), mass: toPtr[float32](mass), gravcomp: toPtr[float32](gravcomp),
    xipos: toPtr[float32](xipos), subtreeCom: toPtr[float32](subtreeCom), cdof: toPtr[float32](cdof),
    qfrcGravcomp: toPtr[float32](qfrcGravcomp),
    bodyRootid: toPtr[int32](bodyRootid), bodyIsdofancestor: toPtr[int32](bodyIsdofancestor),
  )
  runJob(gravityForceWorker, addr args, nworld)

# ------------------------------------------------------------------------------
# NxN broadphase (mirror of collision_driver._nxn_broadphase).
#
# Two passes: a parallel count per world, then a parallel fill emitting pairs in
# world-major order, which reproduces the serial Warp CPU ordering exactly.

type
  BroadphaseArgs = object
    nworld, npairs, naconmax, filter: int
    ngeom: int
    rboundWorlds, marginWorlds, gapWorlds, dataidWorlds, aabbWorlds: int
    awakeStride: int
    geomType, geomBodyid: ptr UncheckedArray[int32]
    geomDataid: ptr UncheckedArray[int32]
    geomAabb, geomRbound, geomMargin, geomGap: ptr UncheckedArray[float32]
    nxnPair, nxnPairid: ptr UncheckedArray[int32]
    geomXpos, geomXmat: ptr UncheckedArray[float32]
    bodyAwake, bodyAwakePrev: ptr UncheckedArray[int32]
    ncollision: ptr UncheckedArray[int32]
    counts: ptr UncheckedArray[int32]
    flags: ptr UncheckedArray[uint8]
    collisionPair, collisionPairid, collisionWorldid: ptr UncheckedArray[int32]
    checkMesh, enableSleep, incremental: bool

func matVec(m: array[9, float32], v: Vec3f): Vec3f {.inline.} =
  Vec3f([
    m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
    m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
    m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
  ])


func bpRbound(a: ptr BroadphaseArgs, w, geom: int): float32 {.inline.} =
  a.geomRbound[(w mod a.rboundWorlds) * a.ngeom + geom]

func bpMargin(a: ptr BroadphaseArgs, w, geom: int): float32 {.inline.} =
  a.geomMargin[(w mod a.marginWorlds) * a.ngeom + geom]

func bpGap(a: ptr BroadphaseArgs, w, geom: int): float32 {.inline.} =
  a.geomGap[(w mod a.gapWorlds) * a.ngeom + geom]

func bpXpos(a: ptr BroadphaseArgs, w, geom: int): Vec3f {.inline.} =
  let off = (w * a.ngeom + geom) * 3
  Vec3f([a.geomXpos[off], a.geomXpos[off + 1], a.geomXpos[off + 2]])

func bpXmat(a: ptr BroadphaseArgs, w, geom: int): array[9, float32] {.inline.} =
  let off = (w * a.ngeom + geom) * 9
  [
    a.geomXmat[off], a.geomXmat[off + 1], a.geomXmat[off + 2],
    a.geomXmat[off + 3], a.geomXmat[off + 4], a.geomXmat[off + 5],
    a.geomXmat[off + 6], a.geomXmat[off + 7], a.geomXmat[off + 8],
  ]

func bpSphereFilter(rbound1, rbound2, margin1, margin2: float32, xpos1, xpos2: Vec3f): bool {.inline.} =
  let bound = rbound1 + rbound2 + margin1 + margin2
  let dx = xpos2[0] - xpos1[0]
  let dy = xpos2[1] - xpos1[1]
  let dz = xpos2[2] - xpos1[2]
  dx * dx + dy * dy + dz * dz <= bound * bound

func bpAabbFilter(center1, center2, size1, size2: Vec3f, margin1, margin2: float32, xpos1, xpos2: Vec3f, xmat1, xmat2: array[9, float32]): bool {.inline.} =
  var c1 = matVec(xmat1, center1)
  c1[0] += xpos1[0]
  c1[1] += xpos1[1]
  c1[2] += xpos1[2]
  var c2 = matVec(xmat2, center2)
  c2[0] += xpos2[0]
  c2[1] += xpos2[1]
  c2[2] += xpos2[2]
  let margin = margin1 + margin2

  let signs = [-1.0'f32, 1.0'f32]
  var min1 = [MjMaxVal, MjMaxVal, MjMaxVal]
  var max1 = [-MjMaxVal, -MjMaxVal, -MjMaxVal]
  var min2 = [MjMaxVal, MjMaxVal, MjMaxVal]
  var max2 = [-MjMaxVal, -MjMaxVal, -MjMaxVal]
  for i in 0 ..< 2:
    for j in 0 ..< 2:
      for k in 0 ..< 2:
        let corner1 = Vec3f([signs[i] * size1[0], signs[j] * size1[1], signs[k] * size1[2]])
        let pos1 = matVec(xmat1, corner1)
        let corner2 = Vec3f([signs[i] * size2[0], signs[j] * size2[1], signs[k] * size2[2]])
        let pos2 = matVec(xmat2, corner2)
        for c in 0 ..< 3:
          if pos1[c] > max1[c]: max1[c] = pos1[c]
          if pos1[c] < min1[c]: min1[c] = pos1[c]
          if pos2[c] > max2[c]: max2[c] = pos2[c]
          if pos2[c] < min2[c]: min2[c] = pos2[c]

  for c in 0 ..< 3:
    if c1[c] + max1[c] + margin < c2[c] + min2[c]:
      return false
    if c2[c] + max2[c] + margin < c1[c] + min1[c]:
      return false
  true

func bpObbFilter(center1, center2, size1, size2: Vec3f, margin1, margin2: float32, xpos1, xpos2: Vec3f, xmat1, xmat2: array[9, float32]): bool {.inline.} =
  let margin = margin1 + margin2

  var xcenter = [Vec3f([0.0'f32, 0.0'f32, 0.0'f32]), Vec3f([0.0'f32, 0.0'f32, 0.0'f32])]
  xcenter[0] = matVec(xmat1, center1)
  xcenter[0][0] += xpos1[0]
  xcenter[0][1] += xpos1[1]
  xcenter[0][2] += xpos1[2]
  xcenter[1] = matVec(xmat2, center2)
  xcenter[1][0] += xpos2[0]
  xcenter[1][1] += xpos2[1]
  xcenter[1][2] += xpos2[2]

  # columns of each rotation matrix are the box axes
  var normal = [Vec3f([0.0'f32, 0.0'f32, 0.0'f32]), Vec3f([0.0'f32, 0.0'f32, 0.0'f32]),
                Vec3f([0.0'f32, 0.0'f32, 0.0'f32]), Vec3f([0.0'f32, 0.0'f32, 0.0'f32]),
                Vec3f([0.0'f32, 0.0'f32, 0.0'f32]), Vec3f([0.0'f32, 0.0'f32, 0.0'f32])]
  for k in 0 ..< 3:
    normal[k] = Vec3f([xmat1[k], xmat1[3 + k], xmat1[6 + k]])
    normal[3 + k] = Vec3f([xmat2[k], xmat2[3 + k], xmat2[6 + k]])

  var proj: array[2, float32]
  var radius: array[2, float32]
  for j in 0 ..< 2:
    for k in 0 ..< 3:
      let n = normal[3 * j + k]
      for i in 0 ..< 2:
        proj[i] = xcenter[i][0] * n[0] + xcenter[i][1] * n[1] + xcenter[i][2] * n[2]
        let size = if i == 0: size1 else: size2
        let na = normal[3 * i]
        let nb = normal[3 * i + 1]
        let nc = normal[3 * i + 2]
        radius[i] = abs(size[0] * (na[0] * n[0] + na[1] * n[1] + na[2] * n[2])) +
                    abs(size[1] * (nb[0] * n[0] + nb[1] * n[1] + nb[2] * n[2])) +
                    abs(size[2] * (nc[0] * n[0] + nc[1] * n[1] + nc[2] * n[2]))
      if radius[0] + radius[1] + margin < abs(proj[1] - proj[0]):
        return false
  true

func bpFilterPasses(a: ptr BroadphaseArgs, w, geom1, geom2: int): bool {.inline.} =
  let rbound1 = bpRbound(a, w, geom1)
  let rbound2 = bpRbound(a, w, geom2)
  let emargin1 = bpMargin(a, w, geom1) + bpGap(a, w, geom1)
  let emargin2 = bpMargin(a, w, geom2) + bpGap(a, w, geom2)
  let xpos1 = bpXpos(a, w, geom1)
  let xpos2 = bpXpos(a, w, geom2)

  if rbound1 == 0.0'f32 or rbound2 == 0.0'f32:
    if (a.filter and 1) != 0:  # BroadphaseFilter.PLANE
      if rbound1 == 0.0'f32:
        # plane normal = column 2 of xmat1, row-major memory: (mem[2], mem[5], mem[8])
        let xmat1 = bpXmat(a, w, geom1)
        let dist = (xpos2[0] - xpos1[0]) * xmat1[2] + (xpos2[1] - xpos1[1]) * xmat1[5] + (xpos2[2] - xpos1[2]) * xmat1[8]
        return dist <= rbound2 + emargin1 + emargin2
      else:
        let xmat2 = bpXmat(a, w, geom2)
        let dist = (xpos1[0] - xpos2[0]) * xmat2[2] + (xpos1[1] - xpos2[1]) * xmat2[5] + (xpos1[2] - xpos2[2]) * xmat2[8]
        return dist <= rbound1 + emargin1 + emargin2
  else:
    if (a.filter and 2) != 0:  # SPHERE
      if not bpSphereFilter(rbound1, rbound2, emargin1, emargin2, xpos1, xpos2):
        return false
    if (a.filter and 4) != 0 or (a.filter and 8) != 0:  # AABB / OBB
      let aabbId = w mod a.aabbWorlds
      let base1 = (aabbId * a.ngeom + geom1) * 6
      let base2 = (aabbId * a.ngeom + geom2) * 6
      let center1 = Vec3f([a.geomAabb[base1], a.geomAabb[base1 + 1], a.geomAabb[base1 + 2]])
      let size1 = Vec3f([a.geomAabb[base1 + 3], a.geomAabb[base1 + 4], a.geomAabb[base1 + 5]])
      let center2 = Vec3f([a.geomAabb[base2], a.geomAabb[base2 + 1], a.geomAabb[base2 + 2]])
      let size2 = Vec3f([a.geomAabb[base2 + 3], a.geomAabb[base2 + 4], a.geomAabb[base2 + 5]])
      if (a.filter and 4) != 0:
        let xmat1 = bpXmat(a, w, geom1)
        let xmat2 = bpXmat(a, w, geom2)
        if not bpAabbFilter(center1, center2, size1, size2, emargin1, emargin2, xpos1, xpos2, xmat1, xmat2):
          return false
      if (a.filter and 8) != 0:
        let xmat1 = bpXmat(a, w, geom1)
        let xmat2 = bpXmat(a, w, geom2)
        if not bpObbFilter(center1, center2, size1, size2, emargin1, emargin2, xpos1, xpos2, xmat1, xmat2):
          return false
  true

func bpPairPasses(a: ptr BroadphaseArgs, w, geom1, geom2, e: int): bool {.inline.} =
  if a.checkMesh:
    let type1 = a.geomType[geom1]
    let type2 = a.geomType[geom2]
    let dataidId = w mod a.dataidWorlds
    if (type1 == GeomTypeMesh and a.geomDataid[dataidId * a.ngeom + geom1] < 0) or
        (type2 == GeomTypeMesh and a.geomDataid[dataidId * a.ngeom + geom2] < 0):
      return false

  if a.enableSleep:
    let b1 = int(a.geomBodyid[geom1])
    let b2 = int(a.geomBodyid[geom2])
    let s1 = a.bodyAwake[w * a.awakeStride + b1]
    let s2 = a.bodyAwake[w * a.awakeStride + b2]
    if s1 == SleepAsleep and s2 == SleepAsleep:
      return false
    if (s1 == SleepAsleep and s2 == SleepStatic) or (s2 == SleepAsleep and s1 == SleepStatic):
      return false
    if a.incremental:
      let p1 = a.bodyAwakePrev[w * a.awakeStride + b1]
      let p2 = a.bodyAwakePrev[w * a.awakeStride + b2]
      let skippedPass1 =
        (p1 == SleepAsleep and p2 == SleepAsleep) or
        (p1 == SleepAsleep and p2 == SleepStatic) or
        (p2 == SleepAsleep and p1 == SleepStatic)
      if not skippedPass1:
        return false

  if bpFilterPasses(a, w, geom1, geom2):
    return true
  a.nxnPairid[e * 2 + 1] >= 0

proc bpCountWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr BroadphaseArgs](arg)
  while true:
    let w = claimTask()
    if w >= a.nworld:
      break
    var cnt = 0'i32
    if a.flags != nil:
      let fBase = w * a.npairs
      for e in 0 ..< a.npairs:
        let geom1 = int(a.nxnPair[e * 2])
        let geom2 = int(a.nxnPair[e * 2 + 1])
        if bpPairPasses(a, w, geom1, geom2, e):
          a.flags[fBase + e] = 1'u8
          inc cnt
        else:
          a.flags[fBase + e] = 0'u8
    else:
      for e in 0 ..< a.npairs:
        let geom1 = int(a.nxnPair[e * 2])
        let geom2 = int(a.nxnPair[e * 2 + 1])
        if bpPairPasses(a, w, geom1, geom2, e):
          inc cnt
    a.counts[w] = cnt

proc bpFillWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr BroadphaseArgs](arg)
  while true:
    let w = claimTask()
    if w >= a.nworld:
      break
    var cid = int(a.counts[w])
    let fBase = w * a.npairs
    for e in 0 ..< a.npairs:
      let geom1 = int(a.nxnPair[e * 2])
      let geom2 = int(a.nxnPair[e * 2 + 1])
      if a.flags != nil:
        if a.flags[fBase + e] == 0'u8:
          continue
      elif not bpPairPasses(a, w, geom1, geom2, e):
        continue
      if cid < a.naconmax:
        let type1 = a.geomType[geom1]
        let type2 = a.geomType[geom2]
        let p = cid * 2
        if type1 > type2:
          a.collisionPair[p] = int32(geom2)
          a.collisionPair[p + 1] = int32(geom1)
        else:
          a.collisionPair[p] = int32(geom1)
          a.collisionPair[p + 1] = int32(geom2)
        a.collisionPairid[p] = a.nxnPairid[e * 2]
        a.collisionPairid[p + 1] = a.nxnPairid[e * 2 + 1]
        a.collisionWorldid[cid] = int32(w)
      inc cid

proc nxnBroadphaseDense*(
  nworld, npairs, naconmax, filter, ngeom: int,
  rboundWorlds, marginWorlds, gapWorlds, dataidWorlds, aabbWorlds, awakeStride: int,
  checkMesh, enableSleep, incremental: int,
  geomType, geomBodyid, geomDataid: uint64,
  geomAabb, geomRbound, geomMargin, geomGap, nxnPair, nxnPairid: uint64,
  geomXpos, geomXmat, bodyAwake, bodyAwakePrev, ncollision: uint64,
  collisionPair, collisionPairid, collisionWorldid: uint64,
): void {.exportpy: "nxn_broadphase_dense".} =
  if nworld <= 0:
    return
  var args = BroadphaseArgs(
    nworld: nworld, npairs: npairs, naconmax: naconmax, filter: filter, ngeom: ngeom,
    rboundWorlds: rboundWorlds, marginWorlds: marginWorlds, gapWorlds: gapWorlds,
    dataidWorlds: dataidWorlds, aabbWorlds: aabbWorlds, awakeStride: awakeStride,
    checkMesh: checkMesh != 0, enableSleep: enableSleep != 0, incremental: incremental != 0,
    geomType: toPtr[int32](geomType), geomBodyid: toPtr[int32](geomBodyid), geomDataid: toPtr[int32](geomDataid),
    geomAabb: toPtr[float32](geomAabb), geomRbound: toPtr[float32](geomRbound),
    geomMargin: toPtr[float32](geomMargin), geomGap: toPtr[float32](geomGap),
    nxnPair: toPtr[int32](nxnPair), nxnPairid: toPtr[int32](nxnPairid),
    geomXpos: toPtr[float32](geomXpos), geomXmat: toPtr[float32](geomXmat),
    bodyAwake: toPtr[int32](bodyAwake), bodyAwakePrev: toPtr[int32](bodyAwakePrev),
    ncollision: toPtr[int32](ncollision), counts: nil, flags: nil,
    collisionPair: toPtr[int32](collisionPair), collisionPairid: toPtr[int32](collisionPairid),
    collisionWorldid: toPtr[int32](collisionWorldid),
  )
  var counts = newSeq[int32](nworld)
  args.counts = cast[ptr UncheckedArray[int32]](addr counts[0])
  var flags = newSeq[uint8]()
  if nworld * npairs <= BpFlagsCap:
    flags.setLen(nworld * npairs)
    args.flags = cast[ptr UncheckedArray[uint8]](addr flags[0])
  runJob(bpCountWorker, addr args, nworld)

  # exclusive prefix over worlds, continuing from the caller's running count
  let base = int(args.ncollision[0])
  var running = base
  for w in 0 ..< nworld:
    let c = int(args.counts[w])
    args.counts[w] = int32(running)
    running += c
  args.ncollision[0] = int32(running)

  runJob(bpFillWorker, addr args, nworld)

# ------------------------------------------------------------------------------
# Contact-row setup (mirror of constraint._efc_contact_init / _efc_contact_update,
# dense pyramidal path).  Contacts are emitted world-major, so the per-world
# ranges come from one serial scan; the efcid allocation then stays sequential
# within each world, matching the serial Warp ordering.

func kbimp(disableflags: int32, timestep: float32,
           solref: ptr UncheckedArray[float32], solrefOff: int,
           solimp: ptr UncheckedArray[float32], solimpOff: int,
           posImp: float32): (float32, float32, float32) {.inline.} =
  var timeconst = solref[solrefOff]
  let dampratio = solref[solrefOff + 1]
  var dmin = solimp[solimpOff]
  let dmax0 = solimp[solimpOff + 1]
  let widthRaw = solimp[solimpOff + 2]
  var mid = solimp[solimpOff + 3]
  var power = solimp[solimpOff + 4]

  if (disableflags and DisableRefsafe) == 0:
    timeconst = max(timeconst, 2.0'f32 * timestep)

  dmin = clamp(dmin, MjMinImp, MjMaxImp)
  let dmax = clamp(dmax0, MjMinImp, MjMaxImp)
  let width = max(MjMinVal, widthRaw)
  mid = clamp(mid, MjMinImp, MjMaxImp)
  power = max(1.0'f32, power)

  let dmaxSq = dmax * dmax
  var k: float32
  if solref[solrefOff] <= 0.0'f32:
    k = -solref[solrefOff] / max(MjMinVal, dmaxSq)
  else:
    k = 1.0'f32 / max(MjMinVal, dmaxSq * timeconst * timeconst * dampratio * dampratio)
  var b: float32
  if solref[solrefOff + 1] <= 0.0'f32:
    b = -solref[solrefOff + 1] / max(MjMinVal, dmax)
  else:
    b = 2.0'f32 / max(MjMinVal, dmax * timeconst)

  let impX = abs(posImp) / width
  var imp: float32
  if dmin == dmax or widthRaw <= MjMinVal:
    imp = 0.5'f32 * (dmin + dmax)
  elif impX <= 0.0'f32:
    imp = dmin
  elif impX >= 1.0'f32:
    imp = dmax
  elif power == 1.0'f32:
    imp = dmin + impX * (dmax - dmin)
  elif impX <= mid:
    let impY = (1.0'f32 / pow(mid, power - 1.0'f32)) * pow(impX, power)
    imp = clamp(dmin + impY * (dmax - dmin), dmin, dmax)
  else:
    let impY = 1.0'f32 - (1.0'f32 / pow(1.0'f32 - mid, power - 1.0'f32)) * pow(1.0'f32 - impX, power)
    imp = clamp(dmin + impY * (dmax - dmin), dmin, dmax)
  (k, b, imp)

type
  EfcContactInitArgs = object
    nworld, nacon, njmax, nmaxdim: int
    flgAdhesion, isElliptic: bool
    contactType, contactDim, contactWorldid, contactGeom: ptr UncheckedArray[int32]
    contactDist, contactIncludemargin, contactAdhesion: ptr UncheckedArray[float32]
    nefc: ptr UncheckedArray[int32]
    efcAddress: ptr UncheckedArray[int32]
    efcId: ptr UncheckedArray[int32]
    starts: ptr UncheckedArray[int32]

proc efcContactInitWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr EfcContactInitArgs](arg)
  while true:
    let w = claimTask()
    if w >= a.nworld:
      break
    let start = int(a.starts[w])
    let stop = int(a.starts[w + 1])
    if start >= stop:
      continue
    var base = int(a.nefc[w])
    for conid in start ..< stop:
      if (a.contactType[conid] and ContactConstraint) == 0:
        continue
      let condim = int(a.contactDim[conid])
      let pos = a.contactDist[conid] - a.contactIncludemargin[conid]
      let active =
        if a.flgAdhesion: (pos < 0.0'f32) or (a.contactAdhesion[conid] != 0.0'f32)
        else: pos < 0.0'f32
      if not active:
        continue
      let ndim = if a.isElliptic: condim else: (if condim == 1: 1 else: 2 * (condim - 1))
      let baseEfc = base
      base += ndim
      for dim in 0 ..< ndim:
        let efcid = baseEfc + dim
        if efcid >= a.njmax:
          a.efcAddress[conid * a.nmaxdim + dim] = -1'i32
        else:
          a.efcAddress[conid * a.nmaxdim + dim] = int32(efcid)
          a.efcId[w * a.njmax + efcid] = int32(conid)
    a.nefc[w] = int32(base)

proc efcContactInitDense*(
  nworld, nacon, njmax, nmaxdim: int,
  flgAdhesion, isElliptic: int,
  contactType, contactDim, contactWorldid, contactGeom: uint64,
  contactDist, contactIncludemargin, contactAdhesion: uint64,
  nefc, efcAddress, efcId: uint64,
): int {.exportpy: "efc_contact_init_dense".} =
  ## Returns 0 on success, 1 when the contact list is not world-major ordered
  ## (caller falls back to the Warp kernel).
  if nworld <= 0:
    return 1
  var args = EfcContactInitArgs(
    nworld: nworld, nacon: nacon, njmax: njmax, nmaxdim: nmaxdim,
    flgAdhesion: flgAdhesion != 0, isElliptic: isElliptic != 0,
    contactType: toPtr[int32](contactType), contactDim: toPtr[int32](contactDim),
    contactWorldid: toPtr[int32](contactWorldid), contactGeom: toPtr[int32](contactGeom),
    contactDist: toPtr[float32](contactDist), contactIncludemargin: toPtr[float32](contactIncludemargin),
    contactAdhesion: toPtr[float32](contactAdhesion),
    nefc: toPtr[int32](nefc), efcAddress: toPtr[int32](efcAddress), efcId: toPtr[int32](efcId),
    starts: nil,
  )

  var starts = newSeq[int32](nworld + 1)
  var prev = -1'i32
  for c in 0 ..< nacon:
    let cw = args.contactWorldid[c]
    if cw < prev or cw >= int32(nworld):
      return 1
    if cw != prev:
      for w in (int(prev) + 1) .. int(cw):
        starts[w] = int32(c)
      prev = cw
  for w in (int(prev) + 1) .. nworld:
    starts[w] = int32(nacon)
  args.starts = cast[ptr UncheckedArray[int32]](addr starts[0])

  runJob(efcContactInitWorker, addr args, nworld)
  0

type
  EfcContactUpdateArgs = object
    nacon, njmax, nmaxdim: int
    timestepWorlds, impratioWorlds, invweightWorlds, invweightStride: int
    disableflags: int32
    timestep, impratioInvsqrt: ptr UncheckedArray[float32]
    bodyInvweight0: ptr UncheckedArray[float32]
    geomBodyid: ptr UncheckedArray[int32]
    contactDim, contactWorldid, contactGeom, contactType: ptr UncheckedArray[int32]
    contactEfcAddress: ptr UncheckedArray[int32]
    efcJqvel: ptr UncheckedArray[float32]
    contactDist, contactIncludemargin, contactFriction: ptr UncheckedArray[float32]
    contactSolref, contactSolimp: ptr UncheckedArray[float32]
    efcType, efcId: ptr UncheckedArray[int32]
    efcD, efcAref, efcVel, efcPos, efcMargin, efcFrictionloss: ptr UncheckedArray[float32]

proc efcContactUpdateWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr EfcContactUpdateArgs](arg)
  while true:
    let chunk = claimTask()
    let chunkStart = chunk * EfcUpdateChunk
    if chunkStart >= a.nacon:
      break
    let chunkEnd = min(a.nacon, chunkStart + EfcUpdateChunk)
    for conid in chunkStart ..< chunkEnd:
      if (a.contactType[conid] and ContactConstraint) == 0:
        continue

      let condim = int(a.contactDim[conid])
      let worldid = int(a.contactWorldid[conid])
      let geom1 = int(a.contactGeom[conid * 2])
      let geom2 = int(a.contactGeom[conid * 2 + 1])
      let body1 = int(a.geomBodyid[geom1])
      let body2 = int(a.geomBodyid[geom2])

      let includemargin = a.contactIncludemargin[conid]
      let pos = a.contactDist[conid] - includemargin

      var invweight =
        a.bodyInvweight0[(worldid mod a.invweightWorlds) * a.invweightStride + body1 * 2] +
        a.bodyInvweight0[(worldid mod a.invweightWorlds) * a.invweightStride + body2 * 2]

      if condim > 1:
        let fri0 = a.contactFriction[conid * 5]
        invweight = invweight + fri0 * fri0 * invweight
        let impInv = a.impratioInvsqrt[worldid mod a.impratioWorlds]
        invweight = invweight * 2.0'f32 * fri0 * fri0 * impInv * impInv

      let efcType = if condim == 1: ConstraintContactFrictionless else: ConstraintContactPyramidal

      let timestep = a.timestep[worldid mod a.timestepWorlds]
      let (k, b, imp) = kbimp(
        a.disableflags, timestep,
        a.contactSolref, conid * 2, a.contactSolimp, conid * 5, pos)

      let ndim = if condim == 1: 1 else: 2 * (condim - 1)
      let efcOff = worldid * a.njmax
      for dimid in 0 ..< min(ndim, a.nmaxdim):
        let efcid = int(a.contactEfcAddress[conid * a.nmaxdim + dimid])
        if efcid < 0:
          continue
        let Jqvel = a.efcJqvel[efcOff + efcid]

        let D = 1.0'f32 / max(invweight * (1.0'f32 - imp) / imp, MjMinVal)
        let aref = -k * imp * pos - b * Jqvel

        a.efcD[efcOff + efcid] = D
        a.efcAref[efcOff + efcid] = aref
        a.efcVel[efcOff + efcid] = Jqvel
        a.efcPos[efcOff + efcid] = pos + includemargin
        a.efcMargin[efcOff + efcid] = includemargin
        a.efcFrictionloss[efcOff + efcid] = 0.0'f32
        a.efcType[efcOff + efcid] = efcType
        a.efcId[efcOff + efcid] = int32(conid)

proc efcContactUpdateDense*(
  nacon, njmax, nmaxdim: int,
  timestepWorlds, impratioWorlds, invweightWorlds, invweightStride: int,
  disableflags: int,
  timestep, impratioInvsqrt, bodyInvweight0, geomBodyid: uint64,
  contactDim, contactWorldid, contactGeom, contactType: uint64,
  contactEfcAddress, efcJqvel: uint64,
  contactDist, contactIncludemargin, contactFriction, contactSolref, contactSolimp: uint64,
  efcType, efcId: uint64,
  efcD, efcAref, efcVel, efcPos, efcMargin, efcFrictionloss: uint64,
): void {.exportpy: "efc_contact_update_dense".} =
  var args = EfcContactUpdateArgs(
    nacon: nacon, njmax: njmax, nmaxdim: nmaxdim,
    timestepWorlds: timestepWorlds, impratioWorlds: impratioWorlds,
    invweightWorlds: invweightWorlds, invweightStride: invweightStride,
    disableflags: int32(disableflags),
    timestep: toPtr[float32](timestep), impratioInvsqrt: toPtr[float32](impratioInvsqrt),
    bodyInvweight0: toPtr[float32](bodyInvweight0), geomBodyid: toPtr[int32](geomBodyid),
    contactDim: toPtr[int32](contactDim), contactWorldid: toPtr[int32](contactWorldid),
    contactGeom: toPtr[int32](contactGeom), contactType: toPtr[int32](contactType),
    contactEfcAddress: toPtr[int32](contactEfcAddress), efcJqvel: toPtr[float32](efcJqvel),
    contactDist: toPtr[float32](contactDist), contactIncludemargin: toPtr[float32](contactIncludemargin),
    contactFriction: toPtr[float32](contactFriction),
    contactSolref: toPtr[float32](contactSolref), contactSolimp: toPtr[float32](contactSolimp),
    efcType: toPtr[int32](efcType), efcId: toPtr[int32](efcId),
    efcD: toPtr[float32](efcD), efcAref: toPtr[float32](efcAref), efcVel: toPtr[float32](efcVel),
    efcPos: toPtr[float32](efcPos), efcMargin: toPtr[float32](efcMargin),
    efcFrictionloss: toPtr[float32](efcFrictionloss),
  )
  runJob(efcContactUpdateWorker, addr args, nacon)

# ------------------------------------------------------------------------------
# Forward kinematics (mirror of smooth.kinematics): branch walk, body matrices,
# inertial frames, geom and site local->global.  One job, four claim/barrier
# phases, so the dependent stages stay fused without extra wakeups.

type
  KinematicsArgs = object
    nworld, nbody, ngeom, nsite, nbranch, njnt: int
    qpos0Worlds, qpos0Stride: int
    bodyPosWorlds, bodyPosStride: int
    bodyQuatWorlds, bodyQuatStride: int
    jntPosWorlds, jntPosStride: int
    jntAxisWorlds, jntAxisStride: int
    bodyIposWorlds, bodyIposStride: int
    bodyIquatWorlds, bodyIquatStride: int
    geomPosWorlds, geomPosStride: int
    geomQuatWorlds, geomQuatStride: int
    sitePosWorlds, sitePosStride: int
    siteQuatWorlds, siteQuatStride: int
    nmocap: int
    qpos0: ptr UncheckedArray[float32]
    bodyParentid, bodyMocapid, bodyJntnum, bodyJntadr: ptr UncheckedArray[int32]
    jntType, jntQposadr: ptr UncheckedArray[int32]
    bodyBranches, bodyBranchStart: ptr UncheckedArray[int32]
    bodyRootid, bodyWeldid: ptr UncheckedArray[int32]
    geomBodyid, siteBodyid: ptr UncheckedArray[int32]
    bodyPos, bodyQuat, jntPos, jntAxis, bodyIpos, bodyIquat: ptr UncheckedArray[float32]
    geomPos, geomQuat, sitePos, siteQuat: ptr UncheckedArray[float32]
    qpos, mocapPos, mocapQuat: ptr UncheckedArray[float32]
    xpos, xquat, xanchor, xaxis, xmat, xipos, ximat: ptr UncheckedArray[float32]
    geomXpos, geomXmat, siteXpos, siteXmat: ptr UncheckedArray[float32]

func kgVec3(p: ptr UncheckedArray[float32], off: int): Vec3f {.inline.} =
  Vec3f([p[off], p[off + 1], p[off + 2]])

func kgQuat(p: ptr UncheckedArray[float32], off: int): Quatf {.inline.} =
  Quatf([p[off], p[off + 1], p[off + 2], p[off + 3]])

proc kinematicsWorldBranches(a: ptr KinematicsArgs, w: int) {.nimcall, gcsafe.} =
  let qposOff = w * a.qpos0Stride

  for branch in 0 ..< a.nbranch:
    let start = int(a.bodyBranchStart[branch])
    let stop = int(a.bodyBranchStart[branch + 1])

    for i in start ..< stop:
      let bodyid = int(a.bodyBranches[i])
      let jntadr0 = int(a.bodyJntadr[bodyid])
      let jntnum = int(a.bodyJntnum[bodyid])

      if jntnum == 1 and a.jntType[jntadr0] == JointTypeFree:
        let qadr = int(a.jntQposadr[jntadr0])
        let xp = kgVec3(a.qpos, qposOff + qadr)
        let xq = normQuat(kgQuat(a.qpos, qposOff + qadr + 3))
        let xo = (w * a.nbody + bodyid) * 3
        a.xpos[xo] = xp[0]
        a.xpos[xo + 1] = xp[1]
        a.xpos[xo + 2] = xp[2]
        let qo = (w * a.nbody + bodyid) * 4
        a.xquat[qo] = xq[0]
        a.xquat[qo + 1] = xq[1]
        a.xquat[qo + 2] = xq[2]
        a.xquat[qo + 3] = xq[3]
        let jo = (w * a.njnt + jntadr0) * 3
        a.xanchor[jo] = xp[0]
        a.xanchor[jo + 1] = xp[1]
        a.xanchor[jo + 2] = xp[2]
        let jnOff = (w mod a.jntAxisWorlds) * a.jntAxisStride + jntadr0 * 3
        a.xaxis[jo] = a.jntAxis[jnOff]
        a.xaxis[jo + 1] = a.jntAxis[jnOff + 1]
        a.xaxis[jo + 2] = a.jntAxis[jnOff + 2]
        continue

      let pid = int(a.bodyParentid[bodyid])
      let mocapid = int(a.bodyMocapid[bodyid])
      var xp: Vec3f
      var xq: Quatf
      if mocapid >= 0:
        xp = kgVec3(a.mocapPos, (w * a.nmocap + mocapid) * 3)
        xq = kgQuat(a.mocapQuat, (w * a.nmocap + mocapid) * 4)
      else:
        xp = kgVec3(a.bodyPos, (w mod a.bodyPosWorlds) * a.bodyPosStride + bodyid * 3)
        xq = kgQuat(a.bodyQuat, (w mod a.bodyQuatWorlds) * a.bodyQuatStride + bodyid * 4)

      if pid >= 0:
        let pOff = w * a.nbody + pid
        let parentQ = kgQuat(a.xquat, pOff * 4)
        let parentP = kgVec3(a.xpos, pOff * 3)
        let rot = rotVecQuat(xp, parentQ)
        xp = Vec3f([rot[0] + parentP[0], rot[1] + parentP[1], rot[2] + parentP[2]])
        xq = mulQuat(parentQ, xq)

      var jntadr = jntadr0
      for _ in 0 ..< jntnum:
        let qadr = int(a.jntQposadr[jntadr])
        let jntType = a.jntType[jntadr]
        let jnOff = (w mod a.jntAxisWorlds) * a.jntAxisStride + jntadr * 3
        let jpOff = (w mod a.jntPosWorlds) * a.jntPosStride + jntadr * 3
        let jntAxis = kgVec3(a.jntAxis, jnOff)
        let jntPos = kgVec3(a.jntPos, jpOff)
        let xanchor = vadd(rotVecQuat(jntPos, xq), xp)
        let xaxis = rotVecQuat(jntAxis, xq)

        if jntType == JointTypeBall:
          let qloc = normQuat(kgQuat(a.qpos, qposOff + qadr))
          xq = mulQuat(xq, qloc)
          xp = vsub(xanchor, rotVecQuat(jntPos, xq))
        elif jntType == JointTypeSlide:
          let disp = a.qpos[qposOff + qadr] - a.qpos0[(w mod a.qpos0Worlds) * a.qpos0Stride + qadr]
          xp = Vec3f([xp[0] + xaxis[0] * disp, xp[1] + xaxis[1] * disp, xp[2] + xaxis[2] * disp])
        elif jntType == JointTypeHinge:
          let q0 = a.qpos0[(w mod a.qpos0Worlds) * a.qpos0Stride + qadr]
          let qloc = axisAngleToQuat(jntAxis, a.qpos[qposOff + qadr] - q0)
          xq = mulQuat(xq, qloc)
          xp = vsub(xanchor, rotVecQuat(jntPos, xq))

        let jxOff = (w * a.njnt + jntadr) * 3
        a.xanchor[jxOff] = xanchor[0]
        a.xanchor[jxOff + 1] = xanchor[1]
        a.xanchor[jxOff + 2] = xanchor[2]
        a.xaxis[jxOff] = xaxis[0]
        a.xaxis[jxOff + 1] = xaxis[1]
        a.xaxis[jxOff + 2] = xaxis[2]
        inc jntadr

      xq = normQuat(xq)
      let xo = (w * a.nbody + bodyid) * 3
      a.xpos[xo] = xp[0]
      a.xpos[xo + 1] = xp[1]
      a.xpos[xo + 2] = xp[2]
      let qo = (w * a.nbody + bodyid) * 4
      a.xquat[qo] = xq[0]
      a.xquat[qo + 1] = xq[1]
      a.xquat[qo + 2] = xq[2]
      a.xquat[qo + 3] = xq[3]

proc kinematicsWorldBodies(a: ptr KinematicsArgs, w: int) {.nimcall, gcsafe.} =
  for bodyid in 0 ..< a.nbody:
    let q = kgQuat(a.xquat, (w * a.nbody + bodyid) * 4)
    let p = kgVec3(a.xpos, (w * a.nbody + bodyid) * 3)
    let m = quatToMat(q)
    let mo = (w * a.nbody + bodyid) * 9
    for i in 0 ..< 9:
      a.xmat[mo + i] = m[i]
    let ipos = kgVec3(a.bodyIpos, (w mod a.bodyIposWorlds) * a.bodyIposStride + bodyid * 3)
    let iquat = kgQuat(a.bodyIquat, (w mod a.bodyIquatWorlds) * a.bodyIquatStride + bodyid * 4)
    let xi = vadd(rotVecQuat(ipos, q), p)
    let xio = (w * a.nbody + bodyid) * 3
    a.xipos[xio] = xi[0]
    a.xipos[xio + 1] = xi[1]
    a.xipos[xio + 2] = xi[2]
    let im = quatToMat(mulQuat(q, iquat))
    let imo = (w * a.nbody + bodyid) * 9
    for i in 0 ..< 9:
      a.ximat[imo + i] = im[i]

proc kinematicsWorldGeoms(a: ptr KinematicsArgs, w: int) {.nimcall, gcsafe.} =
  for geomid in 0 ..< a.ngeom:
    let bodyid = int(a.geomBodyid[geomid])
    if a.bodyWeldid[bodyid] == 0 and a.bodyMocapid[int(a.bodyRootid[bodyid])] == -1:
      continue
    let xq = kgQuat(a.xquat, (w * a.nbody + bodyid) * 4)
    let xp = kgVec3(a.xpos, (w * a.nbody + bodyid) * 3)
    let gpos = kgVec3(a.geomPos, (w mod a.geomPosWorlds) * a.geomPosStride + geomid * 3)
    let gquat = kgQuat(a.geomQuat, (w mod a.geomQuatWorlds) * a.geomQuatStride + geomid * 4)
    let gp = vadd(rotVecQuat(gpos, xq), xp)
    let go = (w * a.ngeom + geomid) * 3
    a.geomXpos[go] = gp[0]
    a.geomXpos[go + 1] = gp[1]
    a.geomXpos[go + 2] = gp[2]
    let gm = quatToMat(mulQuat(xq, gquat))
    let gmo = (w * a.ngeom + geomid) * 9
    for i in 0 ..< 9:
      a.geomXmat[gmo + i] = gm[i]

proc kinematicsWorldSites(a: ptr KinematicsArgs, w: int) {.nimcall, gcsafe.} =
  for siteid in 0 ..< a.nsite:
    let bodyid = int(a.siteBodyid[siteid])
    let xq = kgQuat(a.xquat, (w * a.nbody + bodyid) * 4)
    let xp = kgVec3(a.xpos, (w * a.nbody + bodyid) * 3)
    let spos = kgVec3(a.sitePos, (w mod a.sitePosWorlds) * a.sitePosStride + siteid * 3)
    let squat = kgQuat(a.siteQuat, (w mod a.siteQuatWorlds) * a.siteQuatStride + siteid * 4)
    let sp = vadd(rotVecQuat(spos, xq), xp)
    let so = (w * a.nsite + siteid) * 3
    a.siteXpos[so] = sp[0]
    a.siteXpos[so + 1] = sp[1]
    a.siteXpos[so + 2] = sp[2]
    let sm = quatToMat(mulQuat(xq, squat))
    let smo = (w * a.nsite + siteid) * 9
    for i in 0 ..< 9:
      a.siteXmat[smo + i] = sm[i]

proc kinematicsWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr KinematicsArgs](arg)
  let chunk = max(1, (a.nworld + nthreads * 4 - 1) div (nthreads * 4))
  while true:
    let first = claimTask() * chunk
    if first >= a.nworld:
      break
    let last = min(a.nworld, first + chunk)
    for w in first ..< last:
      kinematicsWorldBranches(a, w)
      kinematicsWorldBodies(a, w)
      kinematicsWorldGeoms(a, w)
      kinematicsWorldSites(a, w)

proc kinematicsDense*(
  nworld, nbody, ngeom, nsite, nbranch, njnt, nmocap: int,
  qpos0Worlds, qpos0Stride: int,
  bodyPosWorlds, bodyPosStride, bodyQuatWorlds, bodyQuatStride: int,
  jntPosWorlds, jntPosStride, jntAxisWorlds, jntAxisStride: int,
  bodyIposWorlds, bodyIposStride, bodyIquatWorlds, bodyIquatStride: int,
  geomPosWorlds, geomPosStride, geomQuatWorlds, geomQuatStride: int,
  sitePosWorlds, sitePosStride, siteQuatWorlds, siteQuatStride: int,
  qpos0: uint64,
  bodyParentid, bodyMocapid, bodyJntnum, bodyJntadr, jntType, jntQposadr: uint64,
  bodyBranches, bodyBranchStart, bodyRootid, bodyWeldid, geomBodyid, siteBodyid: uint64,
  bodyPos, bodyQuat, jntPos, jntAxis, bodyIpos, bodyIquat: uint64,
  geomPos, geomQuat, sitePos, siteQuat: uint64,
  qpos, mocapPos, mocapQuat: uint64,
  xpos, xquat, xanchor, xaxis, xmat, xipos, ximat: uint64,
  geomXpos, geomXmat, siteXpos, siteXmat: uint64,
): void {.exportpy: "kinematics_dense".} =
  var args = KinematicsArgs(
    nworld: nworld, nbody: nbody, ngeom: ngeom, nsite: nsite, nbranch: nbranch, njnt: njnt,
    qpos0Worlds: qpos0Worlds, qpos0Stride: qpos0Stride,
    bodyPosWorlds: bodyPosWorlds, bodyPosStride: bodyPosStride,
    bodyQuatWorlds: bodyQuatWorlds, bodyQuatStride: bodyQuatStride,
    jntPosWorlds: jntPosWorlds, jntPosStride: jntPosStride,
    jntAxisWorlds: jntAxisWorlds, jntAxisStride: jntAxisStride,
    bodyIposWorlds: bodyIposWorlds, bodyIposStride: bodyIposStride,
    bodyIquatWorlds: bodyIquatWorlds, bodyIquatStride: bodyIquatStride,
    geomPosWorlds: geomPosWorlds, geomPosStride: geomPosStride,
    geomQuatWorlds: geomQuatWorlds, geomQuatStride: geomQuatStride,
    sitePosWorlds: sitePosWorlds, sitePosStride: sitePosStride,
    siteQuatWorlds: siteQuatWorlds, siteQuatStride: siteQuatStride,
    nmocap: nmocap,
    qpos0: toPtr[float32](qpos0),
    bodyParentid: toPtr[int32](bodyParentid), bodyMocapid: toPtr[int32](bodyMocapid),
    bodyJntnum: toPtr[int32](bodyJntnum), bodyJntadr: toPtr[int32](bodyJntadr),
    jntType: toPtr[int32](jntType), jntQposadr: toPtr[int32](jntQposadr),
    bodyBranches: toPtr[int32](bodyBranches), bodyBranchStart: toPtr[int32](bodyBranchStart),
    bodyRootid: toPtr[int32](bodyRootid), bodyWeldid: toPtr[int32](bodyWeldid),
    geomBodyid: toPtr[int32](geomBodyid), siteBodyid: toPtr[int32](siteBodyid),
    bodyPos: toPtr[float32](bodyPos), bodyQuat: toPtr[float32](bodyQuat),
    jntPos: toPtr[float32](jntPos), jntAxis: toPtr[float32](jntAxis),
    bodyIpos: toPtr[float32](bodyIpos), bodyIquat: toPtr[float32](bodyIquat),
    geomPos: toPtr[float32](geomPos), geomQuat: toPtr[float32](geomQuat),
    sitePos: toPtr[float32](sitePos), siteQuat: toPtr[float32](siteQuat),
    qpos: toPtr[float32](qpos), mocapPos: toPtr[float32](mocapPos), mocapQuat: toPtr[float32](mocapQuat),
    xpos: toPtr[float32](xpos), xquat: toPtr[float32](xquat), xanchor: toPtr[float32](xanchor),
    xaxis: toPtr[float32](xaxis), xmat: toPtr[float32](xmat), xipos: toPtr[float32](xipos),
    ximat: toPtr[float32](ximat), geomXpos: toPtr[float32](geomXpos), geomXmat: toPtr[float32](geomXmat),
    siteXpos: toPtr[float32](siteXpos), siteXmat: toPtr[float32](siteXmat),
  )
  runJob(kinematicsWorker, addr args, nworld * (nbranch + nbody + ngeom + nsite))

# ------------------------------------------------------------------------------
# Subtree CoM, composite inertia, dof axes and body velocities (mirror of
# smooth.com_pos / _cinert / _cdof / com_vel).  Body ids are topologically
# ordered, so the tree accumulation runs per world with no cross-thread writes.

type
  ComPosArgs = object
    nworld, nbody, njnt, nlevels: int
    massWorlds, massStride: int
    inertiaWorlds, inertiaStride: int
    subtreemassWorlds, subtreemassStride: int
    xiposStride, ximatStride, comStride, xmatStride, xanchorStride, xaxisStride: int
    cinertStride, cdofStride: int
    bodyParentid, bodyRootid, jntType, jntDofadr, jntBodyid: ptr UncheckedArray[int32]
    bodyTree: ptr UncheckedArray[int32]
    treeOffsets: ptr UncheckedArray[int32]
    bodyMass, bodyInertia, bodySubtreemass: ptr UncheckedArray[float32]
    xipos, ximat, subtreeCom, xmat, xanchor, xaxis: ptr UncheckedArray[float32]
    cinert, cdof: ptr UncheckedArray[float32]

proc comPosWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr ComPosArgs](arg)
  let chunk = max(1, (a.nworld + nthreads * 4 - 1) div (nthreads * 4))
  while true:
    let first = claimTask() * chunk
    if first >= a.nworld:
      break
    let last = min(a.nworld, first + chunk)
    for w in first ..< last:
      let comOff = w * a.comStride
      let xiOff = w * a.xiposStride
      let mOff = (w mod a.massWorlds) * a.massStride
      let smOff = (w mod a.subtreemassWorlds) * a.subtreemassStride

      # subtree_com: mass-weighted xipos, accumulated up the tree, then normalized
      for b in 0 ..< a.nbody:
        let mass = a.bodyMass[mOff + b]
        for c in 0 ..< 3:
          a.subtreeCom[comOff + b * 3 + c] = a.xipos[xiOff + b * 3 + c] * mass

      for level in countdown(a.nlevels - 1, 0):
        for idx in int(a.treeOffsets[level]) ..< int(a.treeOffsets[level + 1]):
          let b = int(a.bodyTree[idx])
          if b == 0:
            continue
          let p = int(a.bodyParentid[b])
          for c in 0 ..< 3:
            a.subtreeCom[comOff + p * 3 + c] += a.subtreeCom[comOff + b * 3 + c]

      for b in 0 ..< a.nbody:
        let mass = a.bodySubtreemass[smOff + b]
        if mass != 0.0'f32:
          for c in 0 ..< 3:
            a.subtreeCom[comOff + b * 3 + c] /= mass

      # composite inertia in the subtree CoM frame
      for b in 0 ..< a.nbody:
        let iOff = (w mod a.inertiaWorlds) * a.inertiaStride + b * 3
        let memOff = w * a.ximatStride + b * 9
        let m0 = a.ximat[memOff]
        let m1 = a.ximat[memOff + 1]
        let m2 = a.ximat[memOff + 2]
        let m3 = a.ximat[memOff + 3]
        let m4 = a.ximat[memOff + 4]
        let m5 = a.ximat[memOff + 5]
        let m6 = a.ximat[memOff + 6]
        let m7 = a.ximat[memOff + 7]
        let m8 = a.ximat[memOff + 8]
        let d0 = a.bodyInertia[iOff]
        let d1 = a.bodyInertia[iOff + 1]
        let d2 = a.bodyInertia[iOff + 2]
        let mass = a.bodyMass[mOff + b]
        let root = int(a.bodyRootid[b])
        let dif0 = a.xipos[xiOff + b * 3] - a.subtreeCom[comOff + root * 3]
        let dif1 = a.xipos[xiOff + b * 3 + 1] - a.subtreeCom[comOff + root * 3 + 1]
        let dif2 = a.xipos[xiOff + b * 3 + 2] - a.subtreeCom[comOff + root * 3 + 2]

        # tmp = mat * diag(inert) * mat^T.  Warp's matrix product uses `muladd`, i.e.
        # fmaf, so the products fuse; mirror that for bit-exact results.  M[i][k] = mem[i * 3 + k]
        let a00 = fma32(m0, d0, 0.0'f32)
        let a01 = fma32(m1, d1, 0.0'f32)
        let a02 = fma32(m2, d2, 0.0'f32)
        let a10 = fma32(m3, d0, 0.0'f32)
        let a11 = fma32(m4, d1, 0.0'f32)
        let a12 = fma32(m5, d2, 0.0'f32)
        let a20 = fma32(m6, d0, 0.0'f32)
        let a21 = fma32(m7, d1, 0.0'f32)
        let a22 = fma32(m8, d2, 0.0'f32)
        var res: array[10, float32]
        res[0] = fma32(a02, m2, fma32(a01, m1, fma32(a00, m0, 0.0'f32)))
        res[1] = fma32(a12, m5, fma32(a11, m4, fma32(a10, m3, 0.0'f32)))
        res[2] = fma32(a22, m8, fma32(a21, m7, fma32(a20, m6, 0.0'f32)))
        res[3] = fma32(a02, m5, fma32(a01, m4, fma32(a00, m3, 0.0'f32)))
        res[4] = fma32(a02, m8, fma32(a01, m7, fma32(a00, m6, 0.0'f32)))
        res[5] = fma32(a12, m8, fma32(a11, m7, fma32(a10, m6, 0.0'f32)))
        res[0] += mass * (dif1 * dif1 + dif2 * dif2)
        res[1] += mass * (dif0 * dif0 + dif2 * dif2)
        res[2] += mass * (dif0 * dif0 + dif1 * dif1)
        res[3] -= mass * dif0 * dif1
        res[4] -= mass * dif0 * dif2
        res[5] -= mass * dif1 * dif2
        res[6] = mass * dif0
        res[7] = mass * dif1
        res[8] = mass * dif2
        res[9] = mass
        let ciOff = w * a.cinertStride + b * 10
        for c in 0 ..< 10:
          a.cinert[ciOff + c] = res[c]

      # dof axes in the subtree CoM frame
      for j in 0 ..< a.njnt:
        let bodyid = int(a.jntBodyid[j])
        let dofid = int(a.jntDofadr[j])
        let axisOff = w * a.xaxisStride + j * 3
        let xax0 = a.xaxis[axisOff]
        let xax1 = a.xaxis[axisOff + 1]
        let xax2 = a.xaxis[axisOff + 2]
        let memOff = w * a.xmatStride + bodyid * 9
        # xmat = transpose(xmat_in): row i of the body rotation lives at mem[i], mem[i+3], mem[i+6]
        let row0 = [a.xmat[memOff], a.xmat[memOff + 3], a.xmat[memOff + 6]]
        let row1 = [a.xmat[memOff + 1], a.xmat[memOff + 4], a.xmat[memOff + 7]]
        let row2 = [a.xmat[memOff + 2], a.xmat[memOff + 5], a.xmat[memOff + 8]]
        let root = int(a.bodyRootid[bodyid])
        let ancOff = w * a.xanchorStride + j * 3
        let off0 = a.subtreeCom[comOff + root * 3] - a.xanchor[ancOff]
        let off1 = a.subtreeCom[comOff + root * 3 + 1] - a.xanchor[ancOff + 1]
        let off2 = a.subtreeCom[comOff + root * 3 + 2] - a.xanchor[ancOff + 2]
        let cdo = w * a.cdofStride + dofid * 6
        let jntType = a.jntType[j]
        if jntType == JointTypeFree:
          for i in 0 ..< 3:
            let base = cdo + i * 6
            for c in 0 ..< 6:
              a.cdof[base + c] = 0.0'f32
            a.cdof[base + 3 + i] = 1.0'f32
          for i in 0 ..< 3:
            let row = if i == 0: row0 elif i == 1: row1 else: row2
            let base = cdo + (3 + i) * 6
            a.cdof[base] = row[0]
            a.cdof[base + 1] = row[1]
            a.cdof[base + 2] = row[2]
            a.cdof[base + 3] = row[1] * off2 - row[2] * off1
            a.cdof[base + 4] = row[2] * off0 - row[0] * off2
            a.cdof[base + 5] = row[0] * off1 - row[1] * off0
        elif jntType == JointTypeBall:
          for i in 0 ..< 3:
            let row = if i == 0: row0 elif i == 1: row1 else: row2
            let base = cdo + i * 6
            a.cdof[base] = row[0]
            a.cdof[base + 1] = row[1]
            a.cdof[base + 2] = row[2]
            a.cdof[base + 3] = row[1] * off2 - row[2] * off1
            a.cdof[base + 4] = row[2] * off0 - row[0] * off2
            a.cdof[base + 5] = row[0] * off1 - row[1] * off0
        elif jntType == JointTypeSlide:
          a.cdof[cdo] = 0.0'f32
          a.cdof[cdo + 1] = 0.0'f32
          a.cdof[cdo + 2] = 0.0'f32
          a.cdof[cdo + 3] = xax0
          a.cdof[cdo + 4] = xax1
          a.cdof[cdo + 5] = xax2
        else:  # hinge
          a.cdof[cdo] = xax0
          a.cdof[cdo + 1] = xax1
          a.cdof[cdo + 2] = xax2
          a.cdof[cdo + 3] = xax1 * off2 - xax2 * off1
          a.cdof[cdo + 4] = xax2 * off0 - xax0 * off2
          a.cdof[cdo + 5] = xax0 * off1 - xax1 * off0

proc smoothComPosDense*(
  nworld, nbody, njnt, nlevels: int,
  massWorlds, massStride, inertiaWorlds, inertiaStride, subtreemassWorlds, subtreemassStride: int,
  xiposStride, ximatStride, comStride, xmatStride, xanchorStride, xaxisStride, cinertStride, cdofStride: int,
  bodyParentid, bodyRootid, jntType, jntDofadr, jntBodyid, bodyTree, treeOffsets: uint64,
  bodyMass, bodyInertia, bodySubtreemass: uint64,
  xipos, ximat, subtreeCom, xmat, xanchor, xaxis, cinert, cdof: uint64,
): void {.exportpy: "smooth_com_pos_dense".} =
  var args = ComPosArgs(
    nworld: nworld, nbody: nbody, njnt: njnt, nlevels: nlevels,
    massWorlds: massWorlds, massStride: massStride,
    inertiaWorlds: inertiaWorlds, inertiaStride: inertiaStride,
    subtreemassWorlds: subtreemassWorlds, subtreemassStride: subtreemassStride,
    xiposStride: xiposStride, ximatStride: ximatStride, comStride: comStride,
    xmatStride: xmatStride, xanchorStride: xanchorStride, xaxisStride: xaxisStride,
    cinertStride: cinertStride, cdofStride: cdofStride,
    bodyParentid: toPtr[int32](bodyParentid), bodyRootid: toPtr[int32](bodyRootid),
    jntType: toPtr[int32](jntType), jntDofadr: toPtr[int32](jntDofadr), jntBodyid: toPtr[int32](jntBodyid),
    bodyTree: toPtr[int32](bodyTree), treeOffsets: toPtr[int32](treeOffsets),
    bodyMass: toPtr[float32](bodyMass), bodyInertia: toPtr[float32](bodyInertia),
    bodySubtreemass: toPtr[float32](bodySubtreemass),
    xipos: toPtr[float32](xipos), ximat: toPtr[float32](ximat), subtreeCom: toPtr[float32](subtreeCom),
    xmat: toPtr[float32](xmat), xanchor: toPtr[float32](xanchor), xaxis: toPtr[float32](xaxis),
    cinert: toPtr[float32](cinert), cdof: toPtr[float32](cdof),
  )
  runJob(comPosWorker, addr args, nworld)

# Body spatial velocities and cdof time derivatives.

type
  ComVelArgs = object
    nworld, nbranch: int
    qvelStride, cdofStride, cvelStride, cdofDotStride: int
    bodyParentid, bodyJntnum, bodyJntadr, bodyDofadr, jntType: ptr UncheckedArray[int32]
    bodyBranches, bodyBranchStart: ptr UncheckedArray[int32]
    qvel, cdof, cvel, cdofDot: ptr UncheckedArray[float32]

func mcross(u: array[6, float32], v: array[6, float32]): array[6, float32] {.inline.} =
  array[6, float32]([
    u[1] * v[2] - u[2] * v[1],
    u[2] * v[0] - u[0] * v[2],
    u[0] * v[1] - u[1] * v[0],
    u[4] * v[2] - u[5] * v[1] + (u[1] * v[5] - u[2] * v[4]),
    u[5] * v[0] - u[3] * v[2] + (u[2] * v[3] - u[0] * v[5]),
    u[3] * v[1] - u[4] * v[0] + (u[0] * v[4] - u[1] * v[3]),
  ])

proc comVelWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr ComVelArgs](arg)
  let chunk = max(1, (a.nworld + nthreads * 4 - 1) div (nthreads * 4))
  while true:
    let first = claimTask() * chunk
    if first >= a.nworld:
      break
    let last = min(a.nworld, first + chunk)
    for w in first ..< last:
      let cvOff = w * a.cvelStride
      let cdOff = w * a.cdofStride
      let cddOff = w * a.cdofDotStride
      let qvOff = w * a.qvelStride
      for c in 0 ..< 6:
        a.cvel[cvOff + c] = 0.0'f32

      for branch in 0 ..< a.nbranch:
        for i in int(a.bodyBranchStart[branch]) ..< int(a.bodyBranchStart[branch + 1]):
          let bodyid = int(a.bodyBranches[i])
          let pid = int(a.bodyParentid[bodyid])
          var cv: array[6, float32]
          for c in 0 ..< 6:
            cv[c] = a.cvel[cvOff + pid * 6 + c]
          var dofid = int(a.bodyDofadr[bodyid])
          let jntid = int(a.bodyJntadr[bodyid])
          let jntnum = int(a.bodyJntnum[bodyid])
          if jntnum == 0:
            for c in 0 ..< 6:
              a.cvel[cvOff + bodyid * 6 + c] = cv[c]
            continue

          for j in jntid ..< jntid + jntnum:
            let jntType = a.jntType[j]
            if jntType == JointTypeFree:
              for k in 0 ..< 3:
                let q = a.qvel[qvOff + dofid + k]
                for c in 0 ..< 6:
                  cv[c] += a.cdof[cdOff + (dofid + k) * 6 + c] * q
              for c in 0 ..< 6:
                a.cdofDot[cddOff + dofid * 6 + c] = 0.0'f32
                a.cdofDot[cddOff + (dofid + 1) * 6 + c] = 0.0'f32
                a.cdofDot[cddOff + (dofid + 2) * 6 + c] = 0.0'f32
              for k in 3 ..< 6:
                var cdo: array[6, float32]
                for c in 0 ..< 6:
                  cdo[c] = a.cdof[cdOff + (dofid + k) * 6 + c]
                let res = mcross(cv, cdo)
                for c in 0 ..< 6:
                  a.cdofDot[cddOff + (dofid + k) * 6 + c] = res[c]
              for k in 3 ..< 6:
                let q = a.qvel[qvOff + dofid + k]
                for c in 0 ..< 6:
                  cv[c] += a.cdof[cdOff + (dofid + k) * 6 + c] * q
              dofid += 6
            elif jntType == JointTypeBall:
              for k in 0 ..< 3:
                var cdo: array[6, float32]
                for c in 0 ..< 6:
                  cdo[c] = a.cdof[cdOff + (dofid + k) * 6 + c]
                let res = mcross(cv, cdo)
                for c in 0 ..< 6:
                  a.cdofDot[cddOff + (dofid + k) * 6 + c] = res[c]
              for k in 0 ..< 3:
                let q = a.qvel[qvOff + dofid + k]
                for c in 0 ..< 6:
                  cv[c] += a.cdof[cdOff + (dofid + k) * 6 + c] * q
              dofid += 3
            else:
              var cdo: array[6, float32]
              for c in 0 ..< 6:
                cdo[c] = a.cdof[cdOff + dofid * 6 + c]
              let res = mcross(cv, cdo)
              for c in 0 ..< 6:
                a.cdofDot[cddOff + dofid * 6 + c] = res[c]
              let q = a.qvel[qvOff + dofid]
              for c in 0 ..< 6:
                cv[c] += cdo[c] * q
              dofid += 1

          for c in 0 ..< 6:
            a.cvel[cvOff + bodyid * 6 + c] = cv[c]

proc smoothComVelDense*(
  nworld, nbranch: int,
  qvelStride, cdofStride, cvelStride, cdofDotStride: int,
  bodyParentid, bodyJntnum, bodyJntadr, bodyDofadr, jntType, bodyBranches, bodyBranchStart: uint64,
  qvel, cdof, cvel, cdofDot: uint64,
): void {.exportpy: "smooth_com_vel_dense".} =
  var args = ComVelArgs(
    nworld: nworld, nbranch: nbranch,
    qvelStride: qvelStride, cdofStride: cdofStride, cvelStride: cvelStride, cdofDotStride: cdofDotStride,
    bodyParentid: toPtr[int32](bodyParentid), bodyJntnum: toPtr[int32](bodyJntnum),
    bodyJntadr: toPtr[int32](bodyJntadr), bodyDofadr: toPtr[int32](bodyDofadr), jntType: toPtr[int32](jntType),
    bodyBranches: toPtr[int32](bodyBranches), bodyBranchStart: toPtr[int32](bodyBranchStart),
    qvel: toPtr[float32](qvel), cdof: toPtr[float32](cdof), cvel: toPtr[float32](cvel),
    cdofDot: toPtr[float32](cdofDot),
  )
  runJob(comVelWorker, addr args, nworld)

# ------------------------------------------------------------------------------
# Position/velocity advance (mirror of forward._next_position / _next_velocity)
# and applied-force mapping (support._apply_ft).

type
  AdvanceArgs = object
    nworld, njnt, nv: int
    timestepWorlds: int
    qposStride, qvelStride, qaccStride, qvelInStride: int
    timestep: ptr UncheckedArray[float32]
    jntType, jntQposadr, jntDofadr: ptr UncheckedArray[int32]
    qpos, qvel, qvelIn, qacc: ptr UncheckedArray[float32]

func quatIntegrate(q: Quatf, v: Vec3f, dt: float32): Quatf {.inline.} =
  let vlen = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
  var vn = Vec3f([0.0'f32, 0.0'f32, 0.0'f32])
  if vlen > 0.0'f32:
    vn = Vec3f([v[0] / vlen, v[1] / vlen, v[2] / vlen])
  let angle = dt * vlen
  let qRes = axisAngleToQuat(vn, angle)
  normQuat(mulQuat(normQuat(q), qRes))

proc advanceWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr AdvanceArgs](arg)
  let chunk = max(1, (a.nworld + nthreads * 4 - 1) div (nthreads * 4))
  while true:
    let first = claimTask() * chunk
    if first >= a.nworld:
      break
    let last = min(a.nworld, first + chunk)
    for w in first ..< last:
      let dt = a.timestep[w mod a.timestepWorlds]
      let qpOff = w * a.qposStride
      let qvOff = w * a.qvelStride
      let qviOff = w * a.qvelInStride
      let qaOff = w * a.qaccStride

      # velocity first: Euler integrates positions with the updated qvel
      for dof in 0 ..< a.nv:
        a.qvel[qvOff + dof] += a.qacc[qaOff + dof] * dt

      for jnt in 0 ..< a.njnt:
        let jnttype = a.jntType[jnt]
        let qadr = int(a.jntQposadr[jnt])
        let dadr = int(a.jntDofadr[jnt])

        if jnttype == JointTypeFree:
          let vx = a.qvelIn[qviOff + dadr] * dt
          let vy = a.qvelIn[qviOff + dadr + 1] * dt
          let vz = a.qvelIn[qviOff + dadr + 2] * dt
          a.qpos[qpOff + qadr] += vx
          a.qpos[qpOff + qadr + 1] += vy
          a.qpos[qpOff + qadr + 2] += vz

          let q = Quatf([
            a.qpos[qpOff + qadr + 3], a.qpos[qpOff + qadr + 4],
            a.qpos[qpOff + qadr + 5], a.qpos[qpOff + qadr + 6],
          ])
          let vang = Vec3f([
            a.qvelIn[qviOff + dadr + 3], a.qvelIn[qviOff + dadr + 4], a.qvelIn[qviOff + dadr + 5],
          ])
          let qn = quatIntegrate(q, vang, dt)
          a.qpos[qpOff + qadr + 3] = qn[0]
          a.qpos[qpOff + qadr + 4] = qn[1]
          a.qpos[qpOff + qadr + 5] = qn[2]
          a.qpos[qpOff + qadr + 6] = qn[3]
        elif jnttype == JointTypeBall:
          let q = Quatf([
            a.qpos[qpOff + qadr], a.qpos[qpOff + qadr + 1],
            a.qpos[qpOff + qadr + 2], a.qpos[qpOff + qadr + 3],
          ])
          let vang = Vec3f([a.qvelIn[qviOff + dadr], a.qvelIn[qviOff + dadr + 1], a.qvelIn[qviOff + dadr + 2]])
          let qn = quatIntegrate(q, vang, dt)
          a.qpos[qpOff + qadr] = qn[0]
          a.qpos[qpOff + qadr + 1] = qn[1]
          a.qpos[qpOff + qadr + 2] = qn[2]
          a.qpos[qpOff + qadr + 3] = qn[3]
        else:
          a.qpos[qpOff + qadr] += dt * a.qvelIn[qviOff + dadr]

proc advanceDense*(
  nworld, njnt, nv, timestepWorlds: int,
  qposStride, qvelStride, qaccStride, qvelInStride: int,
  timestep: uint64,
  jntType, jntQposadr, jntDofadr: uint64,
  qpos, qvel, qvelIn, qacc: uint64,
): void {.exportpy: "advance_dense".} =
  var args = AdvanceArgs(
    nworld: nworld, njnt: njnt, nv: nv, timestepWorlds: timestepWorlds,
    qposStride: qposStride, qvelStride: qvelStride, qaccStride: qaccStride, qvelInStride: qvelInStride,
    timestep: toPtr[float32](timestep),
    jntType: toPtr[int32](jntType), jntQposadr: toPtr[int32](jntQposadr), jntDofadr: toPtr[int32](jntDofadr),
    qpos: toPtr[float32](qpos), qvel: toPtr[float32](qvel), qvelIn: toPtr[float32](qvelIn),
    qacc: toPtr[float32](qacc),
  )
  runJob(advanceWorker, addr args, nworld)

# Applied forces: map body forces to dof space via the dof's subtree.

type
  ApplyFtArgs = object
    nworld, nbody, nv, nvPad: int
    flgAdd: bool
    xiposStride, comStride, cdofStride, ftStride, qfrcStride, isdofStride: int
    xipos, subtreeCom, cdof, ft, qfrc: ptr UncheckedArray[float32]
    bodyParentid, bodyRootid, dofBodyid: ptr UncheckedArray[int32]

proc applyFtWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr ApplyFtArgs](arg)
  while true:
    let w = claimTask()
    if w >= a.nworld:
      break
    let ftOff = w * a.ftStride
    var nz = 0
    for b in 0 ..< a.nbody:
      var any = false
      for c in 0 ..< 6:
        if a.ft[ftOff + b * 6 + c] != 0.0'f32:
          any = true
          break
      if any:
        nz += 1
    if nz == 0:
      if not a.flgAdd:
        for dof in 0 ..< a.nv:
          a.qfrc[w * a.qfrcStride + dof] = 0.0'f32
      continue

    let cdofOff = w * a.cdofStride
    let xiOff = w * a.xiposStride
    let comOff = w * a.comStride
    let qfOff = w * a.qfrcStride
    for dof in 0 ..< a.nv:
      var accumul = 0.0'f32
      let c0 = a.cdof[cdofOff + dof * 6]
      let c1 = a.cdof[cdofOff + dof * 6 + 1]
      let c2 = a.cdof[cdofOff + dof * 6 + 2]
      let c3 = a.cdof[cdofOff + dof * 6 + 3]
      let c4 = a.cdof[cdofOff + dof * 6 + 4]
      let c5 = a.cdof[cdofOff + dof * 6 + 5]
      let dofbodyid = int(a.dofBodyid[dof])
      for b in 0 ..< a.nbody:
        var zero = true
        for c in 0 ..< 6:
          if a.ft[ftOff + b * 6 + c] != 0.0'f32:
            zero = false
            break
        if zero:
          continue
        # body b is in the subtree of dofbodyid iff dofbodyid is an ancestor-or-self of b
        var p = b
        while p != 0 and p != dofbodyid:
          p = int(a.bodyParentid[p])
        if p == 0:
          continue
        let root = int(a.bodyRootid[b])
        let ox = a.xipos[xiOff + b * 3] - a.subtreeCom[comOff + root * 3]
        let oy = a.xipos[xiOff + b * 3 + 1] - a.subtreeCom[comOff + root * 3 + 1]
        let oz = a.xipos[xiOff + b * 3 + 2] - a.subtreeCom[comOff + root * 3 + 2]
        # cross(rotational_cdof, offset)
        let cx = c1 * oz - c2 * oy
        let cy = c2 * ox - c0 * oz
        let cz = c0 * oy - c1 * ox
        # jac = (lin, ang) = (c3..c5, c0..c2)
        accumul +=
          c3 * a.ft[ftOff + b * 6] + c4 * a.ft[ftOff + b * 6 + 1] + c5 * a.ft[ftOff + b * 6 + 2] +
          c0 * a.ft[ftOff + b * 6 + 3] + c1 * a.ft[ftOff + b * 6 + 4] + c2 * a.ft[ftOff + b * 6 + 5] +
          cx * a.ft[ftOff + b * 6] + cy * a.ft[ftOff + b * 6 + 1] + cz * a.ft[ftOff + b * 6 + 2]
      if a.flgAdd:
        a.qfrc[qfOff + dof] += accumul
      else:
        a.qfrc[qfOff + dof] = accumul

proc applyFtDense*(
  nworld, nbody, nv, nvPad: int,
  flgAdd: int,
  xiposStride, comStride, cdofStride, ftStride, qfrcStride: int,
  xipos, subtreeCom, cdof, ft, qfrc: uint64,
  bodyParentid, bodyRootid, dofBodyid: uint64,
): void {.exportpy: "apply_ft_dense".} =
  var args = ApplyFtArgs(
    nworld: nworld, nbody: nbody, nv: nv, nvPad: nvPad, flgAdd: flgAdd != 0,
    xiposStride: xiposStride, comStride: comStride, cdofStride: cdofStride,
    ftStride: ftStride, qfrcStride: qfrcStride,
    xipos: toPtr[float32](xipos), subtreeCom: toPtr[float32](subtreeCom), cdof: toPtr[float32](cdof),
    ft: toPtr[float32](ft), qfrc: toPtr[float32](qfrc),
    bodyParentid: toPtr[int32](bodyParentid), bodyRootid: toPtr[int32](bodyRootid),
    dofBodyid: toPtr[int32](dofBodyid),
  )
  runJob(applyFtWorker, addr args, nworld)

# ------------------------------------------------------------------------------
# Actuation fast paths (joint transmissions to scalar dofs, fixed gains, no bias
# or activation dynamics).  Mirrors smooth._transmission, forward._actuator_velocity,
# forward._actuator_force and forward._qfrc_actuator for that subset; the Python
# side falls back to the Warp kernels otherwise.

type
  TransmissionArgs = object
    nworld, nu: int
    gearWorlds, gearStride: int
    qposStride: int
    lengthStride, momentStride, rowNnzStride, rowAdrStride, colindStride: int
    trnid, jntDofadr, jntQposadr: ptr UncheckedArray[int32]
    gear, qpos, actuatorLength, actuatorMoment: ptr UncheckedArray[float32]
    momentRownnz, momentRowadr, momentColind: ptr UncheckedArray[int32]

proc transmissionWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr TransmissionArgs](arg)
  let chunk = max(1, (a.nworld + nthreads * 4 - 1) div (nthreads * 4))
  while true:
    let first = claimTask() * chunk
    if first >= a.nworld:
      break
    let last = min(a.nworld, first + chunk)
    for w in first ..< last:
      let qpOff = w * a.qposStride
      let gwOff = (w mod a.gearWorlds) * a.gearStride
      let lenOff = w * a.lengthStride
      let rnOff = w * a.rowNnzStride
      let raOff = w * a.rowAdrStride
      let colOff = w * a.colindStride
      let momOff = w * a.momentStride
      for actid in 0 ..< a.nu:
        let jntid = int(a.trnid[actid * 2])
        let vadr = int(a.jntDofadr[jntid])
        let qadr = int(a.jntQposadr[jntid])
        let gear0 = a.gear[gwOff + actid * 6]
        a.actuatorLength[lenOff + actid] = a.qpos[qpOff + qadr] * gear0
        a.momentRownnz[rnOff + actid] = 1'i32
        a.momentRowadr[raOff + actid] = int32(actid)
        a.momentColind[colOff + actid] = int32(vadr)
        a.actuatorMoment[momOff + actid] = gear0

proc transmissionDense*(
  nworld, nu, gearWorlds, gearStride, qposStride: int,
  lengthStride, momentStride, rowNnzStride, rowAdrStride, colindStride: int,
  trnid, jntDofadr, jntQposadr, gear, qpos: uint64,
  actuatorLength, actuatorMoment, momentRownnz, momentRowadr, momentColind: uint64,
): void {.exportpy: "transmission_dense".} =
  var args = TransmissionArgs(
    nworld: nworld, nu: nu, gearWorlds: gearWorlds, gearStride: gearStride, qposStride: qposStride,
    lengthStride: lengthStride, momentStride: momentStride, rowNnzStride: rowNnzStride,
    rowAdrStride: rowAdrStride, colindStride: colindStride,
    trnid: toPtr[int32](trnid), jntDofadr: toPtr[int32](jntDofadr), jntQposadr: toPtr[int32](jntQposadr),
    gear: toPtr[float32](gear), qpos: toPtr[float32](qpos),
    actuatorLength: toPtr[float32](actuatorLength), actuatorMoment: toPtr[float32](actuatorMoment),
    momentRownnz: toPtr[int32](momentRownnz), momentRowadr: toPtr[int32](momentRowadr),
    momentColind: toPtr[int32](momentColind),
  )
  runJob(transmissionWorker, addr args, nworld)

type
  ActuatorVelocityArgs = object
    nworld, nu: int
    qvelStride, actuatorVelocityStride: int
    rowNnzStride, rowAdrStride, colindStride, momentStride: int
    momentRownnz, momentRowadr, momentColind: ptr UncheckedArray[int32]
    actuatorMoment, qvel, actuatorVelocity: ptr UncheckedArray[float32]

proc actuatorVelocityWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr ActuatorVelocityArgs](arg)
  let chunk = max(1, (a.nworld + nthreads * 4 - 1) div (nthreads * 4))
  while true:
    let first = claimTask() * chunk
    if first >= a.nworld:
      break
    let last = min(a.nworld, first + chunk)
    for w in first ..< last:
      let qvOff = w * a.qvelStride
      let avOff = w * a.actuatorVelocityStride
      let rnOff = w * a.rowNnzStride
      let raOff = w * a.rowAdrStride
      let colOff = w * a.colindStride
      let momOff = w * a.momentStride
      for actid in 0 ..< a.nu:
        let rownnz = int(a.momentRownnz[rnOff + actid])
        let rowadr = int(a.momentRowadr[raOff + actid])
        var vel = 0.0'f32
        for i in 0 ..< rownnz:
          let sparseid = rowadr + i
          let colind = int(a.momentColind[colOff + sparseid])
          vel += a.actuatorMoment[momOff + sparseid] * a.qvel[qvOff + colind]
        a.actuatorVelocity[avOff + actid] = vel

proc actuatorVelocityDense*(
  nworld, nu, qvelStride, actuatorVelocityStride: int,
  rowNnzStride, rowAdrStride, colindStride, momentStride: int,
  momentRownnz, momentRowadr, momentColind: uint64,
  actuatorMoment, qvel, actuatorVelocity: uint64,
): void {.exportpy: "actuator_velocity_dense".} =
  var args = ActuatorVelocityArgs(
    nworld: nworld, nu: nu,
    qvelStride: qvelStride, actuatorVelocityStride: actuatorVelocityStride,
    rowNnzStride: rowNnzStride, rowAdrStride: rowAdrStride,
    colindStride: colindStride, momentStride: momentStride,
    momentRownnz: toPtr[int32](momentRownnz), momentRowadr: toPtr[int32](momentRowadr),
    momentColind: toPtr[int32](momentColind),
    actuatorMoment: toPtr[float32](actuatorMoment), qvel: toPtr[float32](qvel),
    actuatorVelocity: toPtr[float32](actuatorVelocity),
  )
  runJob(actuatorVelocityWorker, addr args, nworld)

type
  ActuatorForceArgs = object
    nworld, nu, nv, actuatorForceStride: int
    clampCtrl: bool
    gravityEnabled: bool
    ctrlWorlds, ctrlRangeWorlds, gainWorlds, gainStride: int
    forceRangeWorlds, forceRangeStride: int
    actFrcRangeWorlds, actFrcRangeStride: int
    ctrlStride, ctrlRangeStride, forceStride, qfrcStride, gravcompStride, qvelStride: int
    rowNnzStride, rowAdrStride, colindStride, momentStride: int
    ctrladr: ptr UncheckedArray[int32]
    trnid: ptr UncheckedArray[int32]
    dofJntid, jntActgravcomp: ptr UncheckedArray[int32]
    ctrllimited, forcelimited, jntActfrclimited: ptr UncheckedArray[bool]
    ctrl, ctrlRange, gainprm, forcerange, jntActfrcrange: ptr UncheckedArray[float32]
    actuatorMoment, actuatorVelocity, actuatorForce, qfrcActuator, qfrcGravcomp: ptr UncheckedArray[float32]
    momentRownnz, momentRowadr, momentColind: ptr UncheckedArray[int32]
    qfrcActuatorStride: int

proc actuatorForceWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr ActuatorForceArgs](arg)
  let chunk = max(1, (a.nworld + nthreads * 4 - 1) div (nthreads * 4))
  while true:
    let first = claimTask() * chunk
    if first >= a.nworld:
      break
    let last = min(a.nworld, first + chunk)
    for w in first ..< last:
      let awOff = w * a.actuatorForceStride
      let qfOff = a.qfrcActuatorStride
      let rnOff = w * a.rowNnzStride
      let raOff = w * a.rowAdrStride
      let colOff = w * a.colindStride
      let momOff = w * a.momentStride
      let crOff = (w mod a.ctrlRangeWorlds) * a.ctrlRangeStride
      let grOff = (w mod a.gainWorlds) * a.gainStride
      let frOff = (w mod a.forceRangeWorlds) * a.forceRangeStride
      # forces first (accumulated into a zeroed qfrc_actuator)
      for actid in 0 ..< a.nu:
        let uadr = int(a.ctrladr[actid])
        var ctrl = a.ctrl[(w mod a.ctrlWorlds) * a.ctrlStride + uadr]
        if a.ctrllimited[uadr] and not a.clampCtrl:
          let lo = a.ctrlRange[crOff + uadr * 2]
          let hi = a.ctrlRange[crOff + uadr * 2 + 1]
          if ctrl < lo: ctrl = lo
          elif ctrl > hi: ctrl = hi
        var force = a.gainprm[grOff + actid * 10] * ctrl
        if a.forcelimited[actid]:
          let lo = a.forcerange[frOff + actid * 2]
          let hi = a.forcerange[frOff + actid * 2 + 1]
          if force < lo: force = lo
          elif force > hi: force = hi
        a.actuatorForce[w * a.forceStride + actid] = force

        let rownnz = int(a.momentRownnz[rnOff + actid])
        let rowadr = int(a.momentRowadr[raOff + actid])
        for i in 0 ..< rownnz:
          let sparseid = rowadr + i
          let colind = int(a.momentColind[colOff + sparseid])
          a.qfrcActuator[w * qfOff + colind] += a.actuatorMoment[momOff + sparseid] * force

      # gravcomp + joint actuator force limits
      let arOff = (w mod a.actFrcRangeWorlds) * a.actFrcRangeStride
      for dof in 0 ..< a.nv:
        let jntid = int(a.dofJntid[dof])
        var qfrc = a.qfrcActuator[w * qfOff + dof]
        if a.gravityEnabled and a.jntActgravcomp[jntid] != 0:
          qfrc += a.qfrcGravcomp[w * a.gravcompStride + dof]
        if a.jntActfrclimited[jntid]:
          let lo = a.jntActfrcrange[arOff + jntid * 2]
          let hi = a.jntActfrcrange[arOff + jntid * 2 + 1]
          if qfrc < lo: qfrc = lo
          elif qfrc > hi: qfrc = hi
        a.qfrcActuator[w * qfOff + dof] = qfrc

proc actuatorForceDense*(
  nworld, nu, nv, clampCtrl, gravityEnabled: int,
  ctrlWorlds, ctrlRangeWorlds, gainWorlds, gainStride, forceRangeWorlds, forceRangeStride: int,
  actFrcRangeWorlds, actFrcRangeStride: int,
  ctrlStride, ctrlRangeStride, forceStride, qfrcStride, gravcompStride, qvelStride: int,
  rowNnzStride, rowAdrStride, colindStride, momentStride: int,
  ctrladr, trnid, ctrllimited, forcelimited, dofJntid, jntActgravcomp, jntActfrclimited: uint64,
  ctrl, ctrlRange, gainprm, forcerange, jntActfrcrange: uint64,
  actuatorMoment, actuatorVelocity, actuatorForce, qfrcActuator, qfrcGravcomp: uint64,
  momentRownnz, momentRowadr, momentColind: uint64,
): void {.exportpy: "actuator_force_dense".} =
  var args = ActuatorForceArgs(
    nworld: nworld, nu: nu, nv: nv, actuatorForceStride: forceStride,
    clampCtrl: clampCtrl != 0, gravityEnabled: gravityEnabled != 0,
    ctrlWorlds: ctrlWorlds, ctrlRangeWorlds: ctrlRangeWorlds, gainWorlds: gainWorlds, gainStride: gainStride,
    forceRangeWorlds: forceRangeWorlds, forceRangeStride: forceRangeStride,
    actFrcRangeWorlds: actFrcRangeWorlds, actFrcRangeStride: actFrcRangeStride,
    ctrlStride: ctrlStride, ctrlRangeStride: ctrlRangeStride, forceStride: forceStride,
    qfrcStride: qfrcStride, gravcompStride: gravcompStride, qvelStride: qvelStride,
    ctrladr: toPtr[int32](ctrladr), trnid: toPtr[int32](trnid),
    ctrllimited: toPtr[bool](ctrllimited), forcelimited: toPtr[bool](forcelimited),
    dofJntid: toPtr[int32](dofJntid), jntActgravcomp: toPtr[int32](jntActgravcomp),
    jntActfrclimited: toPtr[bool](jntActfrclimited),
    ctrl: toPtr[float32](ctrl), ctrlRange: toPtr[float32](ctrlRange), gainprm: toPtr[float32](gainprm),
    forcerange: toPtr[float32](forcerange), jntActfrcrange: toPtr[float32](jntActfrcrange),
    actuatorMoment: toPtr[float32](actuatorMoment), actuatorVelocity: toPtr[float32](actuatorVelocity),
    actuatorForce: toPtr[float32](actuatorForce), qfrcActuator: toPtr[float32](qfrcActuator),
    qfrcGravcomp: toPtr[float32](qfrcGravcomp),
    momentRownnz: toPtr[int32](momentRownnz), momentRowadr: toPtr[int32](momentRowadr),
    momentColind: toPtr[int32](momentColind),
    qfrcActuatorStride: qfrcStride,
    rowNnzStride: rowNnzStride, rowAdrStride: rowAdrStride,
    colindStride: colindStride, momentStride: momentStride,
  )
  runJob(actuatorForceWorker, addr args, nworld)

# ------------------------------------------------------------------------------
# Passive spring/damper dof forces and the qfrc_passive reduction
# (mirror of passive._spring_damper_dof_passive / _qfrc_passive_kernel).

func polyForce(linear: float32, p0, p1, x: float32, flgOdd: bool): float32 {.inline.} =
  let xv = if flgOdd: abs(x) else: x
  linear + p0 * xv + p1 * xv * xv

func quatToVel(q: Quatf): Vec3f {.inline.} =
  let ax = q[1]
  let ay = q[2]
  let az = q[3]
  let sinA2 = sqrt(ax * ax + ay * ay + az * az)
  if sinA2 == 0.0'f32:
    return Vec3f([0.0'f32, 0.0'f32, 0.0'f32])
  var speed = 2.0'f32 * arctan2(sinA2, q[0])
  if speed > PI:
    speed -= 2.0'f32 * PI
  Vec3f([ax * speed / sinA2, ay * speed / sinA2, az * speed / sinA2])

func quatSub(qa, qb: Quatf): Vec3f {.inline.} =
  let qneg = Quatf([qb[0], -qb[1], -qb[2], -qb[3]])
  quatToVel(mulQuat(qneg, qa))

type
  SpringDamperArgs = object
    nworld, njnt: int
    dsblSpring, dsblDamper: bool
    stiffnessWorlds, stiffnessStride, stiffpolyWorlds, stiffpolyStride: int
    dampingWorlds, dampingStride, dampingpolyWorlds, dampingpolyStride: int
    qposspringWorlds, qposspringStride, qposStride, qvelStride: int
    springStride, damperStride: int
    jntType, jntQposadr, jntDofadr: ptr UncheckedArray[int32]
    jntStiffness, jntStiffnessPoly, dofDamping, dofDampingPoly, qposSpring: ptr UncheckedArray[float32]
    qpos, qvel, qfrcSpring, qfrcDamper: ptr UncheckedArray[float32]

proc springDamperWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr SpringDamperArgs](arg)
  let chunk = max(1, (a.nworld + nthreads * 4 - 1) div (nthreads * 4))
  while true:
    let first = claimTask() * chunk
    if first >= a.nworld:
      break
    let last = min(a.nworld, first + chunk)
    for w in first ..< last:
      let qpOff = w * a.qposStride
      let qvOff = w * a.qvelStride
      let spOff = w * a.springStride
      let dmOff = w * a.damperStride
      let jw = w mod a.stiffnessWorlds
      let djw = w mod a.dampingWorlds
      let pw = w mod a.stiffpolyWorlds
      let pdw = w mod a.dampingpolyWorlds
      let qsw = w mod a.qposspringWorlds

      for jntid in 0 ..< a.njnt:
        let dofid = int(a.jntDofadr[jntid])
        let jnttype = a.jntType[jntid]
        let stiffness = a.jntStiffness[jw * a.stiffnessStride + jntid]
        let sp0 = a.jntStiffnessPoly[pw * a.stiffpolyStride + jntid * 2]
        let sp1 = a.jntStiffnessPoly[pw * a.stiffpolyStride + jntid * 2 + 1]
        let hasStiffness = (stiffness != 0.0'f32 or sp0 != 0.0'f32 or sp1 != 0.0'f32) and not a.dsblSpring

        var ndof = 1
        if jnttype == JointTypeFree:
          ndof = 6
        elif jnttype == JointTypeBall:
          ndof = 3

        if a.dsblDamper:
          for i in 0 ..< ndof:
            a.qfrcDamper[dmOff + dofid + i] = 0.0'f32
        else:
          for i in 0 ..< ndof:
            let dof = dofid + i
            let damping = a.dofDamping[djw * a.dampingStride + dof]
            let dp0 = a.dofDampingPoly[pdw * a.dampingpolyStride + dof * 2]
            let dp1 = a.dofDampingPoly[pdw * a.dampingpolyStride + dof * 2 + 1]
            if damping != 0.0'f32 or dp0 != 0.0'f32 or dp1 != 0.0'f32:
              let v = a.qvel[qvOff + dof]
              a.qfrcDamper[dmOff + dof] = -v * polyForce(damping, dp0, dp1, v, true)
            else:
              a.qfrcDamper[dmOff + dof] = 0.0'f32

        if not hasStiffness:
          for i in 0 ..< ndof:
            a.qfrcSpring[spOff + dofid + i] = 0.0'f32
          continue

        let qadr = int(a.jntQposadr[jntid])
        let qsOff = qsw * a.qposspringStride
        if jnttype == JointTypeFree:
          let d0 = a.qpos[qpOff + qadr] - a.qposSpring[qsOff + qadr]
          let d1 = a.qpos[qpOff + qadr + 1] - a.qposSpring[qsOff + qadr + 1]
          let d2 = a.qpos[qpOff + qadr + 2] - a.qposSpring[qsOff + qadr + 2]
          let r = sqrt(d0 * d0 + d1 * d1 + d2 * d2)
          let k = polyForce(stiffness, sp0, sp1, r, false)
          a.qfrcSpring[spOff + dofid] = -k * d0
          a.qfrcSpring[spOff + dofid + 1] = -k * d1
          a.qfrcSpring[spOff + dofid + 2] = -k * d2

          let rot = normQuat(Quatf([
            a.qpos[qpOff + qadr + 3], a.qpos[qpOff + qadr + 4],
            a.qpos[qpOff + qadr + 5], a.qpos[qpOff + qadr + 6],
          ]))
          let qref = Quatf([
            a.qposSpring[qsOff + qadr + 3], a.qposSpring[qsOff + qadr + 4],
            a.qposSpring[qsOff + qadr + 5], a.qposSpring[qsOff + qadr + 6],
          ])
          let dif = quatSub(rot, qref)
          let rRot = sqrt(dif[0] * dif[0] + dif[1] * dif[1] + dif[2] * dif[2])
          let kRot = polyForce(stiffness, sp0, sp1, rRot, false)
          a.qfrcSpring[spOff + dofid + 3] = -kRot * dif[0]
          a.qfrcSpring[spOff + dofid + 4] = -kRot * dif[1]
          a.qfrcSpring[spOff + dofid + 5] = -kRot * dif[2]
        elif jnttype == JointTypeBall:
          let rot = normQuat(Quatf([
            a.qpos[qpOff + qadr], a.qpos[qpOff + qadr + 1],
            a.qpos[qpOff + qadr + 2], a.qpos[qpOff + qadr + 3],
          ]))
          let qref = Quatf([
            a.qposSpring[qsOff + qadr], a.qposSpring[qsOff + qadr + 1],
            a.qposSpring[qsOff + qadr + 2], a.qposSpring[qsOff + qadr + 3],
          ])
          let dif = quatSub(rot, qref)
          let r = sqrt(dif[0] * dif[0] + dif[1] * dif[1] + dif[2] * dif[2])
          let k = polyForce(stiffness, sp0, sp1, r, false)
          a.qfrcSpring[spOff + dofid] = -k * dif[0]
          a.qfrcSpring[spOff + dofid + 1] = -k * dif[1]
          a.qfrcSpring[spOff + dofid + 2] = -k * dif[2]
        else:
          let fdif = a.qpos[qpOff + qadr] - a.qposSpring[qsOff + qadr]
          a.qfrcSpring[spOff + dofid] = -fdif * polyForce(stiffness, sp0, sp1, fdif, false)

proc springDamperDense*(
  nworld, njnt, dsblSpring, dsblDamper: int,
  stiffnessWorlds, stiffnessStride, stiffpolyWorlds, stiffpolyStride: int,
  dampingWorlds, dampingStride, dampingpolyWorlds, dampingpolyStride: int,
  qposspringWorlds, qposspringStride, qposStride, qvelStride, springStride, damperStride: int,
  jntType, jntQposadr, jntDofadr: uint64,
  jntStiffness, jntStiffnessPoly, dofDamping, dofDampingPoly, qposSpring: uint64,
  qpos, qvel, qfrcSpring, qfrcDamper: uint64,
): void {.exportpy: "spring_damper_dense".} =
  var args = SpringDamperArgs(
    nworld: nworld, njnt: njnt, dsblSpring: dsblSpring != 0, dsblDamper: dsblDamper != 0,
    stiffnessWorlds: stiffnessWorlds, stiffnessStride: stiffnessStride,
    stiffpolyWorlds: stiffpolyWorlds, stiffpolyStride: stiffpolyStride,
    dampingWorlds: dampingWorlds, dampingStride: dampingStride,
    dampingpolyWorlds: dampingpolyWorlds, dampingpolyStride: dampingpolyStride,
    qposspringWorlds: qposspringWorlds, qposspringStride: qposspringStride,
    qposStride: qposStride, qvelStride: qvelStride, springStride: springStride, damperStride: damperStride,
    jntType: toPtr[int32](jntType), jntQposadr: toPtr[int32](jntQposadr), jntDofadr: toPtr[int32](jntDofadr),
    jntStiffness: toPtr[float32](jntStiffness), jntStiffnessPoly: toPtr[float32](jntStiffnessPoly),
    dofDamping: toPtr[float32](dofDamping), dofDampingPoly: toPtr[float32](dofDampingPoly),
    qposSpring: toPtr[float32](qposSpring),
    qpos: toPtr[float32](qpos), qvel: toPtr[float32](qvel),
    qfrcSpring: toPtr[float32](qfrcSpring), qfrcDamper: toPtr[float32](qfrcDamper),
  )
  runJob(springDamperWorker, addr args, nworld)

type
  QfrcPassiveArgs = object
    nworld, nv: int
    gravityEnabled, hasFluid, flgAdhesion: bool
    springStride, damperStride, gravcompStride, fluidStride, adhesionStride, passiveStride: int
    jntActgravcomp, dofJntid: ptr UncheckedArray[int32]
    qfrcSpring, qfrcDamper, qfrcGravcomp, qfrcFluid, qfrcAdhesion, qfrcPassive: ptr UncheckedArray[float32]

proc qfrcPassiveWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr QfrcPassiveArgs](arg)
  let chunk = max(1, (a.nworld + nthreads * 4 - 1) div (nthreads * 4))
  while true:
    let first = claimTask() * chunk
    if first >= a.nworld:
      break
    let last = min(a.nworld, first + chunk)
    for w in first ..< last:
      for dof in 0 ..< a.nv:
        var p = a.qfrcSpring[w * a.springStride + dof] + a.qfrcDamper[w * a.damperStride + dof]
        if a.gravityEnabled and a.jntActgravcomp[int(a.dofJntid[dof])] == 0:
          p += a.qfrcGravcomp[w * a.gravcompStride + dof]
        if a.hasFluid:
          p += a.qfrcFluid[w * a.fluidStride + dof]
        if a.flgAdhesion:
          p += a.qfrcAdhesion[w * a.adhesionStride + dof]
        a.qfrcPassive[w * a.passiveStride + dof] = p

proc qfrcPassiveDense*(
  nworld, nv, gravityEnabled, hasFluid, flgAdhesion: int,
  springStride, damperStride, gravcompStride, fluidStride, adhesionStride, passiveStride: int,
  jntActgravcomp, dofJntid: uint64,
  qfrcSpring, qfrcDamper, qfrcGravcomp, qfrcFluid, qfrcAdhesion, qfrcPassive: uint64,
): void {.exportpy: "qfrc_passive_dense".} =
  var args = QfrcPassiveArgs(
    nworld: nworld, nv: nv, gravityEnabled: gravityEnabled != 0, hasFluid: hasFluid != 0,
    flgAdhesion: flgAdhesion != 0,
    springStride: springStride, damperStride: damperStride, gravcompStride: gravcompStride,
    fluidStride: fluidStride, adhesionStride: adhesionStride, passiveStride: passiveStride,
    jntActgravcomp: toPtr[int32](jntActgravcomp), dofJntid: toPtr[int32](dofJntid),
    qfrcSpring: toPtr[float32](qfrcSpring), qfrcDamper: toPtr[float32](qfrcDamper),
    qfrcGravcomp: toPtr[float32](qfrcGravcomp), qfrcFluid: toPtr[float32](qfrcFluid),
    qfrcAdhesion: toPtr[float32](qfrcAdhesion), qfrcPassive: toPtr[float32](qfrcPassive),
  )
  runJob(qfrcPassiveWorker, addr args, nworld)

# ------------------------------------------------------------------------------
# Composite rigid body inertias and the joint-space inertia matrix
# (mirror of smooth.crb / _M / _crb_accumulate), plus the qfrc_smooth reduction.

type
  CrbArgs = object
    nworld, nbody, nlevels: int
    cinertStride, crbStride: int
    bodyParentid, bodyTree: ptr UncheckedArray[int32]
    treeOffsets: ptr UncheckedArray[int32]
    cinert, crb: ptr UncheckedArray[float32]

proc crbWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr CrbArgs](arg)
  let chunk = max(1, (a.nworld + nthreads * 4 - 1) div (nthreads * 4))
  while true:
    let first = claimTask() * chunk
    if first >= a.nworld:
      break
    let last = min(a.nworld, first + chunk)
    for w in first ..< last:
      let ciOff = w * a.cinertStride
      let crOff = w * a.crbStride
      for b in 0 ..< a.nbody * 10:
        a.crb[crOff + b] = a.cinert[ciOff + b]
      for level in countdown(a.nlevels - 1, 0):
        for idx in int(a.treeOffsets[level]) ..< int(a.treeOffsets[level + 1]):
          let b = int(a.bodyTree[idx])
          let p = int(a.bodyParentid[b])
          if p == 0:
            continue  # the world body does not accumulate crb
          for c in 0 ..< 10:
            a.crb[crOff + p * 10 + c] += a.crb[crOff + b * 10 + c]

proc crbDense*(
  nworld, nbody, nlevels: int,
  cinertStride, crbStride: int,
  bodyParentid, bodyTree, treeOffsets: uint64,
  cinert, crb: uint64,
): void {.exportpy: "crb_dense".} =
  var args = CrbArgs(
    nworld: nworld, nbody: nbody, nlevels: nlevels,
    cinertStride: cinertStride, crbStride: crbStride,
    bodyParentid: toPtr[int32](bodyParentid), bodyTree: toPtr[int32](bodyTree),
    treeOffsets: toPtr[int32](treeOffsets),
    cinert: toPtr[float32](cinert), crb: toPtr[float32](crb),
  )
  runJob(crbWorker, addr args, nworld)

func inertVec(c: ptr UncheckedArray[float32], co: int, v: ptr UncheckedArray[float32], vo: int): array[6, float32] {.inline.} =
  # mju_mulInertVec: 6D vector times 6D inertia
  let i0 = c[co]
  let i1 = c[co + 1]
  let i2 = c[co + 2]
  let i3 = c[co + 3]
  let i4 = c[co + 4]
  let i5 = c[co + 5]
  let i6 = c[co + 6]
  let i7 = c[co + 7]
  let i8 = c[co + 8]
  let i9 = c[co + 9]
  array[6, float32]([
    i0 * v[vo] + i3 * v[vo + 1] + i4 * v[vo + 2] - i8 * v[vo + 4] + i7 * v[vo + 5],
    i3 * v[vo] + i1 * v[vo + 1] + i5 * v[vo + 2] + i8 * v[vo + 3] - i6 * v[vo + 5],
    i4 * v[vo] + i5 * v[vo + 1] + i2 * v[vo + 2] - i7 * v[vo + 3] + i6 * v[vo + 4],
    i8 * v[vo + 1] - i7 * v[vo + 2] + i9 * v[vo + 3],
    i6 * v[vo + 2] - i8 * v[vo] + i9 * v[vo + 4],
    i7 * v[vo] - i6 * v[vo + 1] + i9 * v[vo + 5],
  ])

type
  MassMatrixArgs = object
    nworld, nv: int
    armatureWorlds, armatureStride: int
    cdofStride, crbStride, MStride: int
    dofBodyid, dofParentid, mRownnz, mRowadr: ptr UncheckedArray[int32]
    dofArmature, cdof, crb, M: ptr UncheckedArray[float32]

proc massMatrixWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr MassMatrixArgs](arg)
  let chunk = max(1, (a.nworld + nthreads * 4 - 1) div (nthreads * 4))
  while true:
    let first = claimTask() * chunk
    if first >= a.nworld:
      break
    let last = min(a.nworld, first + chunk)
    for w in first ..< last:
      let cdOff = w * a.cdofStride
      let crOff = w * a.crbStride
      let mOff = w * a.MStride
      let armOff = (w mod a.armatureWorlds) * a.armatureStride
      for dof0 in 0 ..< a.nv:
        let bodyid = int(a.dofBodyid[dof0])
        var madrIj = int(a.mRowadr[dof0]) + int(a.mRownnz[dof0]) - 1
        a.M[mOff + madrIj] = a.dofArmature[armOff + dof0]
        let buf = inertVec(a.crb, crOff + bodyid * 10, a.cdof, cdOff + dof0 * 6)
        var dofid = dof0
        while dofid >= 0:
          var dot = 0.0'f32
          for c in 0 ..< 6:
            dot += a.cdof[cdOff + dofid * 6 + c] * buf[c]
          a.M[mOff + madrIj] += dot
          madrIj -= 1
          dofid = int(a.dofParentid[dofid])

proc massMatrixDense*(
  nworld, nv, armatureWorlds, armatureStride, cdofStride, crbStride, MStride: int,
  dofBodyid, dofParentid, mRownnz, mRowadr: uint64,
  dofArmature, cdof, crb, M: uint64,
): void {.exportpy: "mass_matrix_dense".} =
  var args = MassMatrixArgs(
    nworld: nworld, nv: nv, armatureWorlds: armatureWorlds, armatureStride: armatureStride,
    cdofStride: cdofStride, crbStride: crbStride, MStride: MStride,
    dofBodyid: toPtr[int32](dofBodyid), dofParentid: toPtr[int32](dofParentid),
    mRownnz: toPtr[int32](mRownnz), mRowadr: toPtr[int32](mRowadr),
    dofArmature: toPtr[float32](dofArmature), cdof: toPtr[float32](cdof),
    crb: toPtr[float32](crb), M: toPtr[float32](M),
  )
  runJob(massMatrixWorker, addr args, nworld)

type
  QfrcSmoothArgs = object
    nworld, nv: int
    enableSleep: bool
    appliedStride, awakeStride, biasStride, passiveStride, actuatorStride, smoothStride: int
    bodyTreeid, dofBodyid: ptr UncheckedArray[int32]
    qfrcApplied, treeAwake, qfrcBias, qfrcPassive, qfrcActuator, qfrcSmooth: ptr UncheckedArray[float32]

proc qfrcSmoothWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr QfrcSmoothArgs](arg)
  let chunk = max(1, (a.nworld + nthreads * 4 - 1) div (nthreads * 4))
  while true:
    let first = claimTask() * chunk
    if first >= a.nworld:
      break
    let last = min(a.nworld, first + chunk)
    for w in first ..< last:
      for dof in 0 ..< a.nv:
        if a.enableSleep:
          let tree = int(a.bodyTreeid[int(a.dofBodyid[dof])])
          if tree >= 0 and a.treeAwake[w * a.awakeStride + tree] == 0:
            a.qfrcSmooth[w * a.smoothStride + dof] = 0.0'f32
            continue
        a.qfrcSmooth[w * a.smoothStride + dof] =
          a.qfrcPassive[w * a.passiveStride + dof] -
          a.qfrcBias[w * a.biasStride + dof] +
          a.qfrcActuator[w * a.actuatorStride + dof] +
          a.qfrcApplied[w * a.appliedStride + dof]

proc qfrcSmoothDense*(
  nworld, nv, enableSleep: int,
  appliedStride, awakeStride, biasStride, passiveStride, actuatorStride, smoothStride: int,
  bodyTreeid, dofBodyid: uint64,
  qfrcApplied, treeAwake, qfrcBias, qfrcPassive, qfrcActuator, qfrcSmooth: uint64,
): void {.exportpy: "qfrc_smooth_dense".} =
  var args = QfrcSmoothArgs(
    nworld: nworld, nv: nv, enableSleep: enableSleep != 0,
    appliedStride: appliedStride, awakeStride: awakeStride, biasStride: biasStride,
    passiveStride: passiveStride, actuatorStride: actuatorStride, smoothStride: smoothStride,
    bodyTreeid: toPtr[int32](bodyTreeid), dofBodyid: toPtr[int32](dofBodyid),
    qfrcApplied: toPtr[float32](qfrcApplied), treeAwake: toPtr[float32](treeAwake),
    qfrcBias: toPtr[float32](qfrcBias), qfrcPassive: toPtr[float32](qfrcPassive),
    qfrcActuator: toPtr[float32](qfrcActuator), qfrcSmooth: toPtr[float32](qfrcSmooth),
  )
  runJob(qfrcSmoothWorker, addr args, nworld)

# ------------------------------------------------------------------------------
# Primitive narrowphase (mirror of collision_primitive._primitive_narrowphase).
#
# Warp launches one lane per naconmax slot (nworld * nconmax, ~12x the number of
# broadphase pairs on the humanoid), and every active lane runs the whole
# contact_params + geometry + write_contact chain on a single CPU core.  Here the
# pairs are walked directly: contacts are computed per pair in parallel into a
# scratch buffer, then appended in pair order so the contact arrays (and the
# downstream efcid assignment) match the serial Warp layout exactly.

const
  NpKmax = 8  # plane-box emits up to 8 contacts
  NpParamStride = 17  # friction 5, solref 2, solreffriction 2, solimp 5, margin, gap, adhesion
  NpContactStride = 14  # dist, pos 3, frame 9, original contact index
  NpIntStride = 8  # count, condim, pairid0, pairid1, worldid, geom1, geom2, append start
  CtConstraint = 1'i32
  CtSensor = 2'i32

var
  npFbuf: seq[float32]
  npIbuf: seq[int32]

type
  NpGeom = object
    pos, size, normal: Vec3f
    rot: array[9, float32]  # column-major, rot[c * 3 + r]
    index: int32
    margin: float32

  NpArgs = object
    nworld, ncollision, naconmax, njmax, ngeom, npair: int
    marginLead, gapLead, adhesionLead, solmixLead, frictionLead, solrefLead, solimpLead, sizeLead: int
    pmarginLead, pgapLead, pfrictionLead, psolrefLead, psolreffrictionLead, psolimpLead, padhesionLead: int
    geomType, geomCondim, geomPriority, pairDim: ptr UncheckedArray[int32]
    geomSolmix, geomSolref, geomSolimp, geomFriction, geomMargin, geomGap, geomAdhesion, geomSize: ptr UncheckedArray[float32]
    pairSolref, pairSolreffriction, pairSolimp, pairMargin, pairGap, pairAdhesion, pairFriction: ptr UncheckedArray[float32]
    geomXpos, geomXmat: ptr UncheckedArray[float32]
    collisionPair, collisionPairid, collisionWorldid, ncollisionPtr: ptr UncheckedArray[int32]
    contactDist, contactPos, contactFrame, contactIncludemargin, contactFriction: ptr UncheckedArray[float32]
    contactSolref, contactSolreffriction, contactSolimp, contactAdhesion: ptr UncheckedArray[float32]
    contactDim, contactGeom, contactEfcAddress, contactWorldid, contactType, contactGeomCollisionId: ptr UncheckedArray[int32]
    nacon: ptr UncheckedArray[int32]
    fbuf: ptr UncheckedArray[float32]
    ibuf: ptr UncheckedArray[int32]

func `+`(a, b: Vec3f): Vec3f {.inline.} = [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
func `-`(a, b: Vec3f): Vec3f {.inline.} = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
func `-`(a: Vec3f): Vec3f {.inline.} = [-a[0], -a[1], -a[2]]
func `*`(a: Vec3f, s: float32): Vec3f {.inline.} = [a[0] * s, a[1] * s, a[2] * s]
func `*`(s: float32, a: Vec3f): Vec3f {.inline.} = a * s
func `*`(a: Vec3f, b: Vec3f): Vec3f {.inline.} = [a[0] * b[0], a[1] * b[1], a[2] * b[2]]  # cw_mul
func `/`(a: Vec3f, s: float32): Vec3f {.inline.} = [a[0] / s, a[1] / s, a[2] / s]

func npDot(a, b: Vec3f): float32 {.inline.} = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

func npSign(x: float32): float32 {.inline.} =
  if x < 0.0'f32: -1.0'f32 else: 1.0'f32

func npCross(a, b: Vec3f): Vec3f {.inline.} =
  [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

func npLen(a: Vec3f): float32 {.inline.} =
  sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])

func npNormalize(a: Vec3f): Vec3f {.inline.} =
  let l = npLen(a)
  if l > 0.0'f32: a / l else: [0.0'f32, 0.0'f32, 0.0'f32]

func npNormalizeWithNorm(a: Vec3f): (Vec3f, float32) {.inline.} =
  let norm = npLen(a)
  if norm == 0.0'f32: (a, 0.0'f32) else: (a / norm, norm)

func npMatVec(m: array[9, float32], v: Vec3f): Vec3f {.inline.} =
  # warp mul(mat, vec): row-wise dot, r[j] = m[j * 3] * v0 + m[j * 3 + 1] * v1 + ...
  [m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
   m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
   m[6] * v[0] + m[7] * v[1] + m[8] * v[2]]

func npMatTVec(m: array[9, float32], v: Vec3f): Vec3f {.inline.} =
  # transpose(m) @ v = column-wise dot
  let c0 = [m[0], m[1], m[2]] * v[0]
  let c1 = [m[3], m[4], m[5]] * v[1]
  let c2 = [m[6], m[7], m[8]] * v[2]
  c0 + c1 + c2

func npOrthogonals(a: Vec3f): (Vec3f, Vec3f) {.inline.} =
  var b = if -0.5'f32 < a[1] and a[1] < 0.5'f32: [0.0'f32, 1.0'f32, 0.0'f32]
          else: [0.0'f32, 0.0'f32, 1.0'f32]
  b = b - a * npDot(a, b)
  b = npNormalize(b)
  if npLen(a) == 0.0'f32:
    b = [0.0'f32, 0.0'f32, 0.0'f32]
  (b, npCross(a, b))

func npMakeFrame(a: Vec3f): array[9, float32] {.inline.} =
  let n = npNormalize(a)
  let (b, c) = npOrthogonals(n)
  [n[0], n[1], n[2], b[0], b[1], b[2], c[0], c[1], c[2]]

func npClosestSegmentPoint(a, b, pt: Vec3f): Vec3f {.inline.} =
  let ab = b - a
  let t = npDot(pt - a, ab) / (npDot(ab, ab) + 1e-6'f32)
  a + min(max(t, 0.0'f32), 1.0'f32) * ab  # careful: see below

func npPlaneSphere(n, planePos, spherePos: Vec3f, radius: float32): (float32, Vec3f) {.inline.} =
  let dist = npDot(spherePos - planePos, n) - radius
  (dist, spherePos - n * (radius + 0.5'f32 * dist))

func npSphereSphere(pos1: Vec3f, r1: float32, pos2: Vec3f, r2: float32): (float32, Vec3f, Vec3f) {.inline.} =
  let dir = pos2 - pos1
  var dist = npLen(dir)
  let n = if dist == 0.0'f32: [1.0'f32, 0.0'f32, 0.0'f32] else: dir / dist
  dist = dist - (r1 + r2)
  (dist, pos1 + n * (r1 + 0.5'f32 * dist), n)

func npSphereCapsule(spherePos: Vec3f, sphereRadius: float32,
                     capPos, capAxis: Vec3f, capRadius, capHalfLen: float32): (float32, Vec3f, Vec3f) {.inline.} =
  let segment = capAxis * capHalfLen
  let pt = npClosestSegmentPoint(capPos - segment, capPos + segment, spherePos)
  npSphereSphere(spherePos, sphereRadius, pt, capRadius)

func npPlaneCapsule(n, planePos, capPos, capAxis: Vec3f,
                    radius, halfLen: float32): (array[2, float32], array[2, Vec3f], array[9, float32]) {.inline.} =
  var b: Vec3f
  var bNorm: float32
  (b, bNorm) = npNormalizeWithNorm(capAxis - n * npDot(n, capAxis))
  if bNorm < 0.5'f32:
    if -0.5'f32 < n[1] and n[1] < 0.5'f32:
      b = [0.0'f32, 1.0'f32, 0.0'f32]
    else:
      b = [0.0'f32, 0.0'f32, 1.0'f32]
  let c = npCross(n, b)
  let frame = [n[0], n[1], n[2], b[0], b[1], b[2], c[0], c[1], c[2]]
  let segment = capAxis * halfLen
  let (dist1, pos1) = npPlaneSphere(n, planePos, capPos + segment, radius)
  let (dist2, pos2) = npPlaneSphere(n, planePos, capPos - segment, radius)
  ([dist1, dist2], [pos1, pos2], frame)

func npSphereCylinder(spherePos: Vec3f, sphereRadius: float32,
                      cylinderPos, cylinderAxis: Vec3f,
                      cylinderRadius, cylinderHalfHeight: float32): (float32, Vec3f, Vec3f) {.inline.} =
  let vec = spherePos - cylinderPos
  let x = npDot(vec, cylinderAxis)
  let aProj = cylinderAxis * x
  let pProj = vec - aProj
  let pProjSqr = npDot(pProj, pProj)
  var collideSide = abs(x) < cylinderHalfHeight
  let collideCap = pProjSqr < (cylinderRadius * cylinderRadius)
  var useSide = collideSide
  var useCap = collideCap
  if collideSide and collideCap:
    let distCap = cylinderHalfHeight - abs(x)
    let distRadius = cylinderRadius - sqrt(pProjSqr)
    if distCap < distRadius:
      useSide = false
    else:
      useCap = false
  if useSide:
    let posTarget = cylinderPos + aProj
    npSphereSphere(spherePos, sphereRadius, posTarget, cylinderRadius)
  elif useCap:
    var posCap: Vec3f
    var planeNormal: Vec3f
    if x > 0.0'f32:
      posCap = cylinderPos + cylinderAxis * cylinderHalfHeight
      planeNormal = cylinderAxis
    else:
      posCap = cylinderPos - cylinderAxis * cylinderHalfHeight
      planeNormal = -cylinderAxis
    let (dist, pos) = npPlaneSphere(planeNormal, posCap, spherePos, sphereRadius)
    (dist, pos, -planeNormal)
  else:
    let invLen = safeDiv(1.0'f32, sqrt(pProjSqr))
    let pProjScaled = pProj * (cylinderRadius * invLen)
    let capOffset = cylinderAxis * (npSign(x) * cylinderHalfHeight)
    let posCorner = cylinderPos + capOffset + pProjScaled
    npSphereSphere(spherePos, sphereRadius, posCorner, 0.0'f32)

func npPlaneEllipsoid(n, planePos, ellipsoidPos: Vec3f, ellipsoidRot: array[9, float32],
                      ellipsoidSize: Vec3f): (float32, Vec3f) {.inline.} =
  let sphereSupport = -npNormalize(npMatTVec(ellipsoidRot, n) * ellipsoidSize)
  var pos = ellipsoidPos + npMatVec(ellipsoidRot, sphereSupport * ellipsoidSize)
  let dist = npDot(n, pos - planePos)
  pos = pos - n * dist * 0.5'f32
  (dist, pos)

func npPlaneCylinder(n, planePos, cylinderCenter, cylinderAxis: Vec3f,
                     cylinderRadius, cylinderHalfHeight: float32): (array[4, float32], array[4, Vec3f]) {.inline.} =
  var contactDist = [MjMaxVal, MjMaxVal, MjMaxVal, MjMaxVal]
  var contactPos: array[4, Vec3f]
  var axis = cylinderAxis
  var prjaxis = npDot(n, axis)
  if prjaxis > 0.0'f32:
    axis = -axis
    prjaxis = -prjaxis
  let dist0 = npDot(cylinderCenter - planePos, n)
  var vec = axis * prjaxis - n
  let lenSqr = npDot(vec, vec)
  if lenSqr >= 1e-12'f32:
    vec = vec * safeDiv(cylinderRadius, sqrt(lenSqr))
  else:
    vec = [cylinderRadius, 0.0'f32, 0.0'f32]
  let prjvec = npDot(vec, n)
  let axisScaled = axis * cylinderHalfHeight
  let prjaxisScaled = prjaxis * cylinderHalfHeight
  let dist1 = dist0 + prjaxisScaled + prjvec
  let pos1 = cylinderCenter + vec + axisScaled - n * (dist1 * 0.5'f32)
  contactDist[0] = dist1
  contactPos[0] = pos1
  let dist2 = dist0 - prjaxisScaled + prjvec
  let pos2 = cylinderCenter + vec - axisScaled - n * (dist2 * 0.5'f32)
  contactDist[1] = dist2
  contactPos[1] = pos2
  let prjvec1 = -prjvec * 0.5'f32
  let dist3 = dist0 + prjaxisScaled + prjvec1
  var vec1 = npCross(vec, axisScaled)
  vec1 = npNormalize(vec1) * (cylinderRadius * sqrt(3.0'f32) * 0.5'f32)
  let pos3 = cylinderCenter + vec1 + axisScaled - vec * 0.5'f32 - n * (dist3 * 0.5'f32)
  contactDist[2] = dist3
  contactPos[2] = pos3
  let pos4 = cylinderCenter - vec1 + axisScaled - vec * 0.5'f32 - n * (dist3 * 0.5'f32)
  contactDist[3] = dist3
  contactPos[3] = pos4
  (contactDist, contactPos)

func npPlaneBox(n, planePos, boxPos: Vec3f, boxRot: array[9, float32],
                boxSize: Vec3f): (array[8, float32], array[8, Vec3f]) {.inline.} =
  let centerDist = npDot(boxPos - planePos, n)
  var dist: array[8, float32]
  var pos: array[8, Vec3f]
  for i in 0 ..< 8:
    var corner = [
      (if (i and 1) != 0: boxSize[0] else: -boxSize[0]),
      (if (i and 2) != 0: boxSize[1] else: -boxSize[1]),
      (if (i and 4) != 0: boxSize[2] else: -boxSize[2]),
    ]
    corner = npMatVec(boxRot, corner)
    let ldist = npDot(n, corner)
    let cdist = centerDist + ldist
    dist[i] = cdist
    pos[i] = corner + boxPos - 0.5'f32 * n * cdist
  (dist, pos)

func npCapsuleCapsule(cap1Pos, cap1Axis: Vec3f, cap1Radius, cap1HalfLen: float32,
                      cap2Pos, cap2Axis: Vec3f, cap2Radius, cap2HalfLen: float32,
                      margin: float32): (array[2, float32], array[2, Vec3f], array[2, Vec3f]) {.inline.} =
  var contactDist: array[2, float32] = [float32(Inf), float32(Inf)]
  var contactPos: array[2, Vec3f]
  var contactNormal: array[2, Vec3f]
  let axis1 = cap1Axis * cap1HalfLen
  let axis2 = cap2Axis * cap2HalfLen
  let dif = cap1Pos - cap2Pos
  let ma = npDot(axis1, axis1)
  let mb = -npDot(axis1, axis2)
  let mc = npDot(axis2, axis2)
  let u = -npDot(axis1, dif)
  let v = npDot(axis2, dif)
  let det = ma * mc - mb * mb
  if abs(det) >= MjMinVal:
    let invDet = 1.0'f32 / det
    var x1 = (mc * u - mb * v) * invDet
    var x2 = (ma * v - mb * u) * invDet
    if x1 > 1.0'f32:
      x1 = 1.0'f32
      x2 = (v - mb) / mc
    elif x1 < -1.0'f32:
      x1 = -1.0'f32
      x2 = (v + mb) / mc
    if x2 > 1.0'f32:
      x2 = 1.0'f32
      x1 = min(max((u - mb) / ma, -1.0'f32), 1.0'f32)
    elif x2 < -1.0'f32:
      x2 = -1.0'f32
      x1 = min(max((u + mb) / ma, -1.0'f32), 1.0'f32)
    let vec1 = cap1Pos + axis1 * x1
    let vec2 = cap2Pos + axis2 * x2
    let (dist, pos, normal) = npSphereSphere(vec1, cap1Radius, vec2, cap2Radius)
    if dist <= margin:
      contactDist[0] = dist
      contactPos[0] = pos
      contactNormal[0] = normal
  else:
    var contactCount = 0
    block:
      let vec1 = cap1Pos + axis1
      let x2 = min(max((v - mb) / mc, -1.0'f32), 1.0'f32)
      let vec2 = cap2Pos + axis2 * x2
      let (dist, pos, normal) = npSphereSphere(vec1, cap1Radius, vec2, cap2Radius)
      if dist <= margin:
        contactDist[contactCount] = dist
        contactPos[contactCount] = pos
        contactNormal[contactCount] = normal
        inc contactCount
    block:
      let vec1 = cap1Pos - axis1
      let x2 = min(max((v + mb) / mc, -1.0'f32), 1.0'f32)
      let vec2 = cap2Pos + axis2 * x2
      let (dist, pos, normal) = npSphereSphere(vec1, cap1Radius, vec2, cap2Radius)
      if dist <= margin:
        contactDist[contactCount] = dist
        contactPos[contactCount] = pos
        contactNormal[contactCount] = normal
        inc contactCount
    if contactCount < 2:
      let vec2 = cap2Pos + axis2
      let x1 = min(max((u - mb) / ma, -1.0'f32), 1.0'f32)
      let vec1 = cap1Pos + axis1 * x1
      let (dist, pos, normal) = npSphereSphere(vec1, cap1Radius, vec2, cap2Radius)
      if dist <= margin:
        contactDist[contactCount] = dist
        contactPos[contactCount] = pos
        contactNormal[contactCount] = normal
        inc contactCount
    if contactCount < 2:
      let vec2 = cap2Pos - axis2
      let x1 = min(max((u + mb) / ma, -1.0'f32), 1.0'f32)
      let vec1 = cap1Pos + axis1 * x1
      let (dist, pos, normal) = npSphereSphere(vec1, cap1Radius, vec2, cap2Radius)
      if dist <= margin:
        contactDist[contactCount] = dist
        contactPos[contactCount] = pos
        contactNormal[contactCount] = normal
  (contactDist, contactPos, contactNormal)

func npContactMarginGap(a: ptr NpArgs, geoms: array[2, int32], pairid: int32,
                        worldid: int): (float32, float32) {.inline.} =
  if pairid > -1:
    let pr = worldid mod a.pmarginLead
    (a.pairMargin[pr * a.npair + int(pairid)],
     a.pairGap[worldid mod a.pgapLead * a.npair + int(pairid)])
  else:
    let g1 = int(geoms[0])
    let g2 = int(geoms[1])
    let mw = worldid mod a.marginLead
    let gw = worldid mod a.gapLead
    (a.geomMargin[mw * a.ngeom + g1] + a.geomMargin[mw * a.ngeom + g2],
     a.geomGap[gw * a.ngeom + g1] + a.geomGap[gw * a.ngeom + g2])

func npContactMaterial(a: ptr NpArgs, geoms: array[2, int32], pairid: int32, worldid: int,
                       condim: var int32, friction: var array[5, float32],
                       solref: var array[2, float32], solreffriction: var array[2, float32],
                       solimp: var array[5, float32], adhesion: var float32) {.inline.} =
  if pairid > -1:
    let rr = worldid mod a.pfrictionLead
    condim = a.pairDim[int(pairid)]
    for k in 0 ..< 5:
      friction[k] = a.pairFriction[rr * a.npair * 5 + int(pairid) * 5 + k]
    let sr = worldid mod a.psolrefLead
    solref[0] = a.pairSolref[sr * a.npair * 2 + int(pairid) * 2]
    solref[1] = a.pairSolref[sr * a.npair * 2 + int(pairid) * 2 + 1]
    let sfr = worldid mod a.psolreffrictionLead
    solreffriction[0] = a.pairSolreffriction[sfr * a.npair * 2 + int(pairid) * 2]
    solreffriction[1] = a.pairSolreffriction[sfr * a.npair * 2 + int(pairid) * 2 + 1]
    let si = worldid mod a.psolimpLead
    for k in 0 ..< 5:
      solimp[k] = a.pairSolimp[si * a.npair * 5 + int(pairid) * 5 + k]
    adhesion = a.pairAdhesion[worldid mod a.padhesionLead * a.npair + int(pairid)]
  else:
    let g1 = int(geoms[0])
    let g2 = int(geoms[1])
    let smw = worldid mod a.solmixLead
    let frw = worldid mod a.frictionLead
    let srefw = worldid mod a.solrefLead
    let simpw = worldid mod a.solimpLead
    let adw = worldid mod a.adhesionLead
    let solmix1 = a.geomSolmix[smw * a.ngeom + g1]
    let solmix2 = a.geomSolmix[smw * a.ngeom + g2]
    let condim1 = a.geomCondim[g1]
    let condim2 = a.geomCondim[g2]
    let p1 = a.geomPriority[g1]
    let p2 = a.geomPriority[g2]
    let a1 = a.geomAdhesion[adw * a.ngeom + g1]
    let a2 = a.geomAdhesion[adw * a.ngeom + g2]
    var mix: float32
    var maxF: array[3, float32]
    if p1 > p2:
      mix = 1.0'f32
      condim = condim1
      for k in 0 ..< 3:
        maxF[k] = a.geomFriction[frw * a.ngeom + g1 * 3 + k]
      adhesion = a1
    elif p2 > p1:
      mix = 0.0'f32
      condim = condim2
      for k in 0 ..< 3:
        maxF[k] = a.geomFriction[frw * a.ngeom + g2 * 3 + k]
      adhesion = a2
    else:
      mix = safeDiv(solmix1, solmix1 + solmix2)
      if solmix1 < MjMinVal and solmix2 < MjMinVal:
        mix = 0.5'f32
      if solmix1 < MjMinVal and solmix2 >= MjMinVal:
        mix = 0.0'f32
      if solmix1 >= MjMinVal and solmix2 < MjMinVal:
        mix = 1.0'f32
      condim = max(condim1, condim2)
      for k in 0 ..< 3:
        maxF[k] = max(a.geomFriction[frw * a.ngeom + g1 * 3 + k],
                      a.geomFriction[frw * a.ngeom + g2 * 3 + k])
      adhesion = a1 + a2
    friction = [maxF[0], maxF[0], maxF[1], maxF[2], maxF[2]]
    if a.geomSolref[srefw * a.ngeom + g1 * 2] > 0.0'f32 and
       a.geomSolref[srefw * a.ngeom + g2 * 2] > 0.0'f32:
      solref[0] = mix * a.geomSolref[srefw * a.ngeom + g1 * 2] +
                  (1.0'f32 - mix) * a.geomSolref[srefw * a.ngeom + g2 * 2]
      solref[1] = mix * a.geomSolref[srefw * a.ngeom + g1 * 2 + 1] +
                  (1.0'f32 - mix) * a.geomSolref[srefw * a.ngeom + g2 * 2 + 1]
    else:
      solref[0] = min(a.geomSolref[srefw * a.ngeom + g1 * 2],
                      a.geomSolref[srefw * a.ngeom + g2 * 2])
      solref[1] = min(a.geomSolref[srefw * a.ngeom + g1 * 2 + 1],
                      a.geomSolref[srefw * a.ngeom + g2 * 2 + 1])
    solreffriction = [0.0'f32, 0.0'f32]
    for k in 0 ..< 5:
      solimp[k] = mix * a.geomSolimp[simpw * a.ngeom + g1 * 5 + k] +
                  (1.0'f32 - mix) * a.geomSolimp[simpw * a.ngeom + g2 * 5 + k]

  for k in 0 ..< 5:
    friction[k] = max(MjMinMu, friction[k])

func npReadGeom(a: ptr NpArgs, geoms: array[2, int32], worldid, which: int,
                sizeLead: int): NpGeom {.inline.} =
  let g = int(geoms[which])
  let base = worldid * a.ngeom + g
  result.pos = [a.geomXpos[base * 3], a.geomXpos[base * 3 + 1], a.geomXpos[base * 3 + 2]]
  for k in 0 ..< 9:
    result.rot[k] = a.geomXmat[base * 9 + k]
  let sw = worldid mod sizeLead
  result.size = [a.geomSize[sw * a.ngeom + g * 3],
                 a.geomSize[sw * a.ngeom + g * 3 + 1],
                 a.geomSize[sw * a.ngeom + g * 3 + 2]]
  result.normal = [result.rot[2], result.rot[5], result.rot[8]]  # rot[0,2],rot[1,2],rot[2,2]
  result.index = -1
  result.margin = 0.0'f32

# Geometry dispatch for one pair; returns the number of contacts (0 for pairs
# without a fast path -- the Python side only enables the op when every pair
# type present in the model is covered below).
proc npPairContacts(ty1, ty2: int32, g1, g2: NpGeom, margin: float32,
                    dist: var array[NpKmax, float32], pos: var array[NpKmax, Vec3f],
                    frame: var array[NpKmax, array[9, float32]]): int =
  if ty1 == 0 and ty2 == 2:  # plane, sphere
    let (d, p) = npPlaneSphere(g1.normal, g1.pos, g2.pos, g2.size[0])
    dist[0] = d
    pos[0] = p
    frame[0] = npMakeFrame(g1.normal)
    return 1
  if ty1 == 0 and ty2 == 3:  # plane, capsule
    let axis = [g2.rot[2], g2.rot[5], g2.rot[8]]
    let (d, p, f) = npPlaneCapsule(g1.normal, g1.pos, g2.pos, axis, g2.size[0], g2.size[1])
    for i in 0 ..< 2:
      dist[i] = d[i]
      pos[i] = p[i]
      frame[i] = f
    return 2
  if ty1 == 0 and ty2 == 4:  # plane, ellipsoid
    let (d, p) = npPlaneEllipsoid(g1.normal, g1.pos, g2.pos, g2.rot, g2.size)
    dist[0] = d
    pos[0] = p
    frame[0] = npMakeFrame(g1.normal)
    return 1
  if ty1 == 0 and ty2 == 5:  # plane, cylinder
    let axis = [g2.rot[2], g2.rot[5], g2.rot[8]]
    let (d, p) = npPlaneCylinder(g1.normal, g1.pos, g2.pos, axis, g2.size[0], g2.size[1])
    for i in 0 ..< 4:
      dist[i] = d[i]
      pos[i] = p[i]
      frame[i] = npMakeFrame(g1.normal)
    return 4
  if ty1 == 0 and ty2 == 6:  # plane, box
    let (d, p) = npPlaneBox(g1.normal, g1.pos, g2.pos, g2.rot, g2.size)
    for i in 0 ..< 8:
      dist[i] = d[i]
      pos[i] = p[i]
      frame[i] = npMakeFrame(g1.normal)
    return 8
  if ty1 == 2 and ty2 == 2:  # sphere, sphere
    let (d, p, n) = npSphereSphere(g1.pos, g1.size[0], g2.pos, g2.size[0])
    dist[0] = d
    pos[0] = p
    frame[0] = npMakeFrame(n)
    return 1
  if ty1 == 2 and ty2 == 3:  # sphere, capsule
    let axis = [g2.rot[2], g2.rot[5], g2.rot[8]]
    let (d, p, n) = npSphereCapsule(g1.pos, g1.size[0], g2.pos, axis, g2.size[0], g2.size[1])
    dist[0] = d
    pos[0] = p
    frame[0] = npMakeFrame(n)
    return 1
  if ty1 == 2 and ty2 == 5:  # sphere, cylinder
    let axis = [g2.rot[2], g2.rot[5], g2.rot[8]]
    let (d, p, n) = npSphereCylinder(g1.pos, g1.size[0], g2.pos, axis, g2.size[0], g2.size[1])
    dist[0] = d
    pos[0] = p
    frame[0] = npMakeFrame(n)
    return 1
  if ty1 == 3 and ty2 == 3:  # capsule, capsule
    let axis1 = [g1.rot[2], g1.rot[5], g1.rot[8]]
    let axis2 = [g2.rot[2], g2.rot[5], g2.rot[8]]
    let (d, p, n) = npCapsuleCapsule(g1.pos, axis1, g1.size[0], g1.size[1],
                                     g2.pos, axis2, g2.size[0], g2.size[1], margin)
    for i in 0 ..< 2:
      dist[i] = d[i]
      pos[i] = p[i]
      frame[i] = npMakeFrame(n[i])
    return 2
  0

const NpChunk = 16  # pairs per atomic claim

proc claimChunk(): int {.inline, gcsafe.} =
  pool.nextTask.fetchAdd(NpChunk, moRelaxed)

proc npPairWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr NpArgs](arg)
  let last = a.ncollision
  while true:
    let base = claimChunk()
    if base >= last:
      break
    let stop = min(last, base + NpChunk)
    for cid in base ..< stop:
      let geoms = [a.collisionPair[cid * 2], a.collisionPair[cid * 2 + 1]]
      let worldid = int(a.collisionWorldid[cid])
      let pairid = a.collisionPairid[cid * 2]

      var margin, gap: float32
      (margin, gap) = npContactMarginGap(a, geoms, pairid, worldid)
      var condim: int32
      var friction: array[5, float32]
      var solref: array[2, float32]
      var solreffriction: array[2, float32]
      var solimp: array[5, float32]
      var adhesion: float32
      npContactMaterial(a, geoms, pairid, worldid, condim, friction, solref, solreffriction, solimp, adhesion)

      let g1 = npReadGeom(a, geoms, worldid, 0, a.sizeLead)
      let g2 = npReadGeom(a, geoms, worldid, 1, a.sizeLead)
      let ty1 = a.geomType[int(geoms[0])]
      let ty2 = a.geomType[int(geoms[1])]

      var dist: array[NpKmax, float32]
      var pos: array[NpKmax, Vec3f]
      var frame: array[NpKmax, array[9, float32]]
      let count = npPairContacts(ty1, ty2, g1, g2, margin, dist, pos, frame)

      let pOff = cid * NpParamStride
      for k in 0 ..< 5:
        a.fbuf[pOff + k] = friction[k]
      a.fbuf[pOff + 5] = solref[0]
      a.fbuf[pOff + 6] = solref[1]
      a.fbuf[pOff + 7] = solreffriction[0]
      a.fbuf[pOff + 8] = solreffriction[1]
      for k in 0 ..< 5:
        a.fbuf[pOff + 9 + k] = solimp[k]
      a.fbuf[pOff + 14] = margin
      a.fbuf[pOff + 15] = gap
      a.fbuf[pOff + 16] = adhesion

      let cBase = a.ncollision * NpParamStride + cid * NpKmax * NpContactStride
      for i in 0 ..< count:
        let cOff = cBase + i * NpContactStride
        a.fbuf[cOff + 0] = dist[i]
        for k in 0 ..< 3:
          a.fbuf[cOff + 1 + k] = pos[i][k]
        for k in 0 ..< 9:
          a.fbuf[cOff + 4 + k] = frame[i][k]
        a.fbuf[cOff + 13] = float32(i)

      # apply write_contact's filters here so the append can run in parallel:
      # only contacts that would be appended are kept
      let pairid1 = a.collisionPairid[cid * 2 + 1]
      var written = 0
      for i in 0 ..< count:
        let dist0 = a.fbuf[cBase + i * NpContactStride]
        let detected = dist0 < margin + gap
        if (pairid == -2 or not detected) and pairid1 == -1:
          continue
        if written != i:
          let dst = cBase + written * NpContactStride
          let src = cBase + i * NpContactStride
          for k in 0 ..< NpContactStride:
            a.fbuf[dst + k] = a.fbuf[src + k]
        inc written

      let iOff = cid * NpIntStride
      a.ibuf[iOff + 0] = int32(written)
      a.ibuf[iOff + 1] = condim
      a.ibuf[iOff + 2] = pairid
      a.ibuf[iOff + 3] = pairid1
      a.ibuf[iOff + 4] = int32(worldid)
      a.ibuf[iOff + 5] = geoms[0]
      a.ibuf[iOff + 6] = geoms[1]

proc npScanContacts(a: ptr NpArgs) {.nimcall, gcsafe.} =
  # exclusive scan over the per-pair append counts: contact slots follow the
  # serial warp lane order, so the downstream efcid assignment is unchanged
  var run = int(a.nacon[0])
  for c in 0 ..< a.ncollision:
    let iOff = c * NpIntStride
    a.ibuf[iOff + 7] = int32(run)
    run += int(a.ibuf[iOff + 0])
  a.nacon[0] = int32(run)

proc npWriteWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr NpArgs](arg)
  let last = a.ncollision
  while true:
    let base = claimChunk()
    if base >= last:
      break
    let stop = min(last, base + NpChunk)
    for c in base ..< stop:
      let iOff = c * NpIntStride
      let count = int(a.ibuf[iOff + 0])
      if count == 0:
        continue
      let start = int(a.ibuf[iOff + 7])
      let condimIn = a.ibuf[iOff + 1]
      let pairid0 = a.ibuf[iOff + 2]
      let pairid1 = a.ibuf[iOff + 3]
      let worldid = int(a.ibuf[iOff + 4])
      let geom1 = a.ibuf[iOff + 5]
      let geom2 = a.ibuf[iOff + 6]
      let pOff = c * NpParamStride
      let margin = a.fbuf[pOff + 14]
      let gap = a.fbuf[pOff + 15]
      let adhesion = a.fbuf[pOff + 16]
      let cBase = a.ncollision * NpParamStride + c * NpKmax * NpContactStride
      for i in 0 ..< count:
        let slot = start + i
        if slot >= a.naconmax:
          break
        let cOff = cBase + i * NpContactStride
        let dist = a.fbuf[cOff + 0]
        let condim = if (adhesion != 0.0'f32) and (dist >= margin): 1'i32 else: condimIn
        var contactType = 0'i32
        if pairid0 >= -1 and dist < margin + gap:
          contactType = contactType or CtConstraint
        if pairid1 >= 0:
          contactType = contactType or CtSensor
        a.contactDist[slot] = dist
        for k in 0 ..< 3:
          a.contactPos[slot * 3 + k] = a.fbuf[cOff + 1 + k]
        for k in 0 ..< 9:
          a.contactFrame[slot * 9 + k] = a.fbuf[cOff + 4 + k]
        a.contactIncludemargin[slot] = margin
        for k in 0 ..< 5:
          a.contactFriction[slot * 5 + k] = a.fbuf[pOff + k]
        for k in 0 ..< 2:
          a.contactSolref[slot * 2 + k] = a.fbuf[pOff + 5 + k]
          a.contactSolreffriction[slot * 2 + k] = a.fbuf[pOff + 7 + k]
        for k in 0 ..< 5:
          a.contactSolimp[slot * 5 + k] = a.fbuf[pOff + 9 + k]
        a.contactDim[slot] = condim
        a.contactGeom[slot * 2] = geom1
        a.contactGeom[slot * 2 + 1] = geom2
        a.contactWorldid[slot] = int32(worldid)
        a.contactType[slot] = contactType
        a.contactGeomCollisionId[slot] = int32(a.fbuf[cOff + 13])
        a.contactAdhesion[slot] = adhesion
        for k in 0 ..< a.njmax:
          a.contactEfcAddress[slot * a.njmax + k] = -1

proc primitiveNarrowphaseDense*(
  nworld, ncollision, naconmax, njmax, ngeom, npair: int,
  marginLead, gapLead, adhesionLead, solmixLead, frictionLead, solrefLead, solimpLead, sizeLead: int,
  pmarginLead, pgapLead, pfrictionLead, psolrefLead, psolreffrictionLead, psolimpLead, padhesionLead: int,
  geomType, geomCondim, geomPriority, pairDim: uint64,
  geomSolmix, geomSolref, geomSolimp, geomFriction, geomMargin, geomGap, geomAdhesion, geomSize: uint64,
  pairSolref, pairSolreffriction, pairSolimp, pairMargin, pairGap, pairAdhesion, pairFriction: uint64,
  geomXpos, geomXmat: uint64,
  collisionPair, collisionPairid, collisionWorldid, ncollisionIn: uint64,
  contactDist, contactPos, contactFrame, contactIncludemargin, contactFriction: uint64,
  contactSolref, contactSolreffriction, contactSolimp, contactAdhesion: uint64,
  contactDim, contactGeom, contactEfcAddress, contactWorldid, contactType, contactGeomCollisionId: uint64,
  nacon: uint64,
): void {.exportpy: "primitive_narrowphase_dense".} =
  if ncollision <= 0:
    return
  # the broadphase stops recording pairs at naconmax (it keeps counting), and the
  # stock kernel only ever visits the first naconmax lanes: ignore the overflow tail
  let limit = min(ncollision, naconmax)
  if npFbuf.len < limit * NpParamStride + limit * NpKmax * NpContactStride:
    npFbuf.setLen(limit * (NpParamStride + NpKmax * NpContactStride))
  if npIbuf.len < limit * NpIntStride:
    npIbuf.setLen(limit * NpIntStride)
  var args = NpArgs(
    nworld: nworld, ncollision: limit, naconmax: naconmax, njmax: njmax,
    ngeom: ngeom, npair: npair,
    marginLead: marginLead, gapLead: gapLead, adhesionLead: adhesionLead, solmixLead: solmixLead,
    frictionLead: frictionLead, solrefLead: solrefLead, solimpLead: solimpLead, sizeLead: sizeLead,
    pmarginLead: pmarginLead, pgapLead: pgapLead, pfrictionLead: pfrictionLead,
    psolrefLead: psolrefLead, psolreffrictionLead: psolreffrictionLead, psolimpLead: psolimpLead,
    padhesionLead: padhesionLead,
    geomType: toPtr[int32](geomType), geomCondim: toPtr[int32](geomCondim),
    geomPriority: toPtr[int32](geomPriority), pairDim: toPtr[int32](pairDim),
    geomSolmix: toPtr[float32](geomSolmix), geomSolref: toPtr[float32](geomSolref),
    geomSolimp: toPtr[float32](geomSolimp), geomFriction: toPtr[float32](geomFriction),
    geomMargin: toPtr[float32](geomMargin), geomGap: toPtr[float32](geomGap),
    geomAdhesion: toPtr[float32](geomAdhesion), geomSize: toPtr[float32](geomSize),
    pairSolref: toPtr[float32](pairSolref), pairSolreffriction: toPtr[float32](pairSolreffriction),
    pairSolimp: toPtr[float32](pairSolimp), pairMargin: toPtr[float32](pairMargin),
    pairGap: toPtr[float32](pairGap), pairAdhesion: toPtr[float32](pairAdhesion),
    pairFriction: toPtr[float32](pairFriction),
    geomXpos: toPtr[float32](geomXpos), geomXmat: toPtr[float32](geomXmat),
    collisionPair: toPtr[int32](collisionPair), collisionPairid: toPtr[int32](collisionPairid),
    collisionWorldid: toPtr[int32](collisionWorldid), ncollisionPtr: toPtr[int32](ncollisionIn),
    contactDist: toPtr[float32](contactDist), contactPos: toPtr[float32](contactPos),
    contactFrame: toPtr[float32](contactFrame), contactIncludemargin: toPtr[float32](contactIncludemargin),
    contactFriction: toPtr[float32](contactFriction), contactSolref: toPtr[float32](contactSolref),
    contactSolreffriction: toPtr[float32](contactSolreffriction),
    contactSolimp: toPtr[float32](contactSolimp), contactAdhesion: toPtr[float32](contactAdhesion),
    contactDim: toPtr[int32](contactDim), contactGeom: toPtr[int32](contactGeom),
    contactEfcAddress: toPtr[int32](contactEfcAddress), contactWorldid: toPtr[int32](contactWorldid),
    contactType: toPtr[int32](contactType), contactGeomCollisionId: toPtr[int32](contactGeomCollisionId),
    nacon: toPtr[int32](nacon),
    fbuf: cast[ptr UncheckedArray[float32]](addr npFbuf[0]),
    ibuf: cast[ptr UncheckedArray[int32]](addr npIbuf[0]),
  )
  runJob(npPairWorker, addr args, limit)
  npScanContacts(addr args)
  runJob(npWriteWorker, addr args, limit)


# ------------------------------------------------------------------------------
# Recursive Newton-Euler (mirror of smooth.rne): bias forces and internal body
# forces.  Warp runs the five stages as separate single-lane kernels over
# (nworld, *) grids; here one worker walks a whole world, and the branch /
# body-tree order is the serial kernel order, so results stay bit-exact.

type
  RneArgs = object
    nworld, nbody, nbranch, nv, nlevel, gravityWorlds: int
    qvelStride, qaccStride, cdofStride, cdofDotStride, qfrcBiasStride, caccStride: int
    gravityEnabled: bool
    bodyParentid, bodyDofnum, bodyDofadr, bodyBranches, bodyBranchStart, dofBodyid: ptr UncheckedArray[int32]
    bodyTree, treeOffsets: ptr UncheckedArray[int32]
    gravity, qvel, qacc, cdof, cdofDot, cinert, cvel, cacc, cfrcInt, qfrcBias: ptr UncheckedArray[float32]

func inertVecArr(c: ptr UncheckedArray[float32], co: int, v: array[6, float32]): array[6, float32] {.inline.} =
  # mju_mulInertVec: 6D vector times 6D inertia
  let i0 = c[co]
  let i1 = c[co + 1]
  let i2 = c[co + 2]
  let i3 = c[co + 3]
  let i4 = c[co + 4]
  let i5 = c[co + 5]
  let i6 = c[co + 6]
  let i7 = c[co + 7]
  let i8 = c[co + 8]
  let i9 = c[co + 9]
  [i0 * v[0] + i3 * v[1] + i4 * v[2] - i8 * v[4] + i7 * v[5],
   i3 * v[0] + i1 * v[1] + i5 * v[2] + i8 * v[3] - i6 * v[5],
   i4 * v[0] + i5 * v[1] + i2 * v[2] - i7 * v[3] + i6 * v[4],
   i8 * v[1] - i7 * v[2] + i9 * v[3],
   i6 * v[2] - i8 * v[0] + i9 * v[4],
   i7 * v[0] - i6 * v[1] + i9 * v[5]]

func motionCrossForce(v: array[6, float32], f: array[6, float32]): array[6, float32] {.inline.} =
  # ang = cross(v0, f0) + cross(v1, f1); vel = cross(v0, f1)
  let ang0 = (v[1] * f[2] - v[2] * f[1]) + (v[4] * f[5] - v[5] * f[4])
  let ang1 = (v[2] * f[0] - v[0] * f[2]) + (v[5] * f[3] - v[3] * f[5])
  let ang2 = (v[0] * f[1] - v[1] * f[0]) + (v[3] * f[4] - v[4] * f[3])
  let vel0 = v[1] * f[5] - v[2] * f[4]
  let vel1 = v[2] * f[3] - v[0] * f[5]
  let vel2 = v[0] * f[4] - v[1] * f[3]
  [ang0, ang1, ang2, vel0, vel1, vel2]

proc rneWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr RneArgs](arg)
  let chunk = max(1, (a.nworld + nthreads * 4 - 1) div (nthreads * 4))
  while true:
    let first = claimTask() * chunk
    if first >= a.nworld:
      break
    let last = min(a.nworld, first + chunk)
    for w in first ..< last:
      let caccOff = w * a.caccStride
      let cinOff = w * a.nbody * 10  # cinert is 10 wide per body
      let qvOff = w * a.qvelStride
      let qaOff = w * a.qaccStride
      let cdOff = w * a.cdofStride
      let cddOff = w * a.cdofDotStride

      # cacc[0] = spatial_vector(0, -gravity)
      for c in 0 ..< 6:
        a.cacc[caccOff + c] = 0.0'f32
      if a.gravityEnabled:
        let gw = w mod a.gravityWorlds
        for c in 0 ..< 3:
          a.cacc[caccOff + 3 + c] = -a.gravity[gw * 3 + c]

      # forward pass over branches
      for branch in 0 ..< a.nbranch:
        let start = int(a.bodyBranchStart[branch])
        let stop = int(a.bodyBranchStart[branch + 1])
        if start >= stop:
          continue
        let body0 = int(a.bodyBranches[start])
        let pid = int(a.bodyParentid[body0])
        var local: array[6, float32]
        for c in 0 ..< 6:
          local[c] = a.cacc[caccOff + pid * 6 + c]
        for i in start ..< stop:
          let bodyid = int(a.bodyBranches[i])
          let dofnum = int(a.bodyDofnum[bodyid])
          let dofadr = int(a.bodyDofadr[bodyid])
          for j in 0 ..< dofnum:
            let k = 6 * (dofadr + j)
            for c in 0 ..< 6:
              local[c] += a.cdofDot[cddOff + k + c] * a.qvel[qvOff + dofadr + j]
          for c in 0 ..< 6:
            a.cacc[caccOff + bodyid * 6 + c] = local[c]

      # cfrc: body 0 is zero, the rest from cinert, cvel and cacc
      for c in 0 ..< 6:
        a.cfrcInt[caccOff + c] = 0.0'f32
      for bodyid in 1 ..< a.nbody:
        let bo = caccOff + bodyid * 6
        let io = cinOff + bodyid * 10
        var cacc, cvel: array[6, float32]
        for c in 0 ..< 6:
          cacc[c] = a.cacc[bo + c]
          cvel[c] = a.cvel[bo + c]
        let frc = inertVecArr(a.cinert, io, cacc)
        let tmp = inertVecArr(a.cinert, io, cvel)
        let mcf = motionCrossForce(cvel, tmp)
        for c in 0 ..< 6:
          a.cfrcInt[bo + c] = frc[c] + mcf[c]

      # backward pass: level order matches reversed(m.body_tree), node order is
      # the kernel order, so parents accumulate children in the serial order
      for level in countdown(a.nlevel - 1, 0):
        for node in int(a.treeOffsets[level]) ..< int(a.treeOffsets[level + 1]):
          let bodyid = int(a.bodyTree[node])
          if bodyid == 0:
            continue
          let pid = int(a.bodyParentid[bodyid])
          for c in 0 ..< 6:
            a.cfrcInt[caccOff + pid * 6 + c] += a.cfrcInt[caccOff + bodyid * 6 + c]

      # qfrc_bias = cdof . cfrc_int
      for dof in 0 ..< a.nv:
        let bodyid = int(a.dofBodyid[dof])
        let cdofOff = cdOff + dof * 6
        let cfOff = caccOff + bodyid * 6
        var acc = 0.0'f32
        for c in 0 ..< 6:
          acc += a.cdof[cdofOff + c] * a.cfrcInt[cfOff + c]
        a.qfrcBias[w * a.qfrcBiasStride + dof] = acc

proc rneDense*(
  nworld, nbody, nbranch, nv, nlevel, gravityWorlds: int,
  qvelStride, qaccStride, cdofStride, cdofDotStride, qfrcBiasStride, caccStride: int,
  gravityEnabled: int,
  bodyParentid, bodyDofnum, bodyDofadr, bodyBranches, bodyBranchStart, dofBodyid: uint64,
  bodyTree, treeOffsets: uint64,
  gravity, qvel, qacc, cdof, cdofDot, cinert, cvel, cacc, cfrcInt, qfrcBias: uint64,
): void {.exportpy: "rne_dense".} =
  if nworld <= 0:
    return
  var args = RneArgs(
    nworld: nworld, nbody: nbody, nbranch: nbranch, nv: nv, nlevel: nlevel,
    gravityWorlds: gravityWorlds,
    qvelStride: qvelStride, qaccStride: qaccStride, cdofStride: cdofStride,
    cdofDotStride: cdofDotStride, qfrcBiasStride: qfrcBiasStride, caccStride: caccStride,
    gravityEnabled: gravityEnabled != 0,
    bodyParentid: toPtr[int32](bodyParentid), bodyDofnum: toPtr[int32](bodyDofnum),
    bodyDofadr: toPtr[int32](bodyDofadr), bodyBranches: toPtr[int32](bodyBranches),
    bodyBranchStart: toPtr[int32](bodyBranchStart), dofBodyid: toPtr[int32](dofBodyid),
    bodyTree: toPtr[int32](bodyTree), treeOffsets: toPtr[int32](treeOffsets),
    gravity: toPtr[float32](gravity), qvel: toPtr[float32](qvel), qacc: toPtr[float32](qacc),
    cdof: toPtr[float32](cdof), cdofDot: toPtr[float32](cdofDot),
    cinert: toPtr[float32](cinert), cvel: toPtr[float32](cvel), cacc: toPtr[float32](cacc),
    cfrcInt: toPtr[float32](cfrcInt), qfrcBias: toPtr[float32](qfrcBias),
  )
  runJob(rneWorker, addr args, nworld)
