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
import std/cpuinfo
import std/locks
import std/math
import std/typedthreads

import nimpy

const
  Satisfied = 0'i32  # types.ConstraintState.SATISFIED
  Quadratic = 1'i32  # types.ConstraintState.QUADRATIC
  LinearNeg = 2'i32  # types.ConstraintState.LINEARNEG
  LinearPos = 3'i32  # types.ConstraintState.LINEARPOS
  MaxPoolThreads = 64
  AlphaNoiseEps = 8.0'f32 * 1.1920929e-07'f32
  MjMaxVal = 1.0e10'f32
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
  for i in 0 ..< n:
    let iRow = i * stride
    for k in 0 ..< i:
      let kRow = k * stride
      let uki = a[kRow + i]
      for j in i ..< n:
        a[iRow + j] -= uki * a[kRow + j]
    let uii = sqrt(a[iRow + i])
    a[iRow + i] = uii
    let inv = 1.0'f32 / uii
    for j in (i + 1) ..< n:
      a[iRow + j] *= inv

func choleskySolve(a: ptr UncheckedArray[float32], n, stride: int, x: ptr UncheckedArray[float32]) {.inline.} =
  ## In-place solve UᵀUx = b for upper triangular U using unit-stride inner loops.
  for k in 0 ..< n:
    let kRow = k * stride
    let xk = x[k] / a[kRow + k]
    x[k] = xk
    for i in (k + 1) ..< n:
      x[i] -= a[kRow + i] * xk
  for i in countdown(n - 1, 0):
    let iRow = i * stride
    var s = x[i]
    for k in (i + 1) ..< n:
      s -= a[iRow + k] * x[k]
    x[i] = s / a[iRow + i]

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
  w, nv, nvPad, njmax, lsIterations: int,
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
        let jWorldOff = w * njmax * nvPad
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
      w, a.nv, a.nvPad, a.njmax, a.lsIterations,
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
    warmstart, hasHfactor: bool
    optTolerance, optLsTolerance, statMeaninertia: ptr UncheckedArray[float32]
    mColind, mHinitI, mMulmRowadr, mMulmCol, mMulmMadr: ptr UncheckedArray[int32]
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
    let jWorldOff = w * njmax * nvPad
    let gradOff = w * nvPad
    let sOff = w * nv
    let mvOff = w * nv

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
        jr += a.efcJ[jRow + i] * a.qacc[qaOff + i]
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
          a.qfrcConstraint[qcOff + i] += force * a.efcJ[jRow + i]

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
      for i in 0 ..< nv:
        let ji = dk * a.efcJ[jk + i]
        if ji == 0.0'f32:
          continue
        let hRow = i * nvPad
        for jj in i ..< nv:
          hLocal[hRow + jj] += ji * a.efcJ[jk + jj]

    for i in 0 ..< nv:
      let rowOff = i * nvPad + i
      copyMem(addr factorPtr[rowOff], addr hLocal[rowOff], (nv - i) * sizeof(float32))
    choleskyFactorize(factorPtr, nv, nvPad)

    let xw = at(a.search, sOff)
    for i in 0 ..< nv:
      xw[i] = a.grad[gradOff + i]
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
            jvVal += a.efcJ[jRow + i] * a.search[sOff + i]
          a.jv[efcOff + efcid] = jvVal

      linesearchPyramidalWorld(
        w, nv, nvPad, njmax, a.lsIterations,
        a.nTol, a.nLsTol, a.nMeaninertia, a.qfrcSmoothStride, a.qaccStride, a.maStride,
        false, true,
        a.optTolerance, a.optLsTolerance, a.statMeaninertia,
        a.ne, a.nf, a.nefc,
        a.qfrcSmooth, a.efcJ, a.efcD, a.efcFrictionloss,
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
              a.qfrcConstraint[qcOff + i] += force * a.efcJ[jRow + i]

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
            for i in 0 ..< nv:
              let ji = dk * a.efcJ[jk + i]
              if ji == 0.0'f32:
                continue
              let hRow = i * nvPad
              for jj in i ..< nv:
                hLocal[hRow + jj] += ji * a.efcJ[jk + jj]
          for i in 0 ..< nv:
            let rowOff = i * nvPad + i
            copyMem(addr factorPtr[rowOff], addr hLocal[rowOff], (nv - i) * sizeof(float32))
          choleskyFactorize(factorPtr, nv, nvPad)

        for i in 0 ..< nv:
          xw[i] = a.grad[gradOff + i]
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
  )
  runJob(newtonSolveWorker, addr args, if nworld * nv >= 256: nworld else: 1)
