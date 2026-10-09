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

import std/cpuinfo
import std/locks
import std/math
import std/typedthreads

import nimpy

const
  Quadratic = 1  # types.ConstraintState.QUADRATIC
  MaxPoolThreads = 64

type
  Job = proc(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.}

  Pool = object
    lock: Lock
    work, done: Cond
    job: Job
    arg: pointer
    nthreads, generation, remaining: int
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

func chunk(n, tid, nthreads: int): (int, int) {.inline.} =
  ## Half-open range [lo, hi) of [0, n) assigned to thread `tid`.
  let size = max(1, (n + nthreads - 1) div nthreads)
  let lo = tid * size
  (lo, min(n, lo + size))

proc scratch(buf: var seq[float32], n: int): ptr UncheckedArray[float32] {.inline.} =
  if buf.len < n:
    buf.setLen(n)
  result = cast[ptr UncheckedArray[float32]](addr buf[0])
  zeroMem(result, n * sizeof(float32))

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
  numThreads = max(1, min(MaxPoolThreads, countProcessors()))
  workers.setLen(numThreads)
  threads.setLen(numThreads)  # POSIX createThread passes `addr`, so the handles must outlive the workers
  for tid in 0 ..< numThreads:
    workers[tid] = Worker(pool: pool, tid: tid)
    createThread(threads[tid], workerLoop, addr workers[tid])

proc runJob(job: Job, arg: pointer) =
  if pool == nil:
    initPool()
  acquire(pool.lock)
  pool.job = job
  pool.arg = arg
  pool.nthreads = numThreads
  pool.remaining = numThreads
  inc pool.generation
  broadcast(pool.work)
  while pool.remaining > 0:
    wait(pool.done, pool.lock)
  release(pool.lock)

func choleskyFactorize(a: ptr UncheckedArray[float32], n, stride: int) {.inline.} =
  ## In-place upper Cholesky factorization A = UᵀU; only the upper triangle is touched.
  for i in 0 ..< n:
    var d = a[i * stride + i]
    for k in 0 ..< i:
      let uki = a[k * stride + i]
      d -= uki * uki
    let uii = sqrt(d)
    a[i * stride + i] = uii
    let inv = 1.0'f32 / uii
    for j in (i + 1) ..< n:
      var v = a[i * stride + j]
      for k in 0 ..< i:
        v -= a[k * stride + i] * a[k * stride + j]
      a[i * stride + j] = v * inv

func choleskySolve(a: ptr UncheckedArray[float32], n, stride: int, x: ptr UncheckedArray[float32]) {.inline.} =
  ## In-place solve UᵀUx = b for upper triangular U.
  for i in 0 ..< n:
    var s = x[i]
    for k in 0 ..< i:
      s -= a[k * stride + i] * x[k]
    x[i] = s / a[i * stride + i]
  for i in countdown(n - 1, 0):
    var s = x[i]
    for k in (i + 1) ..< n:
      s -= a[i * stride + k] * x[k]
    x[i] = s / a[i * stride + i]

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
  let (lo, hi) = chunk(a.nworld, tid, nthreads)
  for w in lo ..< hi:
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
        for jj in i ..< a.nvPad:
          a.H[hOff + i * a.nvPad + jj] += ji * a.J[jk + jj]

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
  runJob(jtdaJWorker, addr args)

# Cholesky factorize ctx.h (nv x nv upper) and solve for the Newton search direction.

var hCopyBuf {.threadvar.}: seq[float32]

type
  CholeskySolveArgs = object
    nworld, nv, nvPad: int
    skipNoflip: bool
    grad, H, search, searchDot, newtonDecrement: ptr UncheckedArray[float32]
    changed: ptr UncheckedArray[int32]
    done: ptr UncheckedArray[bool]

proc choleskySolveWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr CholeskySolveArgs](arg)
  let hc = scratch(hCopyBuf, a.nvPad * a.nvPad)  # H is reused across iterations, so factorize a copy
  let (lo, hi) = chunk(a.nworld, tid, nthreads)
  for w in lo ..< hi:
    if a.done[w]:
      continue
    if a.skipNoflip and a.changed[w] == 0:
      continue

    let hOff = w * a.nvPad * a.nvPad
    for i in 0 ..< a.nv * a.nvPad:
      hc[i] = a.H[hOff + i]

    choleskyFactorize(hc, a.nv, a.nvPad)

    let gradOff = w * a.nvPad
    let xw = at(a.search, w * a.nv)
    for i in 0 ..< a.nv:
      xw[i] = a.grad[gradOff + i]
    choleskySolve(hc, a.nv, a.nvPad, xw)

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
): void {.exportpy: "cholesky_solve_dense".} =
  var args = CholeskySolveArgs(
    nworld: nworld, nv: nv, nvPad: nvPad, skipNoflip: skipNoflip != 0,
    grad: toPtr[float32](grad), H: toPtr[float32](H), changed: toPtr[int32](changed), done: toPtr[bool](done),
    search: toPtr[float32](search), searchDot: toPtr[float32](searchDot),
    newtonDecrement: toPtr[float32](newtonDecrement),
  )
  runJob(choleskySolveWorker, addr args)

# Dense block Cholesky factorize+solve for the smooth inertia tiles.

var blkBuf {.threadvar.}: seq[float32]

type
  BlockCholeskySolveArgs = object
    nworld, ntiles, size, area, nC, nY, nL: int
    blkAdr, elemid, qldBlockAdr: ptr UncheckedArray[int32]
    M, Y, X, L: ptr UncheckedArray[float32]

proc blockCholeskySolveWorker(arg: pointer, tid, nthreads: int) {.nimcall, gcsafe.} =
  let a = cast[ptr BlockCholeskySolveArgs](arg)
  let bx = scratch(blkBuf, a.area)
  let nTasks = a.nworld * a.ntiles
  let (lo, hi) = chunk(nTasks, tid, nthreads)
  for task in lo ..< hi:
    let w = task div a.ntiles
    let blk = task mod a.ntiles
    let start = int(a.blkAdr[blk])

    let mOff = w * a.nC
    let ebase = blk * a.area
    for s in 0 ..< a.area:
      let idx = int(a.elemid[ebase + s])
      blkBuf[s] = if idx < a.nC: a.M[mOff + idx] else: 0.0'f32

    choleskyFactorize(bx, a.size, a.size)

    let lOff = w * a.nL + int(a.qldBlockAdr[start])
    for s in 0 ..< a.area:
      a.L[lOff + s] = bx[s]

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
  runJob(blockCholeskySolveWorker, addr args)

# Contact Jacobian rows for the dense (non-flex) path.
# One thread per world walks the contact rows and writes J rows plus J @ qvel.

var jacpBuf {.threadvar.}: seq[float32]
var jacrBuf {.threadvar.}: seq[float32]

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
  let jacp = scratch(jacpBuf, nvPad * 3)
  let jacr = scratch(jacrBuf, nvPad * 3)

  let (lo, hi) = chunk(a.nworld, tid, nthreads)
  for w in lo ..< hi:
    let estart = int(a.ne[w]) + int(a.nf[w]) + int(a.nl[w])
    let eend = min(int(a.nefc[w]), a.njmax)
    var prevConid = -1
    var condim = 0
    var f0, f1, f2: array[3, float32]

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
        var off1, off2: array[3, float32]
        for k in 0 ..< 3:
          off1[k] = a.contactPos[pOff + k] - a.subtreeCom[comOff + r1 + k]
          off2[k] = a.contactPos[pOff + k] - a.subtreeCom[comOff + r2 + k]

        f0 = [a.contactFrame[conid * 9 + 0], a.contactFrame[conid * 9 + 1], a.contactFrame[conid * 9 + 2]]
        f1 = [a.contactFrame[conid * 9 + 3], a.contactFrame[conid * 9 + 4], a.contactFrame[conid * 9 + 5]]
        f2 = [a.contactFrame[conid * 9 + 6], a.contactFrame[conid * 9 + 7], a.contactFrame[conid * 9 + 8]]

        for i in 0 ..< nvPad:
          let aff1 = a.bodyIsdofancestor[b1 * nvPad + i] != 0
          let aff2 = a.bodyIsdofancestor[b2 * nvPad + i] != 0
          var ang1, lin1, ang2, lin2: array[3, float32]
          if i < nv:
            let cOff = (w * nv + i) * 6
            ang1 = [a.cdof[cOff + 0], a.cdof[cOff + 1], a.cdof[cOff + 2]]
            lin1 = [a.cdof[cOff + 3], a.cdof[cOff + 4], a.cdof[cOff + 5]]
            ang2 = ang1
            lin2 = lin1
          for k in 0 ..< 3:
            let pp = (k + 1) mod 3
            let qq = (k + 2) mod 3
            let cross1 = ang1[pp] * off1[qq] - ang1[qq] * off1[pp]
            let cross2 = ang2[pp] * off2[qq] - ang2[qq] * off2[pp]
            jacp[i * 3 + k] =
              (if aff2: lin2[k] + cross2 else: 0.0'f32) - (if aff1: lin1[k] + cross1 else: 0.0'f32)
            jacr[i * 3 + k] = (if aff2: ang2[k] else: 0.0'f32) - (if aff1: ang1[k] else: 0.0'f32)

      let eaddr = int(a.efcAddress[conid * a.nEfcAddr])
      let dimid = efcid - eaddr
      let jOff = (w * a.njmax + efcid) * nvPad

      var jq = 0.0'f32
      if a.elliptic:
        let fi = if dimid < 3: dimid else: dimid - 3
        let base = if dimid < 3: jacp else: jacr
        let fx = a.contactFrame[conid * 9 + fi * 3 + 0]
        let fy = a.contactFrame[conid * 9 + fi * 3 + 1]
        let fz = a.contactFrame[conid * 9 + fi * 3 + 2]
        for i in 0 ..< nvPad:
          let jij = base[i * 3 + 0] * fx + base[i * 3 + 1] * fy + base[i * 3 + 2] * fz
          a.J[jOff + i] = jij
          if i < nv:
            jq += jij * a.qvel[w * nv + i]
      else:
        var sign = 0.0'f32
        var dimid2 = 0
        if condim > 1:
          dimid2 = dimid div 2 + 1
          let frii = a.contactFriction[conid * 5 + dimid2 - 1]
          sign = frii * (1.0'f32 - 2.0'f32 * float32(dimid and 1))
        for i in 0 ..< nvPad:
          var jij = jacp[i * 3 + 0] * f0[0] + jacp[i * 3 + 1] * f0[1] + jacp[i * 3 + 2] * f0[2]
          if condim > 1:
            case dimid2
            of 1: jij += sign * (jacp[i * 3 + 0] * f1[0] + jacp[i * 3 + 1] * f1[1] + jacp[i * 3 + 2] * f1[2])
            of 2: jij += sign * (jacp[i * 3 + 0] * f2[0] + jacp[i * 3 + 1] * f2[1] + jacp[i * 3 + 2] * f2[2])
            of 3: jij += sign * (jacr[i * 3 + 0] * f0[0] + jacr[i * 3 + 1] * f0[1] + jacr[i * 3 + 2] * f0[2])
            of 4: jij += sign * (jacr[i * 3 + 0] * f1[0] + jacr[i * 3 + 1] * f1[1] + jacr[i * 3 + 2] * f1[2])
            else: jij += sign * (jacr[i * 3 + 0] * f2[0] + jacr[i * 3 + 1] * f2[1] + jacr[i * 3 + 2] * f2[2])
          a.J[jOff + i] = jij
          if i < nv:
            jq += jij * a.qvel[w * nv + i]

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
  runJob(contactJacWorker, addr args)
