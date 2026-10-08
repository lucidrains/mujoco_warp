"""Batched constraints + mask-based Newton solver + Euler integration on MLX.

Fixed constraint-row slots per world (frictionloss, hinges limits, 8 contact
slots x 4 pyramidal rows) so the solver is branchless: fixed iteration counts,
fixed line-search steps, per-world done masks. Row layout is deterministic; the
inactive rows carry J=0/D=0 so they cancel exactly.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from .batched import BatchedSim, FLOAT, _cross


def _matvec(A, v):
  """Batched (..., r, c) @ (..., c)."""
  return mx.matmul(A, v[..., None])[..., 0]


def _dot(a, b):
  return mx.sum(a * mx.array(b)[None, None, :], axis=-1)



# ---------------------------------------------------------------- row layout
def _efc_const(row):
  return row


class BatchedEngine:
  NF = 14       # frictionloss rows
  NL = 14       # hinge limit rows
  NC = 8        # contact slots (2 feet x 4 points)
  NDIM = 4      # pyramidal rows per contact slot
  NROW = 14 + 14 + 8 * 4  # 60

  def __init__(self, m, nworld: int):
    self.sim = BatchedSim(m, nworld)
    self.m = m
    self.nworld = nworld
    self.nv = m.nv
    self.nrow = self.NROW
    self._prep()

  def _prep(self):
    m = self.m
    self.friction_dofs = [d for d in range(m.nv) if float(m.dof_frictionloss[d]) > 0.0]
    self.limited_hinges = [j for j in range(int(m.njnt)) if bool(m.jnt_limited[j]) and int(m.jnt_type[j]) == 3]
    self.feet = [g for g in range(int(m.ngeom)) if int(m.geom_type[g]) == 7 and (int(m.geom_contype[g]) & 1) and (int(m.geom_conaffinity[g]) & 1)]
    self.foot_chain = []
    for g in self.feet:
      b = int(m.geom_bodyid[g])
      chain = []
      p = b
      while p > 0:
        chain.append(p)
        p = int(m.body_parentid[p])
      self.foot_chain.append(sorted([j for j in range(m.nv) if int(m.dof_bodyid[j]) in chain]))

    self.timestep = float(m.timestep)
    self.impratio_invsqrt = 1.0 / np.sqrt(float(m.impratio))
    self._contact_solref = np.array([0.02, 1.0])
    self._contact_solimp = np.array([0.9, 0.95, 0.001, 0.5, 2.0])
    self.iteration_cap = 4
    self.ls_cap = 2

  # ---------------------------------------------------------------- helpers
  def _efc_row_batched(self, pos_aref, invweight, solref, solimp, margin, vel, ctype_friction):
    """Vectorized mujoco_warp `_efc_row` for non-discrete; arrays are (nw,) or scalars."""
    dt = self.timestep
    timeconst = np.maximum(float(solref[0]), 2.0 * dt)
    dampratio = float(solref[1])
    dmin = min(max(float(solimp[0]), 0.0), 0.95)
    dmax = min(max(float(solimp[1]), 0.0), 0.95)
    width = max(1e-15, float(solimp[2]))
    mid = min(max(float(solimp[3]), 0.0), 0.95)
    power = max(1.0, float(solimp[4]))
    k = 1.0 / max(1e-15, dmax * dmax * timeconst * timeconst * dampratio * dampratio)
    b = 2.0 / max(1e-15, dmax * timeconst)
    if ctype_friction:
      k = 0.0
    imp_x = np.abs(pos_aref) / width
    if dmin == dmax or float(solimp[2]) <= 1e-15:
      imp = np.full_like(imp_x, 0.5 * (dmin + dmax))
    else:
      low = dmin + imp_x * (dmax - dmin)
      mid_curve = dmin + (1.0 / mid ** (power - 1.0)) * imp_x**power * (dmax - dmin)
      high_curve = 1.0 - (1.0 / (1.0 - mid) ** (power - 1.0)) * (1.0 - imp_x) ** power
      imp = np.where(
        imp_x <= 0.0, dmin,
        np.where(
          imp_x >= 1.0, dmax,
          np.where(power == 1.0, low, np.where(imp_x <= mid, mid_curve, high_curve)),
        ),
      )
      imp = np.clip(imp, dmin, dmax)
    D = 1.0 / np.maximum(invweight * (1.0 - imp) / imp, 1e-15)
    aref = -k * imp * pos_aref - b * vel
    return D.astype(np.float32), aref.astype(np.float32)

  # ------------------------------------------------------------ constraints
  def build_constraints(self):
    sim, m = self.sim, self.m
    nw, nv = self.nworld, self.nv
    qpos_np = np.array(sim.qpos)
    qvel_np = np.array(sim.qvel)

    # ---- frictionloss rows (0..NF-1): one-hot J, constant D, aref = -b * vel
    fr_dofs = self.friction_dofs
    nfr = len(fr_dofs)
    E_fr = np.zeros((nfr, nv), np.float32)
    E_fr[np.arange(nfr), fr_dofs] = 1.0
    D_fr = np.zeros(nfr, np.float32)
    b_fr = np.zeros(nfr, np.float32)
    fl_fr = np.zeros(nfr, np.float32)
    for k, dof in enumerate(fr_dofs):
      d, _ = self._efc_row_batched(
        np.zeros(nw, np.float32), float(m.dof_invweight0[dof]), m.dof_solref[dof],
        m.dof_solimp[dof], 0.0, np.zeros(nw, np.float32), True)
      D_fr[k] = d[0]
      b_fr[k] = 2.0 / max(1e-15, 0.95 * max(0.02, 2.0 * self.timestep))
      fl_fr[k] = float(m.dof_frictionloss[dof])
    J_fr = mx.broadcast_to(mx.array(E_fr)[None], (nw, nfr, nv))
    D_fr_b = mx.broadcast_to(mx.array(D_fr)[None], (nw, nfr))
    aref_fr_b = mx.array((-b_fr[None, :] * qvel_np[:, fr_dofs]).astype(np.float32))
    fl_fr_b = mx.broadcast_to(mx.array(fl_fr)[None], (nw, nfr))

    # ---- hinge limit rows (NF..NF+NL-1)
    nl = len(self.limited_hinges)
    E_lim = np.zeros((nl, nv), np.float32)
    signmask = np.zeros((nw, nl), np.float32)
    Dlim = np.zeros((nw, nl), np.float32)
    areflim = np.zeros((nw, nl), np.float32)
    for k, jnt in enumerate(self.limited_hinges):
      qa = int(m.jnt_qposadr[jnt])
      dof = int(m.jnt_dofadr[jnt])
      lo, hi = float(m.jnt_range[jnt, 0]), float(m.jnt_range[jnt, 1])
      margin = float(m.jnt_margin[jnt])
      dmin = qpos_np[:, qa] - lo
      dmax = hi - qpos_np[:, qa]
      pos = np.minimum(dmin, dmax) - margin
      active = pos < 0.0
      Jsign = np.where(dmin < dmax, 1.0, -1.0)
      E_lim[k, dof] = 1.0
      signmask[:, k] = np.where(active, Jsign, 0.0)
      vel = Jsign * qvel_np[:, dof]
      d, a = self._efc_row_batched(
        pos.astype(np.float32), float(m.dof_invweight0[dof]), m.jnt_solref[jnt],
        m.jnt_solimp[jnt], margin, vel, False)
      Dlim[:, k] = np.where(active, d, 0.0)
      areflim[:, k] = np.where(active, a, 0.0)
    J_lim = mx.array(E_lim)[None] * mx.array(signmask.astype(np.float32))[:, :, None]
    D_lim_b = mx.array(Dlim)
    aref_lim_b = mx.array(areflim)
    fl_lim_b = mx.zeros((nw, nl), dtype=FLOAT)

    # ---- contact rows (NF+NL..): single fused MSL kernel
    from .batched_collision import contacts_batched
    from .batched_contact import contact_rows

    dist, cpos = contacts_batched(m, sim, nw)
    cpos_flat = mx.reshape(cpos, (nw, 8 * 3))
    nchain = 16
    chain_buf = np.full((2, nchain), -1, np.int32)
    nchain_buf = np.zeros(2, np.int32)
    root_buf = np.zeros(2, np.int32)
    invw_buf = np.zeros(2, np.float32)
    for f, g in enumerate(self.feet):
      ch = self.foot_chain[f]
      chain_buf[f, : len(ch)] = ch
      nchain_buf[f] = len(ch)
      body = int(m.geom_bodyid[g])
      root_buf[f] = int(m.body_rootid[body])
      invw0 = float(m.body_invweight0[body, 0])
      invw_buf[f] = (invw0 + invw0) * 2.0 * self.impratio_invsqrt**2
    dt = self.timestep
    tc = max(0.02, 2.0 * dt)
    dmax = 0.95
    params = np.array([0.9, dmax, 1e-3, 1e-3, 0.5, 2.0,
                       1.0 / (dmax * dmax * tc * tc), 2.0 / (dmax * tc)], np.float32)
    Jc, Dc, arefc = contact_rows(
      sim, m, dist, cpos_flat,
      mx.array(chain_buf), mx.array(nchain_buf), mx.array(root_buf), mx.array(invw_buf), mx.array(params))

    self.J = mx.concatenate([J_fr, J_lim, Jc], axis=1)
    self.D = mx.concatenate([D_fr_b, D_lim_b, Dc], axis=1)
    self.aref = mx.concatenate([aref_fr_b, aref_lim_b, arefc], axis=1)
    self.fl = mx.concatenate([fl_fr_b, fl_lim_b, mx.zeros((nw, 32), dtype=FLOAT)], axis=1)
    self.contact_active = np.array(dist < 0.0)
    mx.eval(self.J, self.D, self.aref)

  # --------------------------------------------------------------- solver
  def solve(self):
    sim, m = self.sim, self.m
    nw, nv = self.nworld, self.nv
    M = sim.M
    smooth = sim.qfrc_smooth
    J = self.J
    D = self.D
    aref = self.aref
    fl = self.fl

    qacc_smooth = batch_chol_solve(M, smooth)
    qacc = sim.qacc_warmstart
    Ma = _matvec(M, qacc)

    def row_state_force(x):
      neg = x < 0.0
      quad_force = -D * x
      state_quad = neg
      # friction rows handled separately below
      return quad_force, state_quad

    def eval_rows(x):
      # x: (nw, nrow); returns force, state, cost, grad, hess wrt x
      xf = x[:, : self.NF]
      rf = fl[:, : self.NF] / mx.maximum(D[:, : self.NF], 1e-30)
      # friction force / state
      f_force = mx.where(xf <= -rf, fl[:, : self.NF], mx.where(xf >= rf, -fl[:, : self.NF], -D[:, : self.NF] * xf))
      f_cost = mx.where(
        xf <= -rf, -fl[:, : self.NF] * (0.5 * rf + xf),
        mx.where(xf >= rf, -fl[:, : self.NF] * (0.5 * rf - xf), 0.5 * D[:, : self.NF] * xf * xf))
      f_grad = mx.where(xf <= -rf, -fl[:, : self.NF], mx.where(xf >= rf, fl[:, : self.NF], D[:, : self.NF] * xf))
      f_hess = mx.where((xf > -rf) & (xf < rf), D[:, : self.NF], mx.zeros_like(xf))

      xq = x[:, self.NF :]
      neg = xq < 0.0
      o_force = mx.where(neg, -D[:, self.NF :] * xq, mx.zeros_like(xq))
      o_cost = mx.where(neg, 0.5 * D[:, self.NF :] * xq * xq, mx.zeros_like(xq))
      o_grad = mx.where(neg, D[:, self.NF :] * xq, mx.zeros_like(xq))
      o_hess = mx.where(neg, D[:, self.NF :], mx.zeros_like(xq))

      force = mx.concatenate([f_force, o_force], axis=1)
      cost = mx.concatenate([f_cost, o_cost], axis=1)
      grad = mx.concatenate([f_grad, o_grad], axis=1)
      hess = mx.concatenate([f_hess, o_hess], axis=1)
      state_quad = mx.concatenate([(xf > -rf) & (xf < rf), neg], axis=1)
      return force, state_quad, cost, grad, hess

    Jaref = _matvec(J, qacc) - aref
    force, stateq, cost, rgrad, rhess = eval_rows(Jaref)
    activeD = D * stateq.astype(FLOAT)

    def rebuild(qacc_, Ma_):
      qc = mx.sum(J * force[:, :, None], axis=1)
      grad = Ma_ - smooth - qc
      h = M + mx.matmul(mx.swapaxes(J, -1, -2), activeD[:, :, None] * J)
      return grad, h

    grad, h = rebuild(qacc, Ma)
    Mnp = None
    done = mx.zeros(nw, dtype=mx.bool_)
    niter = 0
    for _ in range(self.iteration_cap):
      niter += 1
      x = batch_chol_solve(h, grad)
      search = -x
      mv = _matvec(M, search)
      jv = _matvec(J, search)
      newton_dec = -mx.sum(grad * search, axis=-1)

      # ---- line search: smooth part is exactly quadratic in alpha, so its
      # coefficients are precomputed once; only constraint rows are re-evaluated.
      c_sm0 = 0.5 * mx.sum(qacc * Ma, axis=-1) - mx.sum(qacc * smooth, axis=-1)
      g_sm0 = mx.sum((Ma - smooth) * search, axis=-1)
      hsn_pre = mx.sum(search * mv, axis=-1)

      def phi(alpha):
        xr = Jaref + alpha[:, None] * jv
        _, _, cs, gr, hh = eval_rows(xr)
        c = c_sm0 + alpha * g_sm0 + 0.5 * alpha * alpha * hsn_pre + mx.sum(cs, axis=-1)
        g = g_sm0 + alpha * hsn_pre + mx.sum(gr * jv, axis=-1)
        hs = hsn_pre + mx.sum(hh * jv * jv, axis=-1)
        return c, g, hs

      c0, g0, _ = phi(mx.zeros(nw, dtype=FLOAT))

      # bracket
      a = mx.ones(nw, dtype=FLOAT)
      c_a, g_a, h_a = phi(a)
      lo = mx.zeros(nw, dtype=FLOAT)
      hi = mx.zeros(nw, dtype=FLOAT)
      has_lo = mx.zeros(nw, dtype=mx.bool_)
      has_hi = mx.zeros(nw, dtype=mx.bool_)
      up = g_a < 0.0
      has_lo = has_lo | up
      lo = mx.where(up, a, lo)
      has_hi = has_hi | (~up)
      hi = mx.where(~up, a, hi)
      for _ in range(self.ls_cap):
        a = mx.where(up & ~has_hi, a * 2.0, a)
        a = mx.where(~up & ~has_lo, a * 0.5, a)
        c_a, g_a, h_a = phi(a)
        has_lo = has_lo | (g_a < 0.0)
        lo = mx.where(g_a < 0.0, a, lo)
        has_hi = has_hi | (g_a >= 0.0)
        hi = mx.where(g_a >= 0.0, a, hi)
      # safeguarded newton
      a = 0.5 * (mx.where(has_lo, lo, 0.0) + mx.where(has_hi, hi, 1.0))
      for _ in range(self.ls_cap):
        c_a, g_a, h_a = phi(a)
        lo = mx.where(g_a < 0.0, a, lo)
        hi = mx.where(g_a >= 0.0, a, hi)
        cand = a - g_a / mx.maximum(h_a, 1e-12)
        inside = (cand > lo) & (cand < hi)
        a = mx.where(inside, cand, 0.5 * (lo + hi))
      c_a, g_a, _ = phi(a)
      improvement = c0 - c_a
      alpha = mx.where(g0 < 0.0, a, mx.zeros_like(a))
      alpha = mx.where(done, mx.zeros_like(alpha), alpha)

      qacc = qacc + alpha[:, None] * search
      Ma = Ma + alpha[:, None] * mv
      Jaref = Jaref + alpha[:, None] * jv
      force, stateq, cost, rgrad, rhess = eval_rows(Jaref)
      activeD = D * stateq.astype(FLOAT)
      grad, h = rebuild(qacc, Ma)

      grad_dot = mx.sum(grad * grad, axis=-1)
      rescale = float(self.m.meaninertia) * nv
      newly_done = (alpha == 0.0) | (improvement / rescale < self.m.tolerance) | (mx.sqrt(grad_dot) / rescale < self.m.tolerance)
      done = done | newly_done

    self.qacc = qacc
    self.niter = niter
    mx.eval(self.qacc)

  # ------------------------------------------------------------ integrator
  def euler(self):
    sim, m = self.sim, self.m
    Ma = _matvec(sim.M, self.qacc)
    damp = mx.array(np.asarray(m.dof_damping, np.float32) * np.float32(self.timestep))
    Md = sim.M + mx.diag(damp)[None]
    qacc = batch_chol_solve(Md, Ma)
    self._advance(qacc)

  def _advance(self, qacc):
    sim, m = self.sim, self.m
    dt = self.timestep
    nw = self.nworld
    qvel = sim.qvel + dt * qacc
    qpos = np.array(sim.qpos)
    qvel_np = np.array(qvel)
    for j in range(int(m.njnt)):
      jtype = int(m.jnt_type[j])
      qa = int(m.jnt_qposadr[j])
      dof = int(m.jnt_dofadr[j])
      if jtype == 0:
        qpos[:, qa : qa + 3] += dt * qvel_np[:, dof : dof + 3]
        quat = qpos[:, qa + 3 : qa + 7].astype(np.float64)
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        ang = qvel_np[:, dof + 3 : dof + 6].astype(np.float64)
        nrm = np.linalg.norm(ang, axis=1, keepdims=True)
        axis = np.where(nrm > 0, ang / np.maximum(nrm, 1e-30), np.array([1.0, 0.0, 0.0]))
        half = 0.5 * dt * nrm
        qr = np.concatenate([np.cos(half), axis * np.sin(half)], axis=1)
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        res = np.stack(
          [
            w * qr[:, 0] - x * qr[:, 1] - y * qr[:, 2] - z * qr[:, 3],
            w * qr[:, 1] + x * qr[:, 0] + y * qr[:, 3] - z * qr[:, 2],
            w * qr[:, 2] - x * qr[:, 3] + y * qr[:, 0] + z * qr[:, 1],
            w * qr[:, 3] + x * qr[:, 2] - y * qr[:, 1] + z * qr[:, 0],
          ],
          axis=1,
        )
        res /= np.linalg.norm(res, axis=1, keepdims=True)
        qpos[:, qa + 3 : qa + 7] = res
      elif jtype == 3:
        qpos[:, qa] += dt * qvel_np[:, dof]
    sim.qpos = mx.array(qpos.astype(np.float32))
    sim.qvel = qvel
    sim.qacc_warmstart = self.qacc

  # ------------------------------------------------------- numpy convenience
  def set_state(self, qpos, qvel, ctrl=None):
    """NumPy in: (nworld, nq), (nworld, nv), (nworld, nu)."""
    self.sim.set_state(qpos, qvel, ctrl)

  def set_ctrl(self, ctrl):
    """NumPy in: (nworld, nu)."""
    self.sim.ctrl = mx.array(np.asarray(ctrl, dtype=np.float32))

  def get_state(self):
    """NumPy out: forces evaluation, returns (qpos, qvel)."""
    mx.eval(self.sim.qpos, self.sim.qvel)
    return np.array(self.sim.qpos), np.array(self.sim.qvel)

  def step_np(self, ctrl=None):
    """Optional NumPy ctrl in, NumPy (qpos, qvel) out."""
    if ctrl is not None:
      self.set_ctrl(ctrl)
    self.step()
    return self.get_state()

  def step(self):
    sim = self.sim
    sim.kinematics()
    sim.com_pos()
    sim.crb_compute()
    sim.com_vel()
    sim.rne()
    sim.mass_matrix()
    sim.passive_actuation()
    self.build_constraints()
    self.solve()
    self.euler()


# ------------------------------------------------------------ batched chol

_CHOL_MSL = r"""
    uint w = thread_position_in_grid.x;
    if (w >= NWORLD[0]) return;
    thread float A[N * N];
    thread float x[N];
    for (int i = 0; i < N * N; ++i) A[i] = Ain[w * N * N + i];
    for (int i = 0; i < N; ++i) x[i] = b[w * N + i];
    for (int k = 0; k < N; ++k) {
        float d = metal::sqrt(A[k * N + k]);
        A[k * N + k] = d;
        for (int i = k + 1; i < N; ++i) A[i * N + k] /= d;
        for (int j = k + 1; j < N; ++j) {
            float ajk = A[j * N + k];
            for (int i = j; i < N; ++i) A[i * N + j] -= A[i * N + k] * ajk;
        }
    }
    for (int i = 0; i < N; ++i) {
        float s = x[i];
        for (int j = 0; j < i; ++j) s -= A[i * N + j] * x[j];
        x[i] = s / A[i * N + i];
    }
    for (int i = N - 1; i >= 0; --i) {
        float s = x[i];
        for (int j = i + 1; j < N; ++j) s -= A[j * N + i] * x[j];
        x[i] = s / A[i * N + i];
    }
    for (int i = 0; i < N; ++i) xout[w * N + i] = x[i];
"""

_CHOL_CACHE: dict = {}


def _chol_kernel(n: int, nworld: int):
    key = (n,)
    k = _CHOL_CACHE.get(key)
    if k is None:
        k = mx.fast.metal_kernel(
            name=f"batch_chol_{n}",
            input_names=["Ain", "b", "NWORLD"],
            output_names=["xout"],
            source=_CHOL_MSL,
        )
        _CHOL_CACHE[key] = k
    return k


def batch_chol_solve_gpu(A, b):
    nw, n = A.shape[0], A.shape[1]
    k = _chol_kernel(n, nw)
    nwr = mx.array(np.array([nw], np.int32))
    (x,) = k(
        inputs=[A, b, nwr],
        template=[("N", n)],
        grid=((nw + 31) // 32 * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(nw * n,)],
        output_dtypes=[mx.float32],
    )
    return mx.reshape(x, (nw, n))


def batch_chol_solve(A, b):
  """A x = b for batched (nw, n, n) SPD A (single MSL dispatch, one thread per world)."""
  return batch_chol_solve_gpu(A, b)


def batch_chol_solve_ref(A, b):
  n = A.shape[1]
  L = mx.zeros_like(A)
  Aq = A
  for k in range(n):
    d = mx.sqrt(Aq[:, k, k])
    L = L.at[:, k, k].add(d - L[:, k, k])
    col = Aq[:, k + 1 :, k] / d[:, None]
    L = L.at[:, k + 1 :, k].add(col - L[:, k + 1 :, k])
    upd = col[:, :, None] * col[:, None, :]
    Aq = Aq.at[:, k + 1 :, k + 1 :].add(-upd)
  y = mx.zeros_like(b)
  for i in range(n):
    s = b[:, i] - mx.sum(L[:, i, :i] * y[:, :i], axis=-1)
    y = y.at[:, i].add(s / L[:, i, i] - y[:, i])
  x = mx.zeros_like(b)
  for i in range(n - 1, -1, -1):
    s = y[:, i] - mx.sum(L[:, i + 1 :, i] * x[:, i + 1 :], axis=-1)
    x = x.at[:, i].add(s / L[:, i, i] - x[:, i])
  return x
