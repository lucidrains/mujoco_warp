"""MLX engine: collision, constraints, Newton solver, Euler integration.

Mirrors mujoco_warp's forward pipeline for the microduck model (dense, pyramidal,
Newton). Dynamics/solver run on the MLX device; scalar per-row constraint state and
the line search run in numpy at float64 (as warp does per-lane in float32). Contact
geometry uses the same plane-convex selection as warp.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

from .model import ModelArrays
from .sim import FLOAT, Sim

MJ_MINVAL = 1e-15
MJ_MINIMP = 0.0
MJ_MAXIMP = 0.95
HUGE = 1e6

CONTACT_PYRAMIDAL = 4
LIMIT_JOINT = 6
FRICTION_DOF = 9

STATE_QUADRATIC = 1
STATE_LINEARNEG = 2
STATE_LINEARPOS = 3
STATE_SATISFIED = 4


def _efc_row(timestep, pos_aref, pos_imp, invweight, solref, solimp, margin, vel, frictionloss, ctype):
  timeconst = max(float(solref[0]), 2.0 * timestep)  # REFSAFE on
  dampratio = float(solref[1])
  dmin = min(max(float(solimp[0]), MJ_MINIMP), MJ_MAXIMP)
  dmax = min(max(float(solimp[1]), MJ_MINIMP), MJ_MAXIMP)
  width_raw = float(solimp[2])
  width = max(MJ_MINVAL, width_raw)
  mid = min(max(float(solimp[3]), MJ_MINIMP), MJ_MAXIMP)
  power = max(1.0, float(solimp[4]))
  dmax_sq = dmax * dmax
  if solref[0] <= 0.0:
    k = -solref[0] / max(MJ_MINVAL, dmax_sq)
  else:
    k = 1.0 / max(MJ_MINVAL, dmax_sq * timeconst * timeconst * dampratio * dampratio)
  if solref[1] <= 0.0:
    b = -solref[1] / max(MJ_MINVAL, dmax)
  else:
    b = 2.0 / max(MJ_MINVAL, dmax * timeconst)
  imp_x = abs(pos_imp) / width
  if dmin == dmax or width_raw <= MJ_MINVAL:
    imp = 0.5 * (dmin + dmax)
  elif imp_x <= 0.0:
    imp = dmin
  elif imp_x >= 1.0:
    imp = dmax
  elif power == 1.0:
    imp = dmin + imp_x * (dmax - dmin)
  elif imp_x <= mid:
    imp_y = (1.0 / mid ** (power - 1.0)) * imp_x**power
    imp = min(max(dmin + imp_y * (dmax - dmin), dmin), dmax)
  else:
    imp_y = 1.0 - (1.0 / (1.0 - mid) ** (power - 1.0)) * (1.0 - imp_x) ** power
    imp = min(max(dmin + imp_y * (dmax - dmin), dmin), dmax)
  if ctype == FRICTION_DOF:
    k = 0.0
  D = 1.0 / max(invweight * (1.0 - imp) / imp, MJ_MINVAL)
  aref = -k * imp * pos_aref - b * vel
  return float(D), float(aref), float(pos_aref + margin), float(margin), float(vel)


def _make_frame(a):
  a = np.asarray(a, np.float64)
  a = a / np.linalg.norm(a)
  y = np.array([0.0, 1.0, 0.0])
  z = np.array([0.0, 0.0, 1.0])
  b = y if (-0.5 < a[1] < 0.5) else z
  b = b - a * np.dot(a, b)
  b = b / np.linalg.norm(b) if np.linalg.norm(b) > 0 else b
  c = np.cross(a, b)
  return np.stack([a, b, c])


def _quat_to_mat_np(q):
  w, x, y, z = q
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
  )


def _plane_convex_exhaustive(verts, n, pl):
  supports = (pl[None, :] - verts) @ n
  amax = int(np.argmax(supports))
  max_support = supports[amax]
  if max_support < 0:
    return []
  threshold = max_support - 1e-3
  a = verts[amax]
  mask = np.where(supports > threshold, 0.0, -HUGE)
  bmax = int(np.argmax(np.sum((a[None, :] - verts) ** 2, axis=1) + mask))
  b = verts[bmax]
  ab = np.cross(n, a - b)
  cmax = int(np.argmax(np.abs((a[None, :] - verts) @ ab) + mask))
  c = verts[cmax]
  ac = np.cross(n, a - c)
  bc = np.cross(n, b - c)
  dmax = int(np.argmax(np.abs((a[None, :] - verts) @ ac) + np.abs((b[None, :] - verts) @ bc) + mask))
  return [amax, bmax, cmax, dmax]


def _plane_convex_graph(verts, graph, graphadr, n, pl):
  numvert = int(graph[graphadr])
  vert_edgeadr = graphadr + 2
  vert_globalid = graphadr + 2 + numvert
  edge_localid = graphadr + 2 + 2 * numvert

  def support(idx):
    return float(np.dot(pl - verts[idx], n))

  def climb(imax, score):
    max_score = -HUGE
    while True:
      prev = imax
      i = int(graph[vert_edgeadr + imax])
      while graph[edge_localid + i] >= 0:
        subidx = int(graph[edge_localid + i])
        idx = int(graph[vert_globalid + subidx])
        s = score(idx)
        if s > max_score:
          max_score = s
          imax = subidx
        i += 1
      if imax == prev:
        break
    return imax, max_score

  imax, max_support = climb(0, support)
  if max_support < 0:
    return []
  threshold = max(0.0, max_support - 1e-3)

  imax, _ = climb(imax, lambda idx: support(idx) if support(idx) > threshold else -HUGE)
  aidx = int(graph[vert_globalid + imax])
  a = verts[aidx]

  imax, _ = climb(imax, lambda idx: (float(np.sum((a - verts[idx]) ** 2)) if support(idx) > threshold else -HUGE))
  bidx = int(graph[vert_globalid + imax])
  b = verts[bidx]

  ab = np.cross(n, a - b)
  imax, _ = climb(imax, lambda idx: (abs(float(np.dot(a - verts[idx], ab))) if support(idx) > threshold else -HUGE))
  cidx = int(graph[vert_globalid + imax])
  c = verts[cidx]

  ac = np.cross(n, a - c)
  bc = np.cross(n, b - c)
  imax, _ = climb(
    imax,
    lambda idx: (
      abs(float(np.dot(a - verts[idx], ac))) + abs(float(np.dot(b - verts[idx], bc)))
      if support(idx) > threshold
      else -HUGE
    ),
  )
  didx = int(graph[vert_globalid + imax])
  return [aidx, bidx, cidx, didx]


def _plane_convex(verts, graph, graphadr, n, pl, rot, pos, plane_normal):
  if graphadr == -1 or len(verts) < 10:
    indices = _plane_convex_exhaustive(verts, n, pl)
  else:
    indices = _plane_convex_graph(verts, graph, graphadr, n, pl)
  if not indices:
    return []
  out = []
  for i in (3, 2, 1, 0):
    idx = indices[i]
    if sum(1 for j in range(i + 1) if indices[j] == idx) != 1:
      continue
    local = verts[idx]
    world = pos + rot @ local
    support = float(np.dot(pl - local, n))
    dist_v = -support
    world = world - 0.5 * dist_v * plane_normal
    out.append((dist_v, world))
  return out


class Engine:
  def __init__(self, m: ModelArrays):
    self.sim = Sim(m)
    self.m = m
    self._precompute()

  # ------------------------------------------------------------- model prep
  def _precompute(self):
    m = self.m
    sim = self.sim
    nv, nbody = sim.nv, sim.nbody

    self.body_chain = []
    for b in range(nbody):
      chain = []
      p = b
      while p > 0:
        chain.append(p)
        p = int(m.body_parentid[p])
      dofs = [j for j in range(nv) if int(m.dof_bodyid[j]) in chain]
      self.body_chain.append(sorted(dofs))

    self.limited_hinges = [j for j in range(sim.njnt) if bool(m.jnt_limited[j]) and int(m.jnt_type[j]) == 3]

    self.foot_geoms = [
      g for g in range(int(m.ngeom))
      if int(m.geom_type[g]) == 7 and (int(m.geom_contype[g]) & 1) and (int(m.geom_conaffinity[g]) & 1)
    ]
    self.foot_data = {}
    for foot in self.foot_geoms:
      did = int(m.geom_dataid[foot])
      va, vn = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
      self.foot_data[foot] = (
        np.asarray(m.mesh_vert[va : va + vn], np.float64),
        np.asarray(m.mesh_graph, np.int64),
        int(m.mesh_graphadr[did]),
      )

    self.contact_solref = np.array([0.02, 1.0])
    self.contact_solimp = np.array([0.9, 0.95, 0.001, 0.5, 2.0])
    self.friction = np.array([1.0, 1.0, 0.005, 1e-4, 1e-4])
    self.impratio_invsqrt = 1.0 / np.sqrt(float(m.impratio))
    self.timestep = float(m.timestep)
    self.meaninertia = float(m.meaninertia)
    self.tolerance = float(m.tolerance)
    self.iterations = int(m.iterations)
    self.ls_iterations = int(m.ls_iterations)
    self.ls_tolerance = float(m.ls_tolerance)
    self.warmstart = not (int(m.disableflags) & 1)  # DisableBit.WARMSTART == 1

  # --------------------------------------------------------------- control
  def passive_actuation(self):
    sim = self.sim
    m = self.m
    sim.qfrc_passive = -sim.dof_damping * sim.qvel
    qfrc = mx.zeros(sim.nv, dtype=FLOAT)
    for u in range(sim.nu):
      j = int(m.actuator_trnid[u, 0])
      dof = int(m.jnt_dofadr[j])
      qadr = int(m.jnt_qposadr[j])
      gear = float(m.actuator_gear[u, 0])
      length = sim.qpos[qadr] * gear
      vel = gear * sim.qvel[dof]
      ctrl = sim.ctrl[u]
      if m.actuator_ctrllimited[u]:
        ctrl = mx.minimum(mx.maximum(ctrl, float(m.actuator_ctrlrange[u, 0])), float(m.actuator_ctrlrange[u, 1]))
      force = float(m.actuator_gainprm[u, 0]) * ctrl + (
        float(m.actuator_biasprm[u, 0])
        + float(m.actuator_biasprm[u, 1]) * length
        + float(m.actuator_biasprm[u, 2]) * vel
      )
      if m.actuator_forcelimited[u]:
        force = mx.minimum(mx.maximum(force, float(m.actuator_forcerange[u, 0])), float(m.actuator_forcerange[u, 1]))
      qfrc = qfrc.at[dof].add(gear * force)
    sim.qfrc_actuator = qfrc
    sim.qfrc_smooth = sim.qfrc_passive - sim.qfrc_bias + sim.qfrc_actuator

  # ------------------------------------------------------------- collision
  def collision(self):
    sim = self.sim
    out = []
    plane_normal = np.array([0.0, 0.0, 1.0])
    plane_pos = np.zeros(3)
    for foot in self.foot_geoms:
      body = int(self.m.geom_bodyid[foot])
      rot_b = np.array(sim.xmat[body], np.float64)
      pos_b = np.array(sim.xpos[body], np.float64)
      gmat = _quat_to_mat_np(np.asarray(self.m.geom_quat[foot], np.float64))
      groot = rot_b @ gmat
      gposw = pos_b + rot_b @ np.asarray(self.m.geom_pos[foot], np.float64)
      verts, graph, graphadr = self.foot_data[foot]
      nrm = groot.T @ plane_normal
      pl = groot.T @ (plane_pos - gposw)
      for dist, p in _plane_convex(verts, graph, graphadr, nrm, pl, groot, gposw, plane_normal):
        out.append((dist, p, foot))
    return out

  # ------------------------------------------------------------ constraints
  def make_constraint(self):
    sim = self.sim
    m = self.m
    nv = sim.nv
    rows = []

    nf = 0
    for dof in range(nv):
      fl = float(m.dof_frictionloss[dof])
      if fl <= 0.0:
        continue
      J = np.zeros(nv)
      J[dof] = 1.0
      D, aref, pos_, margin, vel = _efc_row(
        self.timestep, 0.0, 0.0, float(m.dof_invweight0[dof]),
        np.asarray(m.dof_solref[dof]), np.asarray(m.dof_solimp[dof]), 0.0,
        float(sim.qvel[dof]), fl, FRICTION_DOF,
      )
      rows.append((J, D, aref, FRICTION_DOF, fl))
      nf += 1
    self.nf = nf

    nl = 0
    for jnt in self.limited_hinges:
      qadr = int(m.jnt_qposadr[jnt])
      dof = int(m.jnt_dofadr[jnt])
      lo, hi = float(m.jnt_range[jnt, 0]), float(m.jnt_range[jnt, 1])
      qpos = float(sim.qpos[qadr])
      dmin, dmax = qpos - lo, hi - qpos
      margin = float(m.jnt_margin[jnt])
      pos = min(dmin, dmax) - margin
      if pos >= 0.0:
        continue
      Jsign = 1.0 if dmin < dmax else -1.0
      J = np.zeros(nv)
      J[dof] = Jsign
      D, aref, pos_, marg, vel = _efc_row(
        self.timestep, pos, pos, float(m.dof_invweight0[dof]),
        np.asarray(m.jnt_solref[jnt]), np.asarray(m.jnt_solimp[jnt]), margin,
        Jsign * float(sim.qvel[dof]), 0.0, LIMIT_JOINT,
      )
      rows.append((J, D, aref, LIMIT_JOINT, 0.0))
      nl += 1
    self.nl = nl

    frame = _make_frame(np.array([0.0, 0.0, 1.0]))
    self.contacts_cache = [(d, p, f) for d, p, f in self.collision() if d < 0.0]
    for dist, cpos, foot in self.contacts_cache:
      if dist >= 0.0:
        continue
      body = int(m.geom_bodyid[foot])
      root = int(m.body_rootid[body])
      com = np.array(sim.subtree_com[root], np.float64)
      off = np.asarray(cpos, np.float64) - com
      fri0 = self.friction[0]
      invweight = float(m.body_invweight0[body, 0])
      invweight = (invweight + fri0 * fri0 * invweight) * 2.0 * fri0 * fri0 * self.impratio_invsqrt**2
      for dimid in range(2 * (3 - 1)):
        J = np.zeros(nv)
        for dof in self.body_chain[body]:
          cdof = np.array(sim.cdof[dof], np.float64)
          ang, lin = cdof[:3], cdof[3:]
          jacp = lin + np.cross(ang, off)
          jacr = ang
          jrow = float(np.dot(jacp, frame[0]))
          dimid2 = dimid // 2 + 1
          frii = self.friction[dimid2 - 1]
          sign = frii * (1.0 - 2.0 * (dimid & 1))
          if dimid2 == 1:
            jrow += sign * float(np.dot(jacp, frame[1]))
          elif dimid2 == 2:
            jrow += sign * float(np.dot(jacp, frame[2]))
          elif dimid2 == 3:
            jrow += sign * float(np.dot(jacr, frame[0]))
          elif dimid2 == 4:
            jrow += sign * float(np.dot(jacr, frame[1]))
          else:
            jrow += sign * float(np.dot(jacr, frame[2]))
          J[dof] = jrow
        jqvel = float(J @ np.asarray(sim.qvel, np.float64))
        D, aref, pos_, margin, vel = _efc_row(
          self.timestep, dist, dist, invweight, self.contact_solref, self.contact_solimp,
          0.0, jqvel, 0.0, CONTACT_PYRAMIDAL,
        )
        rows.append((np.array(J, np.float32), D, aref, CONTACT_PYRAMIDAL, 0.0))

    self.rows = rows
    self.nefc = len(rows)
    nv = sim.nv
    if rows:
      self.J_np = np.stack([np.asarray(r[0], np.float32) for r in rows])
      self.D_np = np.array([r[1] for r in rows], np.float64)
      self.aref_np = np.array([r[2] for r in rows], np.float64)
      self.fl_np = np.array([r[4] for r in rows], np.float64)
      self.efc_J = mx.array(self.J_np)
      self.efc_D = mx.array(self.D_np.astype(np.float32))
      self.efc_aref = mx.array(self.aref_np.astype(np.float32))
    else:
      self.J_np = np.zeros((0, nv), np.float32)
      self.D_np = np.zeros(0, np.float64)
      self.aref_np = np.zeros(0, np.float64)
      self.fl_np = np.zeros(0, np.float64)
      self.efc_J = mx.zeros((0, nv), dtype=FLOAT)

  # ---------------------------------------------------------------- solver
  def _eval_constraint(self, jaref):
    D, fl = self.D_np, self.fl_np
    force = np.zeros_like(jaref)
    state = np.zeros_like(jaref, dtype=np.int32)
    nf = self.nf
    for k in range(len(jaref)):
      x = jaref[k]
      if k < nf:
        f = fl[k]
        rf = f / D[k] if D[k] != 0.0 else 0.0
        if x <= -rf:
          force[k], state[k] = f, STATE_LINEARNEG
        elif x >= rf:
          force[k], state[k] = -f, STATE_LINEARPOS
        else:
          force[k], state[k] = -D[k] * x, STATE_QUADRATIC
      else:
        if x >= 0.0:
          force[k], state[k] = 0.0, STATE_SATISFIED
        else:
          force[k], state[k] = -D[k] * x, STATE_QUADRATIC
    return force, state

  def _cost_row(self, x, k):
    D, fl = self.D_np[k], self.fl_np[k]
    if k < self.nf:
      rf = fl / D if D != 0.0 else 0.0
      if x <= -rf:
        return (-fl * (0.5 * rf + x), -fl, 0.0)
      if x >= rf:
        return (-fl * (0.5 * rf - x), fl, 0.0)
      return (0.5 * D * x * x, D * x, D)
    if x >= 0.0:
      return (0.0, 0.0, 0.0)
    return (0.5 * D * x * x, D * x, D)

  def solve(self):
    sim = self.sim
    nv = sim.nv
    M = sim.M
    qfrc_smooth = sim.qfrc_smooth
    nrow = self.nefc

    L = cholesky_lower(M)
    qacc_smooth = cholesky_solve(L, qfrc_smooth)

    if nrow == 0:
      sim.qacc = qacc_smooth
      sim.qacc_warmstart = qacc_smooth
      sim.qfrc_constraint = mx.zeros(nv, dtype=FLOAT)
      sim.solver_niter = 0
      return

    qacc = sim.qacc_warmstart if self.warmstart else qacc_smooth
    Ma = M @ qacc
    J = self.efc_J
    J_np = self.J_np
    D, fl, aref = self.D_np, self.fl_np, self.aref_np

    Jaref = np.array(J @ qacc, np.float64) - aref

    def rebuild(qacc_mx, Ma_mx, force, state):
      qfrc_c = mx.array(J_np.T.astype(np.float32)) @ mx.array(force.astype(np.float32))
      grad = Ma_mx - qfrc_smooth - qfrc_c
      Dact = mx.array((D * (state == STATE_QUADRATIC)).astype(np.float32))
      h = M + mx.transpose(J) @ (Dact[:, None] * J)
      return grad, h, qfrc_c

    force, state = self._eval_constraint(Jaref)
    grad, h, qfrc_c = rebuild(qacc, Ma, force, state)
    sim.qfrc_constraint = qfrc_c
    search = cholesky_solve(cholesky_lower(h), -grad)

    Mnp = np.array(M, np.float64)
    bnp = np.array(qfrc_smooth, np.float64)
    niter = 0
    for it in range(self.iterations):
      niter += 1
      mv = M @ search
      jv = np.array(J @ search, np.float64)
      snp = np.array(search, np.float64)
      qnp = np.array(qacc, np.float64)
      hsn = float(snp @ Mnp @ snp)

      def phi(alpha, _Jaref=Jaref, _qnp=qnp, _snp=snp, _jv=jv, _hsn=hsn):
        c = 0.5 * float((_qnp + alpha * _snp) @ Mnp @ (_qnp + alpha * _snp)) - float((_qnp + alpha * _snp) @ bnp)
        g = float((_qnp + alpha * _snp) @ Mnp @ _snp) - float(bnp @ _snp)
        hs = _hsn
        for k in range(nrow):
          ck, gk, hk = self._cost_row(_Jaref[k] + alpha * _jv[k], k)
          c += ck
          g += gk * _jv[k]
          hs += hk * _jv[k] * _jv[k]
        return c, g, hs

      c0, g0, _ = phi(0.0)
      snorm = float(np.sqrt(np.sum(snp * snp)))
      gtol = max(self.tolerance * self.ls_tolerance * snorm * (self.meaninertia * nv), 1e-6)
      alpha = self._line_search(phi, g0, gtol)
      c_alpha = phi(alpha)[0]
      improvement = c0 - c_alpha
      alpha_mx = mx.array(np.float32(alpha))
      qacc = qacc + alpha_mx * search
      Ma = Ma + alpha_mx * mv
      Jaref = Jaref + alpha * jv

      force, state = self._eval_constraint(Jaref)
      grad, h, qfrc_c = rebuild(qacc, Ma, force, state)
      sim.qfrc_constraint = qfrc_c

      # fresh search direction and newton decrement (warp computes these in _update_gradient)
      search = cholesky_solve(cholesky_lower(h), -grad)
      newton_dec = float(np.sum(np.array(grad, np.float64) * (-np.array(search, np.float64))))
      grad_dot = float(np.sum(np.array(grad, np.float64) ** 2))
      rescale = self.meaninertia * nv
      if __import__("os").environ.get("MLX_SOLVER_DEBUG"):
        print(f"  iter {niter}: alpha {alpha:.6f} imp {improvement:.3e} g {np.sqrt(grad_dot):.3e} nd {newton_dec:.3e}")
      if (
        alpha == 0.0
        or (improvement > 0.0 and improvement / rescale < self.tolerance)
        or (np.sqrt(grad_dot) / rescale < self.tolerance)
        or (0.5 * newton_dec / rescale < self.tolerance)
        or niter >= self.iterations
      ):
        break

    sim.qacc = qacc
    sim.solver_niter = niter

  def _line_search(self, phi, g0, gtol):
    if abs(g0) < gtol:
      return 0.0
    alpha = 1.0
    _, g, _ = phi(alpha)
    if g < 0.0:
      lo_a = alpha
      hi_a = None
      for _ in range(self.ls_iterations):
        alpha *= 2.0
        _, g, _ = phi(alpha)
        if g >= 0.0:
          hi_a = alpha
          break
      if hi_a is None:
        return lo_a
    else:
      hi_a = alpha
      lo_a = None
      for _ in range(self.ls_iterations):
        alpha *= 0.5
        _, g, _ = phi(alpha)
        if g <= 0.0:
          lo_a = alpha
          break
        hi_a = alpha
      if lo_a is None:
        return 0.0
    alpha = 0.5 * (lo_a + hi_a)
    for _ in range(self.ls_iterations):
      _, g, h = phi(alpha)
      if abs(g) < gtol:
        return alpha
      if g < 0.0:
        lo_a = alpha
      else:
        hi_a = alpha
      if h > 0.0:
        cand = alpha - g / h
        alpha = cand if lo_a < cand < hi_a else 0.5 * (lo_a + hi_a)
      else:
        alpha = 0.5 * (lo_a + hi_a)
    return alpha

  # ------------------------------------------------------------ integrator
  def euler(self):
    sim = self.sim
    m = self.m
    Ma = sim.M @ sim.qacc
    damp = np.asarray(m.dof_damping, np.float32) * np.float32(self.timestep)
    Md = sim.M + mx.diag(mx.array(damp))
    qacc = cholesky_solve(cholesky_lower(Md), Ma)
    self._advance(qacc)

  def _advance(self, qacc):
    sim = self.sim
    m = self.m
    dt = self.timestep
    qvel_new = sim.qvel + dt * qacc
    qpos = np.array(sim.qpos)
    qvel_np = np.array(qvel_new)
    for j in range(sim.njnt):
      jtype = int(m.jnt_type[j])
      qadr = int(m.jnt_qposadr[j])
      dof = int(m.jnt_dofadr[j])
      if jtype == 0:
        qpos[qadr : qadr + 3] = qpos[qadr : qadr + 3] + dt * qvel_np[dof : dof + 3]
        quat = np.array(qpos[qadr + 3 : qadr + 7], np.float64)
        quat = quat / np.linalg.norm(quat)
        ang = qvel_np[dof + 3 : dof + 6].astype(np.float64)
        nrm = np.linalg.norm(ang)
        axis = ang / nrm if nrm > 0.0 else np.array([1.0, 0.0, 0.0])
        half = 0.5 * dt * nrm
        qr = np.array([np.cos(half), *(axis * np.sin(half))])
        w, x, y, z = quat
        res = np.array(
          [
            w * qr[0] - x * qr[1] - y * qr[2] - z * qr[3],
            w * qr[1] + x * qr[0] + y * qr[3] - z * qr[2],
            w * qr[2] - x * qr[3] + y * qr[0] + z * qr[1],
            w * qr[3] + x * qr[2] - y * qr[1] + z * qr[0],
          ]
        )
        qpos[qadr + 3 : qadr + 7] = (res / np.linalg.norm(res)).astype(np.float32)
      elif jtype == 3:
        qpos[qadr] = qpos[qadr] + dt * qvel_np[dof]
    sim.qpos = mx.array(qpos.astype(np.float32))
    sim.qvel = qvel_new
    sim.qacc_warmstart = sim.qacc

  # ------------------------------------------------------------------ step
  def step(self):
    sim = self.sim
    sim.kinematics()
    sim.com_pos()
    sim.crb_compute()
    sim.com_vel()
    sim.rne()
    self.passive_actuation()
    self.make_constraint()
    self.solve()
    self.euler()


def cholesky_lower(A):
  A = (A + mx.transpose(A)) * mx.array(0.5, dtype=FLOAT)
  n = A.shape[0]
  L = mx.zeros((n, n), dtype=FLOAT)
  for i in range(n):
    for j in range(i + 1):
      s = A[i, j] - mx.sum(L[i, :j] * L[j, :j])
      if i == j:
        L = L.at[i, j].add(mx.sqrt(s) - L[i, j])
      else:
        L = L.at[i, j].add(s / L[j, j] - L[i, j])
  return L


def cholesky_solve(L, b):
  n = L.shape[0]
  y = [None] * n
  for i in range(n):
    s = b[i]
    for j in range(i):
      s = s - L[i, j] * y[j]
    y[i] = s / L[i, i]
  x = [None] * n
  for i in range(n - 1, -1, -1):
    s = y[i]
    for j in range(i + 1, n):
      s = s - L[j, i] * x[j]
    x[i] = s / L[i, i]
  return mx.stack(x)
