"""World-batched MLX engine for rigid-body simulation (mujoco_warp algorithm mirror).

State and all phases carry a leading world dimension, so a single MLX dispatch
processes every world. Tree passes use BFS levels, ancestry tables and scatter-add
instead of per-world Python control flow, which is what makes massive batches
(e.g. nworld in the thousands) viable on Metal.

This module currently covers the position/velocity pipeline; collision,
constraints, the solver and integration are added on top.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mlx.core as mx

from .model import ModelArrays

FLOAT = mx.float32


def _cross(a: mx.array, b: mx.array) -> mx.array:
  return mx.stack(
    [
      a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
      a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
      a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
    ],
    axis=-1,
  )


@dataclass
class BatchModel:
  """Static, world-independent tables derived from ModelArrays."""
  m: ModelArrays
  nworld: int
  # BFS levels, deepest first (for bottom-up accumulation)
  levels_desc: list[np.ndarray]
  # padded ancestry: (nv, max_chain) dof indices, -1 padding
  chain_idx: np.ndarray
  chain_mask: np.ndarray
  # per body: dof range and joint range
  body_jntadr: np.ndarray
  body_jntnum: np.ndarray
  body_dofadr: np.ndarray
  body_dofnum: np.ndarray


def _levels(m: ModelArrays) -> list[np.ndarray]:
  depth = np.zeros(m.nbody, dtype=np.int64)
  for b in range(1, m.nbody):
    depth[b] = depth[int(m.body_parentid[b])] + 1
  maxd = int(depth.max())
  return [np.where(depth == d)[0] for d in range(maxd, -1, -1)]


def _chains(m: ModelArrays) -> tuple[np.ndarray, np.ndarray]:
  nv = m.nv
  chains = []
  maxlen = 0
  for i in range(nv):
    c = []
    j = i
    while j >= 0:
      c.append(j)
      j = int(m.dof_parentid[j])
    chains.append(c)
    maxlen = max(maxlen, len(c))
  idx = np.full((nv, maxlen), -1, dtype=np.int32)
  mask = np.zeros((nv, maxlen), dtype=np.float32)
  for i, c in enumerate(chains):
    idx[i, : len(c)] = c
    mask[i, : len(c)] = 1.0
  return idx, mask


class BatchedSim:
  def __init__(self, m: ModelArrays, nworld: int):
    self.m = m
    self.nworld = nworld
    bm = BatchModel(
      m=m,
      nworld=nworld,
      levels_desc=_levels(m),
      chain_idx=_chains(m)[0],
      chain_mask=_chains(m)[1],
      body_jntadr=np.asarray(m.body_jntadr, np.int32),
      body_jntnum=np.asarray(m.body_jntnum, np.int32),
      body_dofadr=np.asarray(m.body_dofadr, np.int32),
      body_dofnum=np.asarray(m.body_dofnum, np.int32),
    )
    self.bm = bm
    self._to_mx()

  # ------------------------------------------------------------------ setup
  def _to_mx(self):
    m = self.m
    a = lambda x, dt=FLOAT: mx.array(np.asarray(x), dtype=dt)  # noqa: E731
    self.body_parentid = a(m.body_parentid, mx.int32)
    self.body_rootid = a(m.body_rootid, mx.int32)
    self.jnt_type = a(m.jnt_type, mx.int32)
    self.jnt_qposadr = a(m.jnt_qposadr, mx.int32)
    self.jnt_axis = a(m.jnt_axis)
    self.jnt_pos = a(m.jnt_pos)
    self.body_pos = a(m.body_pos)
    self.body_quat = a(m.body_quat)
    self.body_ipos = a(m.body_ipos)
    self.body_iquat = a(m.body_iquat)
    self.body_inertia = a(m.body_inertia)
    self.body_mass = a(m.body_mass)
    self.body_subtreemass = a(m.body_subtreemass)
    self.dof_bodyid = a(m.dof_bodyid, mx.int32)
    self.dof_parentid = a(m.dof_parentid, mx.int32)
    self.dof_armature = a(m.dof_armature)
    self.cdof = None
    self.chain_idx = a(self.bm.chain_idx, mx.int32)
    self.chain_mask = a(self.bm.chain_mask)
    self.levels_desc = [a(lv, mx.int32) for lv in self.bm.levels_desc]
    self.qpos0 = a(m.qpos0)

  # ------------------------------------------------------------------ state
  def set_state(self, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray | None = None):
    """qpos: (nworld, nq), qvel: (nworld, nv)."""
    self.qpos = mx.array(np.asarray(qpos, dtype=np.float32))
    self.qvel = mx.array(np.asarray(qvel, dtype=np.float32))
    nw = self.nworld
    self.qacc = mx.zeros((nw, self.m.nv), dtype=FLOAT)
    self.qacc_warmstart = mx.zeros((nw, self.m.nv), dtype=FLOAT)
    self.ctrl = mx.array(np.asarray(ctrl if ctrl is not None else np.zeros((nw, self.m.nu)), dtype=np.float32))

  # -------------------------------------------------------------- utilities
  @staticmethod
  def _qmul(u, v):
    return mx.stack(
      [
        u[..., 0] * v[..., 0] - u[..., 1] * v[..., 1] - u[..., 2] * v[..., 2] - u[..., 3] * v[..., 3],
        u[..., 0] * v[..., 1] + u[..., 1] * v[..., 0] + u[..., 2] * v[..., 3] - u[..., 3] * v[..., 2],
        u[..., 0] * v[..., 2] - u[..., 1] * v[..., 3] + u[..., 2] * v[..., 0] + u[..., 3] * v[..., 1],
        u[..., 0] * v[..., 3] + u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1] + u[..., 3] * v[..., 0],
      ],
      axis=-1,
    )

  @staticmethod
  def _qrot(v, q):
    s = q[..., 0:1]
    u = q[..., 1:4]
    r = 2.0 * mx.sum(u * v, axis=-1, keepdims=True) * u + (s * s - mx.sum(u * u, axis=-1, keepdims=True)) * v
    return r + 2.0 * s * _cross(u, v)

  @staticmethod
  def _qmat(q):
    q00 = q[..., 0] * q[..., 0]
    q01 = q[..., 0] * q[..., 1]
    q02 = q[..., 0] * q[..., 2]
    q03 = q[..., 0] * q[..., 3]
    q11 = q[..., 1] * q[..., 1]
    q12 = q[..., 1] * q[..., 2]
    q13 = q[..., 1] * q[..., 3]
    q22 = q[..., 2] * q[..., 2]
    q23 = q[..., 2] * q[..., 3]
    q33 = q[..., 3] * q[..., 3]
    return mx.stack(
      [
        mx.stack([q00 + q11 - q22 - q33, 2.0 * (q12 - q03), 2.0 * (q13 + q02)], axis=-1),
        mx.stack([2.0 * (q12 + q03), q00 - q11 + q22 - q33, 2.0 * (q23 - q01)], axis=-1),
        mx.stack([2.0 * (q13 - q02), 2.0 * (q23 + q01), q00 - q11 - q22 + q33], axis=-1),
      ],
      axis=-2,
    )

  # ------------------------------------------------------------ kinematics
  def kinematics(self):
    m = self.m
    nw, nbody = self.nworld, m.nbody
    xpos_l = [None] * nbody
    xquat_l = [None] * nbody
    xanchor = mx.zeros((nw, m.njnt, 3), dtype=FLOAT)
    xaxis = mx.zeros((nw, m.njnt, 3), dtype=FLOAT)
    qpos = self.qpos

    for b in range(nbody):
      pid = int(m.body_parentid[b])
      ja = int(m.body_jntadr[b])
      jn = int(m.body_jntnum[b])
      if jn == 1 and int(m.jnt_type[ja]) == 0:
        qa = int(m.jnt_qposadr[ja])
        xpos = qpos[:, qa : qa + 3]
        xquat = qpos[:, qa + 3 : qa + 7]
        xquat = xquat / mx.sqrt(mx.sum(xquat * xquat, axis=-1, keepdims=True))
        xpos_l[b] = xpos
        xquat_l[b] = xquat
        xanchor = xanchor.at[:, ja].add(xpos)
        xaxis = xaxis.at[:, ja].add(self.jnt_axis[ja])
        continue

      xpos = mx.broadcast_to(self.body_pos[b], (nw, 3))
      xquat = mx.broadcast_to(self.body_quat[b], (nw, 4))
      if pid >= 0 and pid != b:
        xpos = self._qrot(xpos, xquat_l[pid]) + xpos_l[pid]
        xquat = self._qmul(xquat_l[pid], xquat)

      for k in range(jn):
        j = ja + k
        qa = int(m.jnt_qposadr[j])
        jtype = int(m.jnt_type[j])
        axis = self.jnt_axis[j]
        anchor = self._qrot(mx.broadcast_to(self.jnt_pos[j], (nw, 3)), xquat) + xpos
        gaxis = self._qrot(mx.broadcast_to(axis, (nw, 3)), xquat)
        if jtype == 3:  # HINGE
          angle = qpos[:, qa] - self.qpos0[qa]
          s = mx.sin(angle * 0.5)[:, None]
          c = mx.cos(angle * 0.5)[:, None]
          qloc = mx.concatenate([c, mx.broadcast_to(axis, (nw, 3)) * s], axis=-1)
          qloc = qloc / mx.sqrt(mx.sum(qloc * qloc, axis=-1, keepdims=True))
          xquat = self._qmul(xquat, qloc)
          xpos = anchor - self._qrot(mx.broadcast_to(self.jnt_pos[j], (nw, 3)), xquat)
        xanchor = xanchor.at[:, j].add(anchor)
        xaxis = xaxis.at[:, j].add(gaxis)

      xquat = xquat / mx.sqrt(mx.sum(xquat * xquat, axis=-1, keepdims=True))
      xpos_l[b] = xpos
      xquat_l[b] = xquat

    self.xpos = mx.stack(xpos_l, axis=1)
    self.xquat = mx.stack(xquat_l, axis=1)
    self.xmat = self._qmat(self.xquat)
    self.xanchor = xanchor
    self.xaxis = xaxis
    self.xipos = self.xpos + mx.stack(
      [self._qrot(mx.broadcast_to(m.body_ipos[b], (nw, 3)), self.xquat[:, b]) for b in range(nbody)], axis=1
    )
    self.ximat = self._qmat(self._qmul(self.xquat, mx.array(np.asarray(m.body_iquat), dtype=FLOAT)[None]))

  # ------------------------------------------------------------------- com
  def com_pos(self):
    m = self.m
    nw, nbody = self.nworld, m.nbody
    mass = mx.array(np.asarray(m.body_mass), dtype=FLOAT)
    acc = self.xipos * mass[None, :, None]
    for b in range(nbody - 1, 0, -1):
      pid = int(m.body_parentid[b])
      if pid != b:
        acc = acc.at[:, pid].add(acc[:, b])
    sub = acc / mx.maximum(mx.array(np.asarray(m.body_subtreemass), dtype=FLOAT)[None, :, None], 1e-30)
    self.subtree_com = sub

    # subtree mass (needed for com velocity / center of mass frames)
    submass = mx.array(np.asarray(m.body_subtreemass), dtype=FLOAT)

    # cinert (vec10 about subtree root com)
    root = self.body_rootid
    dif = self.xipos - mx.take(sub, root, axis=1)
    inertia = mx.array(np.asarray(m.body_inertia), dtype=FLOAT)
    mat = self.ximat
    inertdiag = mx.eye(3)[None, :, :] * inertia[:, None, :]  # (nbody, 3, 3)
    tmp = mat @ inertdiag[None] @ mx.swapaxes(mat, -1, -2)
    rm = mass[None, :, None]
    res = mx.stack(
      [
        tmp[..., 0, 0] + rm[..., 0] * (dif[..., 1] ** 2 + dif[..., 2] ** 2),
        tmp[..., 1, 1] + rm[..., 0] * (dif[..., 0] ** 2 + dif[..., 2] ** 2),
        tmp[..., 2, 2] + rm[..., 0] * (dif[..., 0] ** 2 + dif[..., 1] ** 2),
        tmp[..., 0, 1] - rm[..., 0] * dif[..., 0] * dif[..., 1],
        tmp[..., 0, 2] - rm[..., 0] * dif[..., 0] * dif[..., 2],
        tmp[..., 1, 2] - rm[..., 0] * dif[..., 1] * dif[..., 2],
        mass[None, :] * dif[..., 0],
        mass[None, :] * dif[..., 1],
        mass[None, :] * dif[..., 2],
        mx.broadcast_to(mass[None, :], (nw, nbody)),
      ],
      axis=-1,
    )
    self.cinert = res
    self._submass = mx.broadcast_to(submass[None, :], (nw, nbody))

    # cdof (nworld, nv, 6): vectorized over worlds
    cdof = mx.zeros((nw, m.nv, 6), dtype=FLOAT)
    for j in range(m.njnt):
      body = int(m.jnt_bodyid[j])
      dof = int(m.jnt_dofadr[j])
      jtype = int(m.jnt_type[j])
      axis = self.xaxis[:, j]
      xmat = mx.swapaxes(self.xmat[:, body], -1, -2)
      offset = mx.take(sub, self.body_rootid[body:body + 1], axis=1)[:, 0] - self.xanchor[:, j]
      if jtype == 0:
        # translation dofs: [0, e_k]
        for k in range(3):
          e = mx.broadcast_to(np.eye(3, dtype=np.float32)[k], (nw, 3))
          cdof = cdof.at[:, dof + k].add(mx.concatenate([mx.zeros((nw, 3)), e], axis=-1))
        # rotation dofs: body-frame axes (columns of xmat)
        for k in range(3):
          a = xmat[..., k, :]
          cdof = cdof.at[:, dof + 3 + k].add(mx.concatenate([a, _cross(a, offset)], axis=-1))
      elif jtype == 3:
        cdof = cdof.at[:, dof].add(mx.concatenate([axis, _cross(axis, offset)], axis=-1))
    self.cdof = cdof

  # ------------------------------------------------------------------- M
  def crb_compute(self):
    m = self.m
    nbody = m.nbody
    cin = self.cinert
    acc = cin
    for b in range(nbody - 1, 0, -1):
      pid = int(m.body_parentid[b])
      if pid != 0 and pid != b:  # warp skips accumulation into the world body
        acc = acc.at[:, pid].add(acc[:, b])
    self.crb = acc

  # ------------------------------------------------------------- velocities
  def com_vel(self):
    m = self.m
    nw, nbody = self.nworld, m.nbody
    cdof_dot = mx.zeros((nw, m.nv, 6), dtype=FLOAT)
    cvel = mx.zeros((nw, nbody, 6), dtype=FLOAT)
    for b in range(1, nbody):
      pid = int(m.body_parentid[b])
      cvel_b = cvel[:, pid]
      dofid = int(m.body_dofadr[b])
      ja = int(m.body_jntadr[b])
      jn = int(m.body_jntnum[b])
      if jn == 0:
        cvel = cvel.at[:, b].add(cvel_b)
        continue
      for j in range(ja, ja + jn):
        jtype = int(m.jnt_type[j])
        if jtype == 0:
          for k in range(3):
            cvel_b = cvel_b + self.cdof[:, dofid + k] * self.qvel[:, dofid + k : dofid + k + 1]
          for k in range(3):
            cdof_dot = cdof_dot.at[:, dofid + 3 + k].add(self._motion_cross(cvel_b, self.cdof[:, dofid + 3 + k]))
          for k in range(3):
            cvel_b = cvel_b + self.cdof[:, dofid + 3 + k] * self.qvel[:, dofid + 3 + k : dofid + 3 + k + 1]
          dofid += 6
        elif jtype == 3:
          cdof_dot = cdof_dot.at[:, dofid].add(self._motion_cross(cvel_b, self.cdof[:, dofid]))
          cvel_b = cvel_b + self.cdof[:, dofid] * self.qvel[:, dofid : dofid + 1]
          dofid += 1
      cvel = cvel.at[:, b].add(cvel_b)
    self.cvel = cvel
    self.cdof_dot = cdof_dot

  def mass_matrix(self):
    """Batched dense M (nworld, nv, nv), symmetric, incl. armature."""
    m = self.m
    nw, nv = self.nworld, m.nv
    body_of_dof = self.dof_bodyid
    buf = self._inert_vec(mx.take(self.crb, body_of_dof, axis=1), self.cdof)  # (nw, nv, 6)
    janc = mx.take(self.cdof, self.chain_idx, axis=1)  # (nw, nv, maxc, 6)
    mchain = mx.sum(janc * buf[:, :, None, :], axis=-1) * self.chain_mask[None]

    lower = mx.zeros((nw, nv, nv), dtype=FLOAT)
    for i in range(nv):
      idx = mx.array(self.chain_idx[i])  # (maxc,), -1 pads masked by zero values
      lower = lower.at[:, i, idx].add(mchain[:, i])
    arm = mx.array(np.asarray(m.dof_armature), dtype=FLOAT)
    diag_l = mx.diagonal(lower, axis1=-2, axis2=-1)  # (nw, nv)
    dyn = diag_l[:, :, None] * mx.eye(nv)[None]
    M = lower + mx.swapaxes(lower, -1, -2) - dyn
    self.M = M + arm[None, :, None] * mx.eye(nv)[None]

  def passive_actuation(self):
    m = self.m
    nw, nv, nu = self.nworld, m.nv, m.nu
    self.qfrc_passive = -mx.array(np.asarray(m.dof_damping), dtype=FLOAT)[None] * self.qvel
    qfrc = mx.zeros((nw, nv), dtype=FLOAT)
    for u in range(nu):
      j = int(m.actuator_trnid[u, 0])
      dof = int(m.jnt_dofadr[j])
      qadr = int(m.jnt_qposadr[j])
      gear = float(m.actuator_gear[u, 0])
      length = self.qpos[:, qadr] * gear
      vel = gear * self.qvel[:, dof]
      ctrl = self.ctrl[:, u]
      if m.actuator_ctrllimited[u]:
        ctrl = mx.minimum(mx.maximum(ctrl, float(m.actuator_ctrlrange[u, 0])), float(m.actuator_ctrlrange[u, 1]))
      force = float(m.actuator_gainprm[u, 0]) * ctrl + (
        float(m.actuator_biasprm[u, 0])
        + float(m.actuator_biasprm[u, 1]) * length
        + float(m.actuator_biasprm[u, 2]) * vel
      )
      if m.actuator_forcelimited[u]:
        force = mx.minimum(mx.maximum(force, float(m.actuator_forcerange[u, 0])), float(m.actuator_forcerange[u, 1]))
      qfrc = qfrc.at[:, dof].add(gear * force)
    self.qfrc_actuator = qfrc
    self.qfrc_smooth = self.qfrc_passive - self.qfrc_bias + self.qfrc_actuator

  @staticmethod
  def _motion_cross(u, v):
    u0, u1 = u[..., :3], u[..., 3:]
    v0, v1 = v[..., :3], v[..., 3:]
    ang = _cross(u0, v0)
    vel = _cross(u1, v0) + _cross(u0, v1)
    return mx.concatenate([ang, vel], axis=-1)

  @staticmethod
  def _inert_vec(i, v):
    return mx.stack(
      [
        i[..., 0] * v[..., 0] + i[..., 3] * v[..., 1] + i[..., 4] * v[..., 2] - i[..., 8] * v[..., 4] + i[..., 7] * v[..., 5],
        i[..., 3] * v[..., 0] + i[..., 1] * v[..., 1] + i[..., 5] * v[..., 2] + i[..., 8] * v[..., 3] - i[..., 6] * v[..., 5],
        i[..., 4] * v[..., 0] + i[..., 5] * v[..., 1] + i[..., 2] * v[..., 2] - i[..., 7] * v[..., 3] + i[..., 6] * v[..., 4],
        i[..., 8] * v[..., 1] - i[..., 7] * v[..., 2] + i[..., 9] * v[..., 3],
        i[..., 6] * v[..., 2] - i[..., 8] * v[..., 0] + i[..., 9] * v[..., 4],
        i[..., 7] * v[..., 0] - i[..., 6] * v[..., 1] + i[..., 9] * v[..., 5],
      ],
      axis=-1,
    )

  @staticmethod
  def _motion_cross_force(v, f):
    v0, v1 = v[..., :3], v[..., 3:]
    f0, f1 = f[..., :3], f[..., 3:]
    ang = _cross(v0, f0) + _cross(v1, f1)
    vel = _cross(v0, f1)
    return mx.concatenate([ang, vel], axis=-1)

  def rne(self):
    m = self.m
    nbody = m.nbody
    nw = self.nworld
    cacc = [None] * nbody
    g = mx.array(np.asarray(m.gravity), dtype=FLOAT)
    cacc[0] = mx.concatenate([mx.zeros((nw, 3)), mx.broadcast_to(-g, (nw, 3))], axis=-1)
    for b in range(1, nbody):
      pid = int(m.body_parentid[b])
      local = cacc[pid]
      da = int(m.body_dofadr[b])
      dn = int(m.body_dofnum[b])
      for d in range(da, da + dn):
        local = local + self.cdof_dot[:, d] * self.qvel[:, d : d + 1]
      cacc[b] = local

    cfrc = [None] * nbody
    cfrc[0] = mx.zeros((nw, 6), dtype=FLOAT)
    for b in range(1, nbody):
      frc = self._inert_vec(self.cinert[:, b], cacc[b])
      frc = frc + self._motion_cross_force(self.cvel[:, b], self._inert_vec(self.cinert[:, b], self.cvel[:, b]))
      cfrc[b] = frc
    for b in range(nbody - 1, 0, -1):
      pid = int(m.body_parentid[b])
      cfrc[pid] = cfrc[pid] + cfrc[b]
    self.cfrc_int = mx.stack(cfrc, axis=1)
    body_of_dof = self.dof_bodyid
    cd = self.cdof
    cf = self.cfrc_int
    contrib = cd * mx.take(cf, body_of_dof, axis=1)
    self.qfrc_bias = mx.sum(contrib, axis=-1)

  def eval(self, *fields):
    mx.eval(*fields)
