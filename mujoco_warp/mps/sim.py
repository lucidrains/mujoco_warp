"""MLX port of the mujoco_warp forward pipeline for a single-world rigid body model.

The algorithm mirrors mujoco_warp's kernels (which are validated against MuJoCo)
so that, for the same initial state and controls, trajectories track the warp CPU
backend. All state lives on the MLX device (Metal); the reference loop can run on
CPU for comparison.

Conventions match warp/MuJoCo:
  * quaternion (w, x, y, z)
  * spatial vector (angular, linear)
  * quat_to_mat returns row-major rotation matrix
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .model import ModelArrays

FLOAT = mx.float32


def cross(a: mx.array, b: mx.array) -> mx.array:
  return mx.stack(
    [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ]
  )


def quat_mul(u: mx.array, v: mx.array) -> mx.array:
  return mx.stack(
    [
      u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
      u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
      u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
      u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
    ]
  )


def quat_rot(vec: mx.array, quat: mx.array) -> mx.array:
  s = quat[0]
  u = quat[1:4]
  r = 2.0 * (mx.sum(u * vec) * u) + (s * s - mx.sum(u * u)) * vec
  return r + 2.0 * s * cross(u, vec)


def quat_normalize(quat: mx.array) -> mx.array:
  return quat / mx.sqrt(mx.sum(quat * quat))


def axis_angle_to_quat(axis: mx.array, angle) -> mx.array:
  s = mx.sin(angle * 0.5)
  c = mx.cos(angle * 0.5)
  parts = [c, axis[0] * s, axis[1] * s, axis[2] * s]
  return mx.stack([mx.array(p, dtype=FLOAT) for p in parts])


def quat_to_mat(quat: mx.array) -> mx.array:
  q = quat
  q00, q01, q02, q03 = q[0] * q[0], q[0] * q[1], q[0] * q[2], q[0] * q[3]
  q11, q12, q13 = q[1] * q[1], q[1] * q[2], q[1] * q[3]
  q22, q23, q33 = q[2] * q[2], q[2] * q[3], q[3] * q[3]
  return mx.stack(
    [
      mx.stack([q00 + q11 - q22 - q33, 2.0 * (q12 - q03), 2.0 * (q13 + q02)]),
      mx.stack([2.0 * (q12 + q03), q00 - q11 + q22 - q33, 2.0 * (q23 - q01)]),
      mx.stack([2.0 * (q13 - q02), 2.0 * (q23 + q01), q00 - q11 - q22 + q33]),
    ]
  )


def inert_vec(i: mx.array, v: mx.array) -> mx.array:
  """mju_mulInertVec: 6x6 inertia (vec10 layout) times spatial vector."""
  return mx.stack(
    [
      i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5],
      i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5],
      i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4],
      i[8] * v[1] - i[7] * v[2] + i[9] * v[3],
      i[6] * v[2] - i[8] * v[0] + i[9] * v[4],
      i[7] * v[0] - i[6] * v[1] + i[9] * v[5],
    ]
  )


def motion_cross(u: mx.array, v: mx.array) -> mx.array:
  u0, u1 = u[:3], u[3:]
  v0, v1 = v[:3], v[3:]
  ang = cross(u0, v0)
  vel = cross(u1, v0) + cross(u0, v1)
  return mx.concatenate([ang, vel])


def motion_cross_force(v: mx.array, f: mx.array) -> mx.array:
  v0, v1 = v[:3], v[3:]
  f0, f1 = f[:3], f[3:]
  ang = cross(v0, f0) + cross(v1, f1)
  vel = cross(v0, f1)
  return mx.concatenate([ang, vel])


class Sim:
  """Single-world MLX rigid body engine mirroring mujoco_warp's algorithms."""

  def __init__(self, m: ModelArrays):
    self.m = m
    self._to_mx()

  # ------------------------------------------------------------------ setup
  def _to_mx(self):
    m = self.m
    a = lambda x, dt=FLOAT: mx.array(np.asarray(x), dtype=dt)  # noqa: E731
    for name in (
      "body_parentid", "body_jntadr", "body_jntnum", "body_dofadr", "body_dofnum",
      "body_rootid", "body_weldid", "body_mocapid", "jnt_type", "jnt_bodyid", "jnt_dofadr",
      "jnt_qposadr", "dof_bodyid", "dof_parentid", "geom_bodyid", "geom_type", "geom_dataid",
      "geom_contype", "geom_conaffinity", "mesh_vertadr", "mesh_vertnum", "mesh_graphadr",
      "actuator_trntype", "actuator_gaintype", "actuator_biastype", "actuator_trnid",
      "geom_condim", "geom_priority",
    ):
      setattr(self, name, a(getattr(m, name), mx.int32))
    for name in (
      "body_pos", "body_quat", "body_mass", "body_subtreemass", "body_ipos", "body_iquat",
      "body_inertia", "jnt_pos", "jnt_axis", "jnt_range", "dof_damping", "dof_armature",
      "dof_frictionloss", "geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_solref",
      "geom_solimp", "geom_margin", "geom_gap", "mesh_vert", "mesh_graph", "actuator_gainprm",
      "actuator_biasprm", "actuator_ctrlrange", "actuator_forcerange", "actuator_gear", "jnt_solref", "jnt_solimp", "jnt_margin", "dof_solref", "dof_solimp",
      "dof_invweight0", "body_invweight0", "geom_solmix", "geom_adhesion",
      "o_solref", "o_solimp", "o_friction", "qpos0", "key_qpos", "key_ctrl",
    ):
      setattr(self, name, a(getattr(m, name)))
    self.gravity = a(m.gravity)
    self.timestep = float(m.timestep)
    self.nbody = m.nbody
    self.nv = m.nv
    self.nq = m.nq
    self.njnt = m.njnt
    self.nu = m.nu

    # dof ancestry chains (including self) for the CRB pass
    parent = np.asarray(m.dof_parentid, dtype=np.int64)
    chains = []
    for i in range(m.nv):
      chain = []
      j = i
      while j >= 0:
        chain.append(j)
        j = int(parent[j])
      chains.append(chain)
    self.dof_chains = chains

    # actuator moment rows: joint transmission -> gear on the joint's dof
    moments = []
    for i in range(m.nu):
      j = int(m.actuator_trnid[i, 0])
      dof = int(m.jnt_dofadr[j])
      moments.append([(dof, float(m.actuator_gear[i, 0]))])
    self.actuator_moments = moments

  # ------------------------------------------------------------------ state
  def set_state(self, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray | None = None):
    self.qpos = mx.array(np.asarray(qpos, dtype=np.float32))
    self.qvel = mx.array(np.asarray(qvel, dtype=np.float32))
    self.qacc = mx.zeros(self.m.nv, dtype=FLOAT)
    self.qacc_warmstart = mx.zeros(self.m.nv, dtype=FLOAT)
    self.ctrl = mx.array(np.asarray(ctrl if ctrl is not None else np.zeros(self.m.nu), dtype=np.float32))

  # ------------------------------------------------------------ kinematics
  def kinematics(self):
    m = self
    nbody = m.nbody
    xpos_all = [mx.zeros(3, dtype=FLOAT) for _ in range(nbody)]
    xquat_all = [mx.array(np.array([1.0, 0, 0, 0], dtype=np.float32)) for _ in range(nbody)]
    xanchor = [None] * m.njnt
    xaxis = [None] * m.njnt
    qpos = self.qpos

    for body in range(nbody):
      pid = int(m.body_parentid[body])
      jntadr = int(m.body_jntadr[body])
      jntnum = int(m.body_jntnum[body])

      if jntnum == 1 and int(m.jnt_type[jntadr]) == 0:  # FREE
        qadr = int(m.jnt_qposadr[jntadr])
        xpos = qpos[qadr : qadr + 3]
        xquat = quat_normalize(qpos[qadr + 3 : qadr + 7])
        xpos_all[body] = xpos
        xquat_all[body] = xquat
        xanchor[jntadr] = xpos
        xaxis[jntadr] = m.jnt_axis[jntadr]
        continue

      xpos = m.body_pos[body]
      xquat = m.body_quat[body]
      if pid >= 0:
        xpos = quat_rot(xpos, xquat_all[pid]) + xpos_all[pid]
        xquat = quat_mul(xquat_all[pid], xquat)

      j = jntadr
      for _ in range(jntnum):
        qadr = int(m.jnt_qposadr[j])
        jtype = int(m.jnt_type[j])
        axis = m.jnt_axis[j]
        anchor = quat_rot(m.jnt_pos[j], xquat) + xpos
        gaxis = quat_rot(axis, xquat)
        if jtype == 3:  # HINGE
          angle = qpos[qadr] - m.qpos0[qadr]
          qloc = axis_angle_to_quat(axis, angle)
          xquat = quat_mul(xquat, qloc)
          xpos = anchor - quat_rot(m.jnt_pos[j], xquat)
        xanchor[j] = anchor
        xaxis[j] = gaxis
        j += 1

      xquat = quat_normalize(xquat)
      xpos_all[body] = xpos
      xquat_all[body] = xquat

    self.xpos = mx.stack(xpos_all)
    self.xquat = mx.stack(xquat_all)
    self.xmat = mx.stack([quat_to_mat(xquat_all[b]) for b in range(nbody)])
    self.xanchor = mx.stack([xanchor[j] if xanchor[j] is not None else mx.zeros(3, dtype=FLOAT) for j in range(m.njnt)])
    self.xaxis = mx.stack([xaxis[j] if xaxis[j] is not None else mx.zeros(3, dtype=FLOAT) for j in range(m.njnt)])
    self.xipos = self.xpos + mx.stack([quat_rot(m.body_ipos[b], self.xquat[b]) for b in range(nbody)])
    self.ximat = mx.stack(
      [quat_to_mat(quat_mul(self.xquat[b], m.body_iquat[b])) for b in range(nbody)]
    )

  # ------------------------------------------------------------------- com
  def com_pos(self):
    m = self
    nbody = m.nbody
    # subtree com (mass weighted, then divide)
    acc = [m.body_mass[b] * self.xipos[b] for b in range(nbody)]
    for b in range(nbody - 1, 0, -1):
      pid = int(m.body_parentid[b])
      if pid >= 0:
        acc[pid] = acc[pid] + acc[b]
    sub = [acc[b] / mx.where(m.body_subtreemass[b] != 0.0, m.body_subtreemass[b], 1.0) for b in range(nbody)]
    self.subtree_com = mx.stack(sub)

    # cinert (vec10, com-based, in subtree root com frame)
    cin = []
    for b in range(nbody):
      mat = self.ximat[b]
      inert = m.body_inertia[b]
      mass = m.body_mass[b]
      dif = self.xipos[b] - self.subtree_com[int(m.body_rootid[b])]
      tmp = mat @ mx.diag(inert) @ mat.T
      vals = [
        tmp[0, 0] + mass * (dif[1] * dif[1] + dif[2] * dif[2]),
        tmp[1, 1] + mass * (dif[0] * dif[0] + dif[2] * dif[2]),
        tmp[2, 2] + mass * (dif[0] * dif[0] + dif[1] * dif[1]),
        tmp[0, 1] - mass * dif[0] * dif[1],
        tmp[0, 2] - mass * dif[0] * dif[2],
        tmp[1, 2] - mass * dif[1] * dif[2],
        mass * dif[0],
        mass * dif[1],
        mass * dif[2],
        mass,
      ]
      cin.append(mx.stack(vals))
    self.cinert = mx.stack(cin)

    # cdof
    cdof = [mx.zeros(6, dtype=FLOAT)] * m.nv
    cdof = list(cdof)
    for j in range(m.njnt):
      body = int(m.jnt_bodyid[j])
      dofid = int(m.jnt_dofadr[j])
      jtype = int(m.jnt_type[j])
      xaxis = self.xaxis[j]
      xmat = self.xmat[body].T
      offset = self.subtree_com[int(m.body_rootid[body])] - self.xanchor[j]
      if jtype == 0:  # FREE
        for k in range(3):
          lin = mx.zeros(3, dtype=FLOAT)
          lin = lin.at[k].add(1.0)
          cdof[dofid + k] = mx.concatenate([mx.zeros(3, dtype=FLOAT), lin])
        for k in range(3):
          axis = xmat[k]
          cdof[dofid + 3 + k] = mx.concatenate([axis, cross(axis, offset)])
      elif jtype == 3:  # HINGE
        cdof[dofid] = mx.concatenate([xaxis, cross(xaxis, offset)])
    self.cdof = mx.stack(cdof)

  # ------------------------------------------------------------------- M / crb
  def crb_compute(self):
    m = self
    nbody = m.nbody
    crb = [self.cinert[b] for b in range(nbody)]
    for b in range(nbody - 1, 0, -1):
      pid = int(m.body_parentid[b])
      if pid > 0:
        crb[pid] = crb[pid] + crb[b]
    self.crb = mx.stack(crb)

    # dense M: row i is dot(cdof[j], crb[body_i] @ cdof[i]) for j in ancestry(i).
    # warp stores only the lower triangle (row >= col); symmetrize for dense solves.
    nv = m.nv
    rows = []
    for i in range(nv):
      bodyid = int(m.dof_bodyid[i])
      buf = inert_vec(self.crb[bodyid], self.cdof[i])
      row = mx.zeros(nv, dtype=FLOAT)
      for j in self.dof_chains[i]:
        row = row.at[j].add(mx.sum(self.cdof[j] * buf))
      row = row.at[i].add(m.dof_armature[i])
      rows.append(row)
    lower = mx.stack(rows)
    diag = mx.diag(mx.diagonal(lower))
    self.M = lower + lower.T - diag

  # ------------------------------------------------------------- velocities
  def com_vel(self):
    m = self
    nbody = m.nbody
    cvel = [mx.zeros(6, dtype=FLOAT) for _ in range(nbody)]
    cdof_dot = [mx.zeros(6, dtype=FLOAT) for _ in range(m.nv)]
    qvel = self.qvel

    for body in range(1, nbody):
      pid = int(m.body_parentid[body])
      cvel_b = cvel[pid]
      dofid = int(m.body_dofadr[body])
      jntadr = int(m.body_jntadr[body])
      jntnum = int(m.body_jntnum[body])
      if jntnum == 0:
        cvel[body] = cvel_b
        continue
      for j in range(jntadr, jntadr + jntnum):
        jtype = int(m.jnt_type[j])
        if jtype == 0:  # FREE
          for k in range(3):
            cvel_b = cvel_b + self.cdof[dofid + k] * qvel[dofid + k]
          for k in range(3):
            cdof_dot[dofid + k] = mx.zeros(6, dtype=FLOAT)
          for k in range(3):
            cdof_dot[dofid + 3 + k] = motion_cross(cvel_b, self.cdof[dofid + 3 + k])
          for k in range(3):
            cvel_b = cvel_b + self.cdof[dofid + 3 + k] * qvel[dofid + 3 + k]
          dofid += 6
        elif jtype == 3:  # HINGE
          cdof_dot[dofid] = motion_cross(cvel_b, self.cdof[dofid])
          cvel_b = cvel_b + self.cdof[dofid] * qvel[dofid]
          dofid += 1
      cvel[body] = cvel_b
    self.cvel = mx.stack(cvel)
    self.cdof_dot = mx.stack(cdof_dot)

  def rne(self):
    """Bias forces (flg_acc=False): gravity + Coriolis."""
    m = self
    nbody = m.nbody
    cacc = [mx.zeros(6, dtype=FLOAT) for _ in range(nbody)]
    cacc[0] = mx.concatenate([mx.zeros(3, dtype=FLOAT), -self.gravity])
    for body in range(1, nbody):
      pid = int(m.body_parentid[body])
      local = cacc[pid]
      dofadr = int(m.body_dofadr[body])
      dofnum = int(m.body_dofnum[body])
      for d in range(dofadr, dofadr + dofnum):
        local = local + self.cdof_dot[d] * self.qvel[d]
      cacc[body] = local

    cfrc = []
    for b in range(nbody):
      if b == 0:
        cfrc.append(mx.zeros(6, dtype=FLOAT))
        continue
      frc = inert_vec(self.cinert[b], cacc[b])
      frc = frc + motion_cross_force(self.cvel[b], inert_vec(self.cinert[b], self.cvel[b]))
      cfrc.append(frc)
    for b in range(nbody - 1, 0, -1):
      pid = int(m.body_parentid[b])
      cfrc[pid] = cfrc[pid] + cfrc[b]
    self.cfrc_int = mx.stack(cfrc)
    self.qfrc_bias = mx.stack(
      [mx.sum(self.cdof[d] * self.cfrc_int[int(m.dof_bodyid[d])]) for d in range(m.nv)]
    )

  # ------------------------------------------------------------- utilities
  def eval(self, *fields):
    return mx.eval(*fields)
