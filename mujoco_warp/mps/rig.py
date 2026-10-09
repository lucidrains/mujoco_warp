"""Static rig data for the world-batched MLX engine.

Ports the pair filtering (`io.py` nxn pair tables), contact material mixing
(`collision_core.contact_material_params`) and constraint sizing to plain numpy,
reusing a `mujoco_warp.put_model` result so candidate pairs match the warp
backend exactly. Everything here is per-run constant; per-world variation lives
in the kernels driven by these tables.
"""

from __future__ import annotations

import mujoco
import numpy as np

import mujoco_warp
from mujoco_warp._src.types import ConeType
from mujoco_warp._src.types import IntegratorType
from mujoco_warp._src.types import SolverType

MJ_MINVAL = 1e-15
MJ_MINMU = 1e-5

# disable bits that change the dynamics; the engine mirrors warp only for
# warmstart / euler damping (and the static nxn pair list honors filterparent)
_DISABLE_UNSUPPORTED = int(
  mujoco_warp.DisableBit.CONSTRAINT
  | mujoco_warp.DisableBit.FRICTIONLOSS
  | mujoco_warp.DisableBit.LIMIT
  | mujoco_warp.DisableBit.CONTACT
  | mujoco_warp.DisableBit.SPRING
  | mujoco_warp.DisableBit.DAMPER
  | mujoco_warp.DisableBit.GRAVITY
  | mujoco_warp.DisableBit.CLAMPCTRL
  | mujoco_warp.DisableBit.ACTUATION
  | mujoco_warp.DisableBit.REFSAFE
)

# primitive narrowphase opcode ids (index = opcode)
OP_PLANE_SPHERE = 0  # 1 point
OP_PLANE_CAPSULE = 1  # 2 points
OP_SPHERE_SPHERE = 2  # 1 point
OP_SPHERE_CAPSULE = 3  # 1 point
OP_CAPSULE_CAPSULE = 4  # 2 points

_GEOM = mujoco.mjtGeom
_OPCODE_TABLE = {
  (_GEOM.mjGEOM_PLANE, _GEOM.mjGEOM_SPHERE): OP_PLANE_SPHERE,
  (_GEOM.mjGEOM_PLANE, _GEOM.mjGEOM_CAPSULE): OP_PLANE_CAPSULE,
  (_GEOM.mjGEOM_SPHERE, _GEOM.mjGEOM_SPHERE): OP_SPHERE_SPHERE,
  (_GEOM.mjGEOM_SPHERE, _GEOM.mjGEOM_CAPSULE): OP_SPHERE_CAPSULE,
  (_GEOM.mjGEOM_CAPSULE, _GEOM.mjGEOM_CAPSULE): OP_CAPSULE_CAPSULE,
}

# max points emitted per opcode (plane-capsule: both capsule ends)
POINTS_PER_OPCODE = (1, 2, 1, 1, 2)


def _pair_material(mjm: mujoco.MjModel, pairid: int, g1: int, g2: int) -> dict:
  """Port of collision_core.contact_material_params / contact_margin_gap."""
  m = mjm

  # explicit <pair> overrides
  if pairid >= 0:
    return dict(
      condim=int(m.pair_dim[pairid]),
      friction=np.maximum(np.asarray(m.pair_friction[pairid], np.float64), MJ_MINMU),
      solref=np.asarray(m.pair_solref[pairid], np.float64),
      solimp=np.asarray(m.pair_solimp[pairid], np.float64),
      margin=float(m.pair_margin[pairid]),
    )

  # geom-level mixing by priority and solmix
  solmix1, solmix2 = float(m.geom_solmix[g1]), float(m.geom_solmix[g2])
  p1, p2 = int(m.geom_priority[g1]), int(m.geom_priority[g2])
  sr1, sr2 = m.geom_solref[g1], m.geom_solref[g2]
  si1, si2 = m.geom_solimp[g1], m.geom_solimp[g2]
  if p1 > p2:
    mix, condim, gf, solref = 1.0, int(m.geom_condim[g1]), m.geom_friction[g1], sr1
  elif p2 > p1:
    mix, condim, gf, solref = 0.0, int(m.geom_condim[g2]), m.geom_friction[g2], sr2
  else:
    mix = solmix1 / (solmix1 + solmix2) if solmix1 + solmix2 > 0 else 0.5
    mix = 0.5 if solmix1 < MJ_MINVAL and solmix2 < MJ_MINVAL else mix
    mix = 0.0 if solmix1 < MJ_MINVAL <= solmix2 else mix
    mix = 1.0 if solmix2 < MJ_MINVAL <= solmix1 else mix
    condim = max(int(m.geom_condim[g1]), int(m.geom_condim[g2]))
    gf = np.maximum(m.geom_friction[g1], m.geom_friction[g2])
    solref = mix * sr1 + (1.0 - mix) * sr2 if sr1[0] > 0 and sr2[0] > 0 else np.minimum(sr1, sr2)

  return dict(
    condim=condim,
    friction=np.maximum(np.array([gf[0], gf[0], gf[1], gf[2], gf[2]], np.float64), MJ_MINMU),
    solref=solref,
    solimp=mix * si1 + (1.0 - mix) * si2,
    margin=float(m.geom_margin[g1] + m.geom_margin[g2]),
  )


class Rig:
  """Static per-model tables for the batched MLX engine (primitive geoms)."""

  def __init__(self, mjm: mujoco.MjModel, contact_row_cap: int = 96):
    """Builds static tables; raises on unsupported model features."""
    m = mujoco_warp.put_model(mjm)
    self.mw = m

    # ---- unsupported-feature gate
    for name, cond in (
      ("tendon", mjm.ntendon > 0),
      ("equality", mjm.neq > 0),
      ("flex", mjm.nflex > 0),
      ("ball joints", bool(np.any(mjm.jnt_type == mujoco.mjtJoint.mjJNT_BALL))),
      ("elliptic cone", ConeType(mjm.opt.cone) != ConeType.PYRAMIDAL),
      ("non-euler integrator", IntegratorType(mjm.opt.integrator) != IntegratorType.EULER),
      ("actuator dynamics", bool(np.any(mjm.actuator_dyntype != int(mujoco.mjtDyn.mjDYN_NONE)))),
      ("non-joint transmissions", bool(np.any(mjm.actuator_trntype != int(mujoco.mjtTrn.mjTRN_JOINT)))),
      ("non-fixed actuator gains", bool(np.any(mjm.actuator_gaintype != int(mujoco.mjtGain.mjGAIN_FIXED)))),
      ("non-none/affine actuator bias", bool(np.any(mjm.actuator_biastype > int(mujoco.mjtBias.mjBIAS_AFFINE)))),
      ("heightfield geoms", bool(np.any(mjm.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD))),
      ("sdf geoms", bool(np.any(mjm.geom_type == mujoco.mjtGeom.mjGEOM_SDF))),
      ("body gravcomp", bool(np.any(np.asarray(mjm.body_gravcomp) != 0.0))),
      ("joint stiffness polynomial", bool(np.any(np.asarray(mjm.jnt_stiffnesspoly) != 0.0))),
      ("dof damping polynomial", bool(np.any(np.asarray(mjm.dof_dampingpoly) != 0.0))),
      (
        "actuator damping",
        bool(np.any(np.asarray(mjm.actuator_damping) != 0.0) or np.any(np.asarray(mjm.actuator_dampingpoly) != 0.0)),
      ),
      ("fluid forces", bool(np.any(np.asarray(mjm.geom_fluid) != 0.0))),
      ("sleep", bool(int(mjm.opt.enableflags) & int(mujoco_warp.EnableBit.SLEEP))),
      ("unsupported disable flags", bool(int(mjm.opt.disableflags) & _DISABLE_UNSUPPORTED)),
      ("non-newton solver", SolverType(mjm.opt.solver) != SolverType.NEWTON),
    ):
      if cond:
        raise ValueError(f"MLX engine: unsupported model feature: {name}")

    self.timestep = float(mjm.opt.timestep)
    self.gravity = np.array(mjm.opt.gravity, np.float32)
    self.impratio = float(mjm.opt.impratio)
    self.impratio_invsqrt = 1.0 / np.sqrt(self.impratio)
    self.meaninertia = float(mjm.stat.meaninertia)
    self.tolerance = float(np.asarray(m.opt.tolerance.numpy()).reshape(-1)[0])  # warp raises the f32 floor to 1e-6
    self.ls_tolerance = float(mjm.opt.ls_tolerance)
    self.iterations = int(mjm.opt.iterations)
    self.ls_iterations = int(mjm.opt.ls_iterations)
    self.disableflags = int(mjm.opt.disableflags)
    self.broadphase_filter = int(m.opt.broadphase_filter)
    self.warmstart = not bool(self.disableflags & mujoco_warp.DisableBit.WARMSTART)
    self.eulerdamp = not bool(self.disableflags & mujoco_warp.DisableBit.EULERDAMP)

    # ---- sizes
    self.nv = int(mjm.nv)
    self.nq = int(mjm.nq)
    self.nbody = int(mjm.nbody)
    self.ngeom = int(mjm.ngeom)
    self.njnt = int(mjm.njnt)
    self.nu = int(mjm.nu)

    # ---- model arrays (float32/int32 copies)
    a = np.asarray

    self.body_parentid = a(mjm.body_parentid, np.int32)
    self.body_rootid = a(mjm.body_rootid, np.int32)
    self.body_dofadr = a(mjm.body_dofadr, np.int32)
    self.body_dofnum = a(mjm.body_dofnum, np.int32)
    self.body_jntadr = a(mjm.body_jntadr, np.int32)
    self.body_jntnum = a(mjm.body_jntnum, np.int32)
    self.body_pos = a(mjm.body_pos, np.float32)
    self.body_quat = a(mjm.body_quat, np.float32)
    self.body_ipos = a(mjm.body_ipos, np.float32)
    self.body_iquat = a(mjm.body_iquat, np.float32)
    self.body_mass = a(mjm.body_mass, np.float32)
    self.body_subtreemass = a(mjm.body_subtreemass, np.float32)
    self.body_inertia = a(mjm.body_inertia, np.float32)
    self.body_invweight0 = a(mjm.body_invweight0, np.float32)

    self.jnt_type = a(mjm.jnt_type, np.int32)
    self.jnt_bodyid = a(mjm.jnt_bodyid, np.int32)
    self.jnt_dofadr = a(mjm.jnt_dofadr, np.int32)
    self.jnt_qposadr = a(mjm.jnt_qposadr, np.int32)
    self.jnt_pos = a(mjm.jnt_pos, np.float32)
    self.jnt_axis = a(mjm.jnt_axis, np.float32)
    self.jnt_range = a(mjm.jnt_range, np.float32)
    self.jnt_solref = a(mjm.jnt_solref, np.float32)
    self.jnt_solimp = a(mjm.jnt_solimp, np.float32)
    self.jnt_margin = a(mjm.jnt_margin, np.float32)
    self.jnt_stiffness = a(mjm.jnt_stiffness, np.float32)

    self.dof_bodyid = a(mjm.dof_bodyid, np.int32)
    self.dof_parentid = a(mjm.dof_parentid, np.int32)
    self.dof_damping = a(mjm.dof_damping, np.float32)
    self.dof_armature = a(mjm.dof_armature, np.float32)
    self.dof_frictionloss = a(mjm.dof_frictionloss, np.float32)
    self.dof_invweight0 = a(mjm.dof_invweight0, np.float32)
    self.dof_solref = a(mjm.dof_solref, np.float32)
    self.dof_solimp = a(mjm.dof_solimp, np.float32)

    self.geom_bodyid = a(mjm.geom_bodyid, np.int32)
    self.geom_type = a(mjm.geom_type, np.int32)
    self.geom_pos = a(mjm.geom_pos, np.float32)
    self.geom_quat = a(mjm.geom_quat, np.float32)
    self.geom_size = a(mjm.geom_size, np.float32)
    self.geom_rbound = np.asarray(m.geom_rbound.numpy(), np.float32).reshape(-1)[: int(mjm.ngeom)]
    self.geom_margin = a(mjm.geom_margin, np.float32)
    self.geom_gap = a(mjm.geom_gap, np.float32)

    self.actuator_trnid = a(mjm.actuator_trnid, np.int32)
    self.actuator_gear = a(mjm.actuator_gear, np.float32)
    self.actuator_gainprm = a(mjm.actuator_gainprm, np.float32)
    self.actuator_biasprm = a(mjm.actuator_biasprm, np.float32)
    self.actuator_ctrlrange = a(mjm.actuator_ctrlrange, np.float32)
    self.actuator_forcerange = a(mjm.actuator_forcerange, np.float32)
    self.actuator_ctrllimited = a(mjm.actuator_ctrllimited, np.int32)
    self.actuator_forcelimited = a(mjm.actuator_forcelimited, np.int32)

    self.qpos_spring = a(mjm.qpos_spring, np.float32)
    self.qpos0 = a(mjm.qpos0, np.float32)

    # dof ancestors per body: (nbody, nv) bitmap, from warp put_model
    self.body_isdofancestor = m.body_isdofancestor.numpy().astype(np.int8)[:, : self.nv].copy()

    # ---- candidate static rows
    self.friction_dofs = np.nonzero(mjm.dof_frictionloss > 0.0)[0]
    self.limited_joints = m.jnt_limited_slide_hinge_adr.numpy().astype(np.int32)
    self.nfr = int(len(self.friction_dofs))
    self.nlim = int(len(self.limited_joints))
    self.nfrlim = self.nfr + self.nlim

    # ---- contact candidate pairs (world-independent)
    gp = m.nxn_geom_pair_filtered.numpy().astype(np.int32)
    pids = m.nxn_pairid_filtered.numpy().astype(np.int32)
    ops, g1s, g2s, mats = [], [], [], []
    for k in range(len(gp)):
      if int(pids[k, 0]) < -1:
        continue
      g1, g2 = int(gp[k, 0]), int(gp[k, 1])
      lo, hi = (g1, g2) if int(mjm.geom_type[g1]) <= int(mjm.geom_type[g2]) else (g2, g1)
      opcode = _OPCODE_TABLE.get((int(mjm.geom_type[lo]), int(mjm.geom_type[hi])))
      if opcode is None:
        raise ValueError(
          f"MLX engine: unsupported collision pair geoms {g1}x{g2} (types {mjm.geom_type[lo]}x{mjm.geom_type[hi]})"
        )
      ops.append(opcode)
      g1s.append(lo)
      g2s.append(hi)
      mats.append(_pair_material(mjm, int(pids[k, 0]), g1, g2))

    self.npairc = len(ops)
    self.pair_op = np.asarray(ops, np.int32)
    self.pair_g1 = np.asarray(g1s, np.int32)
    self.pair_g2 = np.asarray(g2s, np.int32)
    self.pair_body1 = self.geom_bodyid[self.pair_g1].astype(np.int32)
    self.pair_body2 = self.geom_bodyid[self.pair_g2].astype(np.int32)
    if self.npairc > 0:
      self.pair_margin = np.asarray([mat["margin"] for mat in mats], np.float32)
      self.pair_solref = np.asarray([mat["solref"] for mat in mats], np.float32).reshape(self.npairc, 2)
      self.pair_solimp = np.asarray([mat["solimp"] for mat in mats], np.float32).reshape(self.npairc, 5)
      self.pair_friction = np.asarray([mat["friction"] for mat in mats], np.float32).reshape(self.npairc, 5)
      self.pair_condim = np.asarray([mat["condim"] for mat in mats], np.int32)
      if not np.isin(self.pair_condim, (1, 3, 4, 6)).all():
        raise ValueError(f"MLX engine: unsupported contact condim {sorted(set(self.pair_condim.tolist()))}")
      # pyramidal rows per point: condim==1 -> 1 row, else 2*(condim-1)
      self.pair_ndim = np.where(self.pair_condim == 1, 1, 2 * (self.pair_condim - 1))
      b1w = self.body_invweight0[self.pair_body1][:, 0].astype(np.float32)
      b2w = self.body_invweight0[self.pair_body2][:, 0].astype(np.float32)
      fri0 = self.pair_friction[:, 0].astype(np.float32)
      invw_base = (b1w + b2w).astype(np.float32)
      isq2 = np.float32(self.impratio_invsqrt**2)
      invw_scaled = (invw_base + fri0 * fri0 * invw_base) * (2.0 * fri0 * fri0) * isq2
      self.pair_invw = np.where(self.pair_condim > 1, invw_scaled, invw_base).astype(np.float32)
      pts = np.asarray(POINTS_PER_OPCODE, np.int32)[self.pair_op]
      max_rows_needed = int(np.sum(self.pair_ndim * pts))
    else:
      self.pair_margin = np.zeros(0, np.float32)
      self.pair_solref = np.zeros((0, 2), np.float32)
      self.pair_solimp = np.zeros((0, 5), np.float32)
      self.pair_friction = np.zeros((0, 5), np.float32)
      self.pair_condim = np.zeros(0, np.int32)
      self.pair_ndim = np.zeros(0, np.int32)
      self.pair_invw = np.zeros(0, np.float32)
      max_rows_needed = 0
    self.contact_row_cap = min(contact_row_cap, max(1, max_rows_needed))
    self.rowcap = self.nfrlim + self.contact_row_cap

    # per-geom broadphase band (margin+gap) and local aabb (OBB filter)
    self.geom_mg = np.ascontiguousarray(self.geom_margin + self.geom_gap)
    self.geom_aabb = np.ascontiguousarray(m.geom_aabb.numpy().astype(np.float32).reshape(self.ngeom, 6))
