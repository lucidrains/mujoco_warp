"""Convert a mujoco.MjModel into plain numpy arrays for the MLX engine.

Everything is extracted from MjModel (which already compiled the MJCF), so the MLX
engine never parses XML. The model is expected to be pre-modified by the environment
(timestep, actuator gains) before conversion.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class ModelArrays:
  nbody: int
  nv: int
  nq: int
  nu: int
  njnt: int
  ngeom: int
  nsite: int
  # body
  body_parentid: np.ndarray
  body_jntadr: np.ndarray
  body_jntnum: np.ndarray
  body_dofadr: np.ndarray
  body_dofnum: np.ndarray
  body_rootid: np.ndarray
  body_weldid: np.ndarray
  body_mocapid: np.ndarray
  body_pos: np.ndarray
  body_quat: np.ndarray
  body_mass: np.ndarray
  body_subtreemass: np.ndarray
  body_ipos: np.ndarray
  body_iquat: np.ndarray
  body_inertia: np.ndarray
  # joint
  jnt_type: np.ndarray
  jnt_bodyid: np.ndarray
  jnt_dofadr: np.ndarray
  jnt_qposadr: np.ndarray
  jnt_pos: np.ndarray
  jnt_axis: np.ndarray
  jnt_range: np.ndarray
  jnt_limited: np.ndarray
  # dof
  dof_bodyid: np.ndarray
  dof_parentid: np.ndarray
  dof_damping: np.ndarray
  dof_armature: np.ndarray
  dof_frictionloss: np.ndarray
  # geom
  geom_bodyid: np.ndarray
  geom_type: np.ndarray
  geom_dataid: np.ndarray
  geom_pos: np.ndarray
  geom_quat: np.ndarray
  geom_size: np.ndarray
  geom_contype: np.ndarray
  geom_conaffinity: np.ndarray
  geom_friction: np.ndarray
  geom_solref: np.ndarray
  geom_solimp: np.ndarray
  geom_margin: np.ndarray
  geom_gap: np.ndarray
  # mesh (collision hulls)
  mesh_vert: np.ndarray
  mesh_vertadr: np.ndarray
  mesh_vertnum: np.ndarray
  mesh_graphadr: np.ndarray
  mesh_graph: np.ndarray
  # actuator
  actuator_trntype: np.ndarray
  actuator_gaintype: np.ndarray
  actuator_biastype: np.ndarray
  actuator_gainprm: np.ndarray
  actuator_biasprm: np.ndarray
  actuator_ctrlrange: np.ndarray
  actuator_forcerange: np.ndarray
  actuator_gear: np.ndarray
  actuator_trnid: np.ndarray
  actuator_ctrllimited: np.ndarray
  actuator_forcelimited: np.ndarray
  # constraint materials
  jnt_solref: np.ndarray
  jnt_solimp: np.ndarray
  jnt_margin: np.ndarray
  dof_solref: np.ndarray
  dof_solimp: np.ndarray
  dof_invweight0: np.ndarray
  body_invweight0: np.ndarray
  geom_condim: np.ndarray
  geom_priority: np.ndarray
  geom_solmix: np.ndarray
  geom_adhesion: np.ndarray
  # options
  timestep: float
  gravity: np.ndarray
  integrator: int
  solver: int
  cone: int
  iterations: int
  ls_iterations: int
  tolerance: float
  impratio: float
  ls_tolerance: float
  meaninertia: float
  o_solref: np.ndarray
  o_solimp: np.ndarray
  o_friction: np.ndarray
  disableflags: int
  enableflags: int
  # keyframes
  key_qpos: np.ndarray
  key_ctrl: np.ndarray
  qpos0: np.ndarray


def _parent_dofs(mjm: mujoco.MjModel) -> np.ndarray:
  return np.asarray(mjm.dof_parentid, dtype=np.int32).copy()


def convert(mjm: mujoco.MjModel) -> ModelArrays:
  m = mjm
  arrays = dict(
    nbody=m.nbody,
    nv=m.nv,
    nq=m.nq,
    nu=m.nu,
    njnt=m.njnt,
    ngeom=m.ngeom,
    nsite=m.nsite,
    body_parentid=m.body_parentid.copy(),
    body_jntadr=m.body_jntadr.copy(),
    body_jntnum=m.body_jntnum.copy(),
    body_dofadr=m.body_dofadr.copy(),
    body_dofnum=m.body_dofnum.copy(),
    body_rootid=m.body_rootid.copy(),
    body_weldid=m.body_weldid.copy(),
    body_mocapid=m.body_mocapid.copy(),
    body_pos=m.body_pos.copy(),
    body_quat=m.body_quat.copy(),
    body_mass=m.body_mass.copy(),
    body_subtreemass=m.body_subtreemass.copy(),
    body_ipos=m.body_ipos.copy(),
    body_iquat=m.body_iquat.copy(),
    body_inertia=m.body_inertia.copy(),
    jnt_type=m.jnt_type.copy(),
    jnt_bodyid=m.jnt_bodyid.copy(),
    jnt_dofadr=m.jnt_dofadr.copy(),
    jnt_qposadr=m.jnt_qposadr.copy(),
    jnt_pos=m.jnt_pos.copy(),
    jnt_axis=m.jnt_axis.copy(),
    jnt_range=m.jnt_range.copy(),
    jnt_limited=m.jnt_limited.astype(bool),
    dof_bodyid=m.dof_bodyid.copy(),
    dof_parentid=_parent_dofs(m),
    dof_damping=m.dof_damping.copy(),
    dof_armature=m.dof_armature.copy(),
    dof_frictionloss=m.dof_frictionloss.copy(),
    geom_bodyid=m.geom_bodyid.copy(),
    geom_type=m.geom_type.copy(),
    geom_dataid=m.geom_dataid.copy(),
    geom_pos=m.geom_pos.copy(),
    geom_quat=m.geom_quat.copy(),
    geom_size=m.geom_size.copy(),
    geom_contype=m.geom_contype.copy(),
    geom_conaffinity=m.geom_conaffinity.copy(),
    geom_friction=m.geom_friction.copy(),
    geom_solref=m.geom_solref.copy(),
    geom_solimp=m.geom_solimp.copy(),
    geom_margin=m.geom_margin.copy(),
    geom_gap=m.geom_gap.copy(),
    mesh_vert=m.mesh_vert.copy(),
    mesh_vertadr=m.mesh_vertadr.copy(),
    mesh_vertnum=m.mesh_vertnum.copy(),
    mesh_graphadr=m.mesh_graphadr.copy(),
    mesh_graph=m.mesh_graph.copy(),
    actuator_trntype=m.actuator_trntype.copy(),
    actuator_gaintype=m.actuator_gaintype.copy(),
    actuator_biastype=m.actuator_biastype.copy(),
    actuator_gainprm=m.actuator_gainprm.copy(),
    actuator_biasprm=m.actuator_biasprm.copy(),
    actuator_ctrlrange=m.actuator_ctrlrange.copy(),
    actuator_forcerange=m.actuator_forcerange.copy(),
    actuator_gear=m.actuator_gear.copy(),
    actuator_trnid=m.actuator_trnid.copy(),
    actuator_ctrllimited=m.actuator_ctrllimited.astype(bool),
    actuator_forcelimited=m.actuator_forcelimited.astype(bool),
    jnt_solref=m.jnt_solref.copy(),
    jnt_solimp=m.jnt_solimp.copy(),
    jnt_margin=m.jnt_margin.copy(),
    dof_solref=m.dof_solref.copy(),
    dof_solimp=m.dof_solimp.copy(),
    dof_invweight0=m.dof_invweight0.copy(),
    body_invweight0=m.body_invweight0.copy(),
    geom_condim=m.geom_condim.copy(),
    geom_priority=m.geom_priority.copy(),
    geom_solmix=m.geom_solmix.copy(),
    geom_adhesion=m.geom_adhesion.copy(),
    meaninertia=float(m.stat.meaninertia),
    ls_tolerance=float(m.opt.ls_tolerance),
    timestep=float(m.opt.timestep),
    gravity=m.opt.gravity.copy(),
    integrator=int(m.opt.integrator),
    solver=int(m.opt.solver),
    cone=int(m.opt.cone),
    iterations=int(m.opt.iterations),
    ls_iterations=int(m.opt.ls_iterations),
    tolerance=float(m.opt.tolerance),
    impratio=float(m.opt.impratio),
    o_solref=m.opt.o_solref.copy(),
    o_solimp=m.opt.o_solimp.copy(),
    o_friction=m.opt.o_friction.copy(),
    disableflags=int(m.opt.disableflags),
    enableflags=int(m.opt.enableflags),
    key_qpos=m.key_qpos.copy(),
    key_ctrl=m.key_ctrl.copy(),
    qpos0=m.qpos0.copy(),
  )
  return ModelArrays(**arrays)
