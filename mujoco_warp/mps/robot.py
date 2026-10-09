"""World-batched MLX engine over the fused Metal kernels (`mujoco_warp/mps`).

```python
from mujoco_warp.mps.robot import BatchedEngine

eng = BatchedEngine(mjm, nworld=4096)  # one engine, N worlds
eng.set_state(qpos, qvel, ctrl)  # numpy (N, nq/nv/nu)
qpos, qvel = eng.step_np(ctrl)  # numpy out; MLX eval handled internally
```

Supported scope: free/slide/hinge joints, primitive geoms (plane, sphere,
capsule), pyramidal contacts, Euler integrator, joint motor/position actuators.
Everything else raises at construction (`Rig`).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from .fused import dyn_source
from .fused import solve_source
from .fused import solve_source_coop
from .rig import Rig

F = mx.float32

_DYN_TEMPLATE_KEYS = ("NBODY", "NJNT", "NQ", "NV", "NGEOM", "NCHAIN", "NU", "NP", "RC", "NFR", "NLIM", "BF")
_SOLVE_TEMPLATE_KEYS = ("NJNT", "NQ", "NV", "RC", "ITER", "LSITER", "WARMSTART", "EULERDAMP", "TG")

_DYN_INPUTS = ["qpos_in", "qvel_in", "ctrl_in", "nworld_buf", "mbf", "mif"]
_SOLVE_TAIL = ["nworld_buf", "dof_damping", "mif", "opt_buf"]
_SOLVE_INPUTS = [
  "qpos_in",
  "qvel_in",
  "warm_in",
  "M_in",
  "smooth_in",
  "J_in",
  "D_in",
  "aref_in",
  "fl_in",
  "nefc_in",
] + _SOLVE_TAIL

_DYN_BASE_OUTS = [
  "M_out",
  "smooth_out",
  "bias_out",
  "J_out",
  "D_out",
  "aref_out",
  "fl_out",
  "nefc_out",
  "ncon_out",
  "overflow_out",
]
_DYN_KIN_OUTS = [
  "cdof_out",
  "subtree_com_out",
  "geom_xpos_out",
  "geom_xmat_out",
  "xmat_out",
  "xpos_out",
  "cvel_out",
  "xipos_out",
  "cinert_out",
  "pair_conct_out",
]
_SOLVE_OUTS = ["qpos_out", "qvel_out", "warm_out", "qacc_out", "niter_out"]

# packed-buffer layout: floats then ints, in this order
_FLOAT_TABLES = [
  "body_pos",
  "body_quat",
  "body_ipos",
  "body_iquat",
  "body_mass",
  "body_subtreemass",
  "body_inertia",
  "jnt_axis",
  "jnt_pos",
  "qpos0",
  "geom_pos",
  "geom_quat",
  "act_gear",
  "act_gainprm",
  "act_biasprm",
  "act_ctrlrange",
  "act_forcerange",
  "jnt_stiffness",
  "qpos_spring",
  "gravity",
  "jnt_range",
  "jnt_margin",
  "jnt_solref",
  "jnt_solimp",
  "dof_invweight0",
  "fr_D",
  "fr_b",
  "fr_fl",
  "geom_rbound",
  "geom_mg",
  "geom_size",
  "geom_aabb",
  "pair_margin",
  "pair_invw",
  "pair_solref",
  "pair_solimp",
  "pair_friction",
  "dof_armature",
  "dof_damping",
  "opt_buf",
  "mesh_vert",
]
_INT_TABLES = [
  "body_parentid",
  "body_jntadr",
  "body_jntnum",
  "body_dofadr",
  "body_dofnum",
  "jnt_type",
  "jnt_qposadr",
  "jnt_dofadr",
  "jnt_bodyid",
  "body_rootid",
  "dof_bodyid",
  "dof_pad",
  "geom_bodyid_i",
  "act_trnid",
  "act_ctrllimited",
  "act_forcelimited",
  "fr_dofs",
  "lim_jnts",
  "pair_g1",
  "pair_g2",
  "pair_op",
  "pair_body1",
  "pair_body2",
  "pair_ndim",
  "dof_anc",
  "flag_buf",
  "mesh_graph",
  "pair_vadr",
  "pair_gadr",
  "pair_vertnum",
  "pair_usegraph",
]


def _pack_tables(sources: dict, names: list[str], dtype) -> tuple[np.ndarray, dict[str, int]]:
  """Flattens the named source arrays into one buffer; returns (packed, offsets)."""
  offs, flat, cur = {}, [], 0
  for name in names:
    arr = np.ascontiguousarray(np.asarray(sources[name]), dtype=dtype).reshape(-1)
    offs[name] = cur
    cur += arr.size
    flat.append(arr)
  return np.concatenate(flat), offs


def mx_int_arr(x):
  """Device int32 buffer from a numpy array (dtype clamped)."""
  return mx.array(np.ascontiguousarray(np.asarray(x), dtype=np.int32), dtype=mx.int32)


def mx_f32_arr(x):
  """Device float32 buffer from a numpy array (dtype clamped)."""
  return mx.array(np.ascontiguousarray(np.asarray(x), dtype=np.float32))


def _kbimp_np(solref, solimp, pos_imp: float, timestep: float) -> np.ndarray:
  """Python mirror of fused.g_kbimp (float32)."""
  s = np.float32
  timeconst = np.float32(max(float(s(solref[0])), 2.0 * timestep))
  dampratio = s(solref[1])
  dmin = s(min(max(float(solimp[0]), 0.0001), 0.9999))
  dmax = s(min(max(float(solimp[1]), 0.0001), 0.9999))
  width_raw = s(solimp[2])
  width = s(max(1e-15, float(width_raw)))
  mid = s(min(max(float(solimp[3]), 0.0001), 0.9999))
  power = s(max(1.0, float(solimp[4])))
  dmax_sq = s(dmax * dmax)
  if s(solref[0]) <= 0.0:
    k = s(-solref[0] / max(1e-15, float(dmax_sq)))
  else:
    k = s(1.0 / max(1e-15, float(dmax_sq * timeconst * timeconst * dampratio * dampratio)))
  if s(solref[1]) <= 0.0:
    b = s(-solref[1] / max(1e-15, float(dmax)))
  else:
    b = s(2.0 / max(1e-15, float(dmax * timeconst)))
  imp_x = s(abs(pos_imp) / width)
  if dmin == dmax or width_raw <= s(1e-15):
    imp = s(0.5 * (dmin + dmax))
  elif imp_x <= 0.0:
    imp = dmin
  elif imp_x >= 1.0:
    imp = dmax
  elif power == 1.0:
    imp = s(dmin + imp_x * (dmax - dmin))
  elif imp_x <= mid:
    imp_v = s((1.0 / float(mid) ** (power - 1.0)) * float(imp_x) ** float(power))
    imp = s(min(max(dmin + imp_v * (dmax - dmin), dmin), dmax))
  else:
    imp_v = s(1.0 - (1.0 / float(1.0 - mid) ** (power - 1.0)) * float(1.0 - imp_x) ** float(power))
    imp = s(min(max(dmin + imp_v * (dmax - dmin), dmin), dmax))
  return np.array([k, b, imp], np.float32)


class BatchedEngine:
  """Batched MLX engine over the fused ``dyn`` and ``solve`` Metal kernels."""

  def __init__(
    self,
    mjm,
    nworld: int = 4096,
    contact_row_cap: int = 96,
    iterations: int | None = None,
    warmstart: bool = True,
    coop: bool = True,
    threadgroup: int = 32,
    coop_max_worlds: int = 2048,
  ):
    """Builds kernels and static buffers for `nworld` worlds of `mjm`.

    `coop` runs the solve kernel with one threadgroup (`threadgroup` lanes)
    cooperating on each world; it wins on latency for small batches and is
    disabled above `coop_max_worlds` (or when threadgroup memory would exceed
    the device budget), where the one-thread-per-world kernel has better
    throughput.
    """
    if threadgroup < 1 or (threadgroup & (threadgroup - 1)) != 0:
      raise ValueError(f"threadgroup must be a power of two, got {threadgroup}")
    rig = Rig(mjm, contact_row_cap)
    self.rig = rig
    self.nworld = nworld
    self.nv = rig.nv
    self.nq = rig.nq
    self.nbody = rig.nbody
    self.ngeom = rig.ngeom
    self.rowcap = rig.rowcap

    self.iterations = int(iterations if iterations is not None else rig.iterations)
    self.warmstart = bool(warmstart and rig.warmstart)

    # ---- dof ancestor chains for the mass matrix
    nv = rig.nv
    chains = []
    for i in range(nv):
      c = [i]
      j = int(rig.dof_parentid[i])
      while j >= 0:
        c.append(j)
        j = int(rig.dof_parentid[j])
      chains.append(c)
    maxlen = max(len(c) for c in chains)
    pad = np.full((nv, maxlen), -1, dtype=np.int32)
    for i, c in enumerate(chains):
      pad[i, : len(c)] = np.asarray(c, np.int32)
    self.dof_pad = np.ascontiguousarray(pad, np.int32)

    # ---- static friction rows (D and b are world-independent)
    fr_dofs = np.asarray(rig.friction_dofs, np.int32)
    self.fr_dofs = np.ascontiguousarray(fr_dofs)
    fr_D, fr_b, fr_fl = [], [], []
    for dof in fr_dofs:
      kb = _kbimp_np(rig.dof_solref[dof], rig.dof_solimp[dof], 0.0, rig.timestep)
      imp = np.float32(kb[2])
      iw = np.float32(rig.dof_invweight0[dof])
      fr_D.append(np.float32(1.0 / max(iw * np.float32(1.0 - imp) / imp, 1e-15)))
      fr_b.append(kb[1])
      fr_fl.append(np.float32(rig.dof_frictionloss[dof]))
    self.fr_D = np.ascontiguousarray(fr_D, np.float32)
    self.fr_b = np.ascontiguousarray(fr_b, np.float32)
    self.fr_fl = np.ascontiguousarray(fr_fl, np.float32)

    self._t_dyn = dict(
      NBODY=rig.nbody,
      NJNT=rig.njnt,
      NQ=rig.nq,
      NV=rig.nv,
      NGEOM=rig.ngeom,
      NCHAIN=maxlen,
      NU=rig.nu,
      NP=rig.npairc,
      RC=rig.rowcap,
      NFR=len(fr_dofs),
      NLIM=rig.nlim,
      BF=rig.broadphase_filter,
    )
    self._t_solve = dict(
      NJNT=rig.njnt,
      NQ=rig.nq,
      NV=rig.nv,
      RC=rig.rowcap,
      ITER=max(1, self.iterations),
      LSITER=max(1, rig.ls_iterations),
      WARMSTART=1 if self.warmstart else 0,
      EULERDAMP=1 if rig.eulerdamp else 0,
      TG=threadgroup,
    )

    # ---- packed static tables
    self._f_sources = {
      "body_pos": rig.body_pos,
      "body_quat": rig.body_quat,
      "body_ipos": rig.body_ipos,
      "body_iquat": rig.body_iquat,
      "body_mass": rig.body_mass,
      "body_subtreemass": rig.body_subtreemass,
      "body_inertia": rig.body_inertia,
      "jnt_axis": rig.jnt_axis,
      "jnt_pos": rig.jnt_pos,
      "qpos0": rig.qpos0,
      "geom_pos": rig.geom_pos,
      "geom_quat": rig.geom_quat,
      "act_gear": rig.actuator_gear[:, 0],
      "act_gainprm": np.ascontiguousarray(rig.actuator_gainprm[:, :6]),
      "act_biasprm": np.ascontiguousarray(rig.actuator_biasprm[:, :6]),
      "act_ctrlrange": rig.actuator_ctrlrange,
      "act_forcerange": rig.actuator_forcerange,
      "jnt_stiffness": rig.jnt_stiffness,
      "qpos_spring": rig.qpos_spring,
      "gravity": rig.gravity,
      "jnt_range": rig.jnt_range,
      "jnt_margin": rig.jnt_margin,
      "jnt_solref": rig.jnt_solref,
      "jnt_solimp": rig.jnt_solimp,
      "dof_invweight0": rig.dof_invweight0,
      "fr_D": self.fr_D,
      "fr_b": self.fr_b,
      "fr_fl": self.fr_fl,
      "geom_rbound": rig.geom_rbound,
      "geom_mg": rig.geom_mg,
      "geom_size": rig.geom_size,
      "geom_aabb": rig.geom_aabb,
      "pair_margin": rig.pair_margin,
      "pair_invw": rig.pair_invw,
      "pair_solref": rig.pair_solref,
      "pair_solimp": rig.pair_solimp,
      "pair_friction": rig.pair_friction,
      "dof_armature": rig.dof_armature,
      "dof_damping": rig.dof_damping,
      "opt_buf": np.array([rig.timestep], np.float32),
      "mesh_vert": rig.mesh_vert,
    }
    self._i_sources = {
      "body_parentid": rig.body_parentid,
      "body_jntadr": rig.body_jntadr,
      "body_jntnum": rig.body_jntnum,
      "body_dofadr": rig.body_dofadr,
      "body_dofnum": rig.body_dofnum,
      "jnt_type": rig.jnt_type,
      "jnt_qposadr": rig.jnt_qposadr,
      "jnt_dofadr": rig.jnt_dofadr,
      "jnt_bodyid": rig.jnt_bodyid,
      "body_rootid": rig.body_rootid,
      "dof_bodyid": rig.dof_bodyid,
      "dof_pad": self.dof_pad,
      "geom_bodyid_i": rig.geom_bodyid,
      "act_trnid": rig.actuator_trnid[:, 0],
      "act_ctrllimited": rig.actuator_ctrllimited,
      "act_forcelimited": rig.actuator_forcelimited,
      "fr_dofs": self.fr_dofs,
      "lim_jnts": rig.limited_joints,
      "pair_g1": rig.pair_g1,
      "pair_g2": rig.pair_g2,
      "pair_op": rig.pair_op,
      "pair_body1": rig.pair_body1,
      "pair_body2": rig.pair_body2,
      "pair_ndim": rig.pair_ndim,
      "dof_anc": rig.body_isdofancestor,
      "flag_buf": np.array([rig.broadphase_filter], np.int32),
      "mesh_graph": rig.mesh_graph,
      "pair_vadr": rig.pair_vadr,
      "pair_gadr": rig.pair_gadr,
      "pair_vertnum": rig.pair_vertnum,
      "pair_usegraph": rig.pair_usegraph,
    }
    mbf_np, self.adr_f = _pack_tables(self._f_sources, _FLOAT_TABLES, np.float32)
    mif_np, self.adr_i = _pack_tables(self._i_sources, _INT_TABLES, np.int32)
    self.mbf = mx_f32_arr(mbf_np)
    self.mif = mx_int_arr(mif_np)
    self.dof_damping_mx = mx_f32_arr(rig.dof_damping)
    self._solve_opt = mx_f32_arr(np.array([rig.timestep, rig.tolerance, rig.meaninertia, rig.ls_tolerance]))

    src, hdr = dyn_source(self._t_dyn, self.adr_f, self.adr_i, kin=False)
    self.dyn_fn = mx.fast.metal_kernel(
      name="mps_dyn",
      input_names=_DYN_INPUTS,
      output_names=_DYN_BASE_OUTS,
      source=src,
      header=hdr,
    )
    src_ik, hdr_ik = dyn_source(self._t_dyn, self.adr_f, self.adr_i, kin=True)
    self.dyn_kin_fn = mx.fast.metal_kernel(
      name="mps_dyn_kin",
      input_names=_DYN_INPUTS,
      output_names=_DYN_BASE_OUTS + _DYN_KIN_OUTS,
      source=src_ik,
      header=hdr_ik,
    )
    src_s, hdr_s = solve_source(
      self._t_solve,
      {k: self.adr_i[k] for k in ("body_parentid", "body_jntadr", "body_jntnum", "jnt_type", "jnt_qposadr", "jnt_dofadr")},
    )
    # threadgroup memory estimate for the cooperative solve (see fused.solve_source_coop)
    tg_bytes = (2 * rig.nv * rig.nv + 10 * rig.nv + rig.nq + 6 * rig.rowcap + 3 * threadgroup) * 4
    self.coop = bool(coop and nworld <= coop_max_worlds and tg_bytes <= 24 * 1024)
    self.threadgroup = int(threadgroup)
    if self.coop:
      src_s, hdr_s = solve_source_coop(
        self._t_solve,
        {k: self.adr_i[k] for k in ("jnt_type", "jnt_qposadr", "jnt_dofadr")},
        tg=self.threadgroup,
      )
    self.solve_fn = mx.fast.metal_kernel(
      name="mps_solve",
      input_names=_SOLVE_INPUTS,
      output_names=_SOLVE_OUTS,
      source=src_s,
      header=hdr_s,
    )

    self._dyn_template = [(k, self._t_dyn[k]) for k in _DYN_TEMPLATE_KEYS]
    self._solve_template = [(k, self._t_solve[k]) for k in _SOLVE_TEMPLATE_KEYS]

    # ---- state (lazy mx arrays; consumed by the next kernel dispatch)
    self.qpos = mx.zeros((nworld, rig.nq), dtype=F)
    self.qvel = mx.zeros((nworld, rig.nv), dtype=F)
    self.ctrl = mx.zeros((nworld, rig.nu), dtype=F)
    self.warm = mx.zeros((nworld, rig.nv), dtype=F)
    self._nw_buf = mx_int_arr(np.array([nworld], np.int32))

  # --------------------------------------------------------------- state io
  def set_state(self, qpos, qvel, ctrl=None):
    """NumPy in: (nworld, nq), (nworld, nv), optional (nworld, nu)."""
    self.qpos = mx_f32_arr(qpos)
    self.qvel = mx_f32_arr(qvel)
    if ctrl is not None:
      self.ctrl = mx_f32_arr(ctrl)
    self.warm = mx.zeros((self.nworld, self.nv), dtype=F)

  def set_ctrl(self, ctrl):
    """NumPy in: (nworld, nu)."""
    self.ctrl = mx_f32_arr(ctrl)

  def get_state(self):
    """NumPy out (force-evaluates the pending graph): (qpos, qvel)."""
    mx.eval(self.qpos, self.qvel)
    return np.array(self.qpos), np.array(self.qvel)

  def step_np(self, ctrl=None):
    """Optional NumPy ctrl in; NumPy (qpos, qvel) out."""
    if ctrl is not None:
      self.set_ctrl(ctrl)
    self.step()
    return self.get_state()

  def _dyn_dispatch(self, kin: bool):
    nw = self.nworld
    fn = self.dyn_kin_fn if kin else self.dyn_fn
    dyn_out = fn(
      inputs=[self.qpos, self.qvel, self.ctrl, self._nw_buf, self.mbf, self.mif],
      template=self._dyn_template,
      grid=(nw, 1, 1),
      threadgroup=(32, 1, 1),
      output_shapes=[
        (nw, self.nv, self.nv),
        (nw, self.nv),
        (nw, self.nv),
        (nw, self.rowcap, self.nv),
        (nw, self.rowcap),
        (nw, self.rowcap),
        (nw, self.rowcap),
        (nw,),
        (nw,),
        (nw,),
      ]
      + (
        [
          (nw, self.nv, 6),
          (nw, self.nbody, 3),
          (nw, self.ngeom, 3),
          (nw, self.ngeom, 9),
          (nw, self.nbody, 9),
          (nw, self.nbody, 3),
          (nw, self.nbody, 6),
          (nw, self.nbody, 3),
          (nw, self.nbody, 10),
          (nw, self.rig.npairc),
        ]
        if kin
        else []
      ),
      output_dtypes=[F, F, F, F, F, F, F, mx.int32, mx.int32, mx.int32]
      + ([F] * 9 + [mx.int32] if kin else []),
    )
    self.M, self.smooth, self.qfrc_bias, self.J, self.D, self.aref, self.fl, self.nefc, self.ncon, self.overflow = (
      dyn_out[:10]
    )
    if kin:
      (
        self.cdof,
        self.subtree_com,
        self.geom_xpos,
        self.geom_xmat,
        self.xmat,
        self.xpos,
        self.cvel,
        self.xipos,
        self.cinert,
        self.pair_conct,
      ) = dyn_out[10:20]

  # --------------------------------------------------------------- step
  def step(self, kin: bool = False):
    """One simulation step; `kin=True` records validation intermediates."""
    nw = self.nworld
    self._dyn_dispatch(kin)
    out = self.solve_fn(
      inputs=[
        self.qpos,
        self.qvel,
        self.warm,
        self.M,
        self.smooth,
        self.J,
        self.D,
        self.aref,
        self.fl,
        self.nefc,
        self._nw_buf,
        self.dof_damping_mx,
        self.mif,
        self._solve_opt,
      ],
      template=self._solve_template,
      grid=(nw * self.threadgroup if self.coop else nw, 1, 1),
      threadgroup=(self.threadgroup if self.coop else 32, 1, 1),
      output_shapes=[
        (nw, self.nq),
        (nw, self.nv),
        (nw, self.nv),
        (nw, self.nv),
        (nw,),
      ],
      output_dtypes=[F, F, F, F, mx.int32],
    )
    self.qpos, self.qvel, self.warm, self.qacc, self.niter = out
