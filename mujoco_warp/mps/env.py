"""Microduck environment on the MLX engine; mirrors microduck_env.MicroduckGaitEnv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mlx.core as mx

import microduck_env as M
from .engine import Engine
from .model import convert


def _np(a):
  return np.array(a)


class MicroduckMlxEnv:
  def __init__(
    self,
    mjcf_path: str | Path,
    *,
    kp_mult: float = M.KP_MULT,
    kv: float = 0.1,
    horizon: int = 400,
    seed: int = 0,
    cpg: bool = True,
    zmp: bool = False,
    forced_swing: float | None = M.FORCED_SWING,
    fixed_command: tuple | None = None,
  ):
    import mujoco

    self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    self.model.opt.timestep = M.PHYSICS_DT
    self.model.actuator_gainprm[:, 0] *= kp_mult
    self.model.actuator_biasprm[:, 1] = -self.model.actuator_gainprm[:, 0]
    self.model.actuator_biasprm[:, 2] = -kv
    self.engine = Engine(convert(self.model))
    self.sim = self.engine.sim

    stand = self.model.key("STAND")
    self.home_qpos = stand.qpos.copy()
    self.home_ctrl = stand.ctrl.copy()

    self.horizon = horizon
    self.action_dim = self.model.nu
    self.cpg = cpg
    self.obs_dim = M.OBS_GRAVITY_DIM + M.OBS_ANG_VEL_DIM + 3 * self.action_dim + M.COMMAND_DIM + (M.OBS_CPG_DIM if self.cpg else 0)
    self.zmp = zmp
    self.forced_swing = forced_swing
    self.fixed_command = np.asarray(fixed_command, dtype=np.float64) if fixed_command is not None else None
    self._random_commands = self.fixed_command is None

    self.rng = np.random.default_rng(seed)
    self._last_action = np.zeros(self.action_dim)
    self._command = np.zeros(M.COMMAND_DIM)
    self._steps = 0
    self._episodes = 0
    self._foot_air = np.zeros(M.NUM_FEET)
    self._cop = np.zeros(2)
    self._phi = 0.0

    self._foot_geoms = np.array(
      [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision") for side in M.FEET]
    )
    self._foot_bodies = np.array([self.model.geom_bodyid[g] for g in self._foot_geoms])
    self._non_foot_bodies = np.array(
      [i for i in range(self.model.nbody) if i not in self._foot_bodies and i != 0]
    )
    self._floor_clip_eps = 0.02

    joint_id = lambda name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) - 1  # noqa: E731
    self.cpg_ctrl = np.array([joint_id(n) for n in ("left_hip_pitch", "left_knee", "right_hip_pitch", "right_knee")])
    self.direct_ctrl = np.setdiff1d(np.arange(self.action_dim), self.cpg_ctrl)
    self.l_hip_pitch = joint_id("left_hip_pitch")
    self.r_hip_pitch = joint_id("right_hip_pitch")
    self.l_knee = joint_id("left_knee")
    self.r_knee = joint_id("right_knee")
    self.l_ankle = joint_id("left_ankle")
    self.r_ankle = joint_id("right_ankle")
    self.l_hip_roll = joint_id("left_hip_roll")
    self.r_hip_roll = joint_id("right_hip_roll")

  # ------------------------------------------------------------- kinematics
  def _velocity(self, local: bool) -> np.ndarray:
    """mj_objectVelocity for TRUNK_BODY."""
    return self._object_velocity(int(M.TRUNK_BODY), local)

  def _object_velocity(self, body: int, local: bool) -> np.ndarray:
    cvel = _np(self.sim.cvel[body])
    xipos = _np(self.sim.xipos[body])
    ximat = _np(self.sim.ximat[body])
    root = int(self.model.body_rootid[body])
    ang = cvel[:3]
    # cvel is com-based at the subtree root com; shift the linear part to the body's CoM
    lin = cvel[3:] + np.cross(ang, xipos - _np(self.sim.subtree_com[root]))
    if local:
      ang = ximat.T @ ang
      lin = ximat.T @ lin
    return np.concatenate([ang, lin])

  def _geom_xpos(self, geom: int) -> np.ndarray:
    body = int(self.model.geom_bodyid[geom])
    rot = _np(self.sim.xmat[body])
    return _np(self.sim.xpos[body]) + rot @ np.asarray(self.model.geom_pos[geom], np.float64)

  def _contacts(self):
    return getattr(self.engine, "contacts_cache", self.engine.collision())

  # ------------------------------------------------------------------- env
  def _safe_obs(self) -> np.ndarray:
    cpg_obs = np.zeros(M.OBS_CPG_DIM) if self.cpg else np.empty(0)
    return np.concatenate(
      [
        np.zeros(3),
        np.zeros(3),
        np.zeros(self.action_dim),
        np.zeros(self.action_dim),
        self._last_action,
        self._command,
        cpg_obs,
      ]
    ).astype(np.float32)

  def _observe(self) -> np.ndarray:
    xmat = _np(self.sim.xmat[int(M.TRUNK_BODY)])
    g = -xmat.reshape(3, 3).T @ M.WORLD_UP
    cpg_obs = np.array([np.sin(self._phi), np.cos(self._phi)]) if self.cpg else np.empty(0)
    qpos = _np(self.sim.qpos)
    qvel = _np(self.sim.qvel)
    return np.concatenate(
      [
        g,
        self._velocity(local=True)[:3],
        qpos[M.FREE_QPOS_DIM:] - self.home_qpos[M.FREE_QPOS_DIM:],
        qvel[M.FREE_QVEL_DIM:],
        self._last_action,
        self._command,
        cpg_obs,
      ]
    ).astype(np.float32)

  def reset(self, seed: int | None = None) -> np.ndarray:
    if seed is not None:
      self.rng = np.random.default_rng(seed)
    qpos = self.home_qpos.copy()
    qpos[M.SPATIAL_DIM - 1] += self.rng.uniform(*M.RESET_Z_RANGE)
    qpos[M.FREE_QPOS_DIM:] += self.rng.uniform(-M.RESET_JOINT_NOISE, M.RESET_JOINT_NOISE, size=self.action_dim)
    qvel = self.rng.uniform(-M.RESET_VEL_NOISE, M.RESET_VEL_NOISE, size=self.model.nv)
    self.sim.set_state(qpos, qvel, self.home_ctrl)
    self._last_action[:] = 0.0
    self._steps = 0
    self._foot_air[:] = 0
    self._cop[:] = 0
    self._episodes += 1
    self._phi = 0.0
    if self._random_commands and self.rng.uniform() < 0.25:
      self._command[:] = (0.0, 0.0)
    else:
      self._command[:] = (
        (self.rng.uniform(*M.LIN_VEL_RANGE), self.rng.uniform(*M.ANG_VEL_RANGE))
        if self._random_commands
        else (self.fixed_command if self.fixed_command is not None else (M.LIN_VEL_RANGE[1] * 0.5, 0.0))
      )
    self.sim.kinematics()
    self.sim.com_pos()
    self.sim.crb_compute()
    self.sim.com_vel()
    return self._observe()

  def step(self, action: np.ndarray):
    action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
    action_eff = np.clip(action, *M.POLICY_RANGE)
    action_rate = float(np.mean((action_eff - np.clip(self._last_action, *M.POLICY_RANGE)) ** 2))

    if self.cpg:
      t = self._steps * (M.PHYSICS_DT * M.N_SUBSTEPS)
      cycles = t * M.CPG_FREQ
      ramp = min(1.0, cycles / M.CPG_RAMP_CYCLES)
      cmd_speed = float(np.linalg.norm(self._command[:1]))
      speed_scale = np.clip(cmd_speed / M.COMMAND_NOMINAL_SPEED, 0.0, 1.5)
      base_swing = self.forced_swing if self.forced_swing is not None else (M.CPG_SWING_MAX * speed_scale)
      if getattr(self, "swing_blend", 1.0) < 1.0:
        base_swing = (1 - self.swing_blend) * M.default(self.forced_swing, M.CPG_SWING_MAX) + self.swing_blend * base_swing
      swing_amp = base_swing * ramp
      knee_amp = swing_amp * M.CPG_KNEE_SCALE
      ankle_amp = swing_amp * M.CPG_ANKLE_SCALE
      roll_amp = M.CPG_ROLL_AMP * speed_scale * ramp
      turn_bias = float(self._command[1]) * M.CPG_TURN_GAIN
      phi_l = self._phi
      phi_r = self._phi + np.pi
      fwd_l = swing_amp * np.sin(phi_l)
      fwd_r = swing_amp * np.sin(phi_r)
      knee_l = knee_amp * max(0.0, float(np.sin(phi_l)))
      knee_r = knee_amp * max(0.0, float(np.sin(phi_r)))
      ankle_l = ankle_amp * np.sin(phi_l)
      ankle_r = ankle_amp * np.sin(phi_r)
      roll = roll_amp * np.cos(phi_l)
      cpg_target = np.zeros(self.action_dim)
      cpg_target[self.l_hip_pitch] -= fwd_l - turn_bias
      cpg_target[self.r_hip_pitch] += fwd_r + turn_bias
      cpg_target[self.l_knee] += knee_l
      cpg_target[self.r_knee] -= knee_r
      cpg_target[self.l_ankle] += ankle_l
      cpg_target[self.r_ankle] -= ankle_r
      cpg_target[self.l_hip_roll] -= roll
      cpg_target[self.r_hip_roll] -= roll
      ctrl = self.home_ctrl + cpg_target + M.ACTION_SCALE * action_eff
      self._phi = (self._phi + 2 * np.pi * M.CPG_FREQ * (M.PHYSICS_DT * M.N_SUBSTEPS)) % (2 * np.pi)
    else:
      ctrl = self.home_ctrl + M.ACTION_SCALE * action_eff

    self.last_ctrl = ctrl.copy()
    self.sim.ctrl = mx.array(ctrl.astype(np.float32))
    for _ in range(M.N_SUBSTEPS):
      self.engine.step()
    mx.eval(self.sim.qpos, self.sim.qvel)

    self._steps += 1
    self._last_action[:] = action

    qpos = _np(self.sim.qpos)
    qvel = _np(self.sim.qvel)
    if not (np.isfinite(qpos).all() and np.isfinite(qvel).all()):
      return self._safe_obs(), M.NAN_PENALTY, True, False, {}

    w, x, y, z = qpos[M.QUAT_OFFSET : M.QUAT_OFFSET + M.SPATIAL_DIM + 1]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    v_world = self._velocity(local=False)
    vx = np.cos(yaw) * v_world[M.SPATIAL_DIM] + np.sin(yaw) * v_world[M.SPATIAL_DIM + 1]
    wz = self._velocity(local=True)[M.SPATIAL_DIM - 1]

    uprightness = _np(self.sim.xmat[int(M.TRUNK_BODY)]).reshape(9)[M.SPATIAL_DIM * M.SPATIAL_DIM - 1]
    terminated = bool(uprightness < M.MIN_UPRIGHTNESS or qpos[M.SPATIAL_DIM - 1] < M.MIN_TRUNK_Z)
    truncated = self._steps >= self.horizon

    body_z = _np(self.sim.xpos)[self._non_foot_bodies, M.SPATIAL_DIM - 1]
    min_body_z = float(body_z.min())
    floor_clip_depth = max(0.0, -min_body_z)
    if floor_clip_depth > self._floor_clip_eps:
      terminated = True

    reward = M.ALIVE_BONUS
    if floor_clip_depth > 0.0:
      reward += M.FLOOR_CLIP_PENALTY * floor_clip_depth * 100.0
    r_track_lin = M.TRACK_WEIGHT * np.exp(-((vx - self._command[0]) ** 2) / M.TRACK_STD[0])
    r_track_ang = M.TRACK_WEIGHT * np.exp(-((wz - self._command[1]) ** 2) / M.TRACK_STD[1])
    r_upright = M.UPRIGHT_WEIGHT * np.exp(-((1.0 - uprightness) ** 2) / M.UPRIGHT_STD)
    pose_scale = (
      M.POSE_STD_WALKING_SCALE
      if (abs(self._command[0]) > M.COMMAND_EPS or abs(self._command[1]) > M.COMMAND_EPS)
      else M.POSE_STD_STANDING_SCALE
    )
    r_pose = M.POSE_WEIGHT * np.exp(
      -(((qpos[M.FREE_QPOS_DIM:] - self.home_qpos[M.FREE_QPOS_DIM:])[M.LEG_MASK] / (M.POSE_STDS[M.LEG_MASK] * pose_scale)) ** 2).mean()
    )
    reward += r_track_lin + r_track_ang + r_upright + r_pose + M.ACTION_RATE_WEIGHT * action_rate
    if abs(self._command[0]) > M.COMMAND_EPS:
      reward += M.PROGRESS_WEIGHT * vx * np.sign(self._command[0])

    is_standing_gait = bool((uprightness >= 0.85) and (qpos[M.SPATIAL_DIM - 1] >= 0.095) and (floor_clip_depth == 0.0))
    contacts = self._contacts()
    if (abs(self._command[0]) > M.COMMAND_EPS or abs(self._command[1]) > M.COMMAND_EPS) and is_standing_gait:
      for i, g in enumerate(self._foot_geoms):
        if any(foot == g for _, _, foot in contacts):
          self._foot_air[i] = 0
        else:
          self._foot_air[i] += 1
      air_time = self._foot_air * (M.PHYSICS_DT * M.N_SUBSTEPS)
      ramp_ep = min(1.0, self._episodes / M.AIR_TIME_RAMP_EPISODES)
      reward += M.AIR_TIME_WEIGHT * ramp_ep * ((air_time > M.AIR_TIME_MIN) & (air_time < M.AIR_TIME_MAX)).sum()
    else:
      self._foot_air[:] = 0

    foot_z = np.array([self._geom_xpos(g)[2] for g in self._foot_geoms])
    if is_standing_gait:
      reward += M.FOOT_HEIGHT_WEIGHT * np.clip((foot_z - M.FOOT_HEIGHT_CLEAR) / M.FOOT_HEIGHT_CAP, 0.0, 1.0).sum()

    ang_vel = self._velocity(local=True)[:3]
    reward += M.BODY_ANG_VEL_WEIGHT * float(np.sum(ang_vel**2))

    i_comp = _np(self.sim.cinert[int(M.TRUNK_BODY)])[:6]
    inertia = np.array(
      [
        [i_comp[0], i_comp[3], i_comp[4]],
        [i_comp[3], i_comp[1], i_comp[5]],
        [i_comp[4], i_comp[5], i_comp[2]],
      ]
    )
    reward += M.ANGULAR_MOMENTUM_WEIGHT * float(np.sum((inertia @ ang_vel) ** 2))

    self_hits = sum(1 for dist, pos, foot in contacts if foot not in (0,))
    # only plane-foot pairs exist in the MLX engine (self-collisions are separate pairs)
    reward += M.SELF_COLLISION_WEIGHT * float(False)

    if abs(self._command[0]) > M.COMMAND_EPS or abs(self._command[1]) > M.COMMAND_EPS:
      slip = 0.0
      for body in self._foot_bodies:
        v = self._object_velocity(int(body), local=False)
        slip += v[3] ** 2 + v[4] ** 2
      reward += M.FOOT_SLIP_WEIGHT * slip + M.FOOT_CLEARANCE_WEIGHT * np.clip(
        (M.FOOT_CLEARANCE_TARGET - foot_z) / M.FOOT_CLEARANCE_TARGET, 0.0, 1.0
      ).mean()

    return self._observe(), float(reward), terminated, truncated, {}

  def close(self):
    pass
