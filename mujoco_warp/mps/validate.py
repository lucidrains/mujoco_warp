"""Numeric validation of the MLX engine against the mujoco_warp CPU backend.

Checks, on the humanoid scene:
  1. one-step dynamics elements (M, bias, cdof, contact D/aref) at a random state
  2. multi-step trajectory agreement under random controls

Usage:
  uv run python -m mujoco_warp.mps.validate
"""

from __future__ import annotations

import mujoco
import numpy as np

import mujoco_warp
from mujoco_warp.mps.robot import BatchedEngine

SCENE = "mujoco_warp/test_data/humanoid/humanoid.xml"


def compare_step(mjm, with_contacts: bool = True, seed: int = 0) -> dict:
  rng = np.random.default_rng(seed)
  qpos = np.tile(np.array(mjm.qpos0, np.float32), (1, 1))
  if not with_contacts:
    qpos[:, :3] += rng.normal(0, 0.003, (1, 3)).astype(np.float32)
  qvel = rng.normal(0, 0.02, (1, mjm.nv)).astype(np.float32)
  ctrl = rng.uniform(-0.5, 0.5, (1, mjm.nu)).astype(np.float32)

  m = mujoco_warp.put_model(mjm)
  d = mujoco_warp.make_data(mjm, nworld=1)
  d.qpos.numpy()[:] = qpos
  d.qvel.numpy()[:] = qvel
  d.ctrl.numpy()[:] = ctrl
  mujoco_warp.forward(m, d)

  eng = BatchedEngine(mjm, nworld=1)
  eng.set_state(qpos, qvel, ctrl)
  eng.step()

  out = {}
  qacc_w = np.array(d.qacc.numpy())[0]
  qacc_m = np.array(eng.qacc)[0]
  out["qacc"] = float(np.abs(qacc_m - qacc_w).max())
  out["smooth"] = float(np.abs(np.array(eng.smooth)[0] - np.array(d.qfrc_smooth.numpy())[0]).max())
  out["bias"] = float(np.abs(np.array(eng.qfrc_bias)[0] - np.array(d.qfrc_bias.numpy())[0]).max())
  out["nefc"] = (int(np.array(eng.nefc)[0]), int(np.array(d.nefc.numpy())[0]))
  return out


def compare_trajectory(mjm, nworld: int = 2, steps: int = 100, seed: int = 0) -> dict:
  rng = np.random.default_rng(seed)
  qpos = np.tile(np.array(mjm.qpos0, np.float32), (nworld, 1))
  qpos[:, :3] += rng.normal(0, 0.005, (nworld, 3)).astype(np.float32)
  qvel = np.zeros((nworld, mjm.nv), np.float32)
  ctrl = rng.uniform(-0.3, 0.3, (nworld, mjm.nu)).astype(np.float32)

  m = mujoco_warp.put_model(mjm)
  d = mujoco_warp.make_data(mjm, nworld=nworld)
  d.qpos.numpy()[:] = qpos
  d.qvel.numpy()[:] = qvel
  d.ctrl.numpy()[:] = ctrl

  eng = BatchedEngine(mjm, nworld=nworld)
  eng.set_state(qpos, qvel, ctrl)

  errs = {}
  for i in range(1, steps + 1):
    mujoco_warp.step(m, d)
    eng.step()
    if i in (1, 10, 25, 50, 100) and i <= steps:
      qm, vm = eng.get_state()
      qw = np.array(d.qpos.numpy())
      vw = np.array(d.qvel.numpy())
      errs[i] = (float(np.abs(np.array(qm) - qw).max()), float(np.abs(np.array(vm) - vw).max()))
  return errs


def main():
  mjm = mujoco.MjModel.from_xml_path(SCENE)
  print(f"scene: {SCENE}  nq={mjm.nq} nv={mjm.nv} nu={mjm.nu}")

  rc = compare_step(mjm, with_contacts=True)
  print("\none-step elements with contacts (max abs err vs warp-CPU):")
  print(f"  qacc   {rc['qacc']:.3e}")
  print(f"  smooth {rc['smooth']:.3e}")
  print(f"  bias   {rc['bias']:.3e}")
  print(f"  nefc   mine={rc['nefc'][0]} warp={rc['nefc'][1]}")

  rf = compare_step(mjm, with_contacts=False)
  print("\none-step elements in flight (max abs err vs warp-CPU):")
  print(f"  qacc   {rf['qacc']:.3e}")
  print(f"  smooth {rf['smooth']:.3e}")
  print(f"  bias   {rf['bias']:.3e}")
  print(f"  nefc   mine={rf['nefc'][0]} warp={rf['nefc'][1]}")

  print("\ntrajectory (2 worlds, random ctrl, max abs err vs warp-CPU):")
  errs = compare_trajectory(mjm, nworld=2, steps=100)
  for k, (eq, ev) in errs.items():
    print(f"  step {k:4d}: qpos {eq:.3e}  qvel {ev:.3e}")


if __name__ == "__main__":
  main()
