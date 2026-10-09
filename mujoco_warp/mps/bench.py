"""Benchmark the MLX engine against the mujoco_warp CPU backend.

Usage:
  uv run python -m mujoco_warp.mps.bench --scene humanoid --worlds 4096 --steps 50
  uv run python -m mujoco_warp.mps.bench --scene benchmarks/humanoid/humanoid.xml
"""

from __future__ import annotations

import argparse
import time

import mujoco
import numpy as np

SCENES = {
  "humanoid": "mujoco_warp/test_data/humanoid/humanoid.xml",
}


def _load(scene: str):
  return mujoco.MjModel.from_xml_path(SCENES.get(scene, scene))


def bench_mlx(mjm, nworld: int, steps: int, warm: int = 5, eval_every: int = 1) -> float:
  """Returns seconds per substep on MLX GPU."""
  import mlx.core as mx

  from mujoco_warp.mps.robot import BatchedEngine

  eng = BatchedEngine(mjm, nworld=nworld)
  qpos = np.tile(np.array(mjm.qpos0, np.float32), (nworld, 1))
  qvel = np.zeros((nworld, mjm.nv), np.float32)
  ctrl = np.zeros((nworld, mjm.nu), np.float32)
  eng.set_state(qpos, qvel, ctrl)
  for _ in range(warm):
    eng.step()
  mx.eval(eng.qpos, eng.qvel)

  # Reset to initial state so timed steps start from the identical point as CPU
  eng.set_state(qpos, qvel, ctrl)
  mx.eval(eng.qpos, eng.qvel)

  t0 = time.perf_counter()
  if eval_every > 0:
    for i in range(steps):
      eng.step()
      if (i + 1) % eval_every == 0:
        mx.eval(eng.qpos, eng.qvel)
    mx.eval(eng.qpos, eng.qvel)
  else:
    for _ in range(steps):
      eng.step()
    mx.eval(eng.qpos, eng.qvel)
  t1 = time.perf_counter()
  return (t1 - t0) / steps


def bench_warp_cpu(mjm, nworld: int, steps: int, warm: int = 5) -> float:
  """Returns seconds per substep on mujoco_warp CPU."""
  import mujoco_warp

  m = mujoco_warp.put_model(mjm)
  d = mujoco_warp.make_data(mjm, nworld=nworld)
  d.ctrl.numpy()[:] = 0.0
  for _ in range(warm):
    mujoco_warp.step(m, d)

  # Reset to initial state so timed steps start from the identical point as MLX
  d.qpos.numpy()[:] = np.tile(np.array(mjm.qpos0, np.float32), (nworld, 1))
  d.qvel.numpy()[:] = 0.0
  d.ctrl.numpy()[:] = 0.0
  d.qacc_warmstart.numpy()[:] = 0.0

  t0 = time.perf_counter()
  for _ in range(steps):
    mujoco_warp.step(m, d)
  t1 = time.perf_counter()
  return (t1 - t0) / steps


def _worlds(spec: str) -> list[int]:
  return [int(w) for w in spec.split(",") if w]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--scene", default="humanoid", help="scene key or XML path")
  ap.add_argument("--worlds", default="64,256,1024,4096,16384", help="comma-separated MLX world counts")
  ap.add_argument("--steps", type=int, default=50)
  ap.add_argument("--cpu-worlds", default="64,256,1024,4096", help="comma-separated warp-CPU world counts")
  ap.add_argument("--cpu-steps", type=int, default=20)
  ap.add_argument("--eval-every", type=int, default=1, help="force MLX eval every N steps (default 1 for step latency)")
  args = ap.parse_args()

  mjm = _load(args.scene)
  print(f"scene {args.scene}: nq={mjm.nq} nv={mjm.nv} nu={mjm.nu} ngeom={mjm.ngeom}")

  mlx_results: dict[int, float] = {}
  for nw in _worlds(args.worlds):
    dt = bench_mlx(mjm, nw, args.steps, eval_every=args.eval_every)
    mlx_results[nw] = dt
    print(f"MLX      nworld={nw:6d}: {dt * 1e3:9.2f} ms/substep  {dt / nw * 1e6:8.2f} us/world  {nw / dt:12,.0f} worlds/s")

  cpu_results: dict[int, float] = {}
  for nw in _worlds(args.cpu_worlds):
    dt = bench_warp_cpu(mjm, nw, args.cpu_steps)
    cpu_results[nw] = dt
    print(f"warp-CPU nworld={nw:6d}: {dt * 1e3:9.2f} ms/substep  {dt / nw * 1e6:8.2f} us/world  {nw / dt:12,.0f} worlds/s")

  common_worlds = [w for w in mlx_results if w in cpu_results]
  if common_worlds:
    print("\nSpeedup Summary:")
    print(" worlds | MLX ms/substep | warp-CPU ms/substep | speedup")
    print("--------+----------------+---------------------+--------")
    for nw in common_worlds:
      m_ms = mlx_results[nw] * 1e3
      c_ms = cpu_results[nw] * 1e3
      sp = c_ms / m_ms
      print(f"{nw:7d} | {m_ms:14.2f} | {c_ms:19.2f} | {sp:7.1f}x")


if __name__ == "__main__":
  main()
