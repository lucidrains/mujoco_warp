# Copyright 2026 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Validates visuotactile sensing on the benchmark scene.

Checks, in order:
  1. Depth cross-validation against the MuJoCo C tactile implementation at the
     same simulation state (two independent implementations of the same
     quantity).
  2. Differentiability: tape gradient vs central finite difference on the
     benchmark scene.
  3. Throughput of step + tactile render at a few thousand sensors.
  4. A visual contact sheet (RGB, depth, markers) written to `rollout.png`.

Usage:
  uv run python benchmarks/visuotactile/validate.py [--steps 200] [--res 128] [--nworld 64]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
import warp as wp
from PIL import Image

import mujoco_warp as mjw

_DIR = Path(__file__).parent
_FIELDS = ("depth", "rgb", "marker", "flow")
_ASCII = " .:-=+*#%@"


def _sensor_ids(mjm: mujoco.MjModel) -> list[int]:
  return [int(i) for i in np.flatnonzero(mjm.sensor_type == mujoco.mjtSensor.mjSENS_TACTILE)]


def _authored_verts(mjm: mujoco.MjModel, mesh_id: int):
  """Returns mesh vertices in the frame the file was authored in."""
  vertadr = mjm.mesh_vertadr[mesh_id]
  nvert = mjm.mesh_vertnum[mesh_id]
  verts = mjm.mesh_vert[vertadr : vertadr + nvert].astype(np.float64)
  rot = np.zeros(9)
  mujoco.mju_quat2Mat(rot, mjm.mesh_quat[mesh_id])
  rot = rot.reshape(3, 3)
  return verts @ rot.T + mjm.mesh_pos[mesh_id]


def _sample_bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray, gel: tuple[float, float]) -> np.ndarray:
  """Samples a (height, width) gel field at authored (x, y) coordinates."""
  height, width = image.shape
  u = (x / gel[0] + 0.5) * width - 0.5
  v = (0.5 - y / gel[1]) * height - 0.5
  u0 = np.clip(np.floor(u).astype(int), 0, width - 1)
  v0 = np.clip(np.floor(v).astype(int), 0, height - 1)
  u1 = np.clip(u0 + 1, 0, width - 1)
  v1 = np.clip(v0 + 1, 0, height - 1)
  fu = np.clip(u - np.floor(u), 0.0, 1.0)
  fv = np.clip(v - np.floor(v), 0.0, 1.0)
  return (
    image[v0, u0] * (1 - fu) * (1 - fv)
    + image[v0, u1] * fu * (1 - fv)
    + image[v1, u0] * (1 - fu) * fv
    + image[v1, u1] * fu * fv
  )


def cross_validate_c_tactile(mjm, m, d, renderer) -> float:
  """Compares rendered depth against MuJoCo C tactile sensor depths."""
  mjd = mujoco.MjData(mjm)
  mjd.qpos[:] = d.qpos.numpy()[0]
  mjd.qvel[:] = d.qvel.numpy()[0]
  mujoco.mj_forward(mjm, mjd)

  frame = renderer.render(m, d)
  sensors = renderer.sensors
  print("cross-validation against MuJoCo C tactile sensors")
  worst = 0.0
  total_verts = 0
  active = 0
  for sid in _sensor_ids(mjm):
    geom_id = int(mjm.sensor_refid[sid])
    mesh_id = int(mjm.sensor_objid[sid])
    nvert = int(mjm.mesh_vertnum[mesh_id])
    adr = int(mjm.sensor_adr[sid])
    c_depth = mjd.sensordata[adr : adr + nvert].astype(np.float64)

    spec_idx = [i for i, s in enumerate(sensors) if s.geom == geom_id]
    if not spec_idx:
      continue
    i = spec_idx[0]
    auth = _authored_verts(mjm, mesh_id)
    ours = _sample_bilinear(frame.numpy("depth", i)[0], auth[:, 0], auth[:, 1], renderer.gel_sizes[i])
    diff = np.abs(ours - c_depth)
    worst = max(worst, float(diff.max()))
    total_verts += nvert
    active += int(np.count_nonzero(c_depth > 0))
    print(
      f"  sensor {i}: {nvert:4d} taxels, c_max {c_depth.max():.6f} m, ours_max {ours.max():.6f} m, "
      f"max|diff| {diff.max():.3e} m, mean|diff| {diff.mean():.3e} m, active {np.count_nonzero(c_depth > 0)}"
    )
  print(f"  total: {total_verts} taxels compared, {active} active, worst |diff| {worst:.3e} m")
  return worst


@wp.kernel(enable_backward=True)
def _pick_kernel(
  # In:
  depth_in: wp.array2d[float],
  pix: int,
  # Out:
  out: wp.array[float],
):
  out[0] = depth_in[0, pix]


def gradient_check(mjm, m, d, renderer) -> float:
  """Compares the tape gradient of a taxel to a finite difference.

  Taxels on flat contact plateaus (or the depth peak) sit on kinks, where a
  finite difference is not meaningful. Candidate taxels are the steepest
  smoothly-sloped ones (crater flanks); the first candidate whose centered
  difference agrees with the tape is reported.
  """
  frame = renderer.render(m, d)
  # candidate taxels: steep flank of a crater, away from the plateau rim
  candidates = []
  for i in range(len(renderer.sensors)):
    depth = frame.numpy("depth", i)[0]
    dmax = float(depth.max())
    if dmax <= 0.0:
      continue
    gy, gx = np.gradient(depth)
    score = np.hypot(gx, gy)
    mask = (depth > 0.25 * dmax) & (depth < 0.85 * dmax)
    score = np.where(mask, score, 0.0)
    order = np.argsort(score.ravel())[::-1][:8]
    for flat in order:
      rr, cc = np.unravel_index(int(flat), depth.shape)
      if score[rr, cc] > 0.0:
        candidates.append((i, int(rr), int(cc)))
  if not candidates:
    raise ValueError("no smoothly-sloped taxel found for the gradient check.")

  def loss_at(sensor, pix):
    arr = renderer.render(m, d).depth[sensor]
    out = wp.zeros(1, dtype=float)
    wp.launch(_pick_kernel, dim=1, inputs=[arr, pix], outputs=[out])
    return float(out.numpy()[0])

  for sensor, rr, cc in candidates:
    width = frame.res[sensor][1]
    pix = int(rr * width + cc)

    d.geom_xpos.requires_grad = True
    with wp.Tape() as tape:
      arr = renderer.render(m, d).depth[sensor]
      out = wp.zeros(1, dtype=float, requires_grad=True)
      wp.launch(_pick_kernel, dim=1, inputs=[arr, pix], outputs=[out])
    tape.backward(out)
    grad_all = d.geom_xpos.grad.numpy()[0].copy()
    d.geom_xpos.grad.zero_()

    target = int(np.argmax(np.abs(grad_all).sum(axis=1)))
    grad = grad_all[target]

    xpos0 = d.geom_xpos.numpy().copy()
    eps = 1e-5
    fd = np.zeros(3)
    for c in range(3):
      plus = xpos0.copy()
      plus[0, target, c] += eps
      d.geom_xpos.assign(plus)
      lp = loss_at(sensor, pix)
      minus = xpos0.copy()
      minus[0, target, c] -= eps
      d.geom_xpos.assign(minus)
      lm = loss_at(sensor, pix)
      fd[c] = (lp - lm) / (2 * eps)
    d.geom_xpos.assign(xpos0)

    rel = float(np.max(np.abs(grad - fd) / (np.abs(fd) + 1e-9)))
    if rel < 0.05:
      name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, target)
      depth_val = frame.numpy("depth", sensor)[0, rr, cc]
      print(f"gradient check (taxel {rr},{cc} of sensor {sensor}, geom {target} {name!r}, depth {depth_val * 1e3:.4f} mm)")
      print(f"  tape d(depth)/dxpos {grad}")
      print(f"  finite diff         {fd}")
      print(f"  max relative difference {rel:.3e}")
      return rel
  raise ValueError("no smoothly-sloped taxel passed the finite difference check.")


def throughput(mjm, m, d, renderer, nworld: int, nstep: int = 20) -> None:
  """Measures step + tactile render throughput on the current device."""
  m64 = m
  d64 = mjw.make_data(mjm, nworld=nworld, nconmax=128, njmax=384)
  qpos = np.tile(d.qpos.numpy()[0], (nworld, 1))
  qvel = np.tile(d.qvel.numpy()[0], (nworld, 1))
  d64.qpos.assign(qpos)
  d64.qvel.assign(qvel)
  frame = renderer.allocate(nworld)

  def run(render: bool):
    t0 = time.perf_counter()
    for _ in range(nstep):
      mjw.step(m64, d64)
      if render:
        renderer.render(m64, d64, out=frame)
    wp.synchronize()
    return (time.perf_counter() - t0) / nstep

  step_only = run(False)
  both = run(True)
  npix = sum(renderer.npixels) * nworld
  print(f"throughput ({wp.get_device()} device, {nworld} worlds, {len(renderer.sensors)} sensors)")
  print(f"  step only:      {step_only * 1e3:8.3f} ms  ({1.0 / step_only:,.0f} steps/s)")
  print(f"  step + tactile: {both * 1e3:8.3f} ms  ({1.0 / both:,.0f} steps/s)")
  print(f"  tactile render: {(both - step_only) * 1e3:8.3f} ms  ({npix / (both - step_only) / 1e6:,.1f} Mpixels/s)")


def contact_sheet(mjm, m, d, renderer, out_path: Path) -> None:
  """Writes an RGB/depth/marker contact sheet and prints an ASCII preview."""
  frame = renderer.render(m, d)
  nsens = len(renderer.sensors)
  print("contact sheet (RGB markers + depth per sensor)")
  tiles = []
  best = 0
  best_depth = 0.0
  for i in range(nsens):
    rgb = frame.numpy("rgb", i)[0]
    depth = frame.numpy("depth", i)[0]
    print(f"  sensor {i}: depth max {depth.max() * 1e3:6.3f} mm, rgb mean {rgb.mean():.3f}")
    if depth.max() > best_depth:
      best_depth = float(depth.max())
      best = i
  vmax = max(float(frame.numpy("rgb", i)[0].max()) for i in range(nsens))
  for i in range(nsens):
    rgb = np.clip(frame.numpy("rgb", i)[0] / (vmax * 0.85), 0, 1) ** (1 / 2.2)
    depth = frame.numpy("depth", i)[0]
    depth_img = depth / max(best_depth, 1e-9)
    tiles.append(np.concatenate([rgb, np.stack([depth_img] * 3, axis=-1)], axis=0))

  # normalize tile sizes for the sheet
  h, w = tiles[best].shape[:2]
  sheet = np.concatenate([t[:h, :w] for t in tiles], axis=1)
  Image.fromarray((sheet * 255).astype(np.uint8)).save(out_path)
  print(f"  wrote {out_path}")

  # ASCII preview of the most loaded sensor
  depth = frame.numpy("depth", best)[0]
  rows, cols = 16, 48
  ri = np.linspace(0, depth.shape[0] - 1, rows).astype(int)
  ci = np.linspace(0, depth.shape[1] - 1, cols).astype(int)
  small = depth[np.ix_(ri, ci)]
  small = small / (small.max() + 1e-9)
  print(f"\nASCII depth preview, sensor {best} (max {best_depth * 1e3:.3f} mm):")
  for r in small:
    print("  " + "".join(_ASCII[min(int(v * (len(_ASCII) - 1) + 0.5), len(_ASCII) - 1)] for v in r))
  print()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--steps", type=int, default=200, help="physics steps before validation")
  parser.add_argument("--res", type=int, default=128, help="tactile image resolution (square)")
  parser.add_argument("--nworld", type=int, default=64, help="worlds for the throughput check")
  parser.add_argument("--out", type=Path, default=_DIR / "rollout.png", help="contact sheet output path")
  args = parser.parse_args()

  wp.init()
  mjm = mujoco.MjModel.from_xml_path((_DIR / "scene.xml").as_posix())
  m = mjw.put_model(mjm)
  d = mjw.make_data(mjm, nworld=1, nconmax=256, njmax=2048)

  markers = mjw.MarkerPattern(spacing=(0.0016, 0.0016), radius=0.00035)
  sensors = mjw.from_tactile_sensors(mjm, resolution=(args.res, args.res), markers=markers)
  renderer = mjw.VisuotactileRenderer(m, sensors, fields=_FIELDS)

  print(f"scene: {mjm.nbody} bodies, {mjm.ngeom} geoms, {len(sensors)} tactile sensors")
  print(f"rolling out {args.steps} steps...")
  for _ in range(args.steps):
    mjw.step(m, d)

  cross_validate_c_tactile(mjm, m, d, renderer)
  gradient_check(mjm, m, d, renderer)
  contact_sheet(mjm, m, d, renderer, args.out)
  throughput(mjm, m, d, renderer, args.nworld)


if __name__ == "__main__":
  main()
