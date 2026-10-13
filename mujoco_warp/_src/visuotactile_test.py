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
"""Tests for visuotactile (GelSight-style) sensing."""

from __future__ import annotations

import os
import struct
import tempfile

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest
from absl.testing import parameterized

try:
  import torch
except ImportError:
  torch = None

import mujoco_warp as mjw
from mujoco_warp import test_data

_PAD = "0.01 0.008 0.002"
_PAD_TOP = 0.002
_GEL = (0.02, 0.016)
_RADIUS = 0.004


def _scene(z: float = 0.005, offset: str = "0 0", r: float = _RADIUS) -> str:
  return f"""
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <body name="sensor">
        <geom name="pad" type="box" size="{_PAD}"/>
      </body>
      <body name="indenter" pos="{offset} {z}">
        <freejoint/>
        <geom name="ball" type="sphere" size="{r}"/>
      </body>
    </worldbody>
  </mujoco>
  """


def _grid(res, gel):
  """Returns sample coordinates (x, y) for each pixel, shape (height, width)."""
  height, width = res
  dx = gel[0] / width
  dy = gel[1] / height
  x = (np.arange(width) + 0.5 - 0.5 * width) * dx
  y = (0.5 * height - np.arange(height) - 0.5) * dy
  return np.meshgrid(x, y)


def _icosphere(radius, subdiv=3):
  t = (1.0 + 5.0**0.5) / 2.0
  verts = np.array(
    [
      [-1, t, 0],
      [1, t, 0],
      [-1, -t, 0],
      [1, -t, 0],
      [0, -1, t],
      [0, 1, t],
      [0, -1, -t],
      [0, 1, -t],
      [t, 0, -1],
      [t, 0, 1],
      [-t, 0, -1],
      [-t, 0, 1],
    ],
    dtype=np.float64,
  )
  faces = np.array(
    [
      [0, 11, 5],
      [0, 5, 1],
      [0, 1, 7],
      [0, 7, 10],
      [0, 10, 11],
      [1, 5, 9],
      [5, 11, 4],
      [11, 10, 2],
      [10, 7, 6],
      [7, 1, 8],
      [3, 9, 4],
      [3, 4, 2],
      [3, 2, 6],
      [3, 6, 8],
      [3, 8, 9],
      [4, 9, 5],
      [2, 4, 11],
      [6, 2, 10],
      [8, 6, 7],
      [9, 8, 1],
    ]
  )
  for _ in range(subdiv):
    mid = {}
    verts_list = list(verts)

    def midpoint(a, b):
      key = (min(a, b), max(a, b))
      if key not in mid:
        mid[key] = len(verts_list)
        verts_list.append((verts[a] + verts[b]) / 2.0)
      return mid[key]

    new_faces = []
    for a, b, c in faces:
      ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
      new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    verts = np.array(verts_list)
    faces = np.array(new_faces)
  verts /= np.linalg.norm(verts, axis=1, keepdims=True)
  return verts * radius, faces


def _write_icosphere_stl(path, radius, subdiv=3):
  verts, faces = _icosphere(radius, subdiv)
  with open(path, "wb") as f:
    f.write(b"\0" * 80)
    f.write(struct.pack("<I", len(faces)))
    for tri in faces:
      v0, v1, v2 = verts[tri]
      f.write(struct.pack("<3f", 0.0, 0.0, 0.0))
      for v in (v0, v1, v2):
        f.write(struct.pack("<3f", *v))
      f.write(struct.pack("<H", 0))


def _pad_top(d, mjm, geom_name="pad"):
  geom_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
  return float(d.geom_xpos.numpy()[0, geom_id, 2]) + _PAD_TOP


def _ball_center(d, mjm, geom_name="ball"):
  geom_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
  return d.geom_xpos.numpy()[0, geom_id].astype(np.float64)


def _spec(mjm, **kwargs):
  geom_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "pad")
  kwargs.setdefault("resolution", (24, 32))
  kwargs.setdefault("gel_size", _GEL)
  return mjw.VisuotactileSensor(geom=geom_id, **kwargs)


class VisuotactileTest(parameterized.TestCase):
  def test_depth_field_matches_sphere_analytic(self):
    mjm, _, m, d = test_data.fixture(xml=_scene())
    sensor = _spec(mjm)
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth",))
    mjw.fwd_position(m, d)

    frame = renderer.render(m, d)
    depth = frame.numpy("depth", 0)[0]

    center = _ball_center(d, mjm)
    top = _pad_top(d, mjm)
    height, width = sensor.resolution
    gel = sensor.gel_size
    x, y = _grid((height, width), gel)
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (top - center[2]) ** 2)
    expected = np.maximum(_RADIUS - dist, 0.0)

    np.testing.assert_allclose(depth, expected, atol=1e-5)
    self.assertGreater(depth.max(), 5e-4)
    self.assertLess(depth.max(), 1.05e-3)

  def test_depth_zero_and_flat_normals_without_contact(self):
    mjm, _, m, d = test_data.fixture(xml=_scene(z=0.02))
    sensor = _spec(mjm, optics=mjw.GelOptics(leds=(), ambient=0.5))
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth", "force", "normal", "rgb"))
    mjw.fwd_position(m, d)

    frame = renderer.render(m, d)
    self.assertEqual(np.count_nonzero(frame.numpy("depth", 0)), 0)
    self.assertEqual(np.count_nonzero(frame.numpy("force", 0)), 0)
    normal = frame.numpy("normal", 0)
    np.testing.assert_allclose(normal, np.broadcast_to(np.array([0.0, 0.0, 1.0]), normal.shape), atol=1e-6)
    rgb = frame.numpy("rgb", 0)
    np.testing.assert_allclose(rgb, np.broadcast_to(np.array([0.82, 0.82, 0.80]) * 0.5, rgb.shape), atol=1e-6)

  def test_image_orientation(self):
    mjm, _, m, d = test_data.fixture(xml=_scene(z=0.005, offset="0.004 0.003"))
    sensor = _spec(mjm, optics=mjw.GelOptics(leds=()))
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth",))
    mjw.fwd_position(m, d)

    depth = renderer.render(m, d).numpy("depth", 0)[0]
    height, width = sensor.resolution
    row, col = np.unravel_index(np.argmax(depth), depth.shape)

    # +x maps to larger column, +y maps to smaller row.
    self.assertGreater(col, width // 2)
    self.assertLess(row, height // 2)

  def test_normals_match_analytic_sphere(self):
    mjm, _, m, d = test_data.fixture(xml=_scene())
    sensor = _spec(mjm, optics=mjw.GelOptics(leds=(), smoothing=0.0))
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth", "normal"))
    mjw.fwd_position(m, d)

    frame = renderer.render(m, d)
    depth = frame.numpy("depth", 0)[0]
    normal = frame.numpy("normal", 0)[0]

    center = _ball_center(d, mjm)
    top = _pad_top(d, mjm)
    height, width = sensor.resolution
    x, y = _grid((height, width), sensor.gel_size)
    pz = top - center[2]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + pz * pz)
    grad_x = np.where(dist > _RADIUS, 0.0, -(x - center[0]) / np.maximum(dist, 1e-9))
    grad_y = np.where(dist > _RADIUS, 0.0, -(y - center[1]) / np.maximum(dist, 1e-9))
    expected = np.stack([grad_x, grad_y, np.ones_like(grad_x)], axis=-1)
    expected /= np.linalg.norm(expected, axis=-1, keepdims=True)

    # Compare interior pixels whose four neighbors are also inside the cavity
    # (the depth kink at the rim makes central differences disagree there).
    height, width = sensor.resolution
    spacing = max(sensor.gel_size[0] / width, sensor.gel_size[1] / height)
    interior = (dist < _RADIUS - 1.5 * spacing) & (dist > 1e-4)
    np.testing.assert_allclose(normal[interior], expected[interior], atol=2e-3)
    np.testing.assert_allclose(np.linalg.norm(normal, axis=-1), 1.0, atol=1e-6)
    self.assertTrue(np.all(normal[..., 2] > 0.0))

  def test_overhead_led_shading(self):
    mjm, _, m, d = test_data.fixture(xml=_scene())
    optics = mjw.GelOptics(
      leds=(mjw.Led(pos=(0.0, 0.0, 0.01)),),
      ambient=0.0,
      attenuation=0.0,
      shadow_steps=0,
    )
    sensor = _spec(mjm, optics=optics)
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth", "normal", "rgb"))
    mjw.fwd_position(m, d)

    frame = renderer.render(m, d)
    height, width = sensor.resolution
    x, y = _grid((height, width), sensor.gel_size)
    normal = frame.numpy("normal", 0)[0]
    depth = frame.numpy("depth", 0)[0]
    to_led = np.stack([-x, -y, 0.01 + depth], axis=-1)
    dist = np.linalg.norm(to_led, axis=-1)
    ndotl = np.maximum(np.sum(normal * to_led / dist[..., None], axis=-1), 0.0)
    expected = np.stack([0.82 * ndotl, 0.82 * ndotl, 0.80 * ndotl], axis=-1)
    np.testing.assert_allclose(frame.numpy("rgb", 0)[0], expected, atol=2e-3)

  def test_shadow_darkens_cavity(self):
    mjm, _, m, d = test_data.fixture(xml=_scene(z=0.004))
    # Oblique low side LED: rays from the near wall graze the cavity rim.
    led = (mjw.Led(pos=(-0.012, 0.0, 0.005)),)
    shadowed = _spec(
      mjm,
      optics=mjw.GelOptics(leds=led, ambient=0.0, shadow_steps=64, shadow_softness=2e-4, smoothing=0.5),
    )
    unshadowed = _spec(mjm, optics=mjw.GelOptics(leds=led, ambient=0.0, shadow_steps=0, smoothing=0.5))
    renderer_shadow = mjw.VisuotactileRenderer(m, [shadowed], fields=("depth", "rgb"))
    renderer_unshadow = mjw.VisuotactileRenderer(m, [unshadowed], fields=("depth", "rgb"))
    mjw.fwd_position(m, d)

    frame_shadow = renderer_shadow.render(m, d)
    frame_unshadow = renderer_unshadow.render(m, d)
    rgb_shadow = frame_shadow.numpy("rgb", 0)[0]
    rgb_unshadow = frame_unshadow.numpy("rgb", 0)[0]
    depth = frame_shadow.numpy("depth", 0)[0]

    cavity = depth > 0.2e-3
    self.assertGreater(np.count_nonzero(cavity), 0)
    self.assertTrue(np.any(rgb_shadow[cavity] < rgb_unshadow[cavity] - 1e-3))

  def test_marker_pattern_flat(self):
    mjm, _, m, d = test_data.fixture(xml=_scene(z=0.02))
    markers = mjw.MarkerPattern(spacing=(0.002, 0.002), radius=0.0004, softness=0.0001)
    sensor = _spec(mjm, markers=markers, gel_size=(0.016, 0.012))
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth", "marker", "flow"))
    mjw.fwd_position(m, d)

    frame = renderer.render(m, d)
    marker = frame.numpy("marker", 0)[0]
    flow = frame.numpy("flow", 0)[0]

    height, width = sensor.resolution
    x, y = _grid((height, width), sensor.gel_size)
    ci = np.round(x / markers.spacing[0] - 0.5)
    cj = np.round(y / markers.spacing[1] - 0.5)
    cx = (ci + 0.5) * markers.spacing[0]
    cy = (cj + 0.5) * markers.spacing[1]
    dd = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    soft = markers.softness
    t = np.clip((dd - (markers.radius - soft)) / (2.0 * soft), 0.0, 1.0)
    expected = 1.0 - t * t * (3.0 - 2.0 * t)

    np.testing.assert_allclose(marker, expected, atol=1e-5)
    np.testing.assert_allclose(flow, 0.0, atol=1e-7)
    self.assertGreater(np.count_nonzero(marker > 0.5), 0)

  def test_marker_flow_is_outward(self):
    mjm, _, m, d = test_data.fixture(xml=_scene())
    markers = mjw.MarkerPattern(spacing=(0.002, 0.002), radius=0.0003)
    sensor = _spec(mjm, markers=markers)
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth", "flow"))
    mjw.fwd_position(m, d)

    frame = renderer.render(m, d)
    depth = frame.numpy("depth", 0)[0]
    flow = frame.numpy("flow", 0)[0]

    height, width = sensor.resolution
    x, y = _grid((height, width), sensor.gel_size)
    center = _ball_center(d, mjm)
    rx = x - center[0]
    ry = y - center[1]
    radial = rx * flow[..., 0] + ry * flow[..., 1]
    wall = (depth > 0.2e-3) & (depth < 0.8e-3) & (np.sqrt(rx**2 + ry**2) > 1e-3)
    self.assertGreater(np.count_nonzero(wall), 0)
    self.assertTrue(np.all(radial[wall] > 0.0))

  @parameterized.parameters(1, 2)
  def test_batching_and_determinism(self, nworld):
    mjm, _, m, d = test_data.fixture(xml=_scene(), nworld=nworld)
    sensor = _spec(mjm)
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth", "rgb"))

    qpos = np.zeros((nworld, 7), np.float32)
    for w in range(nworld):
      qpos[w] = [0.0, 0.001 * w, 0.005, 1.0, 0.0, 0.0, 0.0]
    d.qpos.assign(qpos)
    mjw.fwd_position(m, d)
    depth_a = renderer.render(m, d).numpy("depth", 0)
    depth_b = renderer.render(m, d).numpy("depth", 0)

    np.testing.assert_array_equal(depth_a, depth_b)
    for w in range(1, nworld):
      self.assertFalse(np.allclose(depth_a[w], depth_a[0]))

  def test_multiple_sensors_heterogeneous_resolution(self):
    xml = f"""
    <mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <body name="sensor0"><geom name="pad0" type="box" size="{_PAD}"/></body>
        <body name="sensor1" pos="0.1 0 0"><geom name="pad1" type="box" size="{_PAD}"/></body>
        <body name="indenter" pos="0 0 0.005">
          <freejoint/>
          <geom name="ball" type="sphere" size="{_RADIUS}"/>
        </body>
      </worldbody>
    </mujoco>
    """
    mjm, _, m, d = test_data.fixture(xml=xml)
    pad0 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "pad0")
    pad1 = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "pad1")
    sensors = [
      mjw.VisuotactileSensor(geom=pad0, resolution=(12, 16), gel_size=(0.02, 0.016)),
      mjw.VisuotactileSensor(geom=pad1, resolution=(24, 32), gel_size=(0.02, 0.016)),
    ]
    renderer = mjw.VisuotactileRenderer(m, sensors, fields=("depth", "rgb"))
    mjw.fwd_position(m, d)
    frame = renderer.render(m, d)

    self.assertEqual(frame.res, ((12, 16), (24, 32)))
    self.assertEqual(frame.numpy("depth", 0).shape, (1, 12, 16))
    self.assertEqual(frame.numpy("depth", 1).shape, (1, 24, 32))
    self.assertGreater(frame.numpy("depth", 0).max(), 0.0)
    self.assertEqual(np.count_nonzero(frame.numpy("depth", 1)), 0)

  def test_tactile_idiom_non_colliding_gel_mesh(self):
    # Standard MuJoCo tactile idiom: gel mesh with contype=0 on a body whose
    # sibling geom collides. The depth field must follow the sibling contacts.
    radius = 0.004
    verts, _ = _icosphere(radius)
    gel = (float(np.ptp(verts[:, 0])), float(np.ptp(verts[:, 1])))
    top = float(verts[:, 2].max())
    with tempfile.TemporaryDirectory() as tmp:
      _write_icosphere_stl(os.path.join(tmp, "gel.stl"), radius)
      xml = f"""
      <mujoco>
        <compiler meshdir="{tmp}"/>
        <option gravity="0 0 0"/>
        <asset><mesh name="gel_mesh" file="gel.stl"/></asset>
        <worldbody>
          <body name="sensor">
            <geom name="pad" type="box" size="{_PAD}"/>
            <geom name="gel" type="mesh" mesh="gel_mesh" contype="0" conaffinity="0"/>
          </body>
          <body name="indenter" pos="0 0 {_PAD_TOP + radius - 0.001}">
            <freejoint/>
            <geom name="ball" type="sphere" size="{radius}"/>
          </body>
        </worldbody>
      </mujoco>
      """
      mjm, _, m, d = test_data.fixture(xml=xml)
      gel_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "gel")
      sensor = mjw.VisuotactileSensor(geom=gel_id, resolution=(24, 32))
      renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth",))
      mjw.fwd_position(m, d)

      depth = renderer.render(m, d).numpy("depth", 0)[0]
      center = _ball_center(d, mjm)
      height, width = sensor.resolution
      x, y = _grid((height, width), gel)
      dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (top - center[2]) ** 2)
      np.testing.assert_allclose(depth, np.maximum(radius - dist, 0.0), atol=1e-5)

  def test_sdf_mesh_indenter(self):
    with tempfile.TemporaryDirectory() as tmp:
      _write_icosphere_stl(os.path.join(tmp, "ball.stl"), _RADIUS)
      xml = f"""
      <mujoco>
        <compiler meshdir="{tmp}"/>
        <option gravity="0 0 0"/>
        <asset><mesh name="ball_mesh" file="ball.stl"/></asset>
        <worldbody>
          <body name="sensor"><geom name="pad" type="box" size="{_PAD}"/></body>
          <body name="indenter" pos="0 0 0.005">
            <freejoint/>
            <geom name="ball" type="sdf" mesh="ball_mesh"/>
          </body>
        </worldbody>
      </mujoco>
      """
      mjm, _, m, d = test_data.fixture(xml=xml)
      sensor = _spec(mjm)
      renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth",))
      mjw.fwd_position(m, d)

      depth = renderer.render(m, d).numpy("depth", 0)[0]
      center = _ball_center(d, mjm)
      top = _pad_top(d, mjm)
      height, width = sensor.resolution
      x, y = _grid((height, width), sensor.gel_size)
      dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (top - center[2]) ** 2)
      np.testing.assert_allclose(depth, np.maximum(_RADIUS - dist, 0.0), atol=1e-4)

  def test_plain_mesh_without_octree_is_skipped(self):
    # Matches MuJoCo tactile sensors: mesh geoms without an octree contribute no depth.
    xml = f"""
    <mujoco>
      <option gravity="0 0 0"/>
      <asset><mesh name="ball_mesh" builtin="sphere" params="0"/></asset>
      <worldbody>
        <body name="sensor"><geom name="pad" type="box" size="{_PAD}"/></body>
        <body name="indenter" pos="0 0 0.005">
          <freejoint/>
          <geom name="ball" type="mesh" mesh="ball_mesh"/>
        </body>
      </worldbody>
    </mujoco>
    """
    mjm, _, m, d = test_data.fixture(xml=xml)
    sensor = _spec(mjm)
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth",))
    mjw.fwd_position(m, d)

    np.testing.assert_allclose(renderer.render(m, d).numpy("depth", 0), 0.0)

  def test_force_is_stiffness_times_depth(self):
    mjm, _, m, d = test_data.fixture(xml=_scene())
    sensor = _spec(mjm, stiffness=1234.0)
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth", "force"))
    mjw.fwd_position(m, d)

    frame = renderer.render(m, d)
    np.testing.assert_allclose(frame.numpy("force", 0), 1234.0 * frame.numpy("depth", 0), rtol=1e-6)

  def test_render_reuses_buffers(self):
    mjm, _, m, d = test_data.fixture(xml=_scene())
    sensor = _spec(mjm)
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth", "rgb", "marker"))
    frame = renderer.allocate(d.nworld)
    mjw.fwd_position(m, d)

    depth_a = renderer.render(m, d, out=frame).numpy("depth", 0).copy()

    qpos = d.qpos.numpy().copy()
    qpos[0, 0] = 0.003
    d.qpos.assign(qpos)
    mjw.fwd_position(m, d)
    depth_b = renderer.render(m, d, out=frame).numpy("depth", 0).copy()

    self.assertFalse(np.array_equal(depth_a, depth_b))
    np.testing.assert_array_equal(depth_b, renderer.render(m, d).numpy("depth", 0))
    self.assertIs(renderer.render(m, d, out=frame), frame)

    partial = mjw.VisuotactileFrame(res=frame.res, depth=frame.depth)
    with self.assertRaises(ValueError):
      renderer.render(m, d, out=partial)

  def test_noise_reproducible(self):
    mjm, _, m, d = test_data.fixture(xml=_scene())
    sensor = _spec(mjm)
    mjw.fwd_position(m, d)

    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("rgb",), noise=mjw.CameraSensor(seed=0))
    rgb_a = renderer.render(m, d).numpy("rgb", 0)
    rgb_b = renderer.render(m, d).numpy("rgb", 0)
    np.testing.assert_array_equal(rgb_a, rgb_b)

    renderer2 = mjw.VisuotactileRenderer(m, [sensor], fields=("rgb",), noise=mjw.CameraSensor(seed=1))
    rgb_c = renderer2.render(m, d).numpy("rgb", 0)
    self.assertFalse(np.allclose(rgb_a, rgb_c))

  def test_display_rgb(self):
    mjm, _, m, d = test_data.fixture(xml=_scene())
    sensor = _spec(mjm)
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("rgb",))
    mjw.fwd_position(m, d)

    image = renderer.render(m, d).display_rgb(0)
    self.assertEqual(image.dtype, np.uint8)
    self.assertEqual(image.shape, (1, 24, 32, 3))
    self.assertGreaterEqual(int(image.min()), 0)
    self.assertLessEqual(int(image.max()), 255)

  def test_field_not_requested_raises(self):
    mjm, _, m, d = test_data.fixture(xml=_scene())
    sensor = _spec(mjm)
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth",))
    mjw.fwd_position(m, d)

    frame = renderer.render(m, d)
    self.assertIsNone(frame.rgb)
    with self.assertRaises(ValueError):
      frame.numpy("rgb", 0)
    with self.assertRaises(ValueError):
      frame.numpy("nope", 0)

  def test_depth_gradient_matches_fd(self):
    mjm, _, m, d = test_data.fixture(xml=_scene())
    sensor = _spec(mjm, optics=mjw.GelOptics(leds=(mjw.Led(pos=(0.0, 0.0, 0.01)),), ambient=0.1))
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth", "rgb"))
    mjw.fwd_position(m, d)
    ball_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball")

    def loss_at(qz):
      qpos = d.qpos.numpy().copy()
      qpos[0, 2] = qz
      d.qpos.assign(qpos)
      mjw.fwd_position(m, d)
      frame = renderer.render(m, d)
      return float(renderer.mse(frame.rgb[0], wp.zeros_like(frame.rgb[0])).numpy()[0]), frame

    d.geom_xpos.requires_grad = True
    with wp.Tape() as tape:
      frame = renderer.render(m, d)
      loss = renderer.mse(frame.rgb[0], wp.zeros_like(frame.rgb[0]))
    tape.backward(loss)
    grad = d.geom_xpos.grad.numpy()[0, ball_id].copy()

    eps = 1e-5
    qz = float(d.qpos.numpy()[0, 2])
    loss_plus, _ = loss_at(qz + eps)
    loss_minus, _ = loss_at(qz - eps)
    qpos = d.qpos.numpy().copy()
    qpos[0, 2] = qz
    d.qpos.assign(qpos)
    mjw.fwd_position(m, d)

    # geom_xpos of a free joint equals qpos; the z gradient is direct.
    fd = (loss_plus - loss_minus) / (2.0 * eps)
    self.assertGreater(abs(grad[2]), 1e-3)
    self.assertAlmostEqual(grad[2], fd, delta=0.01 * abs(grad[2]))

  def test_depth_gradient_zero_without_contact(self):
    mjm, _, m, d = test_data.fixture(xml=_scene(z=0.02))
    sensor = _spec(mjm)
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth",))
    mjw.fwd_position(m, d)
    ball_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball")

    d.geom_xpos.requires_grad = True
    with wp.Tape() as tape:
      frame = renderer.render(m, d)
      loss = renderer.mse(frame.depth[0], wp.zeros_like(frame.depth[0]))
    tape.backward(loss)

    np.testing.assert_allclose(d.geom_xpos.grad.numpy()[0, ball_id], 0.0, atol=1e-9)

  def test_compose_with_physics(self):
    # dL/dq = (dL/dxpos) . (dxpos/dq) for a hinge-driven sphere over the sensor.
    xml = f"""
    <mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <body name="sensor"><geom name="pad" type="box" size="{_PAD}"/></body>
        <body pos="0 0 0.004">
          <joint name="hinge" type="hinge" axis="0 1 0"/>
          <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
          <geom name="ball" pos="0.006 0 0" type="sphere" size="{_RADIUS}"/>
        </body>
      </worldbody>
    </mujoco>
    """
    mjm, _, m, d = test_data.fixture(xml=xml)
    sensor = _spec(mjm)
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth",))
    ball_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball")

    def loss_at(q):
      d.qpos.assign(np.array([q], np.float32))
      mjw.fwd_position(m, d)
      return float(
        renderer.mse(renderer.render(m, d).depth[0], wp.zeros((1, sensor.resolution[0] * sensor.resolution[1]))).numpy()[0]
      )

    def xpos_at(q):
      d.qpos.assign(np.array([q], np.float32))
      mjw.fwd_position(m, d)
      return d.geom_xpos.numpy()[0].copy()

    q0 = 0.1
    d.qpos.assign(np.array([q0], np.float32))
    mjw.fwd_position(m, d)
    d.geom_xpos.requires_grad = True
    with wp.Tape() as tape:
      frame = renderer.render(m, d)
      loss = renderer.mse(frame.depth[0], wp.zeros_like(frame.depth[0]))
    tape.backward(loss)
    grad_xpos = d.geom_xpos.grad.numpy()[0].copy()

    eps = 1e-5
    jacobian = (xpos_at(q0 + eps) - xpos_at(q0 - eps)) / (2.0 * eps)
    chain = float(np.sum(jacobian * grad_xpos))

    fd = (loss_at(q0 + eps) - loss_at(q0 - eps)) / (2.0 * eps)
    self.assertGreater(abs(fd), 1e-6)
    self.assertAlmostEqual(chain, fd, delta=0.05 * abs(fd) + 1e-7)

  def test_from_tactile_sensors(self):
    radius = 0.004
    verts, _ = _icosphere(radius)
    gel = (float(np.ptp(verts[:, 0])), float(np.ptp(verts[:, 1])))
    top = float(verts[:, 2].max())
    with tempfile.TemporaryDirectory() as tmp:
      _write_icosphere_stl(os.path.join(tmp, "gel.stl"), radius)
      xml = f"""
      <mujoco>
        <compiler meshdir="{tmp}"/>
        <option gravity="0 0 0"/>
        <asset><mesh name="gel_mesh" file="gel.stl"/></asset>
        <worldbody>
          <body name="sensor"><geom name="pad" type="mesh" mesh="gel_mesh"/></body>
          <body name="indenter" pos="0 0 {top + radius - 0.001}">
            <freejoint/>
            <geom name="ball" type="sphere" size="{radius}"/>
          </body>
        </worldbody>
        <sensor>
          <tactile geom="pad" mesh="gel_mesh"/>
        </sensor>
      </mujoco>
      """
      mjm, _, m, d = test_data.fixture(xml=xml)
      specs = mjw.from_tactile_sensors(mjm, resolution=(24, 32))
      self.assertLen(specs, 1)
      self.assertEqual(specs[0].geom, mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "pad"))
      renderer = mjw.VisuotactileRenderer(m, specs, fields=("depth",))
      mjw.fwd_position(m, d)

      depth = renderer.render(m, d).numpy("depth", 0)[0]
      center = _ball_center(d, mjm)
      height, width = specs[0].resolution
      x, y = _grid((height, width), gel)
      dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (top - center[2]) ** 2)
      np.testing.assert_allclose(depth, np.maximum(radius - dist, 0.0), atol=1e-5)

    mjm2, _, _, _ = test_data.fixture(xml=_scene())
    with self.assertRaises(ValueError):
      mjw.from_tactile_sensors(mjm2)

  def test_presets(self):
    mjm, _, m, d = test_data.fixture(xml=_scene())
    geom_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "pad")
    mini = mjw.gelsight_mini(geom_id)
    dig = mjw.digit(geom_id)
    self.assertEqual(mini.gel_size, (0.016, 0.012))
    self.assertIsNotNone(mini.markers)
    self.assertIsNone(dig.markers)

    renderer = mjw.VisuotactileRenderer(m, [mini], fields=("depth", "rgb", "marker", "flow"))
    mjw.fwd_position(m, d)
    frame = renderer.render(m, d)
    self.assertEqual(frame.numpy("rgb", 0).shape[:3], (1, 240, 320))
    self.assertEqual(frame.numpy("marker", 0).shape, (1, 240, 320))
    self.assertEqual(frame.numpy("flow", 0).shape, (1, 240, 320, 2))

  def test_errors(self):
    mjm, _, m, d = test_data.fixture(xml=_scene())
    geom_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "pad")
    with self.assertRaises(ValueError):
      mjw.VisuotactileRenderer(m, [])
    with self.assertRaises(ValueError):
      mjw.VisuotactileRenderer(m, [_spec(mjm)], fields=("nope",))
    with self.assertRaises(ValueError):
      mjw.VisuotactileRenderer(m, [mjw.VisuotactileSensor(geom=999)])
    with self.assertRaises(ValueError):
      mjw.VisuotactileSensor(geom=geom_id, resolution=(0, 8))
    with self.assertRaises(ValueError):
      mjw.VisuotactileSensor(geom=geom_id, frame_quat=(1.0, 1.0, 0.0, 0.0), gel_size=(0.01, 0.01))
    with self.assertRaises(ValueError):
      mjw.MarkerPattern(spacing=(0.0, 0.001))
    with self.assertRaises(ValueError):
      mjw.GelOptics(shadow_steps=1, shadow_softness=0.0)

  def test_plane_pad_requires_explicit_geometry(self):
    xml = f"""
    <mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <geom name="pad" type="plane" size="0.01 0.01 0.1"/>
      </worldbody>
    </mujoco>
    """
    mjm, _, m, d = test_data.fixture(xml=xml)
    geom_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "pad")
    with self.assertRaises(ValueError):
      mjw.VisuotactileRenderer(m, [mjw.VisuotactileSensor(geom=geom_id)])
    # explicit geometry works
    sensor = mjw.VisuotactileSensor(geom=geom_id, gel_size=(0.02, 0.02), frame_pos=(0.0, 0.0, 0.0))
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth",))
    mjw.fwd_position(m, d)
    np.testing.assert_allclose(renderer.render(m, d).numpy("depth", 0), 0.0)

  def test_specular_highlight(self):
    mjm, _, m, d = test_data.fixture(xml=_scene(z=0.02))
    led = (mjw.Led(pos=(0.0, 0.0, 0.02), intensity=1.0),)
    sensor_diffuse = _spec(mjm, optics=mjw.GelOptics(leds=led, ambient=0.0, specular=0.0))
    sensor_specular = _spec(mjm, optics=mjw.GelOptics(leds=led, ambient=0.0, specular=1.0, shininess=20.0))
    renderer_diffuse = mjw.VisuotactileRenderer(m, sensor_diffuse, fields=("rgb",))
    renderer_specular = mjw.VisuotactileRenderer(m, sensor_specular, fields=("rgb",))
    mjw.fwd_position(m, d)

    rgb_diffuse = renderer_diffuse.render(m, d).numpy("rgb", 0)[0]
    rgb_specular = renderer_specular.render(m, d).numpy("rgb", 0)[0]

    # Center is directly under the LED; specular reflection should add brightness.
    mid_r, mid_c = rgb_diffuse.shape[0] // 2, rgb_diffuse.shape[1] // 2
    self.assertGreater(float(rgb_specular[mid_r, mid_c, 0]), float(rgb_diffuse[mid_r, mid_c, 0]))

  @absltest.skipIf(torch is None, "torch is not installed")
  def test_torch_interop_and_indexing(self):
    mjm, _, m, d = test_data.fixture(xml=_scene())
    sensor = _spec(mjm)
    renderer = mjw.VisuotactileRenderer(m, sensor, fields=("depth", "rgb"))
    mjw.fwd_position(m, d)
    frame = renderer.render(m, d)

    t_depth = frame.torch("depth")
    self.assertIsInstance(t_depth, torch.Tensor)
    self.assertEqual(t_depth.shape, (1, 24, 32))

    t_rgb = frame.torch("rgb")
    self.assertEqual(t_rgb.shape, (1, 24, 32, 3))

    t_rgb_cf = frame.torch("rgb", channels_first=True)
    self.assertEqual(t_rgb_cf.shape, (1, 3, 24, 32))

    # Indexing access
    np.testing.assert_array_equal(frame["depth"], frame.numpy("depth", 0))
    np.testing.assert_array_equal(frame["rgb", 0], frame.numpy("rgb", 0))

  def test_single_sensor_convenience(self):
    mjm, _, m, d = test_data.fixture(xml=_scene())
    sensor = _spec(mjm)
    renderer = mjw.VisuotactileRenderer(m, sensor, fields=("depth",))
    mjw.fwd_position(m, d)
    frame = renderer.render(m, d)
    self.assertEqual(len(renderer.sensors), 1)
    self.assertGreater(frame["depth"].max(), 0.0)

  def test_depth_only_pipeline(self):
    mjm, _, m, d = test_data.fixture(xml=_scene())
    sensor = _spec(mjm)
    renderer = mjw.VisuotactileRenderer(m, sensor, fields=("depth", "force"))
    mjw.fwd_position(m, d)
    frame = renderer.render(m, d)
    self.assertIsNotNone(frame.depth)
    self.assertIsNotNone(frame.force)
    self.assertIsNone(frame.rgb)
    self.assertFalse(renderer._needs_optics)

  def test_rgb_leds_preset(self):
    mjm, _, m, d = test_data.fixture(xml=_scene())
    geom_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "pad")
    mini = mjw.gelsight_mini(geom_id, rgb_leds=True)
    renderer = mjw.VisuotactileRenderer(m, mini, fields=("rgb",))
    self.assertTrue(mini.optics.rgb_leds)
    rt = renderer._rt[0]
    leds_rgb = rt.leds_rgb.numpy()
    self.assertFalse(np.allclose(leds_rgb[0], leds_rgb[1]))

  def test_self_contact_on_sensor_body_ignored(self):
    # An explicit pair between the sensor body's own geoms, sibling overlapping
    # the gel surface. MuJoCo filters same-body pairs by default, but explicit
    # pairs still emit contacts, and those must not read as indenters.
    xml = f"""
    <mujoco>
      <option gravity="0 0 0"/>
      <contact><pair geom1="pad" geom2="bracket"/></contact>
      <worldbody>
        <body name="sensor">
          <geom name="pad" type="box" size="{_PAD}"/>
          <geom name="bracket" pos="0 0 0.001" type="box" size="{_PAD}"/>
        </body>
        <body name="indenter" pos="0 0 0.05">
          <geom name="ball" type="sphere" size="{_RADIUS}"/>
        </body>
      </worldbody>
    </mujoco>
    """
    mjm, _, m, d = test_data.fixture(xml=xml)
    sensor = _spec(mjm)
    renderer = mjw.VisuotactileRenderer(m, sensor, fields=("depth",))
    mjw.fwd_position(m, d)

    pad = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "pad")
    bracket = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "bracket")
    geoms = d.contact.geom.numpy()[: int(d.nacon.numpy()[0])]
    self.assertTrue(any({int(g) for g in pair} == {pad, bracket} for pair in geoms))

    frame = renderer.render(m, d)
    self.assertEqual(np.count_nonzero(frame["depth"]), 0)


if __name__ == "__main__":
  absltest.main()
