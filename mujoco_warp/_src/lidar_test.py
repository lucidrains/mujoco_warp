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
"""Tests for LiDAR simulation."""

import numpy as np
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp import test_data

_SCENE = """
<mujoco>
  <asset><material name="white" rgba="1 1 1 1"/></asset>
  <worldbody>
    <geom type="plane" size="10 10 0.1" material="white"/>
  </worldbody>
</mujoco>
"""

_OCCLUDED_SCENE = """
<mujoco>
  <asset><material name="white" rgba="1 1 1 1"/></asset>
  <worldbody>
    <geom type="plane" size="10 10 0.1" material="white"/>
    <geom type="box" pos="0 0 1" size="0.2 0.2 0.5"/>
  </worldbody>
</mujoco>
"""


def _normalized(dirs):
  dirs = np.asarray(dirs, dtype=np.float32)
  return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


class LidarTest(absltest.TestCase):
  def test_flat_ranges_and_intensity(self):
    mjm, _, m, d = test_data.fixture(xml=_SCENE)
    rc = mjw.create_render_context(mjm)
    dirs = _normalized([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])
    lidar = mjw.Lidar(dirs, max_range=100.0)

    scan = lidar.scan(m, d, rc, origin=(0.0, 0.0, 2.0))

    np.testing.assert_allclose(scan.ranges.numpy(), [[2.0, 2.0 * np.sqrt(2.0), 2.0 * np.sqrt(2.0)]], rtol=1e-5)
    np.testing.assert_allclose(scan.intensity.numpy(), [[1.0, np.sqrt(0.5), np.sqrt(0.5)]], atol=1e-5)
    np.testing.assert_array_equal(scan.hit.numpy(), [[1, 1, 1]])

  def test_occlusion(self):
    mjm, _, m, d = test_data.fixture(xml=_OCCLUDED_SCENE)
    rc = mjw.create_render_context(mjm)
    lidar = mjw.Lidar(_normalized([[0.0, 0.0, -1.0]]), max_range=100.0)

    scan = lidar.scan(m, d, rc, origin=(0.0, 0.0, 2.0))

    # The box top at z = 1.5 occludes the plane.
    np.testing.assert_allclose(scan.ranges.numpy(), [[0.5]], rtol=1e-5)

  def test_miss(self):
    mjm, _, m, d = test_data.fixture(xml=_SCENE)
    rc = mjw.create_render_context(mjm)
    lidar = mjw.Lidar(_normalized([[0.0, 0.0, 1.0]]), max_range=50.0)

    scan = lidar.scan(m, d, rc, origin=(0.0, 0.0, 2.0))

    np.testing.assert_allclose(scan.ranges.numpy(), [[50.0]])
    np.testing.assert_allclose(scan.intensity.numpy(), [[0.0]])
    np.testing.assert_array_equal(scan.hit.numpy(), [[0]])

  def test_nworld(self):
    mjm, _, m, d = test_data.fixture(xml=_SCENE, nworld=2)
    rc = mjw.create_render_context(mjm, nworld=2)
    lidar = mjw.Lidar(_normalized([[0.0, 0.0, -1.0]]), max_range=100.0)

    scan = lidar.scan(m, d, rc, origin=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]))

    np.testing.assert_allclose(scan.ranges.numpy(), [[2.0], [3.0]], rtol=1e-5)

  def test_rotation(self):
    mjm, _, m, d = test_data.fixture(xml=_SCENE)
    rc = mjw.create_render_context(mjm)
    lidar = mjw.Lidar(_normalized([[0.0, 0.0, -1.0]]), max_range=100.0)

    # Rotate the beam 45 degrees about +Y: local -Z maps to (-sqrt(2)/2, 0, -sqrt(2)/2).
    c = np.sqrt(0.5)
    rotation = np.array([[c, 0.0, -c], [0.0, 1.0, 0.0], [c, 0.0, c]], dtype=np.float32)
    scan = lidar.scan(m, d, rc, origin=(0.0, 0.0, 2.0), rotation=rotation)

    np.testing.assert_allclose(scan.ranges.numpy(), [[2.0 / c]], rtol=1e-5)
    point = scan.points.numpy()[0, 0]
    np.testing.assert_allclose(point[2], 0.0, atol=1e-5)

  def test_range_noise(self):
    mjm, _, m, d = test_data.fixture(xml=_SCENE)
    rc = mjw.create_render_context(mjm)
    down = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (512, 1))
    lidar = mjw.Lidar(down, max_range=50.0, range_sigma=0.01, seed=3)

    scan = lidar.scan(m, d, rc, origin=(0.0, 0.0, 2.0))
    ranges = scan.ranges.numpy().ravel()
    self.assertAlmostEqual(float(ranges.mean()), 2.0, delta=0.002)
    self.assertGreater(float(ranges.std()), 0.0075)
    self.assertLess(float(ranges.std()), 0.0125)

    # Same seed reproduces the scan exactly.
    scan2 = lidar.scan(m, d, rc, origin=(0.0, 0.0, 2.0))
    np.testing.assert_array_equal(scan.ranges.numpy(), scan2.ranges.numpy())

  def test_beam_divergence(self):
    mjm, _, m, d = test_data.fixture(xml=_SCENE)
    rc = mjw.create_render_context(mjm)
    down = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (512, 1))

    spread = mjw.Lidar(down, max_range=50.0, beam_divergence=0.02, seed=1).scan(m, d, rc, origin=(0.0, 0.0, 2.0))
    points = spread.points.numpy()[0]
    self.assertGreater(float(points[:, 0].std()), 0.02)
    self.assertGreater(float(points[:, 1].std()), 0.02)

    sharp = mjw.Lidar(down, max_range=50.0).scan(m, d, rc, origin=(0.0, 0.0, 2.0))
    self.assertAlmostEqual(float(sharp.points.numpy()[0, :, 0].std()), 0.0, delta=1e-6)

  def test_ring_pattern(self):
    dirs = mjw.Lidar.ring_pattern(n_rings=4, points_per_ring=8, vfov=(-10.0, 10.0))
    self.assertEqual(dirs.shape, (32, 3))
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-5)
    self.assertAlmostEqual(float(dirs[:, 2].min()), np.sin(np.deg2rad(-10.0)), places=5)
    self.assertAlmostEqual(float(dirs[:, 2].max()), np.sin(np.deg2rad(10.0)), places=5)


if __name__ == "__main__":
  absltest.main()
