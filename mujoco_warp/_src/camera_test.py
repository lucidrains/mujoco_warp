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
"""Tests for camera and sensor models."""

import numpy as np
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp import test_data

_EMIT_MAT = '<material name="emit" rgba="1 1 1 1" specular="0" shininess="0" emission="1"/>'


def _emitter_xml(pos: str, size: str) -> str:
  return f"""
  <mujoco>
    <asset>{_EMIT_MAT}</asset>
    <worldbody>
      <camera pos="0 0 1.5" xyaxes="1 0 0 0 1 0"/>
      <geom type="sphere" pos="{pos}" size="{size}" material="emit"/>
    </worldbody>
  </mujoco>
  """


def _render(xml: str, camera=None, sensor=None, res: int = 32, samples: int = 4, bounces: int = 1) -> np.ndarray:
  mjm, _, m, d = test_data.fixture(xml=xml)
  rc = mjw.create_render_context(mjm, cam_res=(res, res), render_rgb=True)
  tracer = mjw.PathTracer(rc, max_bounces=bounces, camera=camera, sensor=sensor)
  tracer.render(m, d, samples=samples)
  return tracer.hdr.numpy().reshape(res, res, 3)


def _render_sensor(xml: str, sensor, res: int = 32, samples: int = 1) -> "mjw.PathTracer":
  mjm, _, m, d = test_data.fixture(xml=xml)
  rc = mjw.create_render_context(mjm, cam_res=(res, res), render_rgb=True)
  tracer = mjw.PathTracer(rc, max_bounces=1, sensor=sensor)
  tracer.render(m, d, samples=samples)
  return tracer


def _centroid(img: np.ndarray, channel: int = 0) -> tuple[float, float]:
  w = img[..., channel]
  total = w.sum()
  ys, xs = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
  return float((xs * w).sum() / total), float((ys * w).sum() / total)


_FLAT_XML = """
<mujoco>
  <asset>
    <material name="white" rgba="0.5 0.5 0.5 1" specular="0" shininess="0"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" directional="true"/>
    <geom type="plane" size="10 10 0.1" material="white"/>
    <camera pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""


class CameraTest(absltest.TestCase):
  def test_depth_of_field_keeps_focal_plane_sharp(self):
    xml = _emitter_xml("0.2 0 0.5", "0.05")
    pinhole = _render(xml)
    dof = _render(xml, camera=mjw.CameraModel(aperture=0.05, focus_distance=1.0))

    self.assertAlmostEqual(float(dof.max()), float(pinhole.max()), delta=0.05 * float(pinhole.max()) + 0.01)
    self.assertAlmostEqual(float(dof.sum()), float(pinhole.sum()), delta=0.2 * float(pinhole.sum()) + 0.01)

  def test_depth_of_field_blurs_out_of_focus(self):
    xml = _emitter_xml("0.2 0 0.9", "0.05")
    pinhole = _render(xml)
    dof = _render(xml, camera=mjw.CameraModel(aperture=0.2, focus_distance=1.0), samples=64)

    self.assertLess(float(dof.max()), 0.7 * float(pinhole.max()))
    self.assertGreater(int((dof > 0.05).sum()), 1.5 * int((pinhole > 0.05).sum()))

  def test_distortion_shifts_projection(self):
    xml = _emitter_xml("0.3 0 0.5", "0.06")
    pinhole = _render(xml)
    distorted = _render(xml, camera=mjw.CameraModel(distortion=(2.0, 0.0, 0.0, 0.0)))

    x_ref, _ = _centroid(pinhole)
    x_dist, _ = _centroid(distorted)
    # Positive radial coefficient moves off-axis features toward the center.
    self.assertLess(x_dist, x_ref - 1.0)
    self.assertGreater(int(np.count_nonzero(np.abs(distorted - pinhole) > 0.05)), 20)

  def test_chromatic_aberration_separates_channels(self):
    xml = _emitter_xml("0.25 0 0.5", "0.05")
    camera = mjw.CameraModel(chromatic=(1.12, 1.0, 0.88))
    img = _render(xml, camera=camera, samples=16)

    self.assertGreater(float(img.max(axis=(0, 1)).min()), 0.9)
    x_r, _ = _centroid(img, 0)
    x_g, _ = _centroid(img, 1)
    x_b, _ = _centroid(img, 2)
    # Red is magnified most, so its spot lands closest to the center.
    self.assertLess(x_r, x_g - 0.5)
    self.assertLess(x_g, x_b - 0.5)

  def test_sensor_noise_statistics(self):
    # Signal 0.5 -> 500 e-, shot sigma ~22.4 e-, read sigma 2 e-.
    tracer = _render_sensor(_FLAT_XML, mjw.CameraSensor(exposure=1.0, gain=1000.0, read_noise_e=2.0, seed=5))
    noisy = tracer.sensor_hdr.numpy().reshape(-1, 3)
    self.assertAlmostEqual(float(noisy.mean()), 0.5, delta=0.005)
    self.assertGreater(float(noisy.std()), 0.01)
    self.assertLess(float(noisy.std()), 0.04)

  def test_sensor_saturation(self):
    # 500 e- signal saturates a 100 e- full well.
    tracer = _render_sensor(_FLAT_XML, mjw.CameraSensor(exposure=1.0, gain=1000.0, full_well_e=100.0, read_noise_e=0.0))
    noisy = tracer.sensor_hdr.numpy().reshape(-1, 3)
    self.assertLessEqual(float(noisy.max()), 0.1001)
    self.assertGreater(float(noisy.mean()), 0.09)

  def test_sensor_deterministic(self):
    a = _render_sensor(_FLAT_XML, mjw.CameraSensor(seed=11))
    b = _render_sensor(_FLAT_XML, mjw.CameraSensor(seed=11))
    c = _render_sensor(_FLAT_XML, mjw.CameraSensor(seed=12))
    np.testing.assert_array_equal(a.sensor_hdr.numpy(), b.sensor_hdr.numpy())
    self.assertGreater(int(np.count_nonzero(np.abs(a.sensor_hdr.numpy() - c.sensor_hdr.numpy()) > 1e-6)), 0)

  def test_depth_of_field_requires_focus(self):
    mjm, _, _, _ = test_data.fixture(xml=_emitter_xml("0.2 0 0.5", "0.05"))
    rc = mjw.create_render_context(mjm, cam_res=(8, 8), render_rgb=True)
    with self.assertRaises(ValueError):
      mjw.PathTracer(rc, camera=mjw.CameraModel(aperture=0.05))


if __name__ == "__main__":
  absltest.main()
