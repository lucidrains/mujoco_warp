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
"""Tests for the path tracer."""

import numpy as np
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp import test_data


def _unpack_rgb(packed):
  r = ((packed >> 16) & 0xFF).astype(np.uint8)
  g = ((packed >> 8) & 0xFF).astype(np.uint8)
  b = (packed & 0xFF).astype(np.uint8)
  return np.stack([r, g, b], axis=-1)


def _aces_np(x):
  a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
  return (x * (a * x + b)) / (x * (c * x + d) + e)


_ANALYTIC_XML = """
<mujoco>
  <asset>
    <material name="white" rgba="0.5 0.5 0.5 1" specular="0" shininess="0"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" directional="true" castshadow="true"/>
    <geom type="plane" size="10 10 0.1" material="white"/>
    <camera pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""

_GI_XML = """
<mujoco>
  <asset>
    <material name="floor" rgba="0.8 0.8 0.8 1" specular="0" shininess="0"/>
    <material name="emit" rgba="1 1 1 1" specular="0" shininess="0" emission="1"/>
  </asset>
  <worldbody>
    <geom type="plane" size="5 5 0.1" material="floor"/>
    <geom type="sphere" pos="0.7 0 0.5" size="0.2" material="emit"/>
    <camera pos="0 0 1.5" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""


_FURNACE_XML = """
<mujoco>
  <asset><material name="white" rgba="1 1 1 1" specular="0" shininess="0"/></asset>
  <worldbody>
    <geom type="plane" size="10 10 0.1" material="white"/>
    <camera pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""


_SKYBOX_XML = """
<mujoco>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.55 0.85" rgb2="0.02 0.03 0.08" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="4 4" texuniform="true"/>
  </asset>
  <worldbody>
    <geom type="plane" size="4 4 0.05" material="grid"/>
    <camera pos="0 0 0.2" xyaxes="1 0 0 0 -0.2 1"/>
  </worldbody>
</mujoco>
"""


def _srgb_to_linear_np(c):
  return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))


_SPLAT_XML = """
<mujoco>
  <asset><material name="black" rgba="0.02 0.02 0.02 1" specular="0" shininess="0"/></asset>
  <worldbody>
    <geom type="plane" size="5 5 0.1" material="black"/>
    <camera pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""


def _splat_kwargs(z: float):
  return {
    "splat_position": np.array([[0.0, 0.0, z]], np.float32),
    "splat_rotation": np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
    "splat_scale": np.array([[0.3, 0.3, 0.3]], np.float32),
    "splat_rgba": np.array([[1.0, 0.0, 0.0, 0.95]], np.float32),
  }


class PathTraceTest(absltest.TestCase):
  def test_analytic_direct_lighting(self):
    # A pure diffuse white floor under a unit directional light has radiance
    # albedo * light_diffuse * cos(theta) = 0.5.
    mjm, _, m, d = test_data.fixture(xml=_ANALYTIC_XML)
    rc = mjw.create_render_context(mjm, cam_res=(32, 32), render_rgb=True)
    tracer = mjw.PathTracer(rc)
    tracer.render(m, d, samples=1, exposure=1.0)

    hdr = tracer.hdr.numpy()
    self.assertEqual(hdr.shape, (1, 32 * 32, 3))
    np.testing.assert_allclose(hdr, 0.5, atol=1e-3)

  def test_tonemap_output(self):
    mjm, _, m, d = test_data.fixture(xml=_ANALYTIC_XML)
    exposure = 2.0
    rc = mjw.create_render_context(mjm, cam_res=(16, 16), render_rgb=True)
    tracer = mjw.PathTracer(rc)
    tracer.render(m, d, samples=1, exposure=exposure)

    expected = np.power(np.clip(_aces_np(0.5 * exposure), 0.0, 1.0), 1.0 / 2.2)
    expected_byte = np.floor(expected * 255.0)
    rgb = _unpack_rgb(rc.rgb_data.numpy()).reshape(-1, 3)
    np.testing.assert_allclose(rgb, expected_byte, atol=1)

  def test_global_illumination(self):
    # No lights: the only radiance on the floor comes from bouncing off the
    # emissive sphere. The camera only sees floor (the sphere is outside the
    # view cone), so any signal is multi-bounce global illumination.
    mjm, _, m, d = test_data.fixture(xml=_GI_XML)
    rc = mjw.create_render_context(mjm, cam_res=(24, 24), render_rgb=True)
    tracer = mjw.PathTracer(rc, max_bounces=3)
    tracer.render(m, d, samples=64)

    hdr = tracer.hdr.numpy()
    # Indirect light should be present but dimmer than the emitter radiance.
    indirect = hdr[(hdr > 0.01) & (hdr < 0.5)]
    self.assertGreater(indirect.size, 20)
    self.assertGreater(float(np.mean(hdr[hdr > 0.01])), 0.01)
    # No directly visible emitter or light.
    self.assertLess(float(hdr.max()), 0.6)

  def test_furnace_energy_conservation(self):
    # A white Lambertian surface under a uniform unit environment must return
    # exactly the environment radiance: every bounce is lossless and the
    # diffuse/GGX lobe mixture is unbiased.
    mjm, _, m, d = test_data.fixture(xml=_FURNACE_XML)
    rc = mjw.create_render_context(mjm, cam_res=(16, 16), render_rgb=True, background_color=(1.0, 1.0, 1.0, 1.0))
    tracer = mjw.PathTracer(rc, max_bounces=8, seed=0)
    tracer.render(m, d, samples=256)

    hdr = tracer.hdr.numpy()
    self.assertAlmostEqual(float(hdr.mean()), 1.0, delta=0.01)
    self.assertLess(float(hdr.std()), 0.05)

  def test_skybox_radiance(self):
    # Rays that miss all geometry must return the sRGB-decoded skybox color that
    # the legacy renderer writes for missed rays (identified by depth == 0).
    mjm, _, m, d = test_data.fixture(xml=_SKYBOX_XML)
    rc = mjw.create_render_context(mjm, cam_res=(32, 32), render_rgb=True, render_depth=True, render_skybox=True)
    mjw.render(m, d, rc)
    legacy = _unpack_rgb(rc.rgb_data.numpy()).astype(np.float64) / 255.0
    missed = rc.depth_data.numpy() == 0.0
    self.assertGreater(int(missed.sum()), 50)

    tracer = mjw.PathTracer(rc)
    tracer.render(m, d, samples=1)
    hdr = tracer.hdr.numpy()
    np.testing.assert_allclose(hdr[missed], _srgb_to_linear_np(legacy[missed]), atol=0.02)

  def test_splat_composited_over_geometry(self):
    mjm, _, m, d = test_data.fixture(xml=_SPLAT_XML)
    rc = mjw.create_render_context(mjm, cam_res=(16, 16), render_rgb=True, **_splat_kwargs(1.0))
    tracer = mjw.PathTracer(rc, max_bounces=1)
    tracer.render(m, d, samples=1)

    hdr = tracer.hdr.numpy().reshape(16, 16, 3)
    center = hdr[7:9, 7:9].mean(axis=(0, 1))
    self.assertGreater(float(center[0]), 0.5)
    self.assertGreater(float(center[0]), 4.0 * float(center[1]))

  def test_splat_behind_geometry_is_occluded(self):
    mjm, _, m, d = test_data.fixture(xml=_SPLAT_XML)
    rc = mjw.create_render_context(mjm, cam_res=(16, 16), render_rgb=True, **_splat_kwargs(-0.5))
    tracer = mjw.PathTracer(rc, max_bounces=1)
    tracer.render(m, d, samples=1)

    hdr = tracer.hdr.numpy()
    self.assertLess(float(hdr.max()), 0.05)

  def test_depth_matches_renderer(self):
    mjm, _, m, d = test_data.fixture(xml=_ANALYTIC_XML)
    rc = mjw.create_render_context(mjm, cam_res=(16, 16), render_rgb=True, render_depth=True)
    mjw.render(m, d, rc)
    depth_ref = rc.depth_data.numpy().copy()

    tracer = mjw.PathTracer(rc)
    tracer.render(m, d, samples=1)
    np.testing.assert_allclose(rc.depth_data.numpy(), depth_ref, rtol=1e-5)

  def test_deterministic_and_reset(self):
    mjm, _, m, d = test_data.fixture(xml=_GI_XML)
    rc = mjw.create_render_context(mjm, cam_res=(16, 16), render_rgb=True)
    tracer = mjw.PathTracer(rc, max_bounces=2, seed=7)
    tracer.render(m, d, samples=8)
    first = tracer.hdr.numpy().copy()
    tracer.render(m, d, samples=8)
    np.testing.assert_array_equal(first, tracer.hdr.numpy())

  def test_progressive_accumulation(self):
    mjm, _, m, d = test_data.fixture(xml=_GI_XML)
    rc = mjw.create_render_context(mjm, cam_res=(16, 16), render_rgb=True)
    tracer = mjw.PathTracer(rc, max_bounces=2)
    for _ in range(3):
      tracer.render(m, d, samples=4, accumulate=True)
    self.assertEqual(tracer.sample_count, 12)
    tracer.reset()
    self.assertEqual(tracer.sample_count, 0)
    self.assertEqual(float(np.abs(tracer.hdr.numpy()).max()), 0.0)

  def test_nworld(self):
    mjm, _, m, d = test_data.fixture(xml=_ANALYTIC_XML, nworld=3)
    rc = mjw.create_render_context(mjm, nworld=3, cam_res=(16, 16), render_rgb=True)
    tracer = mjw.PathTracer(rc)
    tracer.render(m, d, samples=1)
    hdr = tracer.hdr.numpy()
    self.assertEqual(hdr.shape, (3, 16 * 16, 3))
    np.testing.assert_allclose(hdr, 0.5, atol=1e-3)


if __name__ == "__main__":
  absltest.main()
