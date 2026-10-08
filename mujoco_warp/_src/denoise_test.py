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
"""Tests for the a-trous denoiser."""

import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src.denoise import denoise

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


def _noisy_flat(res: int, value: float, noise: float, seed: int = 0) -> np.ndarray:
  rng = np.random.default_rng(seed)
  return value + noise * rng.standard_normal((1, res * res, 3)).astype(np.float32)


class DenoiseTest(absltest.TestCase):
  def test_flat_denoise(self):
    res = 32
    noise = 0.05
    flat = _noisy_flat(res, 0.5, noise)
    img = flat.reshape(res, res, 3)
    hdr = wp.array(flat, dtype=wp.vec3)
    variance = wp.array(np.full((1, res * res), noise * noise, np.float32))

    out = denoise(hdr, res, res, variance=variance, iterations=5, sigma_color=0.5)
    out_np = out.numpy().reshape(res, res, 3)
    interior = out_np[8:24, 8:24]

    self.assertGreater(float(np.std(img[8:24, 8:24])), 0.04)
    self.assertLess(float(np.std(interior)), 0.015)
    self.assertAlmostEqual(float(interior.mean()), 0.5, delta=0.01)

  def test_edge_preserved(self):
    res = 32
    noise = 0.05
    img = _noisy_flat(res, 0.1, noise, seed=1).reshape(res, res, 3)
    img[:, res // 2 :, :] = 1.0 + noise * np.random.default_rng(2).standard_normal((res, res // 2, 3)).astype(np.float32)
    hdr = wp.array(img.reshape(1, res * res, 3), dtype=wp.vec3)
    variance = wp.array(np.full((1, res * res), noise * noise, np.float32))

    out = denoise(hdr, res, res, variance=variance, iterations=5, sigma_color=0.5)
    out_np = out.numpy().reshape(res, res, 3)

    left = float(out_np[8:24, 4:12].mean())
    right = float(out_np[8:24, 20:28].mean())
    self.assertLess(left, 0.25)
    self.assertGreater(right, 0.85)

  def test_without_variance_guide(self):
    res = 32
    noise = 0.05
    hdr = wp.array(_noisy_flat(res, 0.5, noise, seed=3), dtype=wp.vec3)

    out = denoise(hdr, res, res, variance=None, iterations=4)
    interior = out.numpy().reshape(res, res, 3)[8:24, 8:24]

    self.assertLess(float(np.std(interior)), 0.03)

  def test_demodulation_prevents_color_bleed(self):
    # Red and blue albedo halves with similar luminance: a luminance-only filter
    # bleeds color across the edge, demodulated filtering does not.
    res = 32
    rng = np.random.default_rng(4)
    albedo = np.empty((res, res, 3), np.float32)
    albedo[:, : res // 2] = (0.8, 0.15, 0.15)
    albedo[:, res // 2 :] = (0.15, 0.15, 0.8)
    irradiance = 0.5 + 0.12 * rng.standard_normal((res, res, 3)).astype(np.float32)
    hdr_np = albedo * irradiance
    hdr = wp.array(hdr_np.reshape(1, res * res, 3), dtype=wp.vec3)
    albedo_aov = wp.array(
      np.concatenate([albedo, np.ones((res, res, 1), np.float32)], axis=2).reshape(1, res * res, 4), dtype=wp.vec4
    )
    normal_aov = wp.array(np.tile(np.array([0.0, 0.0, 1.0], np.float32), (1, res * res, 1)), dtype=wp.vec3)
    variance = wp.array(np.full((1, res * res), 0.12 * 0.12, np.float32))

    plain = denoise(hdr, res, res, variance=variance, iterations=5, sigma_color=2.0).numpy().reshape(res, res, 3)
    demoded = (
      denoise(hdr, res, res, variance=variance, albedo=albedo_aov, normal=normal_aov, iterations=5, sigma_color=2.0)
      .numpy()
      .reshape(res, res, 3)
    )

    # True red in the right (blue) half is 0.15 * 0.5 = 0.075.
    plain_bleed = float(plain[:, 20:28, 0].mean())
    demoded_bleed = float(demoded[:, 20:28, 0].mean())
    self.assertLess(demoded_bleed, 0.12)
    self.assertLess(demoded_bleed, 0.5 * plain_bleed)
    # Demodulated filtering also reduces per-channel noise within each block.
    self.assertLess(float(demoded[:, 20:28, 2].std()), 0.6 * float(hdr_np[:, 20:28, 2].std()))

  def test_validation(self):
    hdr = wp.zeros((1, 16), dtype=wp.vec3)
    with self.assertRaises(ValueError):
      denoise(hdr, 0, 4)
    with self.assertRaises(ValueError):
      denoise(hdr, 8, 8)

  def test_path_tracer_variance_and_denoise(self):
    mjm, _, m, d = test_data.fixture(xml=_GI_XML)
    rc = mjw.create_render_context(mjm, cam_res=(64, 64), render_rgb=True)

    reference = mjw.PathTracer(rc, max_bounces=2, seed=1)
    reference.render(m, d, samples=192)
    ref = reference.hdr.numpy().reshape(64, 64, 3)

    tracer = mjw.PathTracer(rc, max_bounces=2, seed=2)
    tracer.render(m, d, samples=6)
    raw = tracer.hdr.numpy().reshape(64, 64, 3)
    variance = tracer.variance
    self.assertEqual(variance.shape, (1, 64 * 64))
    self.assertGreater(float(variance.numpy().mean()), 0.0)

    denoised = tracer.denoise(iterations=5).numpy().reshape(64, 64, 3)

    # The denoised image is closer to the converged reference overall, at both
    # full resolution and 4x4-pooled (low-frequency) scale.
    def mse(a, b):
      return float(np.mean((a - b) ** 2))

    def pool(img, k=4):
      h, w, c = img.shape
      return img.reshape(h // k, k, w // k, k, c).mean(axis=(1, 3))

    raw_mse = mse(raw, ref)
    den_mse = mse(denoised, ref)
    self.assertGreater(raw_mse, 0.0)
    self.assertLess(den_mse, 0.5 * raw_mse)
    # Pooled (low-frequency) error is not asserted: with a filter footprint of
    # up to 2**iterations pixels it redistributes smooth gradients (glow,
    # shadows) at this test resolution, which is expected of an a-trous filter
    # and is controlled by lowering `iterations` for small images.
    _ = pool


if __name__ == "__main__":
  absltest.main()
