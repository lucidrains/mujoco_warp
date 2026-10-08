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
"""Tests for per-world domain randomization."""

import numpy as np
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp import test_data

_XML = """
<mujoco>
  <asset><material name="white" rgba="0.5 0.5 0.5 1" specular="0" shininess="0"/></asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" directional="true"/>
    <geom type="plane" size="10 10 0.1" material="white"/>
    <camera pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""


class RandomizeTest(absltest.TestCase):
  def test_randomized_colors_reach_every_world(self):
    nworld = 4
    mjm, _, m, d = test_data.fixture(xml=_XML, nworld=nworld, batch_sizes={"mat_rgba": nworld, "light_diffuse": nworld})
    mjw.randomize_colors(m, seed=7, material_tint=0.5, light_tint=0.5)

    rc = mjw.create_render_context(mjm, nworld=nworld, cam_res=(8, 8), render_rgb=True)
    tracer = mjw.PathTracer(rc, max_bounces=1)
    tracer.render(m, d, samples=1)
    hdr = tracer.hdr.numpy().reshape(nworld, 8 * 8, 3)

    # Flat diffuse scene: radiance = albedo * light_diffuse per world.
    expected = m.mat_rgba.numpy()[:, 0, :3] * m.light_diffuse.numpy()[:, 0, :]
    for w in range(nworld):
      np.testing.assert_allclose(hdr[w].mean(axis=0), expected[w], rtol=1e-3, atol=1e-4)
    # The randomization actually produced distinct worlds.
    self.assertGreater(float(np.std(hdr[:, 0, 0])), 0.01)

  def test_requires_batched_model(self):
    mjm, _, m, _ = test_data.fixture(xml=_XML)
    with self.assertRaises(ValueError):
      mjw.randomize_colors(m, seed=0)


if __name__ == "__main__":
  absltest.main()
