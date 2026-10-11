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
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src import randomize

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


class PhysicsRandomizerTest(absltest.TestCase):
  _XML = """
<mujoco>
  <option integrator="Euler"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="b1" pos="0 0 0.1">
      <joint name="j1" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1" condim="3"/>
    </body>
    <body name="b2" pos="0.5 0 0.1">
      <joint name="j2" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1" condim="3"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1"/>
  </actuator>
</mujoco>
"""

  def _fixture(self, nworld=8, **kwargs):
    _, _, m, d = test_data.fixture(
      xml=self._XML,
      nworld=nworld,
      batch_sizes=randomize.physics_batch_sizes(nworld),
      **kwargs,
    )
    return m, d

  def test_batched_gravity_io(self):
    _, _, m, _ = test_data.fixture(xml=self._XML, nworld=5, batch_sizes={"opt.gravity": 5})
    g = m.opt.gravity.numpy()
    self.assertEqual(g.shape[0], 5)
    np.testing.assert_allclose(g[:, 2], -9.81)

  def test_determinism(self):
    m, _ = self._fixture()
    dr = randomize.PhysicsRandomizer(m, body=0.2, friction=0.3, gain=0.1)
    dr.sample(seed=1, epoch=0)
    a = m.body_mass.numpy().copy()
    dr.sample(seed=1, epoch=0)
    np.testing.assert_array_equal(m.body_mass.numpy(), a)
    dr.sample(seed=1, epoch=1)
    self.assertGreater(np.abs(m.body_mass.numpy() - a).max(), 0.0)

  def test_bounds_anchored_to_nominal(self):
    m, _ = self._fixture()
    dr = randomize.PhysicsRandomizer(
      m,
      body=0.25,
      inertia=0.1,
      friction=0.3,
      solref=0.2,
      damping=0.4,
      armature=0.5,
      frictionloss=0.6,
      stiffness=0.3,
      gain=0.2,
      gravity=0.1,
    )
    nominal = {e.name: e.nominal.numpy().copy() for e in dr._entries}
    for epoch in range(4):
      dr.sample(seed=7, epoch=epoch)
      for e in dr._entries:
        amount = e.amount
        val = e.target.numpy()
        base = nominal[e.name]
        self.assertLessEqual(np.abs(val - base).max(), amount * np.abs(base).max() + 1e-6, e.name)
        lo = np.minimum(base * (1.0 - amount), base * (1.0 + amount))
        hi = np.maximum(base * (1.0 - amount), base * (1.0 + amount))
        self.assertTrue(np.all(val >= lo - 1e-6), e.name)
        self.assertTrue(np.all(val <= hi + 1e-6), e.name)
      # gain slots: only gainprm[0] and biasprm[1] move
      np.testing.assert_array_equal(m.actuator_gainprm.numpy()[..., 1:], 0.0)
      np.testing.assert_array_equal(m.actuator_biasprm.numpy()[..., ::2], 0.0)

  def test_per_world_independence(self):
    m, _ = self._fixture(nworld=8)
    dr = randomize.PhysicsRandomizer(m, body=0.2, friction=0.3)
    dr.sample(seed=3, epoch=0)
    mass = m.body_mass.numpy()
    self.assertGreater(np.unique(mass[:, 1]).size, 1)
    self.assertGreater(np.unique(m.geom_friction.numpy()[:, 1, 0]).size, 1)

  def test_mask_only_touches_selected_worlds(self):
    m, _ = self._fixture(nworld=4)
    dr = randomize.PhysicsRandomizer(m, body=0.2)
    dr.sample(seed=0, epoch=0)
    before = m.body_mass.numpy().copy()
    mask = wp.array([True, False, False, True], dtype=bool)
    dr.sample(seed=0, epoch=1, worlds=mask)
    after = m.body_mass.numpy()
    np.testing.assert_array_equal(after[1], before[1])
    np.testing.assert_array_equal(after[2], before[2])
    self.assertGreater(np.abs(after[0] - before[0]).max(), 0.0)
    self.assertGreater(np.abs(after[3] - before[3]).max(), 0.0)

    # the same (seed, epoch, world) draw occurs regardless of the mask
    dr.sample(seed=0, epoch=1)
    np.testing.assert_array_equal(m.body_mass.numpy()[0], after[0])
    np.testing.assert_array_equal(m.body_mass.numpy()[3], after[3])
    # and a masked-out world is left untouched
    before2 = m.body_mass.numpy().copy()
    dr.sample(seed=0, epoch=2, worlds=wp.array([True, False, False, False], dtype=bool))
    np.testing.assert_array_equal(m.body_mass.numpy()[1:], before2[1:])

  def test_mask_length_error(self):
    m, _ = self._fixture(nworld=4)
    dr = randomize.PhysicsRandomizer(m, body=0.2)
    with self.assertRaises(ValueError):
      dr.sample(seed=0, epoch=0, worlds=wp.ones(3, dtype=bool))

  def test_alpha_zero_keeps_nominal(self):
    m, _ = self._fixture(nworld=8)
    dr = randomize.PhysicsRandomizer(m, body=0.2, friction=0.3)
    nominal = {e.name: e.nominal.numpy().copy() for e in dr._entries}
    dr.alpha.assign(np.zeros(8, dtype=np.float32))
    dr.sample(seed=3, epoch=0)
    for e in dr._entries:
      np.testing.assert_array_equal(e.target.numpy(), nominal[e.name])

  def test_alpha_scales_width_per_world(self):
    m, _ = self._fixture(nworld=8)
    dr = randomize.PhysicsRandomizer(m, body=0.2)
    entry = next(e for e in dr._entries if e.name == "body_mass")
    base = entry.nominal.numpy()
    dr.alpha.assign(np.where(np.arange(8) < 4, 0.0, 2.0).astype(np.float32))
    dr.sample(seed=7, epoch=0)
    val = m.body_mass.numpy()
    np.testing.assert_array_equal(val[:4], base[:4])
    self.assertGreater(np.abs(val[4:] - base[4:]).max(), 0.0)
    self.assertLessEqual(np.abs(val[4:] - base[4:]).max(), 2.0 * entry.amount * np.abs(base[4:]).max() + 1e-6)

  def test_adapt_updates_alpha(self):
    m, _ = self._fixture(nworld=6)
    dr = randomize.PhysicsRandomizer(m, body=0.2)
    score = np.array([1.0, 0.0, 0.5, 1.0, 0.0, 0.5], dtype=np.float32)
    dr.adapt(score, expand=1.1, shrink=0.5, low=0.5, high=0.5)
    np.testing.assert_allclose(dr.alpha.numpy(), [1.1, 0.5, 1.0, 1.1, 0.5, 1.0], rtol=1e-6)

    # subset mask touches only selected worlds
    dr.adapt(score, worlds=wp.array([True, False, False, False, False, False], dtype=bool), expand=2.0)
    np.testing.assert_allclose(dr.alpha.numpy(), [2.2, 0.5, 1.0, 1.1, 0.5, 1.0], rtol=1e-6)

    # clamps
    dr.adapt(np.ones(6, dtype=np.float32), expand=10.0, cap=4.0)
    np.testing.assert_allclose(dr.alpha.numpy(), 4.0)
    dr.adapt(np.zeros(6, dtype=np.float32), shrink=0.1, floor=0.5)
    np.testing.assert_allclose(dr.alpha.numpy(), 0.5)

  def test_adapt_validation(self):
    m, _ = self._fixture(nworld=4)
    dr = randomize.PhysicsRandomizer(m, body=0.2)
    with self.assertRaises(ValueError):
      dr.adapt(np.ones(3, dtype=np.float32))
    with self.assertRaises(ValueError):
      dr.adapt(np.ones(4, dtype=np.float32), low=0.8, high=0.2)
    with self.assertRaises(ValueError):
      dr.adapt(np.ones(4, dtype=np.float32), expand=0.0)
    with self.assertRaises(ValueError):
      dr.adapt(np.ones(4, dtype=np.float32), shrink=-1.0)
    with self.assertRaises(ValueError):
      dr.adapt(np.ones(4, dtype=np.float32), floor=1.0, cap=0.5)
    with self.assertRaises(ValueError):
      dr.adapt(np.ones(4, dtype=np.float32), worlds=wp.ones(3, dtype=bool))

  def test_adaptive_reset_loop(self):
    # ADR loop: score -> adapt -> masked resample, no host-side bookkeeping
    m, d = self._fixture(nworld=4)
    dr = randomize.PhysicsRandomizer(m, body=0.2, friction=0.2)
    dr.sample(seed=0, epoch=0)

    done = wp.array([True, False, False, True], dtype=bool)
    score = wp.array(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32), dtype=float)
    dr.adapt(score, worlds=done)
    np.testing.assert_allclose(dr.alpha.numpy()[[0, 3]], 1.05, rtol=1e-6)
    np.testing.assert_allclose(dr.alpha.numpy()[[1, 2]], 1.0, rtol=1e-6)

    before = m.body_mass.numpy().copy()
    randomize.reset_worlds(m, d, done, np.zeros((4, m.nq), dtype=np.float32))
    dr.sample(seed=0, epoch=1, worlds=done)
    after = m.body_mass.numpy()
    np.testing.assert_array_equal(after[[1, 2]], before[[1, 2]])
    self.assertGreater(np.abs(after[[0, 3]] - before[[0, 3]]).max(), 0.0)

  _GAIN_XML = """
<mujoco>
  <option integrator="Euler"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1" friction="0.7 0.1 0.2"/>
    <body name="b1" pos="0 0 0.1">
      <joint name="j1" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1" friction="0.6 0.2 0.3"/>
    </body>
    <body name="b2" pos="0.5 0 0.1">
      <joint name="j2" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="j1" kp="20"/>
    <position joint="j2" kp="15"/>
  </actuator>
</mujoco>
"""

  def test_body_and_gain_groups_share_one_draw(self):
    nworld = 8
    _, _, m, _ = test_data.fixture(xml=self._GAIN_XML, nworld=nworld, batch_sizes=randomize.physics_batch_sizes(nworld))
    dr = randomize.PhysicsRandomizer(m, body=0.3, gain=0.3)
    dr.sample(seed=0, epoch=0)
    nominal = {e.name: e.nominal.numpy() for e in dr._entries}

    # body: mass and inertia scale by the same factor per world
    mass = m.body_mass.numpy()[:, 1:] / nominal["body_mass"][:, 1:]
    inertia = m.body_inertia.numpy()[:, 1:, 0] / nominal["body_inertia"][:, 1:, 0]
    np.testing.assert_allclose(mass, inertia)
    self.assertGreater(np.abs(mass - 1.0).max(), 0.0)

    # gain: gainprm[0] and biasprm[1] scale by the same factor per actuator
    gain = m.actuator_gainprm.numpy()[..., 0] / nominal["actuator_gainprm"][..., 0]
    bias = m.actuator_biasprm.numpy()[..., 1] / nominal["actuator_biasprm"][..., 1]
    np.testing.assert_allclose(gain, bias)
    self.assertGreater(np.abs(gain - 1.0).max(), 0.0)

  def test_friction_only_slide_changes(self):
    m, _ = self._fixture()
    dr = randomize.PhysicsRandomizer(m, friction=0.5)
    base = dr._entries[0].nominal.numpy()
    dr.sample(seed=0, epoch=0)
    friction = m.geom_friction.numpy()
    np.testing.assert_array_equal(friction[..., 1:], base[..., 1:])
    self.assertGreater(np.abs(friction[..., 0] - base[..., 0]).max(), 0.0)

  _GRAVITY_XML = """
<mujoco>
  <option integrator="Euler" gravity="1 -2 -9.81"/>
  <worldbody>
    <geom type="plane" size="5 5 0.1"/>
    <body name="b1" pos="0 0 0.1">
      <joint name="j1" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

  def test_gravity_shared_row_direction_preserved(self):
    # gravity batched at 1 while other fields are per-world must not overrun
    _, _, m, _ = test_data.fixture(xml=self._GRAVITY_XML, nworld=4, batch_sizes={"body_mass": 4})
    self.assertEqual(m.opt.gravity.shape[0], 1)
    nominal = m.opt.gravity.numpy()[0].copy()
    dr = randomize.PhysicsRandomizer(m, body=0.2, gravity=0.3)
    dr.sample(seed=0, epoch=0)

    self.assertEqual(m.opt.gravity.shape[0], 1)
    gravity = m.opt.gravity.numpy()[0]
    ratio = np.linalg.norm(gravity) / np.linalg.norm(nominal)
    self.assertGreaterEqual(ratio, 0.7 - 1e-6)
    self.assertLessEqual(ratio, 1.3 + 1e-6)
    np.testing.assert_allclose(gravity / np.linalg.norm(gravity), nominal / np.linalg.norm(nominal))

  _ACT_XML = """
<mujoco>
  <option integrator="Euler"/>
  <worldbody>
    <body name="b1" pos="0 0 0.1">
      <joint name="j1" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <general joint="j1" dyntype="integrator" gaintype="fixed" biastype="none" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

  def test_reset_worlds_resets_time_and_act(self):
    _, _, m, d = test_data.fixture(xml=self._ACT_XML, nworld=2)
    self.assertEqual(m.na, 1)
    d.time.assign(np.ones(2, dtype=np.float32))
    d.act.assign(np.ones((2, m.na), dtype=np.float32))
    randomize.reset_worlds(m, d, wp.array([True, False], dtype=bool), np.zeros((2, m.nq), dtype=np.float32))
    np.testing.assert_array_equal(d.time.numpy(), np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_array_equal(d.act.numpy(), np.array([[0.0], [1.0]], dtype=np.float32))

  def test_reset_worlds_rejects_wrong_device_shape(self):
    m, d = self._fixture(nworld=2)
    with self.assertRaises(ValueError):
      randomize.reset_worlds(m, d, wp.ones(2, dtype=bool), wp.zeros((2, m.nq + 1), dtype=float))

  def test_unknown_group_and_negative_amount(self):
    m, _ = self._fixture()
    with self.assertRaises(ValueError):
      randomize.PhysicsRandomizer(m, notagroup=0.1)
    with self.assertRaises(ValueError):
      randomize.PhysicsRandomizer(m, body=-0.1)

  def test_reset_worlds_subset(self):
    m, d = self._fixture(nworld=3)
    d.qpos.assign(np.arange(3 * m.nq, dtype=np.float32).reshape(3, m.nq))
    d.qvel.assign(np.ones((3, m.nv), dtype=np.float32))
    d.qacc_warmstart.assign(np.ones((3, m.nv), dtype=np.float32))
    done = wp.array([True, False, True], dtype=bool)
    qpos_init = np.full((3, m.nq), -1.0, dtype=np.float32)
    randomize.reset_worlds(m, d, done, qpos_init)
    np.testing.assert_array_equal(d.qpos.numpy()[[0, 2]], -1.0)
    np.testing.assert_array_equal(d.qpos.numpy()[1], np.arange(m.nq, 2 * m.nq, dtype=np.float32))
    np.testing.assert_array_equal(d.qvel.numpy()[0], 0.0)
    np.testing.assert_array_equal(d.qacc_warmstart.numpy()[0], 0.0)
    np.testing.assert_array_equal(d.qvel.numpy()[1], 1.0)

  def test_physics_divergence(self):
    m, d = self._fixture(nworld=2)
    dr = randomize.PhysicsRandomizer(m, body=0.5, friction=0.5, damping=0.5)
    dr.sample(seed=5, epoch=0)
    d.qpos.assign(np.tile(np.array([0.0, 0.0], dtype=np.float32), (2, 1)))
    d.qvel.assign(np.tile(np.array([0.5, 0.5], dtype=np.float32), (2, 1)))
    d.ctrl.assign(np.ones((2, 1), dtype=np.float32))
    for _ in range(300):
      mjw.step(m, d)
    qpos = d.qpos.numpy()
    self.assertGreater(np.abs(qpos[0] - qpos[1]).max(), 1e-3)

  def test_requires_nworld_batch(self):
    # fields batched at a single shared row still randomize, but identically across worlds
    _, _, m, _ = test_data.fixture(xml=self._XML, nworld=4, batch_sizes=randomize.physics_batch_sizes(1))
    dr = randomize.PhysicsRandomizer(m, body=0.2)
    dr.sample(seed=0, epoch=0)
    self.assertEqual(m.body_mass.shape[0], 1)


class ActionLatencyTest(absltest.TestCase):
  _XML = """
<mujoco>
  <option integrator="Euler"/>
  <worldbody>
    <body pos="0 0 0.1">
      <joint name="j1" type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1"/>
  </actuator>
</mujoco>
"""

  def _fixture(self, nworld=2):
    _, _, m, d = test_data.fixture(xml=self._XML, nworld=nworld)
    return m, d

  def test_delay_semantics(self):
    m, d = self._fixture(nworld=2)
    lat = randomize.ActionLatency(m, d, max_delay=2)
    lat.delays.assign(np.array([0, 2], dtype=np.int32))
    seen = [[], []]
    for t in range(4):
      u = np.full((2, 1), float(t), dtype=np.float32)
      lat.apply(wp.array(u, dtype=float))
      got = d.ctrl.numpy()[:, 0]
      seen[0].append(float(got[0]))
      seen[1].append(float(got[1]))
    np.testing.assert_allclose(seen[0], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(seen[1], [0.0, 0.0, 0.0, 1.0])

  def test_resample_bounds_and_determinism(self):
    m, d = self._fixture(nworld=16)
    lat = randomize.ActionLatency(m, d, max_delay=4)
    lat.resample(seed=1, epoch=0)
    a = lat.delays.numpy().copy()
    lat.resample(seed=1, epoch=0)
    np.testing.assert_array_equal(lat.delays.numpy(), a)
    lat.resample(seed=1, epoch=1)
    self.assertGreater(np.abs(lat.delays.numpy() - a).max(), 0)
    self.assertTrue(np.all(a >= 0) and np.all(a <= 4))
    self.assertGreater(np.unique(a).size, 1)

  def test_resample_mask(self):
    m, d = self._fixture(nworld=4)
    lat = randomize.ActionLatency(m, d, max_delay=4)
    lat.resample(seed=2, epoch=0)
    before = lat.delays.numpy().copy()
    lat.resample(seed=2, epoch=1, worlds=wp.array([True, False, False, True], dtype=bool))
    after = lat.delays.numpy()
    np.testing.assert_array_equal(after[1:3], before[1:3])

  def test_reset_clears_delay_pipeline(self):
    m, d = self._fixture(nworld=2)
    lat = randomize.ActionLatency(m, d, max_delay=2)
    lat.delays.assign(np.array([2, 0], dtype=np.int32))
    lat.apply(wp.array(np.full((2, 1), 5.0, dtype=np.float32), dtype=float))
    lat.reset()
    lat.apply(wp.array(np.full((2, 1), 1.0, dtype=np.float32), dtype=float))
    # no pre-reset history: the delayed world sees zeros, the immediate world sees the input
    np.testing.assert_allclose(d.ctrl.numpy()[0], 0.0)
    np.testing.assert_allclose(d.ctrl.numpy()[1], 1.0)

  def test_invalid(self):
    m, d = self._fixture()
    with self.assertRaises(ValueError):
      randomize.ActionLatency(m, d, max_delay=-1)
    lat = randomize.ActionLatency(m, d, max_delay=2)
    with self.assertRaises(ValueError):
      lat.resample(seed=0, epoch=0, worlds=wp.ones(3, dtype=bool))


class ActionLatencyHoldTest(absltest.TestCase):
  _XML = ActionLatencyTest._XML

  def _fixture(self, nworld=2):
    _, _, m, d = test_data.fixture(xml=self._XML, nworld=nworld)
    return m, d

  def test_hold_reduces_update_rate(self):
    m, d = self._fixture()
    lat = randomize.ActionLatency(m, d, max_delay=0, max_hold=4, seed=3)
    vals = []
    for t in range(60):
      lat.apply(wp.array(np.full((2, 1), float(t), dtype=np.float32), dtype=float))
      vals.append(d.ctrl.numpy()[:, 0].copy())
    vals = np.array(vals)
    # held commands are past inputs: nondecreasing, never ahead of the input
    self.assertTrue(np.all(vals <= np.arange(60)[:, None] + 1e-6))
    self.assertTrue(np.all(np.diff(vals[:, 0]) >= -1e-6))
    self.assertLess(int((np.diff(vals[:, 0]) != 0).sum()), 59)

  def test_hold_determinism_and_seed(self):
    def run(seed):
      m, d = self._fixture()
      lat = randomize.ActionLatency(m, d, max_delay=0, max_hold=4, seed=seed)
      out = []
      for t in range(30):
        lat.apply(wp.array(np.full((2, 1), float(t), dtype=np.float32), dtype=float))
        out.append(d.ctrl.numpy()[:, 0].copy())
      return np.array(out)

    np.testing.assert_array_equal(run(5), run(5))
    self.assertGreater(np.abs(run(5) - run(6)).max(), 0.0)

  def test_reset_makes_next_apply_fresh(self):
    m, d = self._fixture()
    lat = randomize.ActionLatency(m, d, max_delay=0, max_hold=4, seed=0)
    for t in range(5):
      lat.apply(wp.array(np.full((2, 1), float(t), dtype=np.float32), dtype=float))
    lat.reset()
    lat.apply(wp.array(np.full((2, 1), 42.0, dtype=np.float32), dtype=float))
    np.testing.assert_allclose(d.ctrl.numpy(), 42.0)


class ObservationModelTest(absltest.TestCase):
  def test_noise_statistics(self):
    om = randomize.ObservationModel(4096, 1, noise=0.5, seed=3)
    obs = wp.zeros((4096, 1), dtype=float)
    vals = np.concatenate([om.corrupt(obs).numpy() for _ in range(4)], axis=0)
    self.assertLess(abs(float(vals.mean())), 0.05)
    self.assertLess(abs(float(vals.std()) - 0.5), 0.05)

  def test_quantization(self):
    om = randomize.ObservationModel(8, 1, quantum=0.25, seed=0)
    obs_np = np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(8, 1)
    out = om.corrupt(wp.array(obs_np, dtype=float)).numpy()
    np.testing.assert_allclose(out / 0.25, np.round(out / 0.25), atol=1e-5)
    self.assertTrue(np.all(np.abs(out - obs_np) <= 0.125 + 1e-6))

  def test_dropout_holds_previous(self):
    om = randomize.ObservationModel(4, 1, dropout=1.0, seed=0)
    obs = wp.array(np.ones((4, 1), dtype=np.float32), dtype=float)
    np.testing.assert_array_equal(om.corrupt(obs).numpy(), 0.0)

  def test_hold_and_determinism(self):
    def run(seed):
      om = randomize.ObservationModel(2, 1, max_hold=3, seed=seed)
      vals = []
      for t in range(40):
        obs = wp.array(np.full((2, 1), float(t), dtype=np.float32), dtype=float)
        vals.append(om.corrupt(obs).numpy()[:, 0].copy())
      return np.array(vals)

    a, b = run(1), run(1)
    np.testing.assert_array_equal(a, b)
    self.assertGreater(np.abs(a - run(2)).max(), 0.0)
    self.assertLess(int((np.diff(a[:, 0]) != 0).sum()), 39)

  def test_reset_clears_state(self):
    om = randomize.ObservationModel(2, 1, seed=0)
    obs = wp.array(np.full((2, 1), 7.0, dtype=np.float32), dtype=float)
    om.corrupt(obs)
    np.testing.assert_allclose(om._prev.numpy(), 7.0)
    om.reset()
    np.testing.assert_array_equal(om._prev.numpy(), 0.0)

  def test_validation(self):
    with self.assertRaises(ValueError):
      randomize.ObservationModel(2, 1, dropout=1.5)
    with self.assertRaises(ValueError):
      randomize.ObservationModel(2, 1, noise=-1.0)
    with self.assertRaises(ValueError):
      randomize.ObservationModel(2, 1, max_hold=-1)
    om = randomize.ObservationModel(2, 1)
    with self.assertRaises(ValueError):
      om.corrupt(wp.zeros((3, 1), dtype=float))
    with self.assertRaises(ValueError):
      om.reset(worlds=wp.ones(3, dtype=bool))


if __name__ == "__main__":
  absltest.main()
