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
"""Tests for differentiable rendering and physics composition."""

import mujoco
import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp import test_data

_ANALYTIC_XML = """
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

_SPHERE_XML = """
<mujoco>
  <asset>
    <material name="white" rgba="0.5 0.5 0.5 1" specular="0" shininess="0"/>
    <material name="ball" rgba="0.2 0.6 0.3 1" specular="0" shininess="0"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" directional="true"/>
    <geom type="plane" size="10 10 0.1" material="white"/>
    <geom type="sphere" name="ball" pos="0.2 0 0.5" size="0.15" material="ball"/>
    <camera pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""

_PENDULUM_XML = """
<mujoco>
  <asset>
    <material name="white" rgba="0.5 0.5 0.5 1" specular="0" shininess="0"/>
    <material name="ball" rgba="0.2 0.6 0.3 1" specular="0" shininess="0"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1" directional="true"/>
    <geom type="plane" size="10 10 0.1" material="white"/>
    <body>
      <joint type="hinge" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
      <geom type="sphere" pos="0.5 0 0.6" size="0.2" material="ball"/>
    </body>
    <camera pos="0 0 2" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""


def _zeroes(shape):
  return wp.zeros(shape, dtype=wp.vec3)


class DiffRenderTest(absltest.TestCase):
  def test_forward_analytic(self):
    mjm, _, m, d = test_data.fixture(xml=_ANALYTIC_XML)
    rc = mjw.create_render_context(mjm, cam_res=(32, 32), render_rgb=True)
    renderer = mjw.DifferentiableRenderer(m, rc)

    img = renderer.render(m, d).numpy()

    self.assertEqual(img.shape, (1, 32 * 32, 3))
    np.testing.assert_allclose(img, 0.5, atol=1e-4)

  def test_albedo_gradient(self):
    mjm, _, m, d = test_data.fixture(xml=_ANALYTIC_XML)
    rc = mjw.create_render_context(mjm, cam_res=(16, 16), render_rgb=True)
    renderer = mjw.DifferentiableRenderer(m, rc)
    mat_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_MATERIAL, "white")

    m.mat_rgba.requires_grad = True
    with wp.Tape() as tape:
      img = renderer.render(m, d)
      loss = renderer.loss(img, _zeroes(img.shape))
    tape.backward(loss)

    # r = albedo * light_diffuse * cos(theta); dMSE/dalbedo = 2 * r = 1.0.
    grad = m.mat_rgba.grad.numpy()[0, mat_id]
    np.testing.assert_allclose(grad[:3], 1.0, atol=2e-3)
    self.assertAlmostEqual(float(grad[3]), 0.0, delta=1e-4)

  def test_light_gradient(self):
    mjm, _, m, d = test_data.fixture(xml=_ANALYTIC_XML)
    rc = mjw.create_render_context(mjm, cam_res=(16, 16), render_rgb=True)
    renderer = mjw.DifferentiableRenderer(m, rc)

    m.light_diffuse.requires_grad = True
    with wp.Tape() as tape:
      img = renderer.render(m, d)
      loss = renderer.loss(img, _zeroes(img.shape))
    tape.backward(loss)

    # r = albedo * light; dMSE/dlight = 2 * r * albedo = 0.5.
    np.testing.assert_allclose(m.light_diffuse.grad.numpy()[0, 0], 0.5, atol=2e-3)

  def test_geom_position_gradient_matches_fd(self):
    mjm, _, m, d = test_data.fixture(xml=_SPHERE_XML)
    rc = mjw.create_render_context(mjm, cam_res=(16, 16), render_rgb=True)
    renderer = mjw.DifferentiableRenderer(m, rc)
    ball_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball")

    target = _zeroes((1, 16 * 16))
    d.geom_xpos.requires_grad = True
    with wp.Tape() as tape:
      img = renderer.render(m, d)
      loss = renderer.loss(img, target)
    tape.backward(loss)
    grad = d.geom_xpos.grad.numpy()[0, ball_id].copy()

    eps = 1e-3
    xpos = d.geom_xpos.numpy().copy()
    fd = []
    for comp in range(3):
      plus = xpos.copy()
      plus[0, ball_id, comp] += eps
      d.geom_xpos.assign(plus)
      lp = float(renderer.loss(renderer.render(m, d), target).numpy())
      minus = xpos.copy()
      minus[0, ball_id, comp] -= eps
      d.geom_xpos.assign(minus)
      lm = float(renderer.loss(renderer.render(m, d), target).numpy())
      fd.append((lp - lm) / (2.0 * eps))
    d.geom_xpos.assign(xpos)

    np.testing.assert_allclose(grad, fd, rtol=0.05, atol=1e-3)

  def test_camera_gradient_matches_fd(self):
    mjm, _, m, d = test_data.fixture(xml=_SPHERE_XML)
    rc = mjw.create_render_context(mjm, cam_res=(16, 16), render_rgb=True)
    renderer = mjw.DifferentiableRenderer(m, rc)

    target = _zeroes((1, 16 * 16))
    d.cam_xpos.requires_grad = True
    with wp.Tape() as tape:
      img = renderer.render(m, d)
      loss = renderer.loss(img, target)
    tape.backward(loss)
    grad = d.cam_xpos.grad.numpy()[0, 0, 0]

    eps = 1e-3
    cam = d.cam_xpos.numpy().copy()
    plus = cam.copy()
    plus[0, 0, 0] += eps
    d.cam_xpos.assign(plus)
    lp = float(renderer.loss(renderer.render(m, d), target).numpy())
    minus = cam.copy()
    minus[0, 0, 0] -= eps
    d.cam_xpos.assign(minus)
    lm = float(renderer.loss(renderer.render(m, d), target).numpy())
    d.cam_xpos.assign(cam)

    self.assertAlmostEqual(float(grad), (lp - lm) / (2.0 * eps), delta=1e-3)

  def test_compose_with_physics(self):
    # dL/dq = (dL/dxpos) . (dxpos/dq): renderer adjoints chained with the
    # physics Jacobian from MuJoCo Warp kinematics, checked against the full
    # finite difference of the composite objective.
    mjm, _, m, d = test_data.fixture(xml=_PENDULUM_XML)
    rc = mjw.create_render_context(mjm, cam_res=(16, 16), render_rgb=True)
    renderer = mjw.DifferentiableRenderer(m, rc)

    def forward_loss(q):
      d.qpos.assign(np.array([q], np.float32))
      mjw.fwd_position(m, d)
      img = renderer.render(m, d)
      return img

    def xpos_at(q):
      d.qpos.assign(np.array([q], np.float32))
      mjw.fwd_position(m, d)
      return d.geom_xpos.numpy()[0].copy()

    q0 = 0.4
    target = forward_loss(0.0)

    d.qpos.assign(np.array([q0], np.float32))
    mjw.fwd_position(m, d)
    d.geom_xpos.requires_grad = True
    with wp.Tape() as tape:
      img = renderer.render(m, d)
      loss = renderer.loss(img, target)
    tape.backward(loss)
    grad_xpos = d.geom_xpos.grad.numpy()[0].copy()

    eps = 1e-4
    jacobian = (xpos_at(q0 + eps) - xpos_at(q0 - eps)) / (2.0 * eps)
    chain = float(np.sum(jacobian * grad_xpos))

    def loss_at(q):
      img = forward_loss(q)
      return float(renderer.loss(img, target).numpy())

    fd = (loss_at(q0 + eps) - loss_at(q0 - eps)) / (2.0 * eps)
    self.assertGreater(abs(fd), 1e-4)
    self.assertAlmostEqual(chain, fd, delta=0.05 * abs(fd) + 1e-5)


if __name__ == "__main__":
  absltest.main()
