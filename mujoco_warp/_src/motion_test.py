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
"""Tests for rolling shutter and camera motion blur."""

import mujoco
import numpy as np
from absl.testing import absltest

import mujoco_warp as mjw
from mujoco_warp import test_data

_XML = """
<mujoco>
  <asset>
    <material name="emit" rgba="1 1 1 1" specular="0" shininess="0" emission="1"/>
  </asset>
  <worldbody>
    <geom name="ball" type="sphere" pos="0 0 0.5" size="0.2" material="emit"/>
    <camera pos="0 0 1.5" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>
"""

_RES = 32


def _setup():
  mjm, _, m, d = test_data.fixture(xml=_XML)
  rc = mjw.create_render_context(mjm, cam_res=(_RES, _RES), render_rgb=True)
  return m, d, rc


def _move_camera(d, x: float):
  cam = d.cam_xpos.numpy()
  cam[0, 0, 0] = x
  d.cam_xpos.assign(cam)


def _image(tracer, res: int = _RES) -> np.ndarray:
  return tracer.hdr.numpy().reshape(res, res, 3)


def _row_centroids(img: np.ndarray, threshold: float = 0.05):
  lum = img.mean(axis=2)
  xs = np.arange(img.shape[1])
  out = []
  for y in range(img.shape[0]):
    row = lum[y]
    total = row.sum()
    if total > 0.5:
      out.append((y, float((xs * row).sum() / total)))
  return out


def _centroid_x(img: np.ndarray, rows=None, threshold: float = 0.05) -> float:
  lum = img.mean(axis=2)
  ys, xs = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
  mask = lum > threshold
  if rows is not None:
    row_mask = (ys >= rows[0]) & (ys < rows[1])
    mask = mask & row_mask
  weights = lum[mask]
  return float((xs[mask] * weights).sum() / weights.sum())


class MotionTest(absltest.TestCase):
  def test_static_camera_with_shutter_is_sharp(self):
    m, d, rc = _setup()
    tracer = mjw.PathTracer(rc, max_bounces=1, camera_motion=mjw.CameraMotion(exposure=0.5))
    tracer.render(m, d, samples=8)
    first = _image(tracer)
    tracer.render(m, d, samples=8)
    second = _image(tracer)

    # Same pose at both shutter endpoints: no blur.
    np.testing.assert_allclose(first, second, atol=1e-6)
    self.assertAlmostEqual(float(first.max()), 1.0, delta=0.05)

  def test_motion_blur_midpoint(self):
    m, d, rc = _setup()

    sharp = mjw.PathTracer(rc, max_bounces=1, seed=3)
    sharp.render(m, d, samples=2)
    x0 = _centroid_x(_image(sharp))
    _move_camera(d, 0.2)
    sharp.render(m, d, samples=2)
    x1 = _centroid_x(_image(sharp))
    self.assertGreater(abs(x0 - x1), 3.0)  # centroid moves opposite to the camera

    # A tracer that saw the previous frame pose blurs across the interval.
    m2, d2, rc2 = _setup()
    blur = mjw.PathTracer(rc2, max_bounces=1, seed=3, camera_motion=mjw.CameraMotion(exposure=0.8))
    blur.render(m2, d2, samples=1)
    _move_camera(d2, 0.2)
    blur.render(m2, d2, samples=64)
    blurred = _image(blur)

    xb = _centroid_x(blurred)
    lo, hi = min(x0, x1), max(x0, x1)
    self.assertGreater(xb, lo + 0.15 * (hi - lo))
    self.assertLess(xb, hi - 0.15 * (hi - lo))
    # Blur spreads the emitter: bright pixels average well below full radiance.
    self.assertLess(float(blurred[blurred > 0.05].mean()), 0.9)

  def test_rolling_shutter_shear(self):
    m, d, rc = _setup()

    sharp = mjw.PathTracer(rc, max_bounces=1, seed=4)
    sharp.render(m, d, samples=2)
    _move_camera(d, 0.25)
    sharp.render(m, d, samples=2)
    rows = _row_centroids(_image(sharp))
    self.assertGreater(len(rows), 4)
    self.assertLess(rows[0][1] - rows[-1][1], 1.0)  # global shutter: no shear

    m2, d2, rc2 = _setup()
    tracer = mjw.PathTracer(rc2, max_bounces=1, seed=4, camera_motion=mjw.CameraMotion(rolling_shutter=0.95))
    tracer.render(m2, d2, samples=1)
    _move_camera(d2, 0.25)
    tracer.render(m2, d2, samples=16)
    rows = _row_centroids(_image(tracer))
    self.assertGreater(len(rows), 4)
    # Camera translates +x between endpoints; later rows see the sphere shifted -x.
    self.assertGreater(rows[0][1] - rows[-1][1], 2.5)

  def test_deterministic(self):
    m, d, rc = _setup()
    m2, d2, rc2 = _setup()
    a = mjw.PathTracer(rc, max_bounces=1, seed=5, camera_motion=mjw.CameraMotion(exposure=0.6))
    b = mjw.PathTracer(rc2, max_bounces=1, seed=5, camera_motion=mjw.CameraMotion(exposure=0.6))
    a.render(m, d, samples=1)
    b.render(m2, d2, samples=1)
    _move_camera(d, 0.1)
    _move_camera(d2, 0.1)
    a.render(m, d, samples=8)
    b.render(m2, d2, samples=8)
    np.testing.assert_array_equal(a.hdr.numpy(), b.hdr.numpy())

  def test_camera_motion_validation(self):
    with self.assertRaises(ValueError):
      mjw.CameraMotion(exposure=1.2)
    with self.assertRaises(ValueError):
      mjw.CameraMotion(rolling_shutter=-0.1)
    with self.assertRaises(ValueError):
      mjw.CameraMotion(exposure=0.7, rolling_shutter=0.5)
    # Zero shutter is a no-op: tracer stays in sharp mode.
    m, d, rc = _setup()
    tracer = mjw.PathTracer(rc, camera_motion=mjw.CameraMotion())
    self.assertIsNone(tracer.camera_motion)


def _place_ball(d, x: float, mjm, m, rc):
  xpos = d.geom_xpos.numpy().copy()
  ball = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball")
  xpos[0, ball, 0] = x
  d.geom_xpos.assign(xpos)
  mjw.refit_bvh(m, d, rc)


class ObjectMotionTest(absltest.TestCase):
  def test_moving_object_blur(self):
    mjm, _, m, d = test_data.fixture(xml=_XML)
    rc = mjw.create_render_context(mjm, cam_res=(_RES, _RES), render_rgb=True)

    tracer = mjw.PathTracer(
      rc, max_bounces=1, seed=7, camera_motion=mjw.CameraMotion(exposure=1.0), object_motion=True, motion_subframes=8
    )
    _place_ball(d, 0.0, mjm, m, rc)
    tracer.render(m, d, samples=2)
    x0 = _centroid_x(_image(tracer))

    _place_ball(d, 0.3, mjm, m, rc)
    tracer.render(m, d, samples=64)
    blurred = _image(tracer)
    xb = _centroid_x(blurred)

    # Reference sharp pose at the final position.
    sharp = mjw.PathTracer(rc, max_bounces=1, seed=8)
    sharp.render(m, d, samples=2)
    sharp_img = _image(sharp)
    x1 = _centroid_x(sharp_img)

    lo, hi = min(x0, x1), max(x0, x1)
    self.assertGreater(hi - lo, 5.0)
    self.assertGreater(xb, lo + 0.15 * (hi - lo))
    self.assertLess(xb, hi - 0.15 * (hi - lo))
    lit_blur = int((blurred.mean(axis=2) > 0.05).sum())
    lit_sharp = int((sharp_img.mean(axis=2) > 0.05).sum())
    self.assertGreater(lit_blur, lit_sharp)
    self.assertLess(float(blurred[blurred > 0.05].mean()), 0.9)

  def test_static_object_is_sharp(self):
    mjm, _, m, d = test_data.fixture(xml=_XML)
    rc = mjw.create_render_context(mjm, cam_res=(_RES, _RES), render_rgb=True)

    blur = mjw.PathTracer(
      rc, max_bounces=1, seed=9, camera_motion=mjw.CameraMotion(exposure=1.0), object_motion=True, motion_subframes=4
    )
    plain = mjw.PathTracer(rc, max_bounces=1, seed=9)
    blur.render(m, d, samples=1)
    plain.render(m, d, samples=1)
    blur.render(m, d, samples=16)
    plain.render(m, d, samples=16)

    np.testing.assert_allclose(_image(blur), _image(plain), atol=0.01)

  def test_validation(self):
    mjm, _, m, d = test_data.fixture(xml=_XML)
    rc = mjw.create_render_context(mjm, cam_res=(_RES, _RES), render_rgb=True)
    with self.assertRaises(ValueError):
      mjw.PathTracer(rc, object_motion=True)
    with self.assertRaises(ValueError):
      mjw.PathTracer(rc, camera_motion=mjw.CameraMotion(rolling_shutter=0.5), object_motion=True)
    with self.assertRaises(ValueError):
      mjw.PathTracer(rc, camera_motion=mjw.CameraMotion(exposure=0.5), motion_subframes=0)


_FLY_XML = """
<mujoco>
  <option gravity="0 0 0"/>
  <asset>
    <material name="emit" rgba="1 1 1 1" specular="0" shininess="0" emission="1"/>
    <material name="black" rgba="0.02 0.02 0.02 1" specular="0" shininess="0"/>
  </asset>
  <worldbody>
    <geom type="plane" size="5 5 0.1" material="black"/>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom name="ball" type="sphere" size="0.15" material="emit"/>
    </body>
    <camera pos="0 -2.5 0.5" xyaxes="1 0 0 0 0 1" fovy="45"/>
  </worldbody>
</mujoco>
"""


class PhysicsSensorTest(absltest.TestCase):
  def test_object_motion_blur_from_physics_step(self):
    # Ball flies at 2 m/s with no gravity; 50 steps of dt=2ms move it 0.2 m.
    res, depth, vx, steps = 64, 2.5, 2.0, 50
    mjm, _, m, d = test_data.fixture(xml=_FLY_XML)
    rc = mjw.create_render_context(mjm, cam_res=(res, res), render_rgb=True)
    mjw.forward(m, d)
    qvel = d.qvel.numpy()
    qvel[0, 0] = vx
    d.qvel.assign(qvel)

    tracer = mjw.PathTracer(
      rc, max_bounces=1, seed=0, camera_motion=mjw.CameraMotion(exposure=1.0), object_motion=True, motion_subframes=8
    )
    tracer.render(m, d, samples=2)
    x_start = _centroid_x(_image(tracer, res))

    ball = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball")
    x0 = float(d.geom_xpos.numpy()[0, ball, 0])
    for _ in range(steps):
      mjw.step(m, d)
    dx = float(d.geom_xpos.numpy()[0, ball, 0]) - x0
    self.assertLess(abs(dx - vx * steps * mjm.opt.timestep), 0.01)

    tracer.render(m, d, samples=64)
    blurred = _image(tracer, res)

    mjw.refit_bvh(m, d, rc)
    sharp = mjw.PathTracer(rc, max_bounces=1, seed=1)
    sharp.render(m, d, samples=2)
    sharp_img = _image(sharp, res)

    px = res / 2.0 / (np.tan(np.deg2rad(45.0 / 2.0)) * depth)
    predicted_extent = (dx + 2.0 * 0.15) * px
    lit = blurred.mean(axis=2) > 0.1
    xs = np.where(lit.any(axis=0))[0]
    self.assertAlmostEqual(float(xs.max() - xs.min() + 1), predicted_extent, delta=3.0)

    x_blur = _centroid_x(blurred)
    x_sharp = _centroid_x(sharp_img)
    self.assertGreater(x_blur, x_start + 0.2 * (x_sharp - x_start))
    self.assertLess(x_blur, x_sharp - 0.2 * (x_sharp - x_start))


class MovingSceneTest(absltest.TestCase):
  def test_refit_bvh_tracks_geom_motion(self):
    mjm, _, m, d = test_data.fixture(xml=_XML)
    rc = mjw.create_render_context(mjm, cam_res=(_RES, _RES), render_rgb=True)
    ball = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ball")

    tracer = mjw.PathTracer(rc, max_bounces=1, seed=6)

    # Move the emitter and refit the scene BVH, as a physics step would. Both
    # poses keep the sphere clear of the image borders.
    for x, label in ((0.15, "before"), (-0.15, "after")):
      xpos = d.geom_xpos.numpy().copy()
      xpos[0, ball, 0] = x
      d.geom_xpos.assign(xpos)
      mjw.refit_bvh(m, d, rc)
      tracer.reset()
      tracer.render(m, d, samples=2)
      if label == "before":
        before = _centroid_x(_image(tracer))
      else:
        after = _centroid_x(_image(tracer))

    self.assertGreater(before - after, 5.0)


if __name__ == "__main__":
  absltest.main()
