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
"""Tests for the event camera."""

import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjw


def _hdr(values: np.ndarray) -> wp.array2d[wp.vec3]:
  values = np.asarray(values, dtype=np.float32)
  return wp.array(np.repeat(values[..., None], 3, axis=-1), dtype=wp.vec3)


class EventCameraTest(absltest.TestCase):
  def test_events_and_reference(self):
    cam = mjw.EventCamera(nworld=1, npixel=8, threshold=0.2)
    self.assertFalse(cam.initialized)

    flat = np.full((1, 8), 1.0)
    events = cam.update(_hdr(flat)).numpy()
    self.assertTrue(cam.initialized)
    self.assertTrue((events == 0).all())

    # log(1.5) - log(1.0) = 0.405 > threshold: ON events.
    events = cam.update(_hdr(np.full((1, 8), 1.5))).numpy()
    self.assertTrue((events == 1).all())

    # Re-anchored: no event without further change.
    events = cam.update(_hdr(np.full((1, 8), 1.5))).numpy()
    self.assertTrue((events == 0).all())

    # log(1.0) - log(1.5) = -0.405: OFF events.
    events = cam.update(_hdr(flat)).numpy()
    self.assertTrue((events == -1).all())

  def test_subthreshold_change(self):
    cam = mjw.EventCamera(nworld=1, npixel=8, threshold=0.2)
    cam.update(_hdr(np.full((1, 8), 1.0)))
    # log(1.1) - log(1.0) = 0.095 < threshold: no events.
    events = cam.update(_hdr(np.full((1, 8), 1.1))).numpy()
    self.assertTrue((events == 0).all())

  def test_nworld(self):
    cam = mjw.EventCamera(nworld=2, npixel=8, threshold=0.2)
    cam.update(_hdr(np.ones((2, 8))))

    values = np.stack([np.full(8, 1.5), np.full(8, 1.0)])
    events = cam.update(_hdr(values)).numpy()
    self.assertTrue((events[0] == 1).all())
    self.assertTrue((events[1] == 0).all())

  def test_reset(self):
    cam = mjw.EventCamera(nworld=1, npixel=8, threshold=0.2)
    cam.update(_hdr(np.full((1, 8), 1.0)))
    cam.reset()
    self.assertFalse(cam.initialized)
    events = cam.update(_hdr(np.full((1, 8), 0.1))).numpy()
    self.assertTrue((events == 0).all())


if __name__ == "__main__":
  absltest.main()
