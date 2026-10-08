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
"""Event camera (DVS) simulation.

Converts a sequence of HDR radiance images into per-pixel brightness-change
events. Each pixel keeps a log-radiance reference and emits an ON (+1) or OFF
(-1) event when the log-radiance change exceeds the contrast sensitivity, then
re-anchors the reference to the current value.
"""

import warp as wp

from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_EPS = 1.0e-6


@wp.func
def _luma(c: wp.vec3) -> float:
  return wp.dot(c, wp.vec3(0.2126, 0.7152, 0.0722))


@wp.kernel(module="unique", enable_backward=False)
def _event_kernel(
  # In:
  hdr: wp.array2d[wp.vec3],
  threshold: float,
  initialize: int,
  # Out:
  ref_out: wp.array2d[float],
  events_out: wp.array2d[int],
):
  worldid, pixel = wp.tid()
  log_radiance = wp.log(_luma(hdr[worldid, pixel]) + _EPS)
  if initialize != 0:
    ref_out[worldid, pixel] = log_radiance
    events_out[worldid, pixel] = 0
    return
  delta = log_radiance - ref_out[worldid, pixel]
  if delta >= threshold:
    ref_out[worldid, pixel] = log_radiance
    events_out[worldid, pixel] = 1
  elif delta <= -threshold:
    ref_out[worldid, pixel] = log_radiance
    events_out[worldid, pixel] = -1
  else:
    events_out[worldid, pixel] = 0


class EventCamera:
  """Per-pixel brightness-change event sensor.

  Feed successive HDR radiance images (e.g. `PathTracer.hdr`) to `update`; the
  first call initializes the reference and emits no events.
  """

  def __init__(self, nworld: int, npixel: int, threshold: float = 0.2):
    """Initializes the event camera.

    Args:
      nworld: Number of worlds (leading dimension of the radiance images).
      npixel: Number of pixels per image.
      threshold: Contrast sensitivity: the log-radiance change needed to emit an
        event. Typical DVS values are around 0.15 - 0.3.
    """
    if threshold <= 0.0:
      raise ValueError("threshold must be positive.")
    self.ref = wp.zeros((nworld, npixel), dtype=float)
    self.events = wp.zeros((nworld, npixel), dtype=int)
    self.threshold = threshold
    self._initialized = False

  @property
  def initialized(self) -> bool:
    """Whether the reference image has been captured."""
    return self._initialized

  def reset(self):
    """Discards the reference image; the next update re-initializes."""
    self._initialized = False

  @event_scope
  def update(self, hdr: wp.array2d[wp.vec3]) -> wp.array2d[int]:
    """Processes one HDR image and returns the per-pixel event polarity.

    Args:
      hdr: Linear radiance image (nworld, npixel).

    Returns:
      Per-pixel event polarity: +1 (ON), -1 (OFF) or 0, shape (nworld, npixel).
    """
    initialize = 0 if self._initialized else 1
    wp.launch(
      _event_kernel,
      dim=hdr.shape,
      inputs=[hdr, self.threshold, initialize],
      outputs=[self.ref, self.events],
    )
    self._initialized = True
    return self.events
