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
"""Realistic camera and image sensor models.

`CameraModel` adds physical lens effects on top of a perfect pinhole:

  - thin-lens depth of field (aperture radius + focus distance),
  - Brown-Conrady lens distortion (radial k1/k2, tangential p1/p2),
  - lateral chromatic aberration (per-channel radial scale).

`CameraSensor` models the imaging chain: exposure, gain, full-well saturation,
shot noise and read noise.
"""

import dataclasses
from typing import Tuple

import warp as wp

_PI = 3.141592653589793


@dataclasses.dataclass
class CameraModel:
  """Physical lens parameters shared by the cameras of a render context.

  Attributes:
    distortion: Brown-Conrady coefficients (k1, k2, p1, p2). All zero disables
      distortion.
    chromatic: Per-channel radial scale (red, green, blue) applied before
      distortion. (1, 1, 1) disables chromatic aberration.
    aperture: Lens aperture (entrance pupil) radius in meters. Zero is a
      perfect pinhole.
    focus_distance: Distance from the lens to the in-focus plane, in meters.
      Required when aperture > 0.
  """

  distortion: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
  chromatic: tuple[float, float, float] = (1.0, 1.0, 1.0)
  aperture: float = 0.0
  focus_distance: float = 0.0

  @property
  def has_distortion(self) -> bool:
    return any(v != 0.0 for v in self.distortion)

  @property
  def has_chromatic(self) -> bool:
    return any(v != 1.0 for v in self.chromatic)

  @property
  def has_dof(self) -> bool:
    return self.aperture > 0.0


@dataclasses.dataclass
class CameraMotion:
  """Shutter model for motion blur and rolling shutter.

  Times are fractions of the interval between two consecutive `render` calls,
  where t=0 is the previous camera pose and t=1 the current one.

  Attributes:
    exposure: Duration the shutter is open; 0 is an instantaneous global
      shutter and 1 exposes for the whole interval.
    rolling_shutter: Time between exposing the first and last image row; 0 is a
      global shutter. Values of 1 mean the readout sweeps the whole interval.
  """

  exposure: float = 0.0
  rolling_shutter: float = 0.0

  def __post_init__(self):
    if not 0.0 <= self.exposure <= 1.0:
      raise ValueError("exposure must be in [0, 1].")
    if not 0.0 <= self.rolling_shutter <= 1.0:
      raise ValueError("rolling_shutter must be in [0, 1].")
    if self.exposure + self.rolling_shutter > 1.0:
      raise ValueError("exposure + rolling_shutter must be <= 1.")


@dataclasses.dataclass
class CameraSensor:
  """Image sensor settings applied after rendering, before tone mapping.

  The sensor converts scene radiance to electrons with
  `signal_e = radiance * exposure * gain`, saturates at `full_well_e`, adds
  Poisson shot noise and Gaussian read noise, and converts back to radiance.

  Attributes:
    exposure: Exposure multiplier (integration time x throughput).
    gain: Electrons per unit of radiance.
    full_well_e: Saturation capacity in electrons.
    read_noise_e: RMS read noise in electrons.
    seed: Base RNG seed for the noise.
  """

  exposure: float = 1.0
  gain: float = 1.0
  full_well_e: float = 10000.0
  read_noise_e: float = 2.0
  seed: int = 0

  def __post_init__(self):
    if self.exposure <= 0.0 or self.gain <= 0.0:
      raise ValueError("CameraSensor exposure and gain must be positive.")
    if self.full_well_e <= 0.0:
      raise ValueError("CameraSensor full_well_e must be positive.")
    if self.read_noise_e < 0.0:
      raise ValueError("CameraSensor read_noise_e must be non-negative.")


@wp.func
def sample_lens_offset(
  # In:
  aperture: float,
  u1: float,
  u2: float,
) -> wp.vec2:
  # Uniform point in the aperture disk.
  r = aperture * wp.sqrt(u1)
  phi = 2.0 * _PI * u2
  return wp.vec2(r * wp.cos(phi), r * wp.sin(phi))


@wp.func
def distort_direction(
  # In:
  dir_local: wp.vec3,
  distortion: wp.vec4,
  scale: float,
) -> wp.vec3:
  # Pinhole normalized image coordinates: camera looks along -Z.
  x = dir_local[0] * scale / (-dir_local[2])
  y = dir_local[1] * scale / (-dir_local[2])
  r2 = x * x + y * y
  radial = 1.0 + distortion[0] * r2 + distortion[1] * r2 * r2
  xd = x * radial + 2.0 * distortion[2] * x * y + distortion[3] * (r2 + 2.0 * x * x)
  yd = y * radial + distortion[2] * (r2 + 2.0 * y * y) + 2.0 * distortion[3] * x * y
  return wp.normalize(wp.vec3(xd, yd, -1.0))


@wp.func
def thin_lens_direction(
  # In:
  dir_local: wp.vec3,
  lens_offset: wp.vec2,
  focus_distance: float,
) -> wp.vec3:
  # Direction from a point on the lens toward the focal-plane point of dir_local.
  t = focus_distance / wp.max(-dir_local[2], 1.0e-9)
  focus_point = wp.vec3(dir_local[0] * t, dir_local[1] * t, dir_local[2] * t)
  lens_point = wp.vec3(lens_offset[0], lens_offset[1], 0.0)
  return wp.normalize(focus_point - lens_point)


@wp.func
def sensor_sample(
  # In:
  radiance: wp.vec3,
  exposure: float,
  gain: float,
  full_well_e: float,
  read_noise_e: float,
  noise1: wp.vec3,
  noise2: wp.vec3,
) -> wp.vec3:
  """Applies exposure, shot noise, read noise and saturation to linear radiance."""
  signal = wp.cw_mul(radiance, wp.vec3(exposure * gain, exposure * gain, exposure * gain))
  signal = wp.min(wp.max(signal, wp.vec3(0.0, 0.0, 0.0)), wp.vec3(full_well_e, full_well_e, full_well_e))
  # Poisson shot noise approximated as signal + sqrt(signal) * N(0, 1).
  shot = wp.vec3(
    noise1[0] * wp.sqrt(signal[0]),
    noise1[1] * wp.sqrt(signal[1]),
    noise1[2] * wp.sqrt(signal[2]),
  )
  measured = signal + shot + noise2 * read_noise_e
  measured = wp.min(wp.max(measured, wp.vec3(0.0, 0.0, 0.0)), wp.vec3(full_well_e, full_well_e, full_well_e))
  return measured * (1.0 / (exposure * gain))


def _camera_constants(model: CameraModel) -> Tuple[wp.vec4, wp.vec3, float, float]:
  return (
    wp.vec4(*model.distortion),
    wp.vec3(*model.chromatic),
    float(model.aperture),
    float(model.focus_distance),
  )
