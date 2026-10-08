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
"""Ray-bundled LiDAR simulation.

`Lidar` casts a configurable scan pattern as ray bundles accelerated by the
scene BVH, then models beam divergence, Gaussian range noise and
reflectivity-based return intensity. Scans are nworld-batched: every world gets
its own sensor pose, divergence/noise draw and output buffers in one call.
"""

import dataclasses
from typing import Tuple

import numpy as np
import warp as wp

from mujoco_warp._src import types
from mujoco_warp._src.ray import rays
from mujoco_warp._src.render import _geom_rgba
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import RenderContext
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


@wp.func
def _luma(c: wp.vec3) -> float:
  return wp.dot(c, wp.vec3(0.2126, 0.7152, 0.0722))


@wp.kernel(module="unique", enable_backward=False)
def _scan_rays_kernel(
  # In:
  dirs: wp.array[wp.vec3],
  origins: wp.array2d[wp.vec3],
  rots: wp.array2d[wp.mat33],
  divergence: float,
  seed: int,
  # Out:
  pnt_out: wp.array2d[wp.vec3],
  vec_out: wp.array2d[wp.vec3],
):
  worldid, rayid = wp.tid()
  rng = wp.rand_init(seed, worldid * dirs.shape[0] + rayid)
  d = dirs[rayid]
  if divergence > 0.0:
    d = wp.normalize(d + wp.vec3(wp.randn(rng), wp.randn(rng), wp.randn(rng)) * divergence)
  pnt_out[worldid, rayid] = origins[worldid % origins.shape[0], 0]
  vec_out[worldid, rayid] = rots[worldid % rots.shape[0], 0] @ d


@wp.kernel(module="unique", enable_backward=False)
def _scan_post_kernel(
  # Model:
  geom_matid: wp.array2d[int],
  geom_rgba: wp.array2d[wp.vec4],
  mat_rgba: wp.array2d[wp.vec4],
  # In:
  dist: wp.array2d[float],
  geomid: wp.array2d[int],
  normal: wp.array2d[wp.vec3],
  pnt: wp.array2d[wp.vec3],
  vec: wp.array2d[wp.vec3],
  max_range: float,
  range_sigma: float,
  reflectivity: float,
  seed: int,
  # Out:
  ranges_out: wp.array2d[float],
  points_out: wp.array2d[wp.vec3],
  intensity_out: wp.array2d[float],
  hit_out: wp.array2d[int],
):
  worldid, rayid = wp.tid()
  rng = wp.rand_init(seed, worldid * dist.shape[1] + rayid)
  gi = geomid[worldid, rayid]
  if gi < 0:
    ranges_out[worldid, rayid] = max_range
    points_out[worldid, rayid] = pnt[worldid, rayid] + vec[worldid, rayid] * max_range
    intensity_out[worldid, rayid] = 0.0
    hit_out[worldid, rayid] = 0
    return
  r = dist[worldid, rayid]
  if range_sigma > 0.0:
    r += wp.randn(rng) * range_sigma
  r = wp.clamp(r, 0.0, max_range)
  color, _ = _geom_rgba(geom_matid, geom_rgba, mat_rgba, worldid, gi)
  albedo = _luma(wp.vec3(color[0], color[1], color[2]))
  cos_incidence = wp.abs(wp.dot(normal[worldid, rayid], vec[worldid, rayid]))
  ranges_out[worldid, rayid] = r
  points_out[worldid, rayid] = pnt[worldid, rayid] + vec[worldid, rayid] * r
  intensity_out[worldid, rayid] = wp.clamp(reflectivity * albedo * cos_incidence, 0.0, 1.0)
  hit_out[worldid, rayid] = 1


@dataclasses.dataclass
class LidarScan:
  """One LiDAR scan.

  Attributes:
    ranges: Per-ray distance to the first hit, `max_range` on a miss (nworld, nray).
    points: Per-ray hit point in world coordinates, `origin + dir * max_range`
      on a miss (nworld, nray).
    intensity: Reflectivity-weighted return intensity in [0, 1] (nworld, nray).
    hit: 1 for a valid return, 0 for a miss (nworld, nray).
  """

  ranges: wp.array2d[float]
  points: wp.array2d[wp.vec3]
  intensity: wp.array2d[float]
  hit: wp.array2d[int]


def _origin_array(origin, nworld: int):
  """Broadcasts a host (3,), (1, 3) or (nworld, 3) origin to an (rows, 1) vec3 array."""
  arr = np.asarray(origin, dtype=np.float32)
  if arr.ndim == 2 and arr.shape[1] == 3:
    if arr.shape[0] not in (1, nworld):
      raise ValueError(f"origin leading dimension must be 1 or nworld ({nworld}), got {arr.shape[0]}.")
    arr = arr.reshape(arr.shape[0], 1, 3)
  elif arr.ndim == 1 and arr.shape == (3,):
    arr = arr.reshape(1, 1, 3)
  else:
    raise ValueError(f"origin must have shape (3,), (1, 3) or (nworld, 3), got {arr.shape}.")
  return wp.array(arr, dtype=wp.vec3)


def _rotation_array(rotation, nworld: int):
  """Broadcasts a host (3, 3), (1, 3, 3) or (nworld, 3, 3) rotation to an (rows, 1) mat33 array."""
  arr = np.asarray(rotation, dtype=np.float32)
  if arr.ndim == 3 and arr.shape[1:] == (3, 3):
    if arr.shape[0] not in (1, nworld):
      raise ValueError(f"rotation leading dimension must be 1 or nworld ({nworld}), got {arr.shape[0]}.")
    arr = arr.reshape(arr.shape[0], 1, 3, 3)
  elif arr.ndim == 2 and arr.shape == (3, 3):
    arr = arr.reshape(1, 1, 3, 3)
  else:
    raise ValueError(f"rotation must have shape (3, 3), (1, 3, 3) or (nworld, 3, 3), got {arr.shape}.")
  return wp.array(arr, dtype=wp.mat33)


class Lidar:
  """GPU-parallel LiDAR sensor.

  Attributes:
    directions: Local per-ray unit directions (nray, 3) on device.
  """

  def __init__(
    self,
    directions: np.ndarray,
    max_range: float = 100.0,
    range_sigma: float = 0.0,
    beam_divergence: float = 0.0,
    reflectivity: float = 1.0,
    seed: int = 0,
  ):
    """Initializes the LiDAR.

    Args:
      directions: Local unit ray directions with shape (nray, 3). Rows are
        normalized on the host.
      max_range: Maximum range in meters; misses report this range.
      range_sigma: Standard deviation of Gaussian range noise in meters.
      beam_divergence: Standard deviation of the angular jitter applied to each
        ray direction, in radians.
      reflectivity: Global return intensity scale in [0, 1].
      seed: Base RNG seed.
    """
    directions = np.asarray(directions, dtype=np.float32)
    if directions.ndim != 2 or directions.shape[1] != 3 or directions.shape[0] == 0:
      raise ValueError(f"directions must have shape (nray, 3) with nray > 0, got {directions.shape}.")
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
      raise ValueError("directions must be nonzero.")
    if max_range <= 0.0:
      raise ValueError("max_range must be positive.")
    if range_sigma < 0.0 or beam_divergence < 0.0 or reflectivity < 0.0:
      raise ValueError("range_sigma, beam_divergence and reflectivity must be non-negative.")
    self.directions = wp.array(directions / norms, dtype=wp.vec3)
    self.nray = directions.shape[0]
    self.max_range = max_range
    self.range_sigma = range_sigma
    self.beam_divergence = beam_divergence
    self.reflectivity = reflectivity
    self.seed = seed

  @staticmethod
  def ring_pattern(
    n_rings: int = 16,
    points_per_ring: int = 64,
    vfov: Tuple[float, float] = (-15.0, 15.0),
  ) -> np.ndarray:
    """Builds a uniform ring scan pattern.

    Args:
      n_rings: Number of elevation rings.
      points_per_ring: Number of azimuth samples per ring.
      vfov: Minimum and maximum elevation in degrees.

    Returns:
      Local unit directions with shape (n_rings * points_per_ring, 3), with
      azimuth in [-180, 180) and elevation in [vfov[0], vfov[1]] degrees.
    """
    if n_rings < 1 or points_per_ring < 1:
      raise ValueError("n_rings and points_per_ring must be positive.")
    elev = np.deg2rad(np.linspace(vfov[0], vfov[1], n_rings))
    azim = np.deg2rad(np.linspace(-180.0, 180.0, points_per_ring, endpoint=False))
    elev_grid, azim_grid = np.meshgrid(elev, azim, indexing="ij")
    ce = np.cos(elev_grid.ravel())
    directions = np.stack(
      [ce * np.cos(azim_grid.ravel()), ce * np.sin(azim_grid.ravel()), np.sin(elev_grid.ravel())],
      axis=1,
    )
    return directions.astype(np.float32)

  @event_scope
  def scan(self, m: Model, d: Data, rc: RenderContext, origin, rotation=None) -> LidarScan:
    """Casts one scan and applies sensor noise and intensity models.

    Args:
      m: The model containing kinematic and dynamic information (device).
      d: The data object containing the current state (device).
      rc: Render context providing the scene BVH (build once per model).
      origin: Sensor origin in world coordinates, shape (3,) or (nworld, 3).
      rotation: Sensor orientation as a rotation matrix, shape (3, 3) or
        (nworld, 3, 3). None uses the identity.

    Returns:
      The scan outputs, shape (nworld, nray) each.
    """
    nworld = d.nworld
    origins = _origin_array(origin, nworld)
    if rotation is None:
      rotation = np.eye(3, dtype=np.float32)
    rots = _rotation_array(rotation, nworld)

    pnt = wp.zeros((nworld, self.nray), dtype=wp.vec3)
    vec = wp.zeros((nworld, self.nray), dtype=wp.vec3)
    wp.launch(
      _scan_rays_kernel,
      dim=(nworld, self.nray),
      inputs=[self.directions, origins, rots, self.beam_divergence, self.seed],
      outputs=[pnt, vec],
    )

    dist = wp.empty((nworld, self.nray), dtype=float)
    geomid = wp.empty((nworld, self.nray), dtype=int)
    normal = wp.empty((nworld, self.nray), dtype=wp.vec3)
    bodyexclude = wp.full(self.nray, -1, dtype=int)
    rays(m, d, pnt, vec, types.vec6(-1, -1, -1, -1, -1, -1), True, bodyexclude, dist, geomid, normal, rc)

    ranges = wp.empty((nworld, self.nray), dtype=float)
    points = wp.empty((nworld, self.nray), dtype=wp.vec3)
    intensity = wp.empty((nworld, self.nray), dtype=float)
    hit = wp.empty((nworld, self.nray), dtype=int)
    wp.launch(
      _scan_post_kernel,
      dim=(nworld, self.nray),
      inputs=[
        m.geom_matid,
        m.geom_rgba,
        m.mat_rgba,
        dist,
        geomid,
        normal,
        pnt,
        vec,
        self.max_range,
        self.range_sigma,
        self.reflectivity,
        self.seed,
      ],
      outputs=[ranges, points, intensity, hit],
    )
    return LidarScan(ranges=ranges, points=points, intensity=intensity, hit=hit)
