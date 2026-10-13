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
"""Visuotactile (GelSight-style) sensing.

`VisuotactileRenderer` simulates camera-based tactile sensors: a compliant gel
pad with an internal camera and ring illumination, attached to a rigid body.
The pipeline follows the standard decomposition used by GelSight simulators:

  1. Depth: contacts are gathered per sensor body (deduplicated by geom), then
     the signed distance to each contacting geom is sampled on a regular grid on
     the undeformed gel surface, using the same distance functions as MuJoCo
     Warp tactile sensors. Normal indentation is the penetration depth.
  2. Compliance: a separable Gaussian blurs the depth field, standing in for the
     finite stiffness of the gel layer.
  3. Optics: surface normals from depth gradients; Lambertian shading from the
     LED ring with distance attenuation and soft raymarched shadows; an optional
     marker lattice advected by a thin-gel flow model (`u = -t * grad(depth)`).
  4. Sensor: optional shot/read noise and saturation via `CameraSensor`.

The sensor frame is the gel frame: the gel surface is the `z = 0` plane, the
image x axis is +x and the image y axis is +y (row 0 is +y), and the pad body
extends below the surface. By default the frame sits at the top center of the
collision geom.

Outputs are per-sensor fields (depth, force, normal, rgb, marker, flow),
batched over worlds. The smooth parts of the pipeline (depth, smoothing,
normals, shading) are differentiable under `warp.Tape` w.r.t. `d.geom_xpos` and
`d.geom_xmat`. Marker dots and shadow edges are piecewise; disable markers for
gradient-based work.

Example:
  sensor = gelsight_mini(geom_id)
  renderer = VisuotactileRenderer(m, [sensor], fields=("depth", "rgb"))
  frame = renderer.render(m, d)
  image = frame.numpy("rgb", 0)  # (nworld, height, width, 3)
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import mujoco
import numpy as np
import warp as wp

from mujoco_warp._src.camera import CameraSensor
from mujoco_warp._src.camera import sensor_sample
from mujoco_warp._src.collision_sdf import get_sdf_params
from mujoco_warp._src.collision_sdf import sdf
from mujoco_warp._src.math import rot_vec_quat
from mujoco_warp._src.types import MJ_MAXCONPAIR
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec8
from mujoco_warp._src.types import vec8i
from mujoco_warp._src.types import vec_pluginattr
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": True, "default_grid_stride": False})

_FIELDS = ("depth", "force", "normal", "rgb", "marker", "flow")
_FIELD_DTYPE = {"depth": float, "force": float, "normal": wp.vec3, "rgb": wp.vec3, "marker": float, "flow": wp.vec2}
_DIFF_FIELDS = ("depth", "force", "normal", "rgb", "flow")
_VEC_FIELDS = {"normal": 3, "rgb": 3, "flow": 2}
_SUPPORTED_GEOMS = (
  GeomType.BOX,
  GeomType.SPHERE,
  GeomType.CAPSULE,
  GeomType.CYLINDER,
  GeomType.ELLIPSOID,
  GeomType.MESH,
)
_SMOOTHING_MAX = 16.0


def exists(val):
  return val is not None


def default(val, d):
  return val if exists(val) else d


@wp.func
def _smoothstep(e0: float, e1: float, x: float) -> float:
  t = wp.clamp((x - e0) / (e1 - e0), 0.0, 1.0)
  return t * t * (3.0 - 2.0 * t)


@wp.func
def _sample_depth(
  # In:
  grid: wp.array2d[float],
  worldid: int,
  width: int,
  height: int,
  gel_width: float,
  gel_height: float,
  x: float,
  y: float,
) -> float:
  # Bilinear sample of the depth grid at gel-frame (x, y).
  u = (x / gel_width + 0.5) * float(width) - 0.5
  v = (0.5 - y / gel_height) * float(height) - 0.5
  u0 = int(wp.floor(u))
  v0 = int(wp.floor(v))
  u1 = wp.min(u0 + 1, width - 1)
  v1 = wp.min(v0 + 1, height - 1)
  u0 = wp.max(u0, 0)
  v0 = wp.max(v0, 0)
  fu = u - wp.floor(u)
  fv = v - wp.floor(v)
  h00 = grid[worldid, v0 * width + u0]
  h10 = grid[worldid, v0 * width + u1]
  h01 = grid[worldid, v1 * width + u0]
  h11 = grid[worldid, v1 * width + u1]
  return (h00 * (1.0 - fu) + h10 * fu) * (1.0 - fv) + (h01 * (1.0 - fu) + h11 * fu) * fv


@wp.func
def _shadow_visibility(
  # In:
  depth_in: wp.array2d[float],
  worldid: int,
  width: int,
  height: int,
  gel_width: float,
  gel_height: float,
  p: wp.vec3,
  ldir: wp.vec3,
  dist: float,
  steps: int,
  softness: float,
) -> float:
  # March toward the LED; occluded where the surface (z = -depth) rises above the ray.
  vis = float(1.0)
  for k in range(1, steps + 1):
    t = float(k) / float(steps + 1)
    sx = p[0] + ldir[0] * dist * t
    sy = p[1] + ldir[1] * dist * t
    sz = p[2] + ldir[2] * dist * t
    if sx > -0.5 * gel_width and sx < 0.5 * gel_width and sy > -0.5 * gel_height and sy < 0.5 * gel_height:
      hs = _sample_depth(depth_in, worldid, width, height, gel_width, gel_height, sx, sy)
      penetration = -hs - sz
      vis = wp.min(vis, 1.0 - _smoothstep(0.0, softness, penetration))
  return vis


@wp.kernel(module="unique", enable_backward=True)
def _contact_gather_bits(
  # Model:
  geom_bodyid: wp.array[int],
  # Data in:
  contact_geom_in: wp.array[wp.vec2i],
  contact_worldid_in: wp.array[int],
  nacon_in: wp.array[int],
  # In:
  pad_body: wp.array[int],
  # Out:
  seen_out: wp.array3d[wp.uint32],
):
  sens, conid = wp.tid()
  nacon = nacon_in[0]
  if conid >= nacon:
    return

  contact_geom = contact_geom_in[conid]
  geom1 = contact_geom[0]
  geom2 = contact_geom[1]

  # Flex contacts have negative geom indices; unsupported (matches tactile sensors).
  if geom1 < 0 or geom2 < 0:
    return

  # Geoms on the sensor body count even if non-colliding (MuJoCo tactile idiom:
  # gel mesh + colliding pad share a body).
  body = pad_body[sens]
  body1 = geom_bodyid[geom1]
  body2 = geom_bodyid[geom2]
  other = int(-1)
  if body1 == body and body2 != body:
    other = geom2
  elif body2 == body and body1 != body:
    other = geom1
  if other < 0:
    return

  worldid = contact_worldid_in[conid]

  # one bit per geom; no consumed return -> single pass under deterministic mode
  word_idx = other // 32
  bit_mask = wp.uint32(1) << wp.uint32(other % 32)
  wp.atomic_or(seen_out[worldid, sens], word_idx, bit_mask)


@wp.kernel(module="unique", enable_backward=True, grid_stride=False)
def _contact_gather_list(
  # In:
  max_geoms: int,
  # Data in:
  seen_in: wp.array3d[wp.uint32],
  # Out:
  count_out: wp.array2d[int],
  cand_out: wp.array3d[int],
):
  worldid, sens, word_idx = wp.tid()
  bits = seen_in[worldid, sens, word_idx]
  for bit in range(32):
    if (bits & (wp.uint32(1) << wp.uint32(bit))) == wp.uint32(0):
      continue
    other = 32 * word_idx + bit
    # ascending (word, bit) -> canonical candidates
    idx = wp.atomic_add(count_out[worldid], sens, 1)
    if idx < max_geoms:
      cand_out[worldid, sens, idx] = other


@wp.kernel(module="unique", enable_backward=True)
def _depth_kernel(
  # Model:
  oct_child: wp.array[vec8i],
  oct_aabb: wp.array2d[wp.vec3],
  oct_coeff: wp.array[vec8],
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  mesh_octadr: wp.array[int],
  mesh_pos: wp.array[wp.vec3],
  mesh_quat: wp.array[wp.quat],
  plugin: wp.array[int],
  plugin_attr: wp.array[vec_pluginattr],
  geom_plugin_index: wp.array[int],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  # In:
  count_in: wp.array2d[int],
  cand_in: wp.array3d[int],
  sens: int,
  max_geoms: int,
  pad_geom: int,
  frame_pos: wp.vec3,
  frame_x: wp.vec3,
  frame_y: wp.vec3,
  width: int,
  height: int,
  gel_width: float,
  gel_height: float,
  max_depth: float,
  stiffness: float,
  # Out:
  depth_out: wp.array2d[float],
  force_out: wp.array2d[float],
):
  worldid, pix = wp.tid()
  row = pix // width
  col = pix % width

  # Sample point on the undeformed gel surface, frame -> geom -> world.
  x = (float(col) + 0.5 - 0.5 * float(width)) * (gel_width / float(width))
  y = (0.5 * float(height) - float(row) - 0.5) * (gel_height / float(height))
  p_gel = frame_pos + frame_x * x + frame_y * y
  p_world = geom_xmat_in[worldid, pad_geom] @ p_gel + geom_xpos_in[worldid, pad_geom]

  num_cand = wp.min(count_in[worldid, sens], max_geoms)
  depth = float(0.0)
  for k in range(num_cand):
    g = cand_in[worldid, sens, k]
    if g < 0:
      continue
    gtype = geom_type[g]
    dataid = geom_dataid[worldid % geom_dataid.shape[0], g]
    if gtype == GeomType.MESH and mesh_octadr[dataid] == -1:
      continue
    p_local = wp.transpose(geom_xmat_in[worldid, g]) @ (p_world - geom_xpos_in[worldid, g])
    plugin_id = geom_plugin_index[g]
    # SDF plugins live in the original mesh frame (matches MuJoCo C).
    if gtype == GeomType.SDF and plugin_id != -1 and dataid >= 0:
      p_local = rot_vec_quat(p_local, mesh_quat[dataid]) + mesh_pos[dataid]
    attrs, pidx, volume_data, mesh_data = get_sdf_params(
      oct_child,
      oct_aabb,
      oct_coeff,
      mesh_octadr,
      plugin,
      plugin_attr,
      gtype,
      geom_size[worldid % geom_size.shape[0], g],
      plugin_id,
      dataid,
    )
    dist = sdf(gtype, p_local, attrs, pidx, volume_data, mesh_data)
    depth = wp.max(depth, -wp.min(dist, 0.0))

  depth = wp.min(depth, max_depth)
  depth_out[worldid, pix] = depth
  force_out[worldid, pix] = stiffness * depth


@wp.kernel(module="unique", enable_backward=True)
def _blur_kernel(
  # In:
  src: wp.array2d[float],
  weights: wp.array[float],
  radius: int,
  width: int,
  height: int,
  axis: int,
  # Out:
  blur_out: wp.array2d[float],
):
  worldid, pix = wp.tid()
  row = pix // width
  col = pix % width
  acc = float(0.0)
  for k in range(2 * radius + 1):
    i = wp.max(wp.min(col + k - radius, width - 1), 0)
    j = wp.max(wp.min(row + k - radius, height - 1), 0)
    if axis == 0:
      acc += weights[k] * src[worldid, row * width + i]
    else:
      acc += weights[k] * src[worldid, j * width + col]
  blur_out[worldid, pix] = acc


@wp.kernel(module="unique", enable_backward=True)
def _shade_kernel(
  # In:
  depth_in: wp.array2d[float],
  leds_pos: wp.array[wp.vec3],
  leds_rgb: wp.array[wp.vec3],
  leds_intensity: wp.array[float],
  # In:
  base_rgb: wp.vec3,
  marker_rgb: wp.vec3,
  width: int,
  height: int,
  gel_width: float,
  gel_height: float,
  ambient: float,
  attenuation: float,
  specular: float,
  shininess: float,
  shadow_steps: int,
  shadow_softness: float,
  shadow_strength: float,
  marker_spacing_x: float,
  marker_spacing_y: float,
  marker_radius: float,
  marker_softness: float,
  marker_alpha: float,
  flow_scale: float,
  use_markers: int,
  # Out:
  rgb_out: wp.array2d[wp.vec3],
  normal_out: wp.array2d[wp.vec3],
  marker_out: wp.array2d[float],
  flow_out: wp.array2d[wp.vec2],
):
  worldid, pix = wp.tid()
  row = pix // width
  col = pix % width
  dx = gel_width / float(width)
  dy = gel_height / float(height)
  x = (float(col) + 0.5 - 0.5 * float(width)) * dx
  y = (0.5 * float(height) - float(row) - 0.5) * dy

  # Central differences on the smoothed depth; one sided at the border.
  col_l = wp.max(col - 1, 0)
  col_r = wp.min(col + 1, width - 1)
  row_t = wp.max(row - 1, 0)
  row_b = wp.min(row + 1, height - 1)
  hl = depth_in[worldid, row * width + col_l]
  hr = depth_in[worldid, row * width + col_r]
  ht = depth_in[worldid, row_t * width + col]
  hb = depth_in[worldid, row_b * width + col]
  dhx = (hr - hl) / (float(col_r - col_l) * dx)
  dhy = (ht - hb) / (float(row_b - row_t) * dy)
  normal = wp.normalize(wp.vec3(dhx, dhy, 1.0))

  # Surface height s = -depth, so the point at height -h is the cavity floor.
  h = depth_in[worldid, pix]
  p = wp.vec3(x, y, -h)
  # Viewer on the LED side (+z), mirroring the physical internal camera.
  view = wp.vec3(0.0, 0.0, 1.0)

  irradiance = wp.vec3(ambient, ambient, ambient)
  for i in range(leds_pos.shape[0]):
    to_led = leds_pos[i] - p
    dist = wp.length(to_led)
    ldir = to_led / wp.max(dist, 1.0e-9)
    ndotl = wp.max(wp.dot(normal, ldir), 0.0)
    if ndotl > 0.0:
      att = float(1.0)
      if attenuation > 0.0:
        att = 1.0 / (1.0 + (dist / attenuation) * (dist / attenuation))
      vis = float(1.0)
      if shadow_steps > 0:
        vis = _shadow_visibility(
          depth_in, worldid, width, height, gel_width, gel_height, p, ldir, dist, shadow_steps, shadow_softness
        )
        vis = 1.0 - shadow_strength * (1.0 - vis)
      irradiance += leds_rgb[i] * (leds_intensity[i] * ndotl * att * vis)
      if specular > 0.0:
        hvec = wp.normalize(ldir + view)
        irradiance += leds_rgb[i] * (leds_intensity[i] * specular * wp.pow(wp.max(wp.dot(normal, hvec), 0.0), shininess) * vis)

  rgb = wp.cw_mul(base_rgb, irradiance)

  # Marker lattice advected by the thin-gel flow model u = -flow_scale * grad(h).
  flow = wp.vec2(-flow_scale * dhx, -flow_scale * dhy)
  marker = float(0.0)
  if use_markers == 1:
    qx = x - flow[0]
    qy = y - flow[1]
    ci = wp.round(qx / marker_spacing_x - 0.5)
    cj = wp.round(qy / marker_spacing_y - 0.5)
    cx = (ci + 0.5) * marker_spacing_x
    cy = (cj + 0.5) * marker_spacing_y
    dd = wp.sqrt((qx - cx) * (qx - cx) + (qy - cy) * (qy - cy))
    cov = 1.0 - _smoothstep(marker_radius - marker_softness, marker_radius + marker_softness, dd)
    marker = cov
    mix = marker_alpha * cov
    rgb = rgb * (1.0 - mix) + marker_rgb * mix

  rgb_out[worldid, pix] = rgb
  normal_out[worldid, pix] = normal
  marker_out[worldid, pix] = marker
  flow_out[worldid, pix] = flow


@wp.kernel(module="unique", enable_backward=False)
def _noise_kernel(
  # In:
  rgb_in: wp.array2d[wp.vec3],
  exposure: float,
  gain: float,
  full_well_e: float,
  read_noise_e: float,
  seed: int,
  # Out:
  rgb_out: wp.array2d[wp.vec3],
):
  worldid, pix = wp.tid()
  rng = wp.rand_init(seed, worldid * rgb_in.shape[1] + pix)
  noise1 = wp.vec3(wp.randn(rng), wp.randn(rng), wp.randn(rng))
  noise2 = wp.vec3(wp.randn(rng), wp.randn(rng), wp.randn(rng))
  rgb_out[worldid, pix] = sensor_sample(rgb_in[worldid, pix], exposure, gain, full_well_e, read_noise_e, noise1, noise2)


@wp.kernel(module="unique", enable_backward=True)
def _mse_vec3_kernel(
  # In:
  pred: wp.array2d[wp.vec3],
  target: wp.array2d[wp.vec3],
  inv_count: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid, pix = wp.tid()
  diff = pred[worldid, pix] - target[worldid, pix]
  wp.atomic_add(loss_out, 0, wp.dot(diff, diff) * inv_count)


@wp.kernel(module="unique", enable_backward=True)
def _mse_float_kernel(
  # In:
  pred: wp.array2d[float],
  target: wp.array2d[float],
  inv_count: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid, pix = wp.tid()
  diff = pred[worldid, pix] - target[worldid, pix]
  wp.atomic_add(loss_out, 0, diff * diff * inv_count)


@dataclasses.dataclass
class Led:
  """A single LED of the sensor illumination.

  Attributes:
    pos: LED position in the gel frame (meters). The gel surface is `z = 0`.
    rgb: LED color, linear.
    intensity: Radiant intensity scale.
  """

  pos: tuple[float, float, float]
  rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
  intensity: float = 1.0

  def __post_init__(self):
    if self.intensity < 0.0:
      raise ValueError("Led intensity must be non-negative.")


@dataclasses.dataclass
class GelOptics:
  """Optical parameters of the sensor.

  Attributes:
    base_rgba: Diffuse color of the reflective gel coating.
    leds: Illumination. None places four white LEDs at the gel corners.
    ambient: Ambient irradiance.
    attenuation: Distance falloff length scale in meters; 0 disables falloff.
    specular: Specular reflectance; 0 disables the highlight.
    shininess: Blinn-Phong exponent.
    smoothing: Gaussian sigma in pixels applied to the depth field before
      shading (gel compliance). 0 disables smoothing.
    shadow_steps: Heightfield raymarch steps per LED; 0 disables shadows.
    shadow_softness: Shadow penumbra in meters; keeps gradients smooth.
    shadow_strength: Shadow darkness in [0, 1].
    rgb_leds: If True and leds is None, uses directional RGB illumination.
  """

  base_rgba: tuple[float, float, float] = (0.82, 0.82, 0.80)
  leds: tuple[Led, ...] | None = None
  ambient: float = 0.2
  attenuation: float = 0.0
  specular: float = 0.0
  shininess: float = 20.0
  smoothing: float = 1.5
  shadow_steps: int = 16
  shadow_softness: float = 0.002
  shadow_strength: float = 1.0
  rgb_leds: bool = False

  def __post_init__(self):
    if self.ambient < 0.0:
      raise ValueError("GelOptics ambient must be non-negative.")
    if self.attenuation < 0.0:
      raise ValueError("GelOptics attenuation must be non-negative.")
    if self.specular < 0.0:
      raise ValueError("GelOptics specular must be non-negative.")
    if self.shininess <= 0.0:
      raise ValueError("GelOptics shininess must be positive.")
    if self.smoothing < 0.0 or self.smoothing > _SMOOTHING_MAX:
      raise ValueError(f"GelOptics smoothing must be in [0, {_SMOOTHING_MAX}].")
    if self.shadow_steps < 0:
      raise ValueError("GelOptics shadow_steps must be non-negative.")
    if self.shadow_steps > 0 and self.shadow_softness <= 0.0:
      raise ValueError("GelOptics shadow_softness must be positive when shadow_steps > 0.")
    if not 0.0 <= self.shadow_strength <= 1.0:
      raise ValueError("GelOptics shadow_strength must be in [0, 1].")


@dataclasses.dataclass
class MarkerPattern:
  """Printed marker lattice on the gel surface.

  Attributes:
    spacing: Dot lattice spacing (x, y) in meters.
    radius: Dot radius in meters.
    softness: Dot edge width in meters.
    rgb: Dot color, linear.
    alpha: Dot coverage opacity in [0, 1].
    flow_scale: Marker displacement per unit depth gradient in meters. 0 uses
      the gel thickness (thin-gel incompressible flow).
  """

  spacing: tuple[float, float] = (0.0016, 0.0016)
  radius: float = 0.00035
  softness: float = 0.0001
  rgb: tuple[float, float, float] = (0.12, 0.12, 0.12)
  alpha: float = 1.0
  flow_scale: float = 0.0

  def __post_init__(self):
    if self.spacing[0] <= 0.0 or self.spacing[1] <= 0.0:
      raise ValueError("MarkerPattern spacing must be positive.")
    if self.radius < 0.0:
      raise ValueError("MarkerPattern radius must be non-negative.")
    if self.softness <= 0.0:
      raise ValueError("MarkerPattern softness must be positive.")
    if self.flow_scale < 0.0:
      raise ValueError("MarkerPattern flow_scale must be non-negative.")
    if not 0.0 <= self.alpha <= 1.0:
      raise ValueError("MarkerPattern alpha must be in [0, 1].")


@dataclasses.dataclass
class VisuotactileSensor:
  """Specification of one visuotactile sensor.

  The sensor is attached to `geom`, the gel pad collision geom. The gel frame
  is the optical frame: the gel surface is `z = 0` with the pad below, the
  image x axis is +x, the image y axis is +y (row 0 is +y), and the optical
  view direction is +z, the same side as the LED ring. This mirrors the
  physical sensor, whose camera looks up at the reflective pad from inside.

  Attributes:
    geom: Gel geom id. Contacts involving any geom on this geom's body determine
      the depth field, so the gel geom itself may be non-colliding (the standard
      MuJoCo tactile idiom: tactile mesh plus a colliding pad on the same body).
    resolution: Image resolution (height, width).
    gel_size: Rendered gel extent (x, y) in meters. None uses the geom
      footprint. Interpreted in the geom's authored frame: the frame a mesh file
      was authored in (MuJoCo stores mesh vertices in a principal frame, so the
      renderer converts through `mesh_pos`/`mesh_quat` internally).
    frame_pos: Gel frame origin in the geom's authored frame. None places it at
      the top center of the geom.
    frame_quat: Gel frame orientation (w, x, y, z) relative to the geom's
      authored frame.
    gel_thickness: Gel layer thickness in meters; caps the depth field and sets
      the default marker flow scale.
    stiffness: Linear force proxy scale: force = stiffness * depth, per pixel.
    optics: Optical parameters.
    markers: Marker lattice; None disables markers.
  """

  geom: int
  resolution: tuple[int, int] = (120, 160)
  gel_size: tuple[float, float] | None = None
  frame_pos: tuple[float, float, float] | None = None
  frame_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
  gel_thickness: float = 0.005
  stiffness: float = 0.0
  optics: GelOptics = dataclasses.field(default_factory=GelOptics)
  markers: MarkerPattern | None = None

  def __post_init__(self):
    if self.resolution[0] < 1 or self.resolution[1] < 1:
      raise ValueError("VisuotactileSensor resolution must be positive.")
    if exists(self.gel_size) and (self.gel_size[0] <= 0.0 or self.gel_size[1] <= 0.0):
      raise ValueError("VisuotactileSensor gel_size must be positive.")
    if self.gel_thickness <= 0.0:
      raise ValueError("VisuotactileSensor gel_thickness must be positive.")
    if self.stiffness < 0.0:
      raise ValueError("VisuotactileSensor stiffness must be non-negative.")
    norm = float(np.linalg.norm(np.asarray(self.frame_quat, dtype=np.float64)))
    if not np.isclose(norm, 1.0, atol=1e-4):
      raise ValueError("VisuotactileSensor frame_quat must be a unit quaternion (w, x, y, z).")


@dataclasses.dataclass
class VisuotactileFrame:
  """One rendered frame of all visuotactile sensors.

  Fields hold one entry per sensor, or None if the field was not requested (or
  not available for that sensor). Each entry is an (nworld, height * width)
  warp array; `numpy` reshapes to (nworld, height, width[, channels]).

  Attributes:
    res: Per-sensor (height, width).
    depth: Gel indentation in meters, >= 0.
    force: `stiffness * depth` linear force proxy.
    normal: Gel-frame surface unit normals.
    rgb: Linear radiance image.
    marker: Dot coverage in [0, 1] (0 when the sensor has no markers).
    flow: Marker displacement field in meters.
  """

  res: tuple[tuple[int, int], ...]
  depth: list | None = None
  force: list | None = None
  normal: list | None = None
  rgb: list | None = None
  marker: list | None = None
  flow: list | None = None

  def __getitem__(self, item) -> np.ndarray:
    """Allows frame[field] or frame[field, sensor] index access."""
    if isinstance(item, tuple):
      field, sensor = item
      return self.numpy(field, sensor)
    return self.numpy(item, 0)

  def numpy(self, field: str, sensor: int = 0) -> np.ndarray:
    """Returns one field of one sensor as a numpy array.

    Args:
      field: One of "depth", "force", "normal", "rgb", "marker", "flow".
      sensor: Sensor index.

    Returns:
      An array of shape (nworld, height, width) or (nworld, height, width, C).
    """
    if field not in _FIELDS:
      raise ValueError(f"unknown field '{field}'; expected one of {_FIELDS}.")
    arrays = getattr(self, field)
    if not exists(arrays):
      raise ValueError(f"field '{field}' was not requested.")
    if not exists(arrays[sensor]):
      raise ValueError(f"field '{field}' is not available for sensor {sensor}.")
    arr = arrays[sensor].numpy()
    height, width = self.res[sensor]
    channels = _VEC_FIELDS.get(field, 1)
    shape = (arr.shape[0], height, width) if channels == 1 else (arr.shape[0], height, width, channels)
    return arr.reshape(shape)  # (nworld, h * w[, C]) -> (nworld, h, w[, C])

  def torch(self, field: str, sensor: int = 0, channels_first: bool = False):
    """Returns one field of one sensor as a PyTorch tensor (zero-copy).

    Args:
      field: One of "depth", "force", "normal", "rgb", "marker", "flow".
      sensor: Sensor index.
      channels_first: If True and the field has channels (e.g. RGB), returns
        shape (nworld, C, height, width) instead of (nworld, height, width, C).

    Returns:
      A PyTorch tensor of shape (nworld, height, width) or (nworld, [C,] H, W).
    """
    if field not in _FIELDS:
      raise ValueError(f"unknown field '{field}'; expected one of {_FIELDS}.")
    arrays = getattr(self, field)
    if not exists(arrays):
      raise ValueError(f"field '{field}' was not requested.")
    if not exists(arrays[sensor]):
      raise ValueError(f"field '{field}' is not available for sensor {sensor}.")
    tensor = wp.to_torch(arrays[sensor])
    height, width = self.res[sensor]
    channels = _VEC_FIELDS.get(field, 1)
    if channels == 1:
      return tensor.view(-1, height, width)  # (nworld, h * w) -> (nworld, h, w)
    tensor = tensor.view(-1, height, width, channels)  # (nworld, h * w, C) -> (nworld, h, w, C)
    if channels_first:
      tensor = tensor.permute(0, 3, 1, 2)
    return tensor

  def display_rgb(self, sensor: int = 0, gamma: float = 2.2) -> np.ndarray:
    """Returns the RGB image tone-mapped to uint8 for display.

    Args:
      sensor: Sensor index.
      gamma: Display gamma.

    Returns:
      A uint8 array of shape (nworld, height, width, 3).
    """
    linear = np.clip(self.numpy("rgb", sensor), 0.0, 1.0)
    return (np.power(linear, 1.0 / gamma) * 255.0 + 0.5).astype(np.uint8)


def _quat_to_mat(quat: Sequence[float]) -> np.ndarray:
  w, x, y, z = (float(v) for v in quat)
  return np.array(
    [
      [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
      [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
      [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ],
    dtype=np.float64,
  )


def _mesh_geom_extent(m: Model, geom_id: int) -> tuple[tuple[float, float], float, np.ndarray, np.ndarray]:
  """Returns the authored footprint, top z and the authored-to-local maps of a mesh geom.

  MuJoCo stores mesh vertices in a principal frame and relates it to the frame
  the mesh was authored in via `mesh_pos`/`mesh_quat`. Gel geometry is defined in
  the authored frame, so this returns the footprint and top z there along with
  the rotation and offset converting authored coordinates to the geom-local
  coordinates used by the kernels.
  """
  dataid = int(m.geom_dataid.numpy()[0, geom_id])
  vertadr = int(m.mesh_vertadr.numpy()[dataid])
  vertnum = int(m.mesh_vertnum.numpy()[dataid])
  verts = m.mesh_vert.numpy()[vertadr : vertadr + vertnum].astype(np.float64)
  mesh_pos = m.mesh_pos.numpy()[dataid].astype(np.float64)
  rot = _quat_to_mat(m.mesh_quat.numpy()[dataid])
  authored = verts @ rot.T + mesh_pos
  footprint = (float(np.ptp(authored[:, 0])), float(np.ptp(authored[:, 1])))
  rot_la = rot.T
  return footprint, float(authored[:, 2].max()), rot_la, -rot_la @ mesh_pos


def _geom_extent(m: Model, geom_id: int) -> tuple[tuple[float, float], float, np.ndarray, np.ndarray]:
  """Returns the gel footprint, top z in the authored frame and the authored-to-local maps."""
  geom_type = GeomType(int(m.geom_type.numpy()[geom_id]))
  size = m.geom_size.numpy()[0, geom_id].astype(np.float64)
  identity = np.eye(3)
  zero = np.zeros(3)
  if geom_type == GeomType.BOX:
    return (float(size[0]), float(size[1])), float(size[2]), identity, zero
  if geom_type == GeomType.SPHERE:
    return (float(size[0]), float(size[0])), float(size[0]), identity, zero
  if geom_type == GeomType.CAPSULE:
    return (float(size[0]), float(size[0])), float(size[0] + size[1]), identity, zero
  if geom_type == GeomType.CYLINDER:
    return (float(size[0]), float(size[0])), float(size[1]), identity, zero
  if geom_type == GeomType.ELLIPSOID:
    return (float(size[0]), float(size[1])), float(size[2]), identity, zero
  if geom_type == GeomType.MESH:
    return _mesh_geom_extent(m, geom_id)
  raise ValueError(
    f"cannot infer gel geometry for geom {geom_id} of type {geom_type.name}; set gel_size and frame_pos explicitly."
  )


@dataclasses.dataclass
class _SensorRuntime:
  """Per-sensor resolved parameters."""

  pad_geom: int
  max_depth: float
  stiffness: float
  height: int
  width: int
  npix: int
  gel_width: float
  gel_height: float
  frame_pos: wp.vec3
  frame_x: wp.vec3
  frame_y: wp.vec3
  leds_pos: wp.array
  leds_rgb: wp.array
  leds_intensity: wp.array
  weights: wp.array
  radius: int
  marker_scale: float
  base_rgba: wp.vec3
  ambient: float
  attenuation: float
  specular: float
  shininess: float
  shadow_steps: int
  shadow_softness: float
  shadow_strength: float
  marker_spacing_x: float
  marker_spacing_y: float
  marker_radius: float
  marker_softness: float
  marker_alpha: float
  marker_rgb: wp.vec3
  use_markers: int


def _to_vec3(values: Sequence[float]) -> wp.vec3:
  return wp.vec3(float(values[0]), float(values[1]), float(values[2]))


def _resolve_leds(optics: GelOptics, gel_width: float, gel_height: float) -> tuple[Led, ...]:
  if exists(optics.leds):
    return optics.leds
  hx = 0.35 * gel_width
  hy = 0.35 * gel_height
  if optics.rgb_leds:
    return (
      Led((hx, 0.0, 0.01), rgb=(1.0, 0.1, 0.1)),
      Led((-hx, 0.0, 0.01), rgb=(0.1, 1.0, 0.1)),
      Led((0.0, hy, 0.01), rgb=(0.1, 0.1, 1.0)),
      Led((0.0, -hy, 0.01), rgb=(0.8, 0.8, 0.8)),
    )
  return (
    Led((hx, hy, 0.01)),
    Led((-hx, hy, 0.01)),
    Led((-hx, -hy, 0.01)),
    Led((hx, -hy, 0.01)),
  )


class VisuotactileRenderer:
  """Batched visuotactile renderer.

  Example:
    sensor = mjw.gelsight_mini(geom_id)
    renderer = mjw.VisuotactileRenderer(m, [sensor], fields=("depth", "rgb"))
    frame = renderer.render(m, d)

    # Gradients through the smooth pipeline.
    d.geom_xpos.requires_grad = True
    with wp.Tape() as tape:
      frame = renderer.render(m, d)
      loss = renderer.rgb_loss(frame, [target])
    tape.backward(loss)
  """

  def __init__(
    self,
    m: Model,
    sensors: Sequence[VisuotactileSensor] | VisuotactileSensor,
    fields: Sequence[str] = ("depth", "rgb"),
    noise: CameraSensor | None = None,
    max_geoms: int = MJ_MAXCONPAIR,
    seed: int = 0,
  ):
    """Initializes the renderer.

    Args:
      m: The model on device.
      sensors: One or more sensor specifications.
      fields: Output fields to compute and store, any of "depth", "force",
        "normal", "rgb", "marker", "flow".
      noise: Optional image sensor noise model; None renders noise-free.
      max_geoms: Maximum number of contacting geoms tracked per sensor and
        world.
      seed: Base RNG seed for the noise model.
    """
    if isinstance(sensors, VisuotactileSensor):
      sensors = (sensors,)
    elif not sensors:
      raise ValueError("at least one sensor is required.")
    unknown = [f for f in fields if f not in _FIELDS]
    if unknown:
      raise ValueError(f"unknown fields {unknown}; expected any of {_FIELDS}.")
    if max_geoms < 1:
      raise ValueError("max_geoms must be positive.")
    self.sensors = tuple(sensors)
    self.fields = tuple(fields)
    self.noise = noise
    self.max_geoms = int(max_geoms)
    self.seed = int(seed)
    self._needs_optics = any(f in self.fields for f in ("rgb", "normal", "marker", "flow"))

    ngeom = m.geom_type.shape[0]
    self._pad_body = np.zeros(len(sensors), dtype=np.int32)
    self._rt = []
    for i, spec in enumerate(self.sensors):
      if not 0 <= spec.geom < ngeom:
        raise ValueError(f"geom {spec.geom} is out of range (ngeom={ngeom}).")
      geom_type = GeomType(int(m.geom_type.numpy()[spec.geom]))
      if geom_type in _SUPPORTED_GEOMS:
        footprint, top, rot_la, pos_l = _geom_extent(m, spec.geom)
      elif exists(spec.gel_size) and exists(spec.frame_pos):
        footprint, top, rot_la, pos_l = (0.0, 0.0), 0.0, np.eye(3), np.zeros(3)
      else:
        raise ValueError(f"sensor {i}: geom {spec.geom} has type {geom_type.name}; set gel_size and frame_pos explicitly.")
      gel_size = (float(spec.gel_size[0]), float(spec.gel_size[1])) if exists(spec.gel_size) else footprint
      if exists(spec.frame_pos):
        frame_pos_auth = np.asarray(spec.frame_pos, dtype=np.float64)
      else:
        frame_pos_auth = np.array([0.0, 0.0, top], dtype=np.float64)
      # gel geometry is authored in the file frame; convert to geom-local
      rot = rot_la @ _quat_to_mat(spec.frame_quat)
      frame_pos = rot_la @ frame_pos_auth + pos_l

      height, width = spec.resolution
      optics = spec.optics
      leds = _resolve_leds(optics, gel_size[0], gel_size[1])
      if optics.smoothing > 0.0:
        radius = max(1, int(np.ceil(3.0 * optics.smoothing)))
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        weights = np.exp(-0.5 * (offsets / optics.smoothing) ** 2)
        weights /= weights.sum()
      else:
        radius = 0
        weights = np.ones(1, dtype=np.float64)
      markers = spec.markers
      if exists(markers):
        marker_scale = markers.flow_scale if markers.flow_scale > 0.0 else spec.gel_thickness
        marker_spacing_x = float(markers.spacing[0])
        marker_spacing_y = float(markers.spacing[1])
        marker_radius = float(markers.radius)
        marker_softness = float(markers.softness)
        marker_alpha = float(markers.alpha)
        marker_rgb = _to_vec3(markers.rgb)
        use_markers = 1
      else:
        marker_scale = 0.0
        marker_spacing_x = 1.0
        marker_spacing_y = 1.0
        marker_radius = 0.0
        marker_softness = 1.0
        marker_alpha = 0.0
        marker_rgb = wp.vec3(0.0, 0.0, 0.0)
        use_markers = 0

      self._pad_body[i] = int(m.geom_bodyid.numpy()[spec.geom])
      self._rt.append(
        _SensorRuntime(
          pad_geom=spec.geom,
          max_depth=float(spec.gel_thickness),
          stiffness=float(spec.stiffness),
          height=height,
          width=width,
          npix=height * width,
          gel_width=gel_size[0],
          gel_height=gel_size[1],
          frame_pos=_to_vec3(frame_pos),
          frame_x=_to_vec3(rot[:, 0]),
          frame_y=_to_vec3(rot[:, 1]),
          leds_pos=wp.array([_to_vec3(l.pos) for l in leds], dtype=wp.vec3),
          leds_rgb=wp.array([_to_vec3(l.rgb) for l in leds], dtype=wp.vec3),
          leds_intensity=wp.array([float(l.intensity) for l in leds], dtype=float),
          weights=wp.array(weights, dtype=float),
          radius=radius,
          marker_scale=marker_scale,
          base_rgba=_to_vec3(optics.base_rgba),
          ambient=float(optics.ambient),
          attenuation=float(optics.attenuation),
          specular=float(optics.specular),
          shininess=float(optics.shininess),
          shadow_steps=int(optics.shadow_steps),
          shadow_softness=float(optics.shadow_softness),
          shadow_strength=float(optics.shadow_strength),
          marker_spacing_x=marker_spacing_x,
          marker_spacing_y=marker_spacing_y,
          marker_radius=marker_radius,
          marker_softness=marker_softness,
          marker_alpha=marker_alpha,
          marker_rgb=marker_rgb,
          use_markers=use_markers,
        )
      )
    self._pad_body = wp.array(self._pad_body, dtype=int)
    self._nwords = (m.geom_type.shape[0] + 31) // 32
    self._cache_nworld = None
    self._buf = None
    self._noise_params = None
    if exists(self.noise):
      self._noise_params = (
        float(self.noise.exposure),
        float(self.noise.gain),
        float(self.noise.full_well_e),
        float(self.noise.read_noise_e),
        int(self.noise.seed) + self.seed,
      )

  @property
  def gel_sizes(self) -> tuple[tuple[float, float], ...]:
    """Resolved gel extents (width, height) per sensor, in meters."""
    return tuple((rt.gel_width, rt.gel_height) for rt in self._rt)

  @property
  def npixels(self) -> tuple[int, ...]:
    """Pixels per sensor image."""
    return tuple(rt.npix for rt in self._rt)

  def allocate(self, nworld: int) -> VisuotactileFrame:
    """Allocates a frame of output buffers for `render(..., out=frame)`.

    Args:
      nworld: Number of worlds.

    Returns:
      A frame with one buffer per requested field and sensor. Differentiable
      fields are allocated with `requires_grad=True`.
    """
    frame = VisuotactileFrame(res=tuple((rt.height, rt.width) for rt in self._rt))
    for field in self.fields:
      dtype = _FIELD_DTYPE[field]
      requires_grad = field in _DIFF_FIELDS
      arrays = [wp.zeros((nworld, rt.npix), dtype=dtype, requires_grad=requires_grad) for rt in self._rt]
      setattr(frame, field, arrays)
    return frame

  def _backing(self, nworld: int) -> dict:
    """Returns reused backing buffers, reallocating when nworld changes."""
    if self._cache_nworld != nworld:
      sensors = []
      for rt in self._rt:
        bufs = {}
        if "depth" not in self.fields:
          bufs["depth"] = wp.zeros((nworld, rt.npix), dtype=float, requires_grad=True)
        if "force" not in self.fields:
          bufs["force"] = wp.zeros((nworld, rt.npix), dtype=float, requires_grad=True)
        if self._needs_optics:
          if "rgb" not in self.fields or exists(self.noise):
            bufs["rgb"] = wp.zeros((nworld, rt.npix), dtype=wp.vec3, requires_grad=True)
          if "normal" not in self.fields:
            bufs["normal"] = wp.zeros((nworld, rt.npix), dtype=wp.vec3, requires_grad=True)
          if "marker" not in self.fields:
            bufs["marker"] = wp.zeros((nworld, rt.npix), dtype=float)
          if "flow" not in self.fields:
            bufs["flow"] = wp.zeros((nworld, rt.npix), dtype=wp.vec2, requires_grad=True)
          if rt.radius > 0:
            bufs["tmp"] = wp.zeros((nworld, rt.npix), dtype=float, requires_grad=True)
            bufs["smoothed"] = wp.zeros((nworld, rt.npix), dtype=float, requires_grad=True)
        sensors.append(bufs)
      self._buf = {
        "seen": wp.zeros((nworld, len(self._rt), self._nwords), dtype=wp.uint32),
        "count": wp.zeros((nworld, len(self._rt)), dtype=int),
        "cand": wp.full((nworld, len(self._rt), self.max_geoms), -1, dtype=int),
        "sensors": sensors,
      }
      self._cache_nworld = nworld
    return self._buf

  def _validate_out(self, out: VisuotactileFrame, nworld: int):
    res = tuple((rt.height, rt.width) for rt in self._rt)
    if out.res != res:
      raise ValueError(f"out.res {out.res} does not match renderer res {res}.")
    for field in self.fields:
      arrays = getattr(out, field)
      if not exists(arrays) or len(arrays) != len(self._rt):
        raise ValueError(f"out is missing the '{field}' field.")
      for arr, rt in zip(arrays, self._rt):
        if tuple(arr.shape) != (nworld, rt.npix):
          raise ValueError(f"out.{field} shape {arr.shape} does not match {(nworld, rt.npix)}.")

  @event_scope
  def render(self, m: Model, d: Data, out: VisuotactileFrame | None = None) -> VisuotactileFrame:
    """Renders all sensors.

    Internal scratch buffers are reused while `nworld` is unchanged, so back out
    a pending `wp.Tape` before the next `render` (warp tapes do not copy their
    inputs).

    Args:
      m: The model on device.
      d: The data on device; contacts must be current (e.g. after `step1`).
      out: Optional frame from `allocate` holding reused output buffers. This
        allocation-free path suits long RL loops and CUDA graph capture; the
        buffers (and their gradients) are overwritten on every call.

    Returns:
      The rendered fields.
    """
    nworld = d.nworld
    nsens = len(self.sensors)
    if exists(out):
      self._validate_out(out, nworld)
    else:
      out = self.allocate(nworld)

    buf = self._backing(nworld)
    seen, count, cand = buf["seen"], buf["count"], buf["cand"]
    seen.zero_()
    count.zero_()
    cand.fill_(-1)
    wp.launch(
      _contact_gather_bits,
      dim=(nsens, d.naconmax),
      inputs=[m.geom_bodyid, d.contact.geom, d.contact.worldid, d.nacon, self._pad_body],
      outputs=[seen],
    )
    wp.launch(
      _contact_gather_list,
      dim=(nworld, nsens, self._nwords),
      inputs=[self.max_geoms, seen],
      outputs=[count, cand],
    )

    for i, rt in enumerate(self._rt):
      bufs = buf["sensors"][i]
      depth = out.depth[i] if exists(out.depth) else bufs["depth"]
      force = out.force[i] if exists(out.force) else bufs["force"]
      wp.launch(
        _depth_kernel,
        dim=(nworld, rt.npix),
        inputs=[
          m.oct_child,
          m.oct_aabb,
          m.oct_coeff,
          m.geom_type,
          m.geom_dataid,
          m.geom_size,
          m.mesh_octadr,
          m.mesh_pos,
          m.mesh_quat,
          m.plugin,
          m.plugin_attr,
          m.geom_plugin_index,
          d.geom_xpos,
          d.geom_xmat,
          count,
          cand,
          i,
          self.max_geoms,
          rt.pad_geom,
          rt.frame_pos,
          rt.frame_x,
          rt.frame_y,
          rt.width,
          rt.height,
          rt.gel_width,
          rt.gel_height,
          rt.max_depth,
          rt.stiffness,
        ],
        outputs=[depth, force],
      )

      if not self._needs_optics:
        continue

      smoothed = depth
      if rt.radius > 0:
        tmp = bufs["tmp"]
        smoothed = bufs["smoothed"]
        wp.launch(
          _blur_kernel,
          dim=(nworld, rt.npix),
          inputs=[depth, rt.weights, rt.radius, rt.width, rt.height, 0],
          outputs=[tmp],
        )
        wp.launch(
          _blur_kernel,
          dim=(nworld, rt.npix),
          inputs=[tmp, rt.weights, rt.radius, rt.width, rt.height, 1],
          outputs=[smoothed],
        )

      rgb = out.rgb[i] if exists(out.rgb) else bufs["rgb"]
      rgb_shaded = bufs["rgb"] if exists(self.noise) and exists(out.rgb) else rgb
      normal = out.normal[i] if exists(out.normal) else bufs["normal"]
      marker = out.marker[i] if exists(out.marker) else bufs["marker"]
      flow = out.flow[i] if exists(out.flow) else bufs["flow"]
      wp.launch(
        _shade_kernel,
        dim=(nworld, rt.npix),
        inputs=[
          smoothed,
          rt.leds_pos,
          rt.leds_rgb,
          rt.leds_intensity,
          rt.base_rgba,
          rt.marker_rgb,
          rt.width,
          rt.height,
          rt.gel_width,
          rt.gel_height,
          rt.ambient,
          rt.attenuation,
          rt.specular,
          rt.shininess,
          rt.shadow_steps,
          rt.shadow_softness,
          rt.shadow_strength,
          rt.marker_spacing_x,
          rt.marker_spacing_y,
          rt.marker_radius,
          rt.marker_softness,
          rt.marker_alpha,
          rt.marker_scale,
          rt.use_markers,
        ],
        outputs=[rgb_shaded, normal, marker, flow],
      )

      if exists(self._noise_params) and exists(out.rgb):
        wp.launch(
          _noise_kernel,
          dim=(nworld, rt.npix),
          inputs=[rgb_shaded, *self._noise_params],
          outputs=[rgb],
        )

    return out

  def mse(self, pred: wp.array2d, target: wp.array2d) -> wp.array:
    """Returns the scalar MSE between two arrays as a `Tape.backward` root.

    Args:
      pred: Prediction array, float or vec3.
      target: Target array with the same shape and dtype.

    Returns:
      A scalar loss array with `requires_grad=True`.
    """
    if pred.shape != target.shape:
      raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}.")
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    inv_count = 1.0 / float(pred.shape[0] * pred.shape[1])
    if pred.dtype == wp.vec3:
      kernel = _mse_vec3_kernel
    elif pred.dtype in (wp.float32, wp.float64, float):
      kernel = _mse_float_kernel
    else:
      raise ValueError(f"mse supports float and vec3 arrays, got {pred.dtype}.")
    wp.launch(kernel, dim=pred.shape, inputs=[pred, target, inv_count], outputs=[loss])
    return loss

  def rgb_loss(self, frame: VisuotactileFrame, targets: Sequence[wp.array2d]) -> wp.array:
    """Returns the mean RGB MSE over all sensors as a `Tape.backward` root.

    Args:
      frame: A frame with the "rgb" field.
      targets: Per-sensor target images with the same shape as `frame.rgb`.

    Returns:
      A scalar loss array with `requires_grad=True`.
    """
    if not exists(frame.rgb):
      raise ValueError('rgb_loss requires the "rgb" field.')
    if len(targets) != len(self.sensors):
      raise ValueError(f"expected {len(self.sensors)} targets, got {len(targets)}.")
    total = sum(t.shape[0] * t.shape[1] for t in targets)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    for rgb, target in zip(frame.rgb, targets):
      if rgb.shape != target.shape:
        raise ValueError(f"target shape {target.shape} != rgb shape {rgb.shape}.")
      wp.launch(_mse_vec3_kernel, dim=rgb.shape, inputs=[rgb, target, 1.0 / float(total)], outputs=[loss])
    return loss


def from_tactile_sensors(
  mjm: mujoco.MjModel,
  resolution: tuple[int, int] = (120, 160),
  **kwargs,
) -> list[VisuotactileSensor]:
  """Builds sensor specs from the model's MuJoCo tactile sensors.

  Args:
    mjm: The host model containing the tactile sensors.
    resolution: Image resolution for all sensors.
    **kwargs: Additional `VisuotactileSensor` arguments.

  Returns:
    One `VisuotactileSensor` per MuJoCo `<tactile>` sensor.
  """
  sensor_ids = np.flatnonzero(mjm.sensor_type == mujoco.mjtSensor.mjSENS_TACTILE)
  if len(sensor_ids) == 0:
    raise ValueError("the model has no MuJoCo tactile sensors.")
  return [VisuotactileSensor(geom=int(mjm.sensor_refid[i]), resolution=resolution, **kwargs) for i in sensor_ids]


def _preset(
  geom: int,
  resolution: tuple[int, int],
  rgb_leds: bool,
  gel_size: tuple[float, float],
  markers: MarkerPattern | None,
  ambient: float,
  **kwargs,
) -> VisuotactileSensor:
  """Shared flat-pad preset: LED ring, attenuation and soft shadows."""
  optics = default(
    kwargs.pop("optics", None),
    GelOptics(ambient=ambient, attenuation=0.02, shadow_steps=16, shadow_softness=0.002, rgb_leds=rgb_leds),
  )
  defaults = dict(gel_size=gel_size, gel_thickness=0.004, optics=optics, markers=markers)
  defaults.update(kwargs)
  return VisuotactileSensor(geom=geom, resolution=resolution, **defaults)


def gelsight_mini(
  geom: int,
  resolution: tuple[int, int] = (240, 320),
  rgb_leds: bool = False,
  **kwargs,
) -> VisuotactileSensor:
  """Returns sensor parameters for a GelSight Mini-class flat-pad sensor.

  The defaults are approximate and intended as a starting point; tune
  `gel_size`, `gel_thickness` and the optics to your unit.

  Args:
    geom: Pad collision geom id.
    resolution: Image resolution (height, width).
    rgb_leds: If True, uses red/green/blue directional illumination.
    **kwargs: Overrides for `VisuotactileSensor` fields.

  Returns:
    The sensor specification.
  """
  return _preset(
    geom,
    resolution,
    rgb_leds,
    gel_size=(0.016, 0.012),
    markers=MarkerPattern(spacing=(0.0016, 0.0016), radius=0.00035),
    ambient=0.25,
    **kwargs,
  )


def digit(
  geom: int,
  resolution: tuple[int, int] = (240, 320),
  rgb_leds: bool = False,
  **kwargs,
) -> VisuotactileSensor:
  """Returns sensor parameters for a DIGIT-class flat-pad sensor.

  The defaults are approximate and intended as a starting point; tune
  `gel_size`, `gel_thickness` and the optics to your unit.

  Args:
    geom: Pad collision geom id.
    resolution: Image resolution (height, width).
    rgb_leds: If True, uses red/green/blue directional illumination.
    **kwargs: Overrides for `VisuotactileSensor` fields.

  Returns:
    The sensor specification.
  """
  return _preset(
    geom,
    resolution,
    rgb_leds,
    gel_size=(0.019, 0.016),
    markers=None,
    ambient=0.3,
    **kwargs,
  )
