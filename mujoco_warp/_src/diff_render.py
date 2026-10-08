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
"""Differentiable rendering.

`DifferentiableRenderer` produces images whose analytic gradients can be taken
with `warp.Tape` w.r.t. scene parameters: geom poses (`d.geom_xpos`,
`d.geom_xmat`), geom sizes, material albedo/emission, light parameters and the
camera pose.

Visibility is made differentiable with a soft closest-hit: every enabled
primitive's intersection distance is evaluated and blended with a softmin at
temperature `temperature`. Spheres use a distance that is continuous across the
silhouette, so geometry gradients are correct as objects move in and out of a
ray's path; other primitives fall back to hard misses. Shading is Lambertian
direct lighting plus emission and background; shadows and textures are omitted.
The image is written per camera in separate launches so no data-dependent output
index is computed inside the kernel (Warp's adjoint drops those gradients).

The renderer consumes the same `d.geom_xpos` / `d.geom_xmat` tensors as the
rest of MJWarp, so it composes with physics under the chain rule:
`dL/dq = (dL/dxpos) . (dxpos/dq)`, with the second factor provided by the
physics engine (`derivative.py`, analytic Jacobians, or finite differences).
Requires primitive geoms only (plane, sphere, box, ellipsoid, capsule,
cylinder); meshes, heightfields and flexes are not differentiable here.
"""

from typing import Tuple

import numpy as np
import warp as wp

from mujoco_warp._src.ray import ray_box
from mujoco_warp._src.ray import ray_capsule
from mujoco_warp._src.ray import ray_cylinder
from mujoco_warp._src.ray import ray_ellipsoid
from mujoco_warp._src.ray import ray_plane
from mujoco_warp._src.render import _geom_rgba
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import RenderContext
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"default_grid_stride": False, "enable_backward": True})

_PI = 3.141592653589793
_FAR = 1.0e6
_NGEOM_TYPES = (GeomType.PLANE, GeomType.SPHERE, GeomType.BOX, GeomType.ELLIPSOID, GeomType.CAPSULE, GeomType.CYLINDER)


@wp.func
def _geom_distance(
  # Model:
  geom_type: wp.array[int],
  geom_size: wp.array2d[wp.vec3],
  # Data in:
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  # In:
  worldid: int,
  gi: int,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> Tuple[float, wp.vec3]:
  d = float(-1.0)
  n = wp.vec3(0.0, 0.0, 1.0)
  gtype = geom_type[gi]
  size = geom_size[worldid % geom_size.shape[0], gi]
  if gtype == GeomType.PLANE:
    d, n = ray_plane(geom_xpos_in[worldid, gi], geom_xmat_in[worldid, gi], size, pnt, vec)
  elif gtype == GeomType.SPHERE:
    # Smooth sphere distance: continuous across the silhouette so that gradient
    # flows when geometry moves in and out of a ray's path.
    oc = geom_xpos_in[worldid, gi] - pnt
    t = wp.dot(oc, vec)
    d2 = wp.dot(oc, oc) - t * t
    r = size[0]
    d = t - wp.sqrt(wp.max(r * r - d2, 0.0)) + wp.sqrt(wp.max(d2 - r * r, 0.0))
    n = wp.normalize(pnt + t * vec - geom_xpos_in[worldid, gi])
  elif gtype == GeomType.BOX:
    d, _, n = ray_box(geom_xpos_in[worldid, gi], geom_xmat_in[worldid, gi], size, pnt, vec)
  elif gtype == GeomType.ELLIPSOID:
    d, n = ray_ellipsoid(geom_xpos_in[worldid, gi], geom_xmat_in[worldid, gi], size, pnt, vec)
  elif gtype == GeomType.CAPSULE:
    d, n = ray_capsule(geom_xpos_in[worldid, gi], geom_xmat_in[worldid, gi], size, pnt, vec)
  elif gtype == GeomType.CYLINDER:
    d, n = ray_cylinder(geom_xpos_in[worldid, gi], geom_xmat_in[worldid, gi], size, pnt, vec)
  return d, n


def _build_kernel(m: Model, rc: RenderContext, temperature: float):
  """Builds the specialized differentiable shading kernel."""
  ngeom = rc.enabled_geom_ids.shape[0]
  M_NLIGHT = m.nlight
  bg = int(rc.background_color)
  background = wp.vec3(
    float((bg >> 16) & 0xFF) / 255.0,
    float((bg >> 8) & 0xFF) / 255.0,
    float(bg & 0xFF) / 255.0,
  )
  light_attenuation_is_default = rc.light_attenuation_is_default

  @wp.kernel(module="unique", enable_backward=True, grid_stride=False)
  def _diff_render_kernel(
    # Model:
    geom_type: wp.array[int],
    geom_matid: wp.array2d[int],
    geom_size: wp.array2d[wp.vec3],
    geom_rgba: wp.array2d[wp.vec4],
    light_type: wp.array2d[int],
    light_active: wp.array2d[bool],
    light_attenuation: wp.array2d[wp.vec3],
    light_diffuse: wp.array2d[wp.vec3],
    mat_emission: wp.array2d[float],
    mat_rgba: wp.array2d[wp.vec4],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    cam_xpos_in: wp.array2d[wp.vec3],
    cam_xmat_in: wp.array2d[wp.mat33],
    light_xpos_in: wp.array2d[wp.vec3],
    light_xdir_in: wp.array2d[wp.vec3],
    # In:
    enabled_geom_ids: wp.array[int],
    ray: wp.array[wp.vec3],
    ray0: int,
    pix0: int,
    camid: int,
    # Out:
    rgb_out: wp.array2d[wp.vec3],
  ):
    worldid, rayid = wp.tid()

    ray_dir_world = cam_xmat_in[worldid, camid] @ ray[ray0 + rayid]
    ray_origin_world = cam_xpos_in[worldid, camid]

    # Soft closest hit: blend primitive intersections with a softmin.
    if wp.static(ngeom > 0):
      d_min = float(_FAR)
      for i in range(ngeom):
        d, _ = _geom_distance(
          geom_type, geom_size, geom_xpos_in, geom_xmat_in, worldid, enabled_geom_ids[i], ray_origin_world, ray_dir_world
        )
        d_min = wp.min(d_min, wp.where(d >= 0.0, d, _FAR))

      w_sum = float(0.0)
      hit_point = wp.vec3(0.0, 0.0, 0.0)
      normal = wp.vec3(0.0, 0.0, 0.0)
      albedo = wp.vec3(0.0, 0.0, 0.0)
      emission = wp.vec3(0.0, 0.0, 0.0)
      for i in range(ngeom):
        gi = enabled_geom_ids[i]
        d, n = _geom_distance(geom_type, geom_size, geom_xpos_in, geom_xmat_in, worldid, gi, ray_origin_world, ray_dir_world)
        d = wp.where(d >= 0.0, d, _FAR)
        w = wp.exp(-(d - d_min) / temperature)
        color, mat_id = _geom_rgba(geom_matid, geom_rgba, mat_rgba, worldid, gi)
        albedo_i = wp.vec3(color[0], color[1], color[2])
        emis_i = float(0.0)
        if mat_id >= 0:
          emis_i = mat_emission[worldid % mat_emission.shape[0], mat_id]
        hit_point += w * (ray_origin_world + d * ray_dir_world)
        normal += w * n
        albedo += w * albedo_i
        emission += w * (albedo_i * emis_i)
        w_sum += w

      hit_point = hit_point / w_sum
      normal = wp.normalize(normal)
      albedo = albedo / w_sum
      emission = emission / w_sum

      # Direct lighting (Lambertian, no shadows).
      irradiance = wp.vec3(0.0, 0.0, 0.0)
      if wp.static(M_NLIGHT > 0):
        for li in range(wp.static(M_NLIGHT)):
          if light_active[worldid % light_active.shape[0], li]:
            l_dir = wp.vec3(0.0, 0.0, 1.0)
            light_radiance = wp.vec3(0.0, 0.0, 0.0)
            if light_type[worldid % light_type.shape[0], li] == 1:  # directional
              l_dir = -light_xdir_in[worldid, li]
              light_radiance = light_diffuse[worldid % light_diffuse.shape[0], li] * _PI
            else:
              to_light = light_xpos_in[worldid, li] - hit_point
              dist_light = wp.length(to_light)
              l_dir = to_light / wp.max(dist_light, 1.0e-9)
              attenuation = 1.0
              if wp.static(not light_attenuation_is_default):
                att = light_attenuation[worldid % light_attenuation.shape[0], li]
                attenuation = wp.safe_div(1.0, wp.dot(wp.vec3(1.0, dist_light, dist_light * dist_light), att))
              light_radiance = light_diffuse[worldid % light_diffuse.shape[0], li] * _PI * attenuation
            ndotl = wp.max(wp.dot(normal, l_dir), 0.0)
            irradiance += light_radiance * ndotl

      # Hard visibility against the background: straight-through at silhouettes.
      background_weight = wp.where(d_min >= _FAR, 1.0, 0.0)
      radiance = emission + wp.cw_mul(albedo, irradiance) * (1.0 / _PI)
      rgb_out[worldid, pix0 + rayid] = radiance * (1.0 - background_weight) + background * background_weight
    else:
      rgb_out[worldid, pix0 + rayid] = wp.static(background)

  return _diff_render_kernel


@wp.kernel(module="unique", enable_backward=True)
def _mse_kernel(
  # In:
  pred: wp.array2d[wp.vec3],
  target: wp.array2d[wp.vec3],
  inv_count: float,
  # Out:
  loss_out: wp.array[float],
):
  worldid, pixel = wp.tid()
  diff = pred[worldid, pixel] - target[worldid, pixel]
  wp.atomic_add(loss_out, 0, wp.dot(diff, diff) * inv_count)


class DifferentiableRenderer:
  """Soft ray-traced renderer with analytic adjoints.

  Example:
    d.geom_xpos.requires_grad = True
    renderer = mjw.DifferentiableRenderer(m, rc)
    with wp.Tape() as tape:
      img = renderer.render(m, d)
      loss = renderer.loss(img, target)
    tape.backward(loss)
    grad = d.geom_xpos.grad  # dL/dgeom_xpos
  """

  def __init__(self, m: Model, rc: RenderContext, temperature: float = 0.05):
    """Initializes the renderer.

    Args:
      m: The model on device.
      rc: Render context providing camera rays and enabled geoms. Textures,
        shadows, splats and non-primitive geoms are ignored.
      temperature: Softmin temperature in meters. Smaller values approach hard
        closest-hit at the cost of noisier gradients.
    """
    if temperature <= 0.0:
      raise ValueError("temperature must be positive.")
    if rc.rgb_data.shape[1] == 0:
      raise ValueError("DifferentiableRenderer requires render_rgb=True for at least one camera.")
    enabled_types = np.unique(m.geom_type.numpy()[rc.enabled_geom_ids.numpy()])
    unsupported = [GeomType(int(t)).name for t in enabled_types if int(t) not in _NGEOM_TYPES]
    if unsupported:
      raise NotImplementedError(f"DifferentiableRenderer does not support geom types: {unsupported}.")
    self.rc = rc
    self.temperature = temperature
    self._cam_res = rc.cam_res.numpy()
    self._cam_ids = rc.cam_id_map.numpy()
    self._render_rgb = rc.render_rgb.numpy()
    self._rgb_adr = rc.rgb_adr.numpy()
    self._ray_offsets = []
    offset = 0
    for idx in range(rc.nrender):
      self._ray_offsets.append(offset)
      offset += int(self._cam_res[idx][0]) * int(self._cam_res[idx][1])
    self._kernel = _build_kernel(m, rc, temperature)

  @event_scope
  def render(self, m: Model, d: Data) -> wp.array2d[wp.vec3]:
    """Renders linear radiance (nworld, n_pixel) differentiably.

    Args:
      m: The model on device.
      d: The data on device.

    Returns:
      Per-pixel radiance for all RGB cameras, concatenated in camera order.
    """
    rc = self.rc
    rgb_out = wp.zeros((d.nworld, rc.rgb_data.shape[1]), dtype=wp.vec3, requires_grad=True)
    for idx in range(rc.nrender):
      npix = int(self._cam_res[idx][0]) * int(self._cam_res[idx][1])
      if bool(self._render_rgb[idx]):
        wp.launch(
          kernel=self._kernel,
          dim=(d.nworld, npix),
          inputs=[
            m.geom_type,
            m.geom_matid,
            m.geom_size,
            m.geom_rgba,
            m.light_type,
            m.light_active,
            m.light_attenuation,
            m.light_diffuse,
            m.mat_emission,
            m.mat_rgba,
            d.geom_xpos,
            d.geom_xmat,
            d.cam_xpos,
            d.cam_xmat,
            d.light_xpos,
            d.light_xdir,
            rc.enabled_geom_ids,
            rc.ray,
            self._ray_offsets[idx],
            int(self._rgb_adr[idx]),
            int(self._cam_ids[idx]),
          ],
          outputs=[rgb_out],
        )
    return rgb_out

  def loss(self, image: wp.array2d[wp.vec3], target: wp.array2d[wp.vec3]) -> wp.array:
    """Returns the scalar mean squared error between two radiance images.

    Args:
      image: Rendered radiance (nworld, n_pixel).
      target: Target radiance with the same shape.

    Returns:
      A scalar loss array with `requires_grad=True`, as a `Tape.backward` root.
    """
    if image.shape != target.shape:
      raise ValueError(f"image shape {image.shape} != target shape {target.shape}.")
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    wp.launch(
      _mse_kernel,
      dim=image.shape,
      inputs=[image, target, 1.0 / float(image.shape[0] * image.shape[1])],
      outputs=[loss],
    )
    return loss
