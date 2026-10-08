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
"""Physically based Monte Carlo path tracer.

`PathTracer` renders a `RenderContext` with multi-bounce global illumination
instead of direct lighting only. It shares the renderer's scene BVHs, camera
rays, textures and geom support, and adds:

  - a Lambertian + GGX microfacet BSDF with Schlick Fresnel and Smith masking,
  - next-event estimation against MuJoCo lights (directional, point, spot),
  - emissive materials and the MuJoCo skybox as radiance sources,
  - Russian roulette termination,
  - a linear HDR radiance buffer with progressive accumulation across calls,
  - exposure and ACES tone mapping into `rc.rgb_data` for downstream consumers.

Rendering is nworld-batched: every world renders its own randomized scene with
its own materials and lights in a single kernel launch.
"""

from typing import Tuple

import warp as wp

from mujoco_warp._src.bvh import refit_bvh
from mujoco_warp._src.camera import CameraModel
from mujoco_warp._src.camera import CameraMotion
from mujoco_warp._src.camera import CameraSensor
from mujoco_warp._src.camera import distort_direction
from mujoco_warp._src.camera import sample_lens_offset
from mujoco_warp._src.camera import sensor_sample
from mujoco_warp._src.camera import thin_lens_direction
from mujoco_warp._src.denoise import denoise
from mujoco_warp._src.render import _geom_rgba
from mujoco_warp._src.render import _make_cast_ray
from mujoco_warp._src.render import _make_sample_texture
from mujoco_warp._src.render import sample_skybox
from mujoco_warp._src.render import shade_splats
from mujoco_warp._src.render_util import pack_rgba_to_uint32
from mujoco_warp._src.types import MJ_MAXVAL
from mujoco_warp._src.types import Data
from mujoco_warp._src.types import GeomType
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import RenderContext
from mujoco_warp._src.warp_util import event_scope

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})

_PI = 3.141592653589793
_INV_PI = 1.0 / _PI
# MuJoCo's default material parameters, mirrored from render.py.
_DEFAULT_SPECULAR = 0.5
_DEFAULT_SHININESS = 0.5
_MAX_SHININESS = 128.0
# Dielectric F0 for MuJoCo's default specular = 0.5.
_SPECULAR_TO_F0 = 0.08
_MIN_ROUGHNESS = 0.02
# Radiance scale: MuJoCo diffuse shading is albedo * light_diffuse * cos, so a
# physical BRDF (albedo / pi) needs light radiance = light_diffuse * pi.
_LIGHT_RADIANCE_SCALE = _PI
# Shadow ray offset along the shading normal.
_SHADOW_EPS = 1.0e-4


@wp.func
def _luminance(c: wp.vec3) -> float:
  return wp.dot(c, wp.vec3(0.2126, 0.7152, 0.0722))


@wp.func
def _interp_camera(
  # In:
  pos0: wp.vec3,
  pos1: wp.vec3,
  mat0: wp.mat33,
  mat1: wp.mat33,
  t: float,
) -> Tuple[wp.vec3, wp.mat33]:
  # Lerp translation and slerp rotation between shutter endpoints.
  pos = pos0 * (1.0 - t) + pos1 * t
  q0 = wp.quat_from_matrix(mat0)
  q1 = wp.quat_from_matrix(mat1)
  if wp.dot(q0, q1) < 0.0:
    q1 = -q1
  return pos, wp.quat_to_matrix(wp.quat_slerp(q0, q1, t))


@wp.kernel(module="unique", enable_backward=False)
def _interp_geom_kernel(
  # In:
  xpos0: wp.array2d[wp.vec3],
  xpos1: wp.array2d[wp.vec3],
  xmat0: wp.array2d[wp.mat33],
  xmat1: wp.array2d[wp.mat33],
  t: float,
  # Out:
  pos_out: wp.array2d[wp.vec3],
  mat_out: wp.array2d[wp.mat33],
):
  worldid, geomid = wp.tid()
  pos_out[worldid, geomid] = xpos0[worldid, geomid] * (1.0 - t) + xpos1[worldid, geomid] * t
  q0 = wp.quat_from_matrix(xmat0[worldid, geomid])
  q1 = wp.quat_from_matrix(xmat1[worldid, geomid])
  if wp.dot(q0, q1) < 0.0:
    q1 = -q1
  mat_out[worldid, geomid] = wp.quat_to_matrix(wp.quat_slerp(q0, q1, t))


@wp.func
def _srgb_channel(u: float) -> float:
  if u <= 0.04045:
    return u * (1.0 / 12.92)
  return wp.pow((u + 0.055) * (1.0 / 1.055), 2.4)


@wp.func
def _srgb_to_linear(c: wp.vec3) -> wp.vec3:
  return wp.vec3(_srgb_channel(c[0]), _srgb_channel(c[1]), _srgb_channel(c[2]))


@wp.func
def _onb(n: wp.vec3) -> Tuple[wp.vec3, wp.vec3]:
  # Build an orthonormal basis around n.
  a = wp.where(wp.abs(n[0]) > 0.9, wp.vec3(0.0, 1.0, 0.0), wp.vec3(1.0, 0.0, 0.0))
  t = wp.normalize(wp.cross(a, n))
  b = wp.cross(n, t)
  return t, b


@wp.func
def _sample_cosine_hemisphere(n: wp.vec3, t: wp.vec3, b: wp.vec3, u1: float, u2: float) -> wp.vec3:
  r = wp.sqrt(u1)
  phi = 2.0 * _PI * u2
  local = wp.vec3(r * wp.cos(phi), r * wp.sin(phi), wp.sqrt(wp.max(0.0, 1.0 - u1)))
  return local[0] * t + local[1] * b + local[2] * n


@wp.func
def _g1_ggx(ndotx: float, alpha: float) -> float:
  if ndotx <= 0.0:
    return 0.0
  a2 = alpha * alpha
  return (2.0 * ndotx) / (ndotx + wp.sqrt(a2 + (1.0 - a2) * ndotx * ndotx))


@wp.func
def _d_ggx(ndoth: float, alpha: float) -> float:
  a2 = alpha * alpha
  d = ndoth * ndoth * (a2 - 1.0) + 1.0
  return a2 / (_PI * d * d)


@wp.func
def _fresnel_schlick(f0: wp.vec3, cos_theta: float) -> wp.vec3:
  return f0 + (wp.vec3(1.0, 1.0, 1.0) - f0) * wp.pow(wp.clamp(1.0 - cos_theta, 0.0, 1.0), 5.0)


@wp.func
def _sample_ggx_vndf(n: wp.vec3, t: wp.vec3, b: wp.vec3, v: wp.vec3, alpha: float, u1: float, u2: float) -> wp.vec3:
  # Heitz 2018, "Sampling the GGX Distribution of Visible Normals", local frame.
  v_local = wp.vec3(wp.dot(v, t), wp.dot(v, b), wp.dot(v, n))
  vh = wp.normalize(wp.vec3(alpha * v_local[0], alpha * v_local[1], v_local[2]))
  lensq = vh[0] * vh[0] + vh[1] * vh[1]
  t1 = wp.vec3(1.0, 0.0, 0.0)
  if lensq > 0.0:
    t1 = wp.vec3(-vh[1], vh[0], 0.0) / wp.sqrt(lensq)
  t2 = wp.cross(vh, t1)
  r = wp.sqrt(u1)
  phi = 2.0 * _PI * u2
  p1 = r * wp.cos(phi)
  p2 = r * wp.sin(phi)
  s = 0.5 * (1.0 + vh[2])
  p2 = (1.0 - s) * wp.sqrt(wp.max(0.0, 1.0 - p1 * p1)) + s * p2
  nh = p1 * t1 + p2 * t2 + wp.sqrt(wp.max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh
  ne = wp.normalize(wp.vec3(alpha * nh[0], alpha * nh[1], wp.max(0.0, nh[2])))
  h = ne[0] * t + ne[1] * b + ne[2] * n
  return 2.0 * wp.dot(v, h) * h - v  # reflect the view about the sampled normal


@wp.func
def _pdf_ggx_vndf(n: wp.vec3, v: wp.vec3, alpha: float, l: wp.vec3) -> float:
  ndotv = wp.dot(n, v)
  if ndotv <= 0.0:
    return 0.0
  h = v + l
  if wp.dot(h, h) <= 0.0:
    return 0.0
  ndoth = wp.dot(n, wp.normalize(h))
  if ndoth <= 0.0:
    return 0.0
  return _d_ggx(ndoth, alpha) * _g1_ggx(ndotv, alpha) / (4.0 * ndotv)


@wp.func
def _bsdf_eval(albedo: wp.vec3, f0: wp.vec3, alpha: float, n: wp.vec3, v: wp.vec3, l: wp.vec3) -> wp.vec3:
  ndotl = wp.dot(n, l)
  ndotv = wp.dot(n, v)
  if ndotl <= 0.0 or ndotv <= 0.0:
    return wp.vec3(0.0, 0.0, 0.0)
  h = v + l
  if wp.dot(h, h) <= 0.0:
    return wp.vec3(0.0, 0.0, 0.0)
  h = wp.normalize(h)
  ndoth = wp.max(wp.dot(n, h), 0.0)
  vdoth = wp.max(wp.dot(v, h), 0.0)
  f = _fresnel_schlick(f0, vdoth)
  d = _d_ggx(ndoth, alpha)
  g = _g1_ggx(ndotv, alpha) * _g1_ggx(ndotl, alpha)
  spec = f * (d * g / (4.0 * ndotv * ndotl))
  diff = wp.cw_mul(wp.vec3(1.0, 1.0, 1.0) - f, albedo) * _INV_PI
  return diff + spec


@wp.func
def _aces(x: wp.vec3) -> wp.vec3:
  # Narkowicz 2015 ACES fitted curve.
  a = 2.51
  b = 0.03
  c = 2.43
  d = 0.59
  e = 0.14
  num = wp.cw_mul(x, x * a + wp.vec3(b, b, b))
  den = wp.cw_mul(x, x * c + wp.vec3(d, d, d)) + wp.vec3(e, e, e)
  return wp.cw_div(num, den)


def _make_surface(sample_texture: wp.Function, use_textures: bool, use_vertex_normals: bool) -> wp.Function:
  """Build a func that resolves hit surface shading attributes."""

  @wp.func
  def surface(
    # Model:
    geom_type: wp.array[int],
    geom_matid: wp.array2d[int],
    geom_size: wp.array2d[wp.vec3],
    geom_rgba: wp.array2d[wp.vec4],
    mesh_faceadr: wp.array[int],
    mesh_normaladr: wp.array[int],
    mesh_normal: wp.array[wp.vec3],
    mat_texid: wp.array3d[int],
    mat_texuniform: wp.array2d[bool],
    mat_texrepeat: wp.array2d[wp.vec2],
    mat_emission: wp.array2d[float],
    mat_specular: wp.array2d[float],
    mat_shininess: wp.array2d[float],
    mat_rgba: wp.array2d[wp.vec4],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    # In:
    flex_rgba: wp.array[wp.vec4],
    mesh_facetexcoord: wp.array[wp.vec3i],
    mesh_facenormal: wp.array[wp.vec3i],
    mesh_texcoord: wp.array[wp.vec2],
    mesh_texcoord_offsets: wp.array[int],
    textures: wp.array[wp.Texture2D],
    worldid: int,
    geom_id: int,
    mesh_id: int,
    bary_u: float,
    bary_v: float,
    f: int,
    normal_in: wp.vec3,
    hit_point: wp.vec3,
  ) -> Tuple[wp.vec3, wp.vec3, wp.vec3, wp.vec3, float]:
    normal = normal_in

    if geom_id == -2:
      # Flex hit: flat color, no material or texture.
      color = flex_rgba[mesh_id]
      albedo = wp.vec3(color[0], color[1], color[2])
      f0 = wp.vec3(
        _DEFAULT_SPECULAR * _SPECULAR_TO_F0, _DEFAULT_SPECULAR * _SPECULAR_TO_F0, _DEFAULT_SPECULAR * _SPECULAR_TO_F0
      )
      alpha = wp.sqrt(2.0 / (_DEFAULT_SHININESS * _MAX_SHININESS + 2.0))
      return normal, albedo, wp.vec3(0.0, 0.0, 0.0), f0, alpha

    if wp.static(use_vertex_normals):
      gtype = geom_type[geom_id]
      if mesh_id >= 0 and f >= 0:
        if gtype == GeomType.MESH or gtype == GeomType.SDF:
          mat = geom_xmat_in[worldid, geom_id]
          face = wp.transpose(mat) @ normal
          tri = mesh_facenormal[mesh_faceadr[mesh_id] + f]
          adr = mesh_normaladr[mesh_id]
          vec = (
            mesh_normal[adr + tri[0]] * bary_u
            + mesh_normal[adr + tri[1]] * bary_v
            + mesh_normal[adr + tri[2]] * (1.0 - bary_u - bary_v)
          )
          normal = wp.normalize(mat @ vec)

    color, mat_id = _geom_rgba(geom_matid, geom_rgba, mat_rgba, worldid, geom_id)
    albedo = wp.vec3(color[0], color[1], color[2])

    if wp.static(use_textures):
      if mat_id >= 0:
        tex_id = mat_texid[worldid % mat_texid.shape[0], mat_id, 1]
        if tex_id >= 0:
          tex_color = sample_texture(
            geom_type,
            mesh_faceadr,
            geom_id,
            geom_size[worldid % geom_size.shape[0], geom_id],
            mat_texrepeat[worldid % mat_texrepeat.shape[0], mat_id],
            mat_texuniform[worldid % mat_texuniform.shape[0], mat_id],
            textures[tex_id],
            geom_xpos_in[worldid, geom_id],
            geom_xmat_in[worldid, geom_id],
            mesh_facetexcoord,
            mesh_texcoord,
            mesh_texcoord_offsets,
            hit_point,
            normal,
            bary_u,
            bary_v,
            f,
            mesh_id,
          )
          albedo = wp.cw_mul(albedo, tex_color)

    spec = _DEFAULT_SPECULAR
    shininess = _DEFAULT_SHININESS
    emission = float(0.0)
    if mat_id >= 0:
      spec = mat_specular[worldid % mat_specular.shape[0], mat_id]
      shininess = mat_shininess[worldid % mat_shininess.shape[0], mat_id]
      emission = mat_emission[worldid % mat_emission.shape[0], mat_id]

    f0 = wp.vec3(
      wp.clamp(spec, 0.0, 1.0) * _SPECULAR_TO_F0,
      wp.clamp(spec, 0.0, 1.0) * _SPECULAR_TO_F0,
      wp.clamp(spec, 0.0, 1.0) * _SPECULAR_TO_F0,
    )
    # Map the Blinn-Phong exponent to a GGX roughness.
    shin_exp = shininess * _MAX_SHININESS
    alpha = wp.clamp(wp.sqrt(2.0 / (shin_exp + 2.0)), _MIN_ROUGHNESS, 1.0)

    return normal, albedo, albedo * emission, f0, alpha

  return surface


def _build_kernel(m: Model, rc: RenderContext, max_bounces: int, roulette_start: int, camera: CameraModel, motion: wp.vec2):
  """Build the specialized path-tracing megakernel for this context."""
  cast_ray = _make_cast_ray(rc.geom_ray_types, first_hit=False)
  cast_ray_first_hit = _make_cast_ray(rc.geom_ray_types, first_hit=True)
  sample_texture = _make_sample_texture(rc.geom_ray_types)
  surface = _make_surface(sample_texture, rc.use_textures, rc.enable_vertex_normals)

  has_splats = rc.splat_count > 0
  motion_rolling = float(motion[0])
  motion_exposure = float(motion[1])
  has_motion = motion_rolling > 0.0 or motion_exposure > 0.0
  has_distortion = camera.has_distortion
  has_chromatic = camera.has_chromatic
  has_dof = camera.has_dof
  apply_lens = has_distortion or has_chromatic
  M_NLIGHT = m.nlight
  bvh_ngeom = rc.bvh_ngeom
  bvh_nflexgeom = rc.bvh_nflexgeom
  light_attenuation_is_default = rc.light_attenuation_is_default
  has_spot_lights = rc.has_spot_lights
  use_fast_math = rc.use_fast_math
  cull_backfaces = rc.enable_backface_culling
  has_ortho = rc.has_orthographic_camera
  zfar = rc.zfar
  render_skybox = rc.render_skybox
  bg = int(rc.background_color)
  background = wp.vec3(
    float((bg >> 16) & 0xFF) / 255.0,
    float((bg >> 8) & 0xFF) / 255.0,
    float(bg & 0xFF) / 255.0,
  )

  @wp.func
  def env_radiance(
    # In:
    textures: wp.array[wp.Texture2D],
    skybox_tex_id: wp.array[int],
    skybox_face_width: wp.array[int],
    worldid: int,
    ray_dir_world: wp.vec3,
  ) -> wp.vec3:
    if wp.static(render_skybox):
      skybox_id = skybox_tex_id[worldid % skybox_tex_id.shape[0]]
      c = sample_skybox(
        textures[skybox_id],
        1.0 / float(skybox_face_width[worldid % skybox_face_width.shape[0]]),
        ray_dir_world,
      )
      return _srgb_to_linear(c)
    return wp.static(background)

  @wp.func
  def accumulate_pixel(
    # In:
    worldid: int,
    pix: int,
    radiance: wp.vec3,
    inv_count: float,
    channel: int,
    # Out:
    hdr_out: wp.array2d[wp.vec3],
    m2_out: wp.array2d[wp.vec3],
  ):
    # Welford running mean (in hdr_out) and second moment (in m2_out).
    old = hdr_out[worldid, pix]
    m2 = m2_out[worldid, pix]
    if wp.static(has_chromatic):
      if channel == 0:
        delta = radiance[0] - old[0]
        new_mean = old[0] + delta * inv_count
        hdr_out[worldid, pix] = wp.vec3(new_mean, old[1], old[2])
        m2_out[worldid, pix] = wp.vec3(m2[0] + delta * (radiance[0] - new_mean), m2[1], m2[2])
      elif channel == 1:
        delta = radiance[1] - old[1]
        new_mean = old[1] + delta * inv_count
        hdr_out[worldid, pix] = wp.vec3(old[0], new_mean, old[2])
        m2_out[worldid, pix] = wp.vec3(m2[0], m2[1] + delta * (radiance[1] - new_mean), m2[2])
      else:
        delta = radiance[2] - old[2]
        new_mean = old[2] + delta * inv_count
        hdr_out[worldid, pix] = wp.vec3(old[0], old[1], new_mean)
        m2_out[worldid, pix] = wp.vec3(m2[0], m2[1], m2[2] + delta * (radiance[2] - new_mean))
    else:
      delta = radiance - old
      new_mean = old + delta * inv_count
      hdr_out[worldid, pix] = new_mean
      m2_out[worldid, pix] = m2 + wp.cw_mul(delta, radiance - new_mean)

  @wp.kernel(module="unique", enable_backward=False, grid_stride=False, module_options={"fast_math": use_fast_math})
  def _path_trace_kernel(
    # Model:
    geom_type: wp.array[int],
    geom_dataid: wp.array2d[int],
    geom_matid: wp.array2d[int],
    geom_size: wp.array2d[wp.vec3],
    geom_rgba: wp.array2d[wp.vec4],
    light_type: wp.array2d[int],
    light_castshadow: wp.array2d[bool],
    light_active: wp.array2d[bool],
    light_attenuation: wp.array2d[wp.vec3],
    light_cutoff: wp.array2d[float],
    light_exponent: wp.array2d[float],
    light_diffuse: wp.array2d[wp.vec3],
    flex_vertadr: wp.array[int],
    flex_edge: wp.array[wp.vec2i],
    flex_radius: wp.array[float],
    mesh_faceadr: wp.array[int],
    mesh_normaladr: wp.array[int],
    mesh_normal: wp.array[wp.vec3],
    mat_texid: wp.array3d[int],
    mat_texuniform: wp.array2d[bool],
    mat_texrepeat: wp.array2d[wp.vec2],
    mat_emission: wp.array2d[float],
    mat_specular: wp.array2d[float],
    mat_shininess: wp.array2d[float],
    mat_rgba: wp.array2d[wp.vec4],
    # Data in:
    geom_xpos_in: wp.array2d[wp.vec3],
    geom_xmat_in: wp.array2d[wp.mat33],
    cam_xpos_in: wp.array2d[wp.vec3],
    cam_xmat_in: wp.array2d[wp.mat33],
    light_xpos_in: wp.array2d[wp.vec3],
    light_xdir_in: wp.array2d[wp.vec3],
    flexvert_xpos_in: wp.array2d[wp.vec3],
    # In:
    mesh_facetexcoord: wp.array[wp.vec3i],
    mesh_facenormal: wp.array[wp.vec3i],
    mesh_texcoord: wp.array[wp.vec2],
    mesh_texcoord_offsets: wp.array[int],
    nrender: int,
    nrays: int,
    cam_res: wp.array[wp.vec2i],
    cam_id_map: wp.array[int],
    ray: wp.array[wp.vec3],
    ray_offset: wp.array[wp.vec3],
    rgb_adr: wp.array[int],
    depth_adr: wp.array[int],
    render_rgb: wp.array[bool],
    render_depth: wp.array[bool],
    bvh_id: wp.uint64,
    group_root: wp.array[int],
    flex_bvh_id: wp.array[wp.uint64],
    flex_group_root: wp.array2d[int],
    enabled_geom_ids: wp.array[int],
    mesh_bvh_id: wp.array[wp.uint64],
    hfield_bvh_id: wp.array[wp.uint64],
    flex_rgba: wp.array[wp.vec4],
    flex_geom_flexid: wp.array[int],
    flex_geom_edgeid: wp.array[int],
    skybox_tex_id: wp.array[int],
    skybox_face_width: wp.array[int],
    textures: wp.array[wp.Texture2D],
    splat_position: wp.array[wp.vec3],
    splat_rotation: wp.array[wp.quat],
    splat_scale: wp.array[wp.vec3],
    splat_rgba: wp.array[wp.vec4],
    splat_bvh_id: wp.uint64,
    splat_group_root: wp.array[int],
    inv_count: float,
    sample_seed: int,
    prev_cam_xpos_in: wp.array2d[wp.vec3],
    prev_cam_xmat_in: wp.array2d[wp.mat33],
    lens_distortion: wp.vec4,
    lens_chromatic: wp.vec3,
    lens_aperture: float,
    lens_focus: float,
    channel: int,
    # Out:
    hdr_out: wp.array2d[wp.vec3],
    m2_out: wp.array2d[wp.vec3],
    depth_out: wp.array2d[float],
    albedo_out: wp.array2d[wp.vec4],
    normal_out: wp.array2d[wp.vec3],
  ):
    worldid, rayid = wp.tid()

    camid = int(-1)
    rayid_local = int(-1)
    accum = int(0)
    for i in range(nrender):
      num_i = cam_res[i][0] * cam_res[i][1]
      if rayid < accum + num_i:
        camid = i
        rayid_local = rayid - accum
        break
      accum += num_i
    if camid == -1:
      return

    render_rgb_cam = render_rgb[camid]
    pix_rgb = int(-1)
    if render_rgb_cam:
      pix_rgb = rgb_adr[camid] + rayid_local

    if not render_rgb_cam and not render_depth[camid]:
      return

    mujoco_cam_id = cam_id_map[camid]
    ray_dir_local_cam = ray[rayid]
    ray_offset_local_cam = ray_offset[rayid]

    rng = wp.rand_init(sample_seed, worldid * nrays + rayid)

    if wp.static(apply_lens):
      scale = float(1.0)
      if wp.static(has_chromatic):
        scale = lens_chromatic[channel]
      ray_dir_local_cam = distort_direction(ray_dir_local_cam, lens_distortion, scale)

    cam_mat_world = cam_xmat_in[worldid, mujoco_cam_id]
    ray_origin_world = cam_xpos_in[worldid, mujoco_cam_id]
    if wp.static(has_motion):
      row = rayid_local // cam_res[camid][0]
      row_norm = float(row) / float(wp.max(cam_res[camid][1] - 1, 1))
      ray_time = wp.clamp(row_norm * wp.static(motion_rolling) + wp.randf(rng) * wp.static(motion_exposure), 0.0, 1.0)
      ray_origin_world, cam_mat_world = _interp_camera(
        prev_cam_xpos_in[worldid, mujoco_cam_id],
        ray_origin_world,
        prev_cam_xmat_in[worldid, mujoco_cam_id],
        cam_mat_world,
        ray_time,
      )
    if wp.static(has_dof):
      lens_uv = sample_lens_offset(lens_aperture, wp.randf(rng), wp.randf(rng))
      ray_origin_world += cam_mat_world @ wp.vec3(lens_uv[0], lens_uv[1], 0.0)
      ray_dir_local_cam = thin_lens_direction(ray_dir_local_cam, lens_uv, lens_focus)
    elif wp.static(has_ortho):
      ray_origin_world += cam_mat_world @ ray_offset_local_cam
    ray_dir_world = cam_mat_world @ ray_dir_local_cam

    if wp.static(zfar > 0.0):
      max_cam_dist = zfar / wp.max(-ray_dir_local_cam[2], 1.0e-6)
    else:
      max_cam_dist = float(MJ_MAXVAL)

    geom_id, dist, normal, bary_u, bary_v, f, mesh_id = cast_ray(
      geom_type,
      geom_dataid,
      geom_matid,
      geom_size,
      geom_rgba,
      flex_vertadr,
      flex_edge,
      flex_radius,
      mat_rgba,
      geom_xpos_in,
      geom_xmat_in,
      flexvert_xpos_in,
      bvh_id,
      group_root[worldid],
      worldid,
      bvh_ngeom,
      bvh_nflexgeom,
      enabled_geom_ids,
      mesh_bvh_id,
      hfield_bvh_id,
      flex_rgba,
      flex_geom_flexid,
      flex_geom_edgeid,
      flex_bvh_id,
      flex_group_root,
      ray_origin_world,
      ray_dir_world,
      max_cam_dist,
      wp.static(cull_backfaces),
    )

    if wp.static(zfar > 0.0):
      if geom_id != -1 and (dist * -ray_dir_local_cam[2]) > zfar:
        geom_id = -1

    splat_color = wp.vec3(0.0, 0.0, 0.0)
    splat_transmittance = float(1.0)
    if wp.static(has_splats):
      splat_color, splat_transmittance, _ = shade_splats(
        splat_position,
        splat_rotation,
        splat_scale,
        splat_rgba,
        splat_bvh_id,
        splat_group_root[worldid],
        ray_origin_world,
        ray_dir_world,
        dist,
      )

    if geom_id == -1:
      if channel == 0 and render_depth[camid]:
        depth_out[worldid, depth_adr[camid] + rayid_local] = 0.0
      if channel == 0 and render_rgb_cam:
        albedo_out[worldid, pix_rgb] = wp.vec4(1.0, 1.0, 1.0, 0.0)
        normal_out[worldid, pix_rgb] = wp.vec3(0.0, 0.0, 0.0)
      if render_rgb_cam:
        radiance = env_radiance(textures, skybox_tex_id, skybox_face_width, worldid, ray_dir_world)
        if wp.static(has_splats):
          radiance = splat_color + radiance * splat_transmittance
        accumulate_pixel(worldid, pix_rgb, radiance, inv_count, channel, hdr_out, m2_out)
      return

    if channel == 0 and render_depth[camid]:
      depth_out[worldid, depth_adr[camid] + rayid_local] = dist * -ray_dir_local_cam[2]

    if not render_rgb_cam:
      return

    hit_point = ray_origin_world + ray_dir_world * dist

    n, albedo, emission, f0, alpha = surface(
      geom_type,
      geom_matid,
      geom_size,
      geom_rgba,
      mesh_faceadr,
      mesh_normaladr,
      mesh_normal,
      mat_texid,
      mat_texuniform,
      mat_texrepeat,
      mat_emission,
      mat_specular,
      mat_shininess,
      mat_rgba,
      geom_xpos_in,
      geom_xmat_in,
      flex_rgba,
      mesh_facetexcoord,
      mesh_facenormal,
      mesh_texcoord,
      mesh_texcoord_offsets,
      textures,
      worldid,
      geom_id,
      mesh_id,
      bary_u,
      bary_v,
      f,
      normal,
      hit_point,
    )
    if wp.static(not cull_backfaces):
      if wp.dot(n, ray_dir_world) > 0.0:
        n = -n

    if channel == 0 and render_rgb_cam:
      albedo_out[worldid, pix_rgb] = wp.vec4(
        wp.max(albedo[0], 1.0e-2), wp.max(albedo[1], 1.0e-2), wp.max(albedo[2], 1.0e-2), 1.0
      )
      normal_out[worldid, pix_rgb] = n

    radiance = emission
    throughput = wp.vec3(1.0, 1.0, 1.0)
    v = -ray_dir_world

    for bounce in range(max_bounces):
      # Next-event estimation: pick one active light uniformly.
      if wp.static(M_NLIGHT > 0):
        nactive = int(0)
        for li in range(wp.static(M_NLIGHT)):
          if light_active[worldid % light_active.shape[0], li]:
            nactive += 1

        if nactive > 0:
          pick = wp.min(int(wp.randf(rng) * float(nactive)), nactive - 1)
          light_index = int(-1)
          seen = int(0)
          for li in range(wp.static(M_NLIGHT)):
            if light_active[worldid % light_active.shape[0], li]:
              if seen == pick:
                light_index = li
              seen += 1

          l_dir = wp.vec3(0.0, 0.0, 1.0)
          light_radiance = wp.vec3(0.0, 0.0, 0.0)
          dist_light = float(MJ_MAXVAL)
          light_type_li = light_type[worldid % light_type.shape[0], light_index]
          if light_type_li == 1:  # directional
            l_dir = -light_xdir_in[worldid, light_index]
            light_radiance = light_diffuse[worldid % light_diffuse.shape[0], light_index] * _LIGHT_RADIANCE_SCALE
          else:
            to_light = light_xpos_in[worldid, light_index] - hit_point
            dist_light = wp.length(to_light)
            l_dir = to_light / wp.max(dist_light, 1.0e-9)
            attenuation = 1.0
            if wp.static(not light_attenuation_is_default):
              att = light_attenuation[worldid % light_attenuation.shape[0], light_index]
              attenuation = wp.safe_div(1.0, wp.dot(wp.vec3(1.0, dist_light, dist_light * dist_light), att))
            light_radiance = light_diffuse[worldid % light_diffuse.shape[0], light_index] * _LIGHT_RADIANCE_SCALE * attenuation
            if wp.static(has_spot_lights):
              if light_type_li == 0:  # spot
                cos_theta = wp.dot(-l_dir, light_xdir_in[worldid, light_index])
                cos_cutoff = wp.cos(light_cutoff[worldid % light_cutoff.shape[0], light_index] * (_PI / 180.0))
                if cos_theta < cos_cutoff:
                  light_radiance = wp.vec3(0.0, 0.0, 0.0)
                else:
                  light_radiance *= wp.pow(
                    wp.max(cos_theta, 0.0), light_exponent[worldid % light_exponent.shape[0], light_index]
                  )

          ndotl = wp.dot(n, l_dir)
          if ndotl > 0.0 and _luminance(light_radiance) > 0.0:
            visible = True
            if light_castshadow[worldid % light_castshadow.shape[0], light_index]:
              shadow_origin = hit_point + n * _SHADOW_EPS
              max_t = float(1.0e8)
              if light_type_li != 1:
                max_t = dist_light - 1.0e-3
              shadow_geom_id, shadow_d, shadow_n, shadow_u, shadow_v, shadow_f, shadow_m = cast_ray_first_hit(
                geom_type,
                geom_dataid,
                geom_matid,
                geom_size,
                geom_rgba,
                flex_vertadr,
                flex_edge,
                flex_radius,
                mat_rgba,
                geom_xpos_in,
                geom_xmat_in,
                flexvert_xpos_in,
                bvh_id,
                group_root[worldid],
                worldid,
                bvh_ngeom,
                bvh_nflexgeom,
                enabled_geom_ids,
                mesh_bvh_id,
                hfield_bvh_id,
                flex_rgba,
                flex_geom_flexid,
                flex_geom_edgeid,
                flex_bvh_id,
                flex_group_root,
                shadow_origin,
                l_dir,
                max_t,
                wp.static(cull_backfaces),
              )
              visible = shadow_geom_id == -1
            if visible:
              f_bsdf = _bsdf_eval(albedo, f0, alpha, n, v, l_dir)
              radiance += wp.cw_mul(wp.cw_mul(throughput, f_bsdf), light_radiance * (ndotl * float(nactive)))

      # Sample the next direction.
      ndotv = wp.dot(n, v)
      if ndotv <= 0.0:
        break
      fresnel_v = _luminance(_fresnel_schlick(f0, ndotv))
      p_spec = wp.clamp(fresnel_v, 0.05, 0.95)
      t1, b1 = _onb(n)
      if wp.randf(rng) < p_spec:
        l = _sample_ggx_vndf(n, t1, b1, v, alpha, wp.randf(rng), wp.randf(rng))
        pdf = _pdf_ggx_vndf(n, v, alpha, l)
      else:
        l = _sample_cosine_hemisphere(n, t1, b1, wp.randf(rng), wp.randf(rng))
        pdf = wp.max(wp.dot(n, l), 0.0) * _INV_PI
      ndotl = wp.dot(n, l)
      if ndotl <= 0.0 or pdf <= 0.0:
        break
      f_bsdf = _bsdf_eval(albedo, f0, alpha, n, v, l)
      throughput = wp.cw_mul(throughput, f_bsdf) * (ndotl / pdf)

      # Trace the bounce ray.
      ray_origin_world = hit_point + n * _SHADOW_EPS
      ray_dir_world = l
      geom_id, dist, normal, bary_u, bary_v, f, mesh_id = cast_ray(
        geom_type,
        geom_dataid,
        geom_matid,
        geom_size,
        geom_rgba,
        flex_vertadr,
        flex_edge,
        flex_radius,
        mat_rgba,
        geom_xpos_in,
        geom_xmat_in,
        flexvert_xpos_in,
        bvh_id,
        group_root[worldid],
        worldid,
        bvh_ngeom,
        bvh_nflexgeom,
        enabled_geom_ids,
        mesh_bvh_id,
        hfield_bvh_id,
        flex_rgba,
        flex_geom_flexid,
        flex_geom_edgeid,
        flex_bvh_id,
        flex_group_root,
        ray_origin_world,
        ray_dir_world,
        float(MJ_MAXVAL),
        wp.static(cull_backfaces),
      )
      if geom_id == -1:
        radiance += wp.cw_mul(throughput, env_radiance(textures, skybox_tex_id, skybox_face_width, worldid, ray_dir_world))
        break

      hit_point = ray_origin_world + ray_dir_world * dist
      n, albedo, emission, f0, alpha = surface(
        geom_type,
        geom_matid,
        geom_size,
        geom_rgba,
        mesh_faceadr,
        mesh_normaladr,
        mesh_normal,
        mat_texid,
        mat_texuniform,
        mat_texrepeat,
        mat_emission,
        mat_specular,
        mat_shininess,
        mat_rgba,
        geom_xpos_in,
        geom_xmat_in,
        flex_rgba,
        mesh_facetexcoord,
        mesh_facenormal,
        mesh_texcoord,
        mesh_texcoord_offsets,
        textures,
        worldid,
        geom_id,
        mesh_id,
        bary_u,
        bary_v,
        f,
        normal,
        hit_point,
      )
      if wp.static(not cull_backfaces):
        if wp.dot(n, ray_dir_world) > 0.0:
          n = -n
      radiance += wp.cw_mul(throughput, emission)
      v = -ray_dir_world

      if bounce >= roulette_start:
        q = wp.clamp(_luminance(throughput), 0.05, 1.0)
        if wp.randf(rng) > q:
          break
        throughput /= q

    if wp.static(has_splats):
      radiance = splat_color + radiance * splat_transmittance
    accumulate_pixel(worldid, pix_rgb, radiance, inv_count, channel, hdr_out, m2_out)

  return _path_trace_kernel


@wp.kernel(module="unique", enable_backward=False)
def _variance_kernel(
  # In:
  m2: wp.array2d[wp.vec3],
  inv_count: float,
  # Out:
  var_out: wp.array2d[float],
):
  worldid, i = wp.tid()
  var_out[worldid, i] = _luminance(m2[worldid, i] * inv_count)


@wp.kernel(module="unique", enable_backward=False)
def _sensor_kernel(
  # In:
  hdr: wp.array2d[wp.vec3],
  exposure: float,
  gain: float,
  full_well_e: float,
  read_noise_e: float,
  seed: int,
  # Out:
  out: wp.array2d[wp.vec3],
):
  worldid, i = wp.tid()
  rng = wp.rand_init(seed, worldid * hdr.shape[1] + i)
  noise_shot = wp.vec3(wp.randn(rng), wp.randn(rng), wp.randn(rng))
  noise_read = wp.vec3(wp.randn(rng), wp.randn(rng), wp.randn(rng))
  out[worldid, i] = sensor_sample(hdr[worldid, i], exposure, gain, full_well_e, read_noise_e, noise_shot, noise_read)


@wp.kernel(module="unique", enable_backward=False)
def _tone_map_kernel(
  # In:
  hdr: wp.array2d[wp.vec3],
  exposure: float,
  # Out:
  rgb_out: wp.array2d[wp.uint32],
):
  worldid, i = wp.tid()
  x = hdr[worldid, i] * exposure
  y = _aces(x)
  y = wp.min(wp.max(y, wp.vec3(0.0, 0.0, 0.0)), wp.vec3(1.0, 1.0, 1.0))
  y = wp.vec3(wp.pow(y[0], 1.0 / 2.2), wp.pow(y[1], 1.0 / 2.2), wp.pow(y[2], 1.0 / 2.2))
  rgb_out[worldid, i] = pack_rgba_to_uint32(y[0] * 255.0, y[1] * 255.0, y[2] * 255.0, 255.0)


class PathTracer:
  """Physically based Monte Carlo path tracer over a `RenderContext`.

  Notes:
    - `rc` must be created with `render_rgb=True` for at least one camera,
      `samples_per_pixel=1` and `use_precomputed_rays=True`.
    - Gaussian splats are composited along camera rays only (they do not
      occlude shadow or bounce rays).
    - HDR radiance is stored in `hdr` as a running average over all accumulated
      samples, before any `CameraSensor` conversion. `render(samples=1,
      accumulate=True)` progressively refines a static scene across calls.
    - Primary-hit albedo and shading normal are stored in `albedo` (rgb +
      validity in alpha) and `normal` for the denoiser's demodulation and edge
      guides.
    - The `camera` and `sensor` models are fixed for the tracer's lifetime.
  """

  def __init__(
    self,
    rc: RenderContext,
    max_bounces: int = 8,
    roulette_start: int = 2,
    seed: int = 0,
    camera: CameraModel | None = None,
    sensor: CameraSensor | None = None,
    camera_motion: CameraMotion | None = None,
    object_motion: bool = False,
    motion_subframes: int = 8,
  ):
    """Initializes the path tracer.

    Args:
      rc: The render context to trace. Its BVHs, rays, textures and materials
        are used directly.
      max_bounces: Maximum number of indirect bounces per path.
      roulette_start: Bounce index at which Russian roulette begins.
      seed: Base RNG seed.
      camera: Physical lens model (distortion, chromatic aberration, depth of
        field). None renders a perfect pinhole.
      sensor: Image sensor model (exposure, gain, saturation, shot and read
        noise). None renders noise-free radiance.
      camera_motion: Shutter model for rolling shutter and camera motion blur.
        Motion is interpolated from the previous `render` call's pose to the
        current one.
      object_motion: Blur moving geometry across the shutter by rendering
        `motion_subframes` sub-exposures, interpolating geom poses from the
        previous `render` call to the current one and refitting the scene BVH
        per sub-exposure. Requires `camera_motion` with exposure > 0.
      motion_subframes: Number of sub-exposures used for object motion blur.
    """
    if rc.rgb_data.shape[1] == 0:
      raise ValueError("PathTracer requires at least one camera with render_rgb=True.")
    if rc.samples_per_pixel != 1:
      raise ValueError("PathTracer does not support samples_per_pixel > 1; use render(samples=...).")
    if not rc.use_precomputed_rays:
      raise ValueError("PathTracer requires use_precomputed_rays=True.")
    self.rc = rc
    self.max_bounces = max_bounces
    self.roulette_start = roulette_start
    self.seed = seed
    self.camera = camera if camera is not None else CameraModel()
    self.sensor = sensor
    if camera_motion is not None and camera_motion.exposure == 0.0 and camera_motion.rolling_shutter == 0.0:
      camera_motion = None
    self.camera_motion = camera_motion
    if motion_subframes < 1:
      raise ValueError("motion_subframes must be at least 1.")
    if object_motion and (camera_motion is None or camera_motion.exposure <= 0.0):
      raise ValueError("object_motion requires camera_motion with exposure > 0.")
    self.object_motion = object_motion
    self.motion_subframes = motion_subframes
    self._prev_cam_xpos = None
    self._prev_cam_xmat = None
    self._prev_geom_xpos = None
    self._prev_geom_xmat = None
    self.sensor_hdr = None
    if sensor is not None:
      self.sensor_hdr = wp.zeros((rc.rgb_data.shape[0], rc.rgb_data.shape[1]), dtype=wp.vec3)
    if self.camera.has_dof and rc.has_orthographic_camera:
      raise ValueError("Depth of field requires perspective cameras.")
    if self.camera.has_dof and self.camera.focus_distance <= 0.0:
      raise ValueError("Depth of field requires a positive focus_distance.")
    self.hdr = wp.zeros((rc.rgb_data.shape[0], rc.rgb_data.shape[1]), dtype=wp.vec3)
    self.m2 = wp.zeros((rc.rgb_data.shape[0], rc.rgb_data.shape[1]), dtype=wp.vec3)
    self.albedo = wp.zeros((rc.rgb_data.shape[0], rc.rgb_data.shape[1]), dtype=wp.vec4)
    self.normal = wp.zeros((rc.rgb_data.shape[0], rc.rgb_data.shape[1]), dtype=wp.vec3)
    self._sample_count = 0
    self._kernel = None

  def reset(self):
    """Clears accumulated radiance and restarts the sample sequence."""
    self.hdr.zero_()
    self.m2.zero_()
    self._sample_count = 0

  @property
  def sample_count(self) -> int:
    """Number of accumulated samples per pixel."""
    return self._sample_count

  @property
  def variance(self) -> wp.array2d[float]:
    """Per-pixel luminance variance of the mean radiance estimate.

    This is the sample variance divided by the sample count, i.e. the quantity
    a variance-guided denoiser expects. Zero when only one sample is
    accumulated.

    Returns:
      A new (nworld, n_pixel) array.
    """
    out = wp.zeros(self.hdr.shape, dtype=float)
    n = max(self._sample_count, 2)
    inv_count = 1.0 / float(n * (n - 1))
    wp.launch(_variance_kernel, dim=self.hdr.shape, inputs=[self.m2, inv_count], outputs=[out])
    return out

  @event_scope
  def render(self, m: Model, d: Data, samples: int = 1, accumulate: bool = False, exposure: float = 1.0):
    """Renders the current frame.

    Accumulated HDR radiance is stored in `self.hdr`; tone-mapped output is
    written to `self.rc.rgb_data` (and depth to `rc.depth_data`).

    Args:
      m: The model on device.
      d: The data on device.
      samples: Paths per pixel added by this call.
      accumulate: If True, continue accumulating on top of previous calls
        instead of restarting. Use for progressive refinement of static scenes.
      exposure: Linear exposure multiplier applied before tone mapping.
    """
    if samples < 1:
      raise ValueError("samples must be at least 1.")

    if self._kernel is None:
      motion = (
        wp.vec2(self.camera_motion.rolling_shutter, self.camera_motion.exposure)
        if self.camera_motion is not None
        else wp.vec2(0.0, 0.0)
      )
      self._kernel = _build_kernel(m, self.rc, self.max_bounces, self.roulette_start, self.camera, motion)

    if not accumulate or self._sample_count == 0:
      self.hdr.zero_()
      self.m2.zero_()
      self._sample_count = 0

    rc = self.rc
    if self.camera_motion is not None:
      if self._prev_cam_xpos is None:
        self._prev_cam_xpos = wp.zeros(d.cam_xpos.shape, dtype=wp.vec3)
        self._prev_cam_xmat = wp.zeros(d.cam_xmat.shape, dtype=wp.mat33)
        self._prev_cam_xpos.assign(d.cam_xpos)
        self._prev_cam_xmat.assign(d.cam_xmat)
      self._launch_prev_cam_xpos = self._prev_cam_xpos
      self._launch_prev_cam_xmat = self._prev_cam_xmat
    else:
      self._launch_prev_cam_xpos = d.cam_xpos
      self._launch_prev_cam_xmat = d.cam_xmat
    channels = 3 if self.camera.has_chromatic else 1
    if self.object_motion:
      curr_xpos_np = d.geom_xpos.numpy().copy()
      curr_xmat_np = d.geom_xmat.numpy().copy()
      if self._prev_geom_xpos is None:
        self._prev_geom_xpos = wp.array(curr_xpos_np, dtype=wp.vec3)
        self._prev_geom_xmat = wp.array(curr_xmat_np, dtype=wp.mat33)
      curr_xpos = wp.array(curr_xpos_np, dtype=wp.vec3)
      curr_xmat = wp.array(curr_xmat_np, dtype=wp.mat33)
      subframes = min(self.motion_subframes, samples)
      base = samples // subframes
      rem = samples % subframes
      for k in range(subframes):
        n_k = base + (1 if k < rem else 0)
        if n_k == 0:
          continue
        t = (float(k) + 0.5) / float(subframes)
        wp.launch(
          _interp_geom_kernel,
          dim=(d.nworld, m.ngeom),
          inputs=[self._prev_geom_xpos, curr_xpos, self._prev_geom_xmat, curr_xmat, t],
          outputs=[d.geom_xpos, d.geom_xmat],
        )
        refit_bvh(m, d, rc)
        self._render_samples(m, d, rc, channels, n_k)
      d.geom_xpos.assign(curr_xpos_np)
      d.geom_xmat.assign(curr_xmat_np)
      refit_bvh(m, d, rc)
      self._prev_geom_xpos = curr_xpos
      self._prev_geom_xmat = curr_xmat
    else:
      self._render_samples(m, d, rc, channels, samples)

    tone_input = self.hdr
    if self.sensor is not None:
      wp.launch(
        kernel=_sensor_kernel,
        dim=rc.rgb_data.shape,
        inputs=[
          self.hdr,
          self.sensor.exposure,
          self.sensor.gain,
          self.sensor.full_well_e,
          self.sensor.read_noise_e,
          self.sensor.seed + self._sample_count,
        ],
        outputs=[self.sensor_hdr],
      )
      tone_input = self.sensor_hdr

    if self.camera_motion is not None:
      self._prev_cam_xpos.assign(d.cam_xpos)
      self._prev_cam_xmat.assign(d.cam_xmat)

    if rc.rgb_data.shape[1] > 0:
      wp.launch(
        kernel=_tone_map_kernel,
        dim=rc.rgb_data.shape,
        inputs=[tone_input, exposure],
        outputs=[rc.rgb_data],
      )

  def _render_samples(self, m: Model, d: Data, rc: RenderContext, channels: int, n_samples: int):
    for _ in range(n_samples):
      self._sample_count += 1
      for channel in range(channels):
        launch_seed = self.seed + channels * self._sample_count + channel
        wp.launch(
          kernel=self._kernel,
          dim=(d.nworld, rc.total_rays),
          inputs=[
            m.geom_type,
            m.geom_dataid,
            m.geom_matid,
            m.geom_size,
            m.geom_rgba,
            m.light_type,
            m.light_castshadow,
            m.light_active,
            m.light_attenuation,
            m.light_cutoff,
            m.light_exponent,
            m.light_diffuse,
            m.flex_vertadr,
            m.flex_edge,
            m.flex_radius,
            m.mesh_faceadr,
            m.mesh_normaladr,
            m.mesh_normal,
            m.mat_texid,
            m.mat_texuniform,
            m.mat_texrepeat,
            m.mat_emission,
            m.mat_specular,
            m.mat_shininess,
            m.mat_rgba,
            d.geom_xpos,
            d.geom_xmat,
            d.cam_xpos,
            d.cam_xmat,
            d.light_xpos,
            d.light_xdir,
            d.flexvert_xpos,
            rc.mesh_facetexcoord,
            rc.mesh_facenormal,
            rc.mesh_texcoord,
            rc.mesh_texcoord_offsets,
            rc.nrender,
            rc.total_rays,
            rc.cam_res,
            rc.cam_id_map,
            rc.ray,
            rc.ray_offset,
            rc.rgb_adr,
            rc.depth_adr,
            rc.render_rgb,
            rc.render_depth,
            rc.bvh_id,
            rc.group_root,
            rc.flex_bvh_id,
            rc.flex_group_root,
            rc.enabled_geom_ids,
            rc.mesh_bvh_id,
            rc.hfield_bvh_id,
            rc.flex_rgba,
            rc.flex_geom_flexid,
            rc.flex_geom_edgeid,
            rc.skybox_tex_id,
            rc.skybox_face_width,
            rc.textures,
            rc.splat_position,
            rc.splat_rotation,
            rc.splat_scale,
            rc.splat_rgba,
            rc.splat_bvh_id,
            rc.splat_group_root,
            1.0 / float(self._sample_count),
            launch_seed,
            self._launch_prev_cam_xpos,
            self._launch_prev_cam_xmat,
            wp.vec4(*self.camera.distortion),
            wp.vec3(*self.camera.chromatic),
            self.camera.aperture,
            self.camera.focus_distance,
            channel,
          ],
          outputs=[
            self.hdr,
            self.m2,
            rc.depth_data,
            self.albedo,
            self.normal,
          ],
        )

  def denoise(self, camera: int = 0, iterations: int = 5, sigma_color: float = 2.0, sigma_depth: float = 0.02):
    """Edge-aware a-trous denoise of the accumulated HDR image.

    Args:
      camera: Index of the active camera to denoise.
      iterations: Number of a-trous passes.
      sigma_color: Luminance falloff; variance-normalized when tracked.
      sigma_depth: Relative depth falloff when the camera renders depth.

    Returns:
      A new (nworld, width * height) denoised radiance buffer.
    """
    rc = self.rc
    if camera < 0 or camera >= rc.nrender:
      raise ValueError(f"camera must be in [0, {rc.nrender}), got {camera}.")
    cam_res = rc.cam_res.numpy()
    width = int(cam_res[camera][0])
    height = int(cam_res[camera][1])
    depth = None
    depth_base = 0
    if bool(rc.render_depth.numpy()[camera]):
      depth = rc.depth_data
      depth_base = int(rc.depth_adr.numpy()[camera])
    return denoise(
      self.hdr,
      width,
      height,
      base=int(rc.rgb_adr.numpy()[camera]),
      variance=self.variance,
      depth=depth,
      depth_base=depth_base,
      albedo=self.albedo,
      normal=self.normal,
      iterations=iterations,
      sigma_color=sigma_color,
      sigma_depth=sigma_depth,
    )
