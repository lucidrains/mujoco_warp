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
"""SVGF-style a-trous denoiser for path-traced radiance.

Pipeline (when primary-hit albedo is provided):

  1. demodulate: irradiance = radiance / albedo, converting the variance too,
  2. filter the irradiance with edge-aware a-trous passes weighted by a spatial
     gaussian, a variance-normalized luminance term, a normal guide and an
     optional depth guide,
  3. remodulate: radiance = filtered_irradiance * albedo.

Filtering in albedo-demodulated irradiance means texture and material edges no
longer look like noise: the filter can smooth illumination freely and the
texture comes back for free. Without albedo it falls back to filtering radiance
directly. Importance weights cap the tolerated luminance difference relative to
the center luminance so bright outliers cannot smear while dark neighbors
reject them (which would leak energy out of highlights).
"""

from typing import Optional

import warp as wp


@wp.func
def _luminance(c: wp.vec3) -> float:
  return wp.dot(c, wp.vec3(0.2126, 0.7152, 0.0722))


@wp.kernel(module="unique", enable_backward=False)
def _demod_kernel(
  # In:
  hdr: wp.array2d[wp.vec3],
  albedo: wp.array2d[wp.vec4],
  variance: wp.array2d[float],
  base: int,
  # Out:
  irr_out: wp.array2d[wp.vec3],
  var_out: wp.array2d[float],
):
  worldid, i = wp.tid()
  a = albedo[worldid, base + i]
  rgb = wp.vec3(a[0], a[1], a[2])
  irr_out[worldid, i] = wp.cw_div(hdr[worldid, base + i], rgb)
  max_a = wp.max(wp.max(rgb[0], rgb[1]), wp.max(rgb[2], 1.0e-2))
  var_out[worldid, i] = variance[worldid, base + i] / (max_a * max_a)


@wp.kernel(module="unique", enable_backward=False)
def _remod_kernel(
  # In:
  irr: wp.array2d[wp.vec3],
  albedo: wp.array2d[wp.vec4],
  base: int,
  # Out:
  out: wp.array2d[wp.vec3],
):
  worldid, i = wp.tid()
  a = albedo[worldid, base + i]
  out[worldid, i] = wp.cw_mul(irr[worldid, i], wp.vec3(a[0], a[1], a[2]))


def _make_atrous(has_variance: bool, has_depth: bool, has_albedo: bool, has_normal: bool):
  """Builds an a-trous pass specialized to the available guides."""

  @wp.kernel(module="unique", enable_backward=False)
  def _atrous(
    # In:
    src: wp.array2d[wp.vec3],
    variance: wp.array2d[float],
    depth: wp.array2d[float],
    albedo: wp.array2d[wp.vec4],
    normal: wp.array2d[wp.vec3],
    width: int,
    height: int,
    base: int,
    variance_base: int,
    depth_base: int,
    albedo_base: int,
    step: int,
    sigma_color: float,
    sigma_depth: float,
    spatial_sigma2: float,
    # Out:
    dst_out: wp.array2d[wp.vec3],
  ):
    worldid, i = wp.tid()
    x = i % width
    y = i // width
    center = src[worldid, base + i]
    center_lum = _luminance(center)
    center_var = float(0.0)
    center_depth = float(1.0)
    center_valid = True
    if wp.static(has_variance):
      center_var = variance[worldid, variance_base + i]
    if wp.static(has_depth):
      center_depth = depth[worldid, depth_base + i]
    if wp.static(has_albedo):
      center_valid = albedo[worldid, albedo_base + i][3] > 0.5

    acc = wp.vec3(0.0, 0.0, 0.0)
    w_sum = float(0.0)
    for dy in range(-2, 3):
      for dx in range(-2, 3):
        sx = wp.min(wp.max(x + dx * step, 0), width - 1)
        sy = wp.min(wp.max(y + dy * step, 0), height - 1)
        j = sy * width + sx
        if wp.static(has_albedo):
          tap_valid = albedo[worldid, albedo_base + j][3] > 0.5
          if tap_valid != center_valid:
            continue
        c = src[worldid, base + j]
        w = wp.exp(-float(dx * dx + dy * dy) / (2.0 * spatial_sigma2))
        # With demodulated irradiance the variance is the right scale and no
        # cap is needed. Filtering raw radiance instead needs a luminance-
        # relative cap: an uncapped variance lets bright outliers average with
        # everything while dark neighbors reject them, leaking energy.
        if wp.static(has_variance):
          if wp.static(has_albedo):
            denom = sigma_color * wp.sqrt(wp.max(center_var, 0.0)) + 1.0e-4
          else:
            denom = wp.min(sigma_color * wp.sqrt(wp.max(center_var, 0.0)) + 1.0e-4, 0.25 * center_lum + 1.0e-2)
          w *= wp.exp(-wp.abs(_luminance(c) - center_lum) / denom)
        else:
          denom = sigma_color + 1.0e-2 * center_lum + 1.0e-4
          w *= wp.exp(-wp.abs(_luminance(c) - center_lum) / denom)
        if wp.static(has_depth):
          d_diff = wp.abs(depth[worldid, depth_base + j] - center_depth)
          w *= wp.exp(-d_diff / (sigma_depth * wp.max(center_depth, 1.0e-3)))
        if wp.static(has_normal):
          if center_valid:
            similarity = wp.max(wp.dot(normal[worldid, albedo_base + i], normal[worldid, albedo_base + j]), 0.0)
            w *= wp.pow(similarity, 16.0)
        acc += c * w
        w_sum += w
    dst_out[worldid, i] = acc / w_sum

  return _atrous


def denoise(
  hdr: wp.array2d[wp.vec3],
  width: int,
  height: int,
  base: int = 0,
  variance: Optional[wp.array2d[float]] = None,
  depth: Optional[wp.array2d[float]] = None,
  depth_base: int = 0,
  albedo: Optional[wp.array2d[wp.vec4]] = None,
  normal: Optional[wp.array2d[wp.vec3]] = None,
  iterations: int = 5,
  sigma_color: float = 2.0,
  sigma_depth: float = 0.02,
  spatial_sigma: float = 1.5,
) -> wp.array2d[wp.vec3]:
  """Denoises a linear radiance image with an edge-aware a-trous filter.

  Args:
    hdr: Radiance image (nworld, n_pixel) with cameras concatenated.
    width: Image width of the region to denoise.
    height: Image height of the region to denoise.
    base: Offset of the region in `hdr` (e.g. `rc.rgb_adr[camera]`).
    variance: Optional per-pixel luminance variance of the mean, e.g. from
      `PathTracer.variance`.
    depth: Optional planar depth image (nworld, ...) used as a guide.
    depth_base: Offset of the region in `depth` (e.g. `rc.depth_adr[camera]`).
    albedo: Optional primary-hit albedo from `PathTracer.albedo` (rgb, validity
      in alpha). Enables demodulated filtering.
    normal: Optional primary-hit shading normals from `PathTracer.normal`.
    iterations: Number of a-trous passes; pass i uses step 2**i.
    sigma_color: Scales the variance-guided luminance tolerance.
    sigma_depth: Relative depth falloff for the depth guide.
    spatial_sigma: Standard deviation of the spatial gaussian.

  Returns:
    A new (nworld, width * height) denoised radiance buffer.
  """
  if width < 1 or height < 1 or iterations < 1:
    raise ValueError("width, height and iterations must be positive.")
  if sigma_color <= 0.0 or sigma_depth <= 0.0 or spatial_sigma <= 0.0:
    raise ValueError("sigma_color, sigma_depth and spatial_sigma must be positive.")
  if hdr.ndim != 2:
    raise ValueError("hdr must be a 2D (nworld, n_pixel) array.")
  npix = width * height
  if hdr.shape[1] < base + npix:
    raise ValueError(f"hdr has {hdr.shape[1]} pixels, need at least {base + npix}.")
  nworld = hdr.shape[0]

  variance_arr = variance
  if variance_arr is None:
    variance_arr = wp.zeros((nworld, base + npix), dtype=float)
  depth_arr = depth
  if depth_arr is None:
    depth_arr = wp.zeros((nworld, depth_base + npix), dtype=float)
  albedo_arr = albedo
  if albedo_arr is None:
    albedo_arr = wp.zeros((nworld, 1), dtype=wp.vec4)
  normal_arr = normal
  if normal_arr is None:
    normal_arr = wp.zeros((nworld, 1), dtype=wp.vec3)

  use_demod = albedo is not None
  use_normal = use_demod and normal is not None
  kernel = _make_atrous(variance is not None, depth is not None, use_demod, use_normal)

  src = hdr
  src_base = base
  var_src = variance_arr
  var_base = base
  albedo_base = base
  if use_demod:
    var_base = 0
    irr = wp.zeros((nworld, npix), dtype=wp.vec3)
    var_irr = wp.zeros((nworld, npix), dtype=float)
    wp.launch(
      _demod_kernel,
      dim=(nworld, npix),
      inputs=[hdr, albedo_arr, variance_arr, base],
      outputs=[irr, var_irr],
    )
    src = irr
    src_base = 0
    var_src = var_irr
    var_base = 0

  buf_a = wp.zeros((nworld, npix), dtype=wp.vec3)
  buf_b = wp.zeros((nworld, npix), dtype=wp.vec3)
  for it in range(iterations):
    dst = buf_a if it % 2 == 0 else buf_b
    wp.launch(
      kernel,
      dim=(nworld, npix),
      inputs=[
        src,
        var_src,
        depth_arr,
        albedo_arr,
        normal_arr,
        width,
        height,
        src_base,
        var_base,
        depth_base,
        albedo_base,
        1 << it,
        sigma_color,
        sigma_depth,
        2.0 * spatial_sigma * spatial_sigma,
      ],
      outputs=[dst],
    )
    src = dst
    src_base = 0

  if use_demod:
    out = wp.zeros((nworld, npix), dtype=wp.vec3)
    wp.launch(_remod_kernel, dim=(nworld, npix), inputs=[src, albedo_arr, albedo_base], outputs=[out])
    return out
  return src
