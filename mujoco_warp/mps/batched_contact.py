"""Fused contact-row assembly: one MSL kernel, one thread per world.

Writes J (nw, 32, nv), D (nw, 32), aref (nw, 32) for the 8 pyramidal contact
slots (2 feet x 4 points x 4 dims) in a single dispatch, replacing ~600 MLX ops.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

_MSL = r"""
    uint w = thread_position_in_grid.x;
    if (w >= NWORLD[0]) return;

    for (int i = 0; i < 32 * NV; ++i) Jc[w * 32 * NV + i] = 0.0f;
    for (int i = 0; i < 32; ++i) { Dc[w * 32 + i] = 0.0f; arefc[w * 32 + i] = 0.0f; }

    for (int f = 0; f < NFOOT; ++f) {
        int root = rootbuf[f];
        float invweight = invwb[f];
        for (int p = 0; p < 4; ++p) {
            int c = f * 4 + p;
            float d0 = dist[w * 8 + c];
            if (d0 >= 0.0f) continue;
            float3 com = float3(
                comb[(w * NBODY + root) * 3 + 0],
                comb[(w * NBODY + root) * 3 + 1],
                comb[(w * NBODY + root) * 3 + 2]);
            float3 off = float3(
                cposb[(w * 8 + c) * 3 + 0] - com.x,
                cposb[(w * 8 + c) * 3 + 1] - com.y,
                cposb[(w * 8 + c) * 3 + 2] - com.z);
            float jq[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            int nch = nchainbuf[f];
            for (int k = 0; k < NCHAIN; ++k) {
                if (k >= nch) break;
                int dof = chainbuf[f * NCHAIN + k];
                int d6 = (w * NV + dof) * 6;
                float3 ang = float3(cdofb[d6 + 0], cdofb[d6 + 1], cdofb[d6 + 2]);
                float3 lin = float3(cdofb[d6 + 3], cdofb[d6 + 4], cdofb[d6 + 5]);
                float3 jacp = lin + cross(ang, off);
                float3 jacr = ang;
                float qv = qvelb[w * NV + dof];
                for (int dim = 0; dim < 4; ++dim) {
                    int dimid2 = dim / 2 + 1;
                    float sgn = 1.0f - 2.0f * float(dim & 1);
                    float row = dot(jacp, float3(0.0f, 0.0f, 1.0f));
                    if (dimid2 == 1) row += sgn * dot(jacp, float3(0.0f, 1.0f, 0.0f));
                    else if (dimid2 == 2) row += sgn * dot(jacp, float3(-1.0f, 0.0f, 0.0f));
                    else if (dimid2 == 3) row += sgn * dot(jacr, float3(0.0f, 0.0f, 1.0f));
                    else if (dimid2 == 4) row += sgn * dot(jacr, float3(0.0f, 1.0f, 0.0f));
                    else row += sgn * dot(jacr, float3(-1.0f, 0.0f, 0.0f));
                    Jc[w * 32 * NV + (c * 4 + dim) * NV + dof] = row;
                    jq[dim] += row * qv;
                }
            }
            float dmin = params[0];
            float dmax = params[1];
            float width = params[2];
            float widthraw = params[3];
            float mid = params[4];
            float power = params[5];
            float x = metal::abs(d0) / width;
            float imp;
            if (dmin == dmax || widthraw <= 1e-15f) {
                imp = 0.5f * (dmin + dmax);
            } else if (x <= 0.0f) {
                imp = dmin;
            } else if (x >= 1.0f) {
                imp = dmax;
            } else {
                float low = dmin + x * (dmax - dmin);
                float mcur = dmin + (1.0f / metal::pow(mid, power - 1.0f)) * metal::pow(x, power) * (dmax - dmin);
                float hcur = dmin + (1.0f - (1.0f / metal::pow(1.0f - mid, power - 1.0f)) * metal::pow(1.0f - x, power)) * (dmax - dmin);
                imp = (power == 1.0f) ? low : ((x <= mid) ? mcur : hcur);
                imp = metal::clamp(imp, dmin, dmax);
            }
            float Dv = 1.0f / metal::max(invweight * (1.0f - imp) / imp, 1e-15f);
            float aref_v = -params[6] * imp * d0 - params[7] * jq[dim_placeholder];
            for (int dim = 0; dim < 4; ++dim) {
                Dc[w * 32 + c * 4 + dim] = Dv;
                arefc[w * 32 + c * 4 + dim] = -params[6] * imp * d0 - params[7] * jq[dim];
            }
        }
    }
"""

# fix the placeholder line above at import time (kept source simple)
_MSL = _MSL.replace("            float aref_v = -params[6] * imp * d0 - params[7] * jq[dim_placeholder];\n", "")

_CACHE: dict = {}


def contact_rows(sim, m, dist, cpos, chain_buf, nchain_buf, root_buf, invw_buf, params):
  nw = sim.nworld
  nv = sim.m.nv
  nbody = int(m.nbody)
  nchain = chain_buf.shape[1]
  key = (nw, nv, nbody, nchain)
  k = _CACHE.get(key)
  if k is None:
    k = mx.fast.metal_kernel(
      name=f"contact_rows_{nv}_{nbody}_{nchain}",
      input_names=["cdofb", "comb", "qvelb", "dist", "cposb", "chainbuf", "nchainbuf", "rootbuf", "invwb", "params", "NWORLD"],
      output_names=["Jc", "Dc", "arefc"],
      source=_MSL,
    )
    _CACHE[key] = k
  nwr = mx.array(np.array([nw], np.int32))
  Jc, Dc, arefc = k(
    inputs=[sim.cdof, sim.subtree_com, sim.qvel, dist, cpos, chain_buf, nchain_buf, root_buf, invw_buf, params, nwr],
    template=[("NV", nv), ("NBODY", nbody), ("NCHAIN", nchain), ("NFOOT", 2)],
    grid=((nw + 31) // 32 * 32, 1, 1),
    threadgroup=(32, 1, 1),
    output_shapes=[(nw * 32 * nv,), (nw * 32,), (nw * 32,)],
    output_dtypes=[mx.float32, mx.float32, mx.float32],
  )
  return mx.reshape(Jc, (nw, 32, nv)), mx.reshape(Dc, (nw, 32)), mx.reshape(arefc, (nw, 32))
