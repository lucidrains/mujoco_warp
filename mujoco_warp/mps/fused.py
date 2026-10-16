"""Fused Metal kernels for the world-batched MLX engine.

One thread per world per kernel, mirroring the `mujoco_warp` pipeline in float32:

  K1 ``dyn``    FK -> geoms -> CoM/cinert -> cdof -> cvel/cdof_dot -> CRB ->
                RNE -> mass matrix -> passive + actuation -> qfrc_smooth, then
                (fused) broadphase + primitive narrowphase + constraint rows
                (friction, hinge limits, pyramidal contacts)
  K2 ``solve``  qacc_smooth via in-thread Cholesky + mask Newton with line
                search, then (fused) Euler integration

Optionally emits per-stage intermediates (``kin=True``) for validation.
"""

from __future__ import annotations

import re

MJ_MINVAL = 1e-15

_MSL_HELPERS = r"""
static inline float3 g_cross(float3 a, float3 b) {
  return float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

static inline float4 g_qmul(float4 u, float4 v) {
  return float4(
    u.x * v.x - u.y * v.y - u.z * v.z - u.w * v.w,
    u.x * v.y + u.y * v.x + u.z * v.w - u.w * v.z,
    u.x * v.z - u.y * v.w + u.z * v.x + u.w * v.y,
    u.x * v.w + u.y * v.z - u.z * v.y + u.w * v.x);
}

static inline float3 g_qrot(float4 q, float3 v) {
  float3 u = q.yzw;
  return v + 2.0f * q.x * g_cross(u, v) + 2.0f * g_cross(u, g_cross(u, v));
}

static inline float4 g_axis_angle(float3 axis, float angle) {
  float s = metal::sin(0.5f * angle);
  return float4(metal::cos(0.5f * angle), axis * s);
}

static inline float3 g_normalize3(float3 v) {
  float n = metal::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  if (n > 0.0f) return v / n;
  return v;
}

static inline float4 g_normalize4(float4 v) {
  float n = metal::sqrt(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
  if (n > 0.0f) return v / n;
  return v;
}

static inline void g_qmat(float4 q, thread float* m) {
  float qw = q.x, qx = q.y, qy = q.z, qz = q.w;
  float q00 = qw * qw, q01 = qw * qx, q02 = qw * qy, q03 = qw * qz;
  float q11 = qx * qx, q12 = qx * qy, q13 = qx * qz;
  float q22 = qy * qy, q23 = qy * qz, q33 = qz * qz;
  m[0] = q00 + q11 - q22 - q33; m[1] = 2.0f * (q12 - q03); m[2] = 2.0f * (q13 + q02);
  m[3] = 2.0f * (q12 + q03);    m[4] = q00 - q11 + q22 - q33; m[5] = 2.0f * (q23 - q01);
  m[6] = 2.0f * (q13 - q02);    m[7] = 2.0f * (q23 + q01);    m[8] = q00 - q11 - q22 + q33;
}

static inline float g_dot3(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

static inline float3 g_mv(const thread float* m, float3 v) {
  return float3(m[0] * v.x + m[1] * v.y + m[2] * v.z,
                m[3] * v.x + m[4] * v.y + m[5] * v.z,
                m[6] * v.x + m[7] * v.y + m[8] * v.z);
}

// 0.5 * |(c - a) x (d - b)|, the quadrilateral area MuJoCo's hull4f maximizes

static inline float g_area4(float3 va, float3 vb, float3 vc, float3 vd) {
  float3 ca = va - vc;
  float3 db = vb - vd;
  float3 cr = g_cross(ca, db);
  return 0.5f * metal::sqrt(g_dot3(cr, cr));
}

static inline void g_motion_cross(const thread float* u, const thread float* v, thread float* out) {
  float3 u0 = float3(u[0], u[1], u[2]);
  float3 u1 = float3(u[3], u[4], u[5]);
  float3 v0 = float3(v[0], v[1], v[2]);
  float3 v1 = float3(v[3], v[4], v[5]);
  float3 ang = g_cross(u0, v0);
  float3 vel = g_cross(u1, v0) + g_cross(u0, v1);
  out[0] = ang.x; out[1] = ang.y; out[2] = ang.z;
  out[3] = vel.x; out[4] = vel.y; out[5] = vel.z;
}

static inline void g_motion_cross_force(const thread float* v, const thread float* f, thread float* out) {
  float3 v0 = float3(v[0], v[1], v[2]);
  float3 v1 = float3(v[3], v[4], v[5]);
  float3 f0 = float3(f[0], f[1], f[2]);
  float3 f1 = float3(f[3], f[4], f[5]);
  float3 ang = g_cross(v0, f0) + g_cross(v1, f1);
  float3 vel = g_cross(v0, f1);
  out[0] = ang.x; out[1] = ang.y; out[2] = ang.z;
  out[3] = vel.x; out[4] = vel.y; out[5] = vel.z;
}

static inline void g_inert_vec(const thread float* i, const thread float* v, thread float* res) {
  res[0] = i[0] * v[0] + i[3] * v[1] + i[4] * v[2] - i[8] * v[4] + i[7] * v[5];
  res[1] = i[3] * v[0] + i[1] * v[1] + i[5] * v[2] + i[8] * v[3] - i[6] * v[5];
  res[2] = i[4] * v[0] + i[5] * v[1] + i[2] * v[2] - i[7] * v[3] + i[6] * v[4];
  res[3] = i[8] * v[1] - i[7] * v[2] + i[9] * v[3];
  res[4] = i[6] * v[2] - i[8] * v[0] + i[9] * v[4];
  res[5] = i[7] * v[0] - i[6] * v[1] + i[9] * v[5];
}

static inline void g_kbimp(
  const device float* solref, const device float* solimp, float pos_imp, float timestep,
  thread float* kbimp
) {
  float timeconst = solref[0];
  float dampratio = solref[1];
  float dmin = solimp[0];
  float dmax = solimp[1];
  float width_raw = solimp[2];
  float width = width_raw;
  float mid = solimp[3];
  float power = solimp[4];

  timeconst = metal::max(timeconst, 2.0f * timestep);

  dmin = metal::clamp(dmin, 0.0001f, 0.9999f);
  dmax = metal::clamp(dmax, 0.0001f, 0.9999f);
  width = metal::max(1e-15f, width);
  mid = metal::clamp(mid, 0.0001f, 0.9999f);
  power = metal::max(1.0f, power);

  float dmax_sq = dmax * dmax;
  float k, b;
  if (solref[0] <= 0.0f) {
    k = -solref[0] / metal::max(1e-15f, dmax_sq);
  } else {
    k = 1.0f / metal::max(1e-15f, dmax_sq * timeconst * timeconst * dampratio * dampratio);
  }
  if (solref[1] <= 0.0f) {
    b = -solref[1] / metal::max(1e-15f, dmax);
  } else {
    b = 2.0f / metal::max(1e-15f, dmax * timeconst);
  }

  float imp_x = metal::abs(pos_imp) / width;
  float imp;
  if (dmin == dmax || width_raw <= 1e-15f) {
    imp = 0.5f * (dmin + dmax);
  } else if (imp_x <= 0.0f) {
    imp = dmin;
  } else if (imp_x >= 1.0f) {
    imp = dmax;
  } else if (power == 1.0f) {
    imp = dmin + imp_x * (dmax - dmin);
  } else if (imp_x <= mid) {
    float ratio = (1.0f / metal::pow(mid, power - 1.0f)) * metal::pow(imp_x, power);
    imp = metal::clamp(dmin + ratio * (dmax - dmin), dmin, dmax);
  } else {
    float ratio = 1.0f - (1.0f / metal::pow(1.0f - mid, power - 1.0f)) * metal::pow(1.0f - imp_x, power);
    imp = metal::clamp(dmin + ratio * (dmax - dmin), dmin, dmax);
  }
  kbimp[0] = k;
  kbimp[1] = b;
  kbimp[2] = imp;
}

static inline void g_make_frame(float3 a, thread float* frame) {
  float3 n = g_normalize3(a);
  float3 y = float3(0.0f, 1.0f, 0.0f);
  float3 z = float3(0.0f, 0.0f, 1.0f);
  float3 v = ((-0.5f < n.y) && (n.y < 0.5f)) ? y : z;
  v = v - n * g_dot3(n, v);
  v = g_normalize3(v);
  float3 c = g_cross(n, v);
  frame[0] = n.x; frame[1] = n.y; frame[2] = n.z;
  frame[3] = v.x; frame[4] = v.y; frame[5] = v.z;
  frame[6] = c.x; frame[7] = c.y; frame[8] = c.z;
}

static inline void g_chol_inplace(thread float* L, int n) {
  for (int k = 0; k < n; ++k) {
    float d = metal::sqrt(L[k * n + k]);
    L[k * n + k] = d;
    for (int i = k + 1; i < n; ++i) L[i * n + k] = L[i * n + k] / d;
    for (int j = k + 1; j < n; ++j) {
      float ajk = L[j * n + k];
      if (ajk != 0.0f) {
        for (int i = j; i < n; ++i) L[i * n + j] -= L[i * n + k] * ajk;
      }
    }
  }
}

static inline void g_chol_solve_fact(const thread float* L, const thread float* b, thread float* x, int n) {
  for (int i = 0; i < n; ++i) {
    float s = b[i];
    for (int j = 0; j < i; ++j) s -= L[i * n + j] * x[j];
    x[i] = s / L[i * n + i];
  }
  for (int i = n - 1; i >= 0; --i) {
    float s = x[i];
    for (int j = i + 1; j < n; ++j) s -= L[j * n + i] * x[j];
    x[i] = s / L[i * n + i];
  }
}

// mask-Newton row cost/grad/hess for one constraint row (batched_engine.eval_rows)
static inline void g_row_cost(float x, float d, float fl, thread float* c, thread float* g, thread float* hs) {
  if (fl > 0.0f) {
    float rf = fl / metal::max(d, 1e-30f);
    if (x <= -rf) {
      *c = -fl * (0.5f * rf + x);
      *g = -fl;
      *hs = 0.0f;
    } else if (x >= rf) {
      *c = -fl * (0.5f * rf - x);
      *g = fl;
      *hs = 0.0f;
    } else {
      *c = 0.5f * d * x * x;
      *g = d * x;
      *hs = d;
    }
  } else {
    if (x >= 0.0f) {
      *c = 0.0f; *g = 0.0f; *hs = 0.0f;
    } else {
      *c = 0.5f * d * x * x;
      *g = d * x;
      *hs = d;
    }
  }
}

// shifted row cost: returns c(x0 + ajv) - c(x0), g(x0 + ajv), h(x0 + ajv)
static inline void g_row_cost_shifted(
  float x0, float ajv, float d, float fl,
  thread float* dc, thread float* g, thread float* hs
) {
  float x = x0 + ajv;
  if (fl > 0.0f) {
    float rf = fl / metal::max(d, 1e-30f);
    if ((x0 > -rf && x0 < rf) && (x > -rf && x < rf)) {
      *dc = d * ajv * (x0 + 0.5f * ajv);
      *g = d * x;
      *hs = d;
      return;
    }
    float c0, g0_u, h0_u, c1, g1, h1;
    g_row_cost(x0, d, fl, &c0, &g0_u, &h0_u);
    g_row_cost(x, d, fl, &c1, &g1, &h1);
    *dc = c1 - c0;
    *g = g1;
    *hs = h1;
  } else {
    if (x0 < 0.0f && x < 0.0f) {
      *dc = d * ajv * (x0 + 0.5f * ajv);
      *g = d * x;
      *hs = d;
    } else if (x0 >= 0.0f && x >= 0.0f) {
      *dc = 0.0f; *g = 0.0f; *hs = 0.0f;
    } else if (x < 0.0f) {
      *dc = 0.5f * d * x * x;
      *g = d * x;
      *hs = d;
    } else {
      *dc = -0.5f * d * x0 * x0;
      *g = 0.0f;
      *hs = 0.0f;
    }
  }
}
"""

# ------------------------------------------------------------------ SDF helpers
# Port of the mjwarp SDF collision path (collision_sdf.py: sdf/clearance/
# compute_grad/gradient_step + halton initpoints) with the metric NIST nut/bolt
# SDFs from examples/alphanist/sdf_m.py, evaluated in unit space and scaled
# back (the fixed absolute line-search constants need unit-sized geometry).
_MSL_SDF = r"""
static inline float g_frac(float x) { return x - metal::floor(x); }

static inline float g_iso_pitch(float r) {
  if (r > 0.0085f) return 0.0025f;   // M20
  if (r > 0.007f) return 0.0020f;    // M16
  if (r > 0.005f) return 0.00175f;   // M12
  if (r > 0.003f) return 0.00125f;   // M8
  return 0.0007f;                    // M4
}

// rotated hex apothem coordinate (kinds shared with the bolt/nut SDFs)
static inline float g_hex_coord_x(float3 p) {
  float k = 6.0f / 3.141592653589793f / 2.0f;
  float angle = -metal::floor(metal::atan2(p.y, p.x) * k + 0.5f) / k;
  float s0 = metal::sin(angle);
  float s1 = metal::sin(angle + 1.570796326794897f);
  return s1 * p.x - s0 * p.y;
}

static inline float g_sdf_thread(float3 p, float r, float pit) {
  float rho2 = metal::sqrt(p.x * p.x + p.y * p.y) - r;
  float az = metal::atan2(p.y, p.x);
  float tri = metal::abs(g_frac(p.z / pit - az / 6.283185307179586f) - 0.5f);
  return (rho2 + 0.5f * pit - tri * pit) * 0.7071067811865476f;
}

static inline float g_sdf_bolt_unit(float3 p, float r, float pit) {
  float U = 12.0f * pit;
  float rho2 = metal::sqrt(p.x * p.x + p.y * p.y) - r;
  float thr = g_sdf_thread(p, r, pit);
  float body = metal::max(thr, -(U * 0.5f - metal::abs(p.z + U * 0.5f)));
  float cone = (p.z - rho2) * 0.7071067811865476f;
  body = metal::max(body, -(cone + U * 0.7071067811865476f));
  float head = g_hex_coord_x(p) - 1.5f * r;
  head = metal::max(head, metal::abs(p.z + U * 0.25f) - U * 0.25f);
  head = metal::max(head, (p.z + rho2 - 0.22f * U) * 0.7071067811865476f);
  return metal::min(body, head);
}

static inline float g_sdf_nut_unit(float3 p, float r, float pit) {
  float U = 12.0f * pit;
  float rho2 = metal::sqrt(p.x * p.x + p.y * p.y) - r;
  float thr = g_sdf_thread(p, r, pit);
  float cone = (p.z - rho2) * 0.7071067811865476f;
  float hole = metal::max(thr, -(cone + U * 0.5f * 0.7071067811865476f));
  hole = metal::min(hole, -cone - 0.05f * U * 0.7071067811865476f);
  float head = g_hex_coord_x(p) - 1.5f * r;
  head = metal::max(head, metal::abs(p.z + U * 0.25f) - U * 0.25f);
  head = metal::max(head, (p.z + rho2 - 0.22f * U) * 0.7071067811865476f);
  return metal::max(head, -hole);
}

static inline float g_sdf_bolt_m(float3 p, float r) {
  float s = 2.0f * r / 0.51f;
  return s * g_sdf_bolt_unit(p / s, r / s, g_iso_pitch(r) / s);
}

static inline float g_sdf_nut_m(float3 p, float r) {
  float s = 2.0f * r / 0.51f;
  return s * g_sdf_nut_unit(p / s, r / s, g_iso_pitch(r) / s);
}

// geom types (mjtGeom): 0 plane, 2 sphere, 3 capsule, 6 box, 8 sdf
static inline float g_sdf_eval(int gtype, float3 p, float3 sz, int kind, float radius) {
  if (gtype == 0) return p.z;
  if (gtype == 2) return metal::sqrt(g_dot3(p, p)) - sz.x;
  if (gtype == 3) {
    float zc = metal::clamp(p.z, -sz.y, sz.y);
    float3 v = float3(p.x, p.y, p.z - zc);
    return metal::sqrt(g_dot3(v, v)) - sz.x;
  }
  if (gtype == 6) {
    float3 a = metal::abs(p) - sz;
    float3 b = metal::max(a, float3(0.0f));
    return metal::sqrt(g_dot3(b, b)) + metal::min(metal::max(a.x, metal::max(a.y, a.z)), 0.0f);
  }
  return kind == 0 ? g_sdf_bolt_m(p, radius) : g_sdf_nut_m(p, radius);
}

static inline float3 g_sdf_grad(int gtype, float3 p, float3 sz, int kind, float radius) {
  const float eps = 1e-6f;
  float f0 = g_sdf_eval(gtype, p, sz, kind, radius);
  float3 dx = float3(eps, 0.0f, 0.0f);
  float3 dy = float3(0.0f, eps, 0.0f);
  float3 dz = float3(0.0f, 0.0f, eps);
  return float3(
    (g_sdf_eval(gtype, p + dx, sz, kind, radius) - f0) / eps,
    (g_sdf_eval(gtype, p + dy, sz, kind, radius) - f0) / eps,
    (g_sdf_eval(gtype, p + dz, sz, kind, radius) - f0) / eps);
}

static inline float g_sdf_clearance(
  int t1, float3 p1, float3 p2, float3 s1, float3 s2, int k1, int k2, float r1, float r2, bool inter
) {
  float a = g_sdf_eval(t1, p1, s1, k1, r1);
  float b = g_sdf_eval(8, p2, s2, k2, r2);
  if (inter) return metal::max(a, b);
  return a + b + metal::abs(metal::max(a, b));
}

static inline float3 g_sdf_compute_grad(
  int t1, float3 p1, float3 p2, float3 s1, float3 s2, int k1, int k2, float r1, float r2,
  const thread float* rel_mat, bool inter
) {
  float A = g_sdf_eval(t1, p1, s1, k1, r1);
  float B = g_sdf_eval(8, p2, s2, k2, r2);
  float3 gr1 = g_sdf_grad(t1, p1, s1, k1, r1);
  float3 gr2 = g_sdf_grad(8, p2, s2, k2, r2);
  float3 g1t = float3(
    rel_mat[0] * gr1.x + rel_mat[3] * gr1.y + rel_mat[6] * gr1.z,
    rel_mat[1] * gr1.x + rel_mat[4] * gr1.y + rel_mat[7] * gr1.z,
    rel_mat[2] * gr1.x + rel_mat[5] * gr1.y + rel_mat[8] * gr1.z);
  if (inter) return A > B ? g1t : gr2;
  float mv = metal::max(A, B);
  float3 mg = A > B ? g1t : gr2;
  float sgn = mv > 0.0f ? 1.0f : (mv < 0.0f ? -1.0f : 0.0f);
  return gr2 + g1t + mg * sgn;
}

// one mjwarp gradient_step (x lives in g2-local, rel_mat maps g2 -> g1)
static inline void g_sdf_step(
  int t1, thread float* x, float3 s1, float3 s2, int k1, int k2, float r1, float r2,
  const thread float* rel_mat, float3 rel_pos, int niter, bool inter, thread float* dist_out, thread float* x_out
) {
  const float amin = 1e-4f;
  const float rho = 0.5f;
  const float c = 0.1f;
  float3 xv = float3(x[0], x[1], x[2]);
  float dist = 1e10f;
  for (int i = 0; i < niter; ++i) {
    float alpha = 2.0f;
    float3 x2 = xv;
    float3 x1 = g_mv(rel_mat, x2) + rel_pos;
    float3 grad = g_sdf_compute_grad(t1, x1, x2, s1, s2, k1, k2, r1, r2, rel_mat, inter);
    float dist0 = g_sdf_clearance(t1, x1, x2, s1, s2, k1, k2, r1, r2, inter);
    float gd = g_dot3(grad, grad);
    if (gd < 1e-12f) {
      dist_out[0] = dist0;
      x_out[0] = xv.x; x_out[1] = xv.y; x_out[2] = xv.z;
      return;
    }
    float wolfe = -c * alpha * gd;
    while (true) {
      alpha *= rho;
      wolfe *= rho;
      xv = x2 - grad * alpha;
      x1 = g_mv(rel_mat, xv) + rel_pos;
      dist = g_sdf_clearance(t1, x1, xv, s1, s2, k1, k2, r1, r2, inter);
      if (alpha <= amin || (dist - dist0) <= wolfe) break;
    }
    if (dist > dist0) {
      dist_out[0] = dist0;
      x_out[0] = x2.x; x_out[1] = x2.y; x_out[2] = x2.z;
      return;
    }
  }
  dist_out[0] = dist;
  x_out[0] = xv.x; x_out[1] = xv.y; x_out[2] = xv.z;
}

static inline float g_halton(int index, int base) {
  float f = 1.0f;
  float r = 0.0f;
  int i = index;
  while (i > 0) {
    f = f / (float)base;
    r = r + f * (float)(i % base);
    i = i / base;
  }
  return r;
}
"""


# --------------------------------------------------------------------- K1 dyn+rows
def dyn_source(
  t: dict,
  adr_f: dict[str, int],
  adr_i: dict[str, int],
  kin: bool = False,
  adr_s: dict[str, int] | None = None,
  packed: bool = True,
  packed_m: bool = False,
  jmeta: bool = False,
) -> tuple[str, str]:
  """Returns (source, header) for the K1 dyn kernel (source is the kernel body).

  Templates: NBODY NJNT NQ NV NGEOM NCHAIN NU NP RC NFR NLIM BF MBS MS.

  `adr_f`/`adr_i` give the packed-buffer offsets of the static model tables
  (fewer device buffers than Metal's per-kernel limit). Float tables are read
  per world (`mbf[w * MBS + adr]`), so any of them (gravity, geom_quat/pos,
  body_mass, actuator gains, pair friction, ...) can differ per world - that is
  what per-world ramps/terrain and per-world domain randomization ride on.
  `kin=True` adds intermediates for validation (cdof, subtree_com, geom poses).
  `packed_m=True` streams the mass matrix as one chain of entries per dof
  (`NV * NCHAIN` floats) instead of the dense `NV * NV` block.  `jmeta=True`
  emits the per-row meta table on the dense row layout (the cooperative v2
  solve walks support pairs while reading dense rows).
  """
  NB, NJ, NQ, NV = t["NBODY"], t["NJNT"], t["NQ"], t["NV"]
  NGE, NCH, NU = t["NGEOM"], t["NCHAIN"], t["NU"]
  NP, RC = t["NP"], t["RC"]
  NFR, NLIM, BF = t["NFR"], t["NLIM"], t["BF"]
  MS = t["MS"]
  MBS = t["MBS"]
  SDFPTS, SDFIT = t["SDFPTS"], t["SDFIT"]

  # mass matrix stream: dense `NV * NV` block, or one ancestor chain per dof
  m_build = (
    f"""  // ==================================================== mass matrix (packed by dof chain)
  // M is a tree matrix: row i holds only its ancestor chain (dof_pad), self first
  device float* Mw = M_out + ((long)w) * {NV} * {NCH};
  for (int i = 0; i < {NV}; ++i) {{
    int b = dof_bodyid[i];
    thread float buf[6];
    g_inert_vec(&crb[b * 10], &cdof[i * 6], buf);
    for (int c = 0; c < {NCH}; ++c) {{
      int j = dof_pad[i * {NCH} + c];
      if (j < 0) break;
      float val = 0.0f;
      for (int s = 0; s < 6; ++s) val += cdof[j * 6 + s] * buf[s];
      if (j == i) val += dof_armature[i];
      Mw[i * {NCH} + c] = val;
    }}
  }}"""
    if packed_m
    else f"""  // ==================================================== mass matrix (streamed to M_out)
  device float* Mw = M_out + ((long)w) * {NV} * {NV};
  for (int i = 0; i < {NV} * {NV}; ++i) Mw[i] = 0.0f;
  for (int i = 0; i < {NV}; ++i) {{
    int b = dof_bodyid[i];
    thread float buf[6];
    g_inert_vec(&crb[b * 10], &cdof[i * 6], buf);
    for (int c = 0; c < {NCH}; ++c) {{
      int j = dof_pad[i * {NCH} + c];
      if (j < 0) break;
      float val = 0.0f;
      for (int s = 0; s < 6; ++s) val += cdof[j * 6 + s] * buf[s];
      if (j == i) {{
        Mw[i * {NV} + i] = val + dof_armature[i];
      }} else {{
        Mw[i * {NV} + j] = val;
        Mw[j * {NV} + i] = val;
      }}
    }}
  }}"""
  )

  kin_geom = (
    f"""
  for (int g = 0; g < {NGE}; ++g) {{
    for (int k = 0; k < 3; ++k) geom_xpos_out[(w * {NGE} + g) * 3 + k] = gp[g * 3 + k];
    for (int k = 0; k < 9; ++k) geom_xmat_out[(w * {NGE} + g) * 9 + k] = gm[g * 9 + k];
  }}
"""
    if kin
    else ""
  )
  kin_sub = (
    f"""
  for (int b = 0; b < {NB}; ++b)
    for (int k = 0; k < 3; ++k)
      subtree_com_out[(w * {NB} + b) * 3 + k] = subcom[b * 3 + k];
"""
    if kin
    else ""
  )
  kin_cdof = (
    f"""
  for (int d = 0; d < {NV}; ++d)
    for (int k = 0; k < 6; ++k) cdof_out[(w * {NV} + d) * 6 + k] = cdof[d * 6 + k];
"""
    if kin
    else ""
  )
  kin_body = (
    f"""
  for (int b = 0; b < {NB}; ++b) {{
    for (int k = 0; k < 3; ++k) xpos_out[(w * {NB} + b) * 3 + k] = xp[b * 3 + k];
    for (int k = 0; k < 9; ++k) xmat_out[(w * {NB} + b) * 9 + k] = xm[b * 9 + k];
    for (int k = 0; k < 6; ++k) cvel_out[(w * {NB} + b) * 6 + k] = cvel[b * 6 + k];
    for (int k = 0; k < 3; ++k) xipos_out[(w * {NB} + b) * 3 + k] = xip[b * 3 + k];
    for (int k = 0; k < 10; ++k) cinert_out[(w * {NB} + b) * 10 + k] = cin[b * 10 + k];
  }}
"""
    if kin
    else ""
  )
  kin_pconct_init = (
    f"""
  for (int p = 0; p < {NP}; ++p) pair_conct_out[w * {NP} + p] = 0;
"""
    if kin
    else ""
  )
  kin_pconct_write = (
    f"""
    pair_conct_out[w * {NP} + p] = pct;
"""
    if kin
    else ""
  )

  # row storage: `packed` writes only the support entries (Jw[r * MS + k], dof
  # sup_dofs[meta * MS + k], meta = jmetaw[r]); else dense nv-wide rows for coop.
  # `jmeta` emits the per-row meta table independently of the row storage (coop-v2
  # walks support pairs while reading dense rows).
  emit_jmeta = packed or jmeta
  jmeta_ptr = f"\n  device int* jmetaw = jmeta_out + ((long)w) * {RC};" if emit_jmeta else ""
  jmeta_fr = "    jmetaw[rowidx] = k;\n" if emit_jmeta else ""
  jmeta_lim = f"    jmetaw[rowidx] = {NFR} + k;\n" if emit_jmeta else ""
  jmeta_pt = f"        jmetaw[rowidx + dim] = {NFR} + {NLIM} + p;\n" if emit_jmeta else ""
  if packed:
    jptr = f"  device float* Jw = J_out + ((long)w) * {RC} * {MS};{jmeta_ptr}"
    fr_write = f"{jmeta_fr}    Jw[rowidx * {MS} + 0] = 1.0f;"
    lim_write = f"{jmeta_lim}    Jw[rowidx * {MS} + 0] = sgn;"
    pt_meta = f"{jmeta_pt}        flw[rowidx + dim] = 0.0f;"
    pt_open = "      int kk = 0;\n"
    pt_val = f"          Jw[(rowidx + dim) * {MS} + kk] = val;"
    pt_close = "        kk += 1;\n"
    # keep bracketed table names out of comments: they get rewritten to mif offsets
    rows_comment = (
      f"  // packed rows: row r holds its support values packed at Jw[r * {MS} + k], k\n"
      "  // below the support's length (sup_num / jmeta give the row's dof list)"
    )
  else:
    jptr = f"  device float* Jw = J_out + ((long)w) * {RC} * {NV};{jmeta_ptr}"
    fr_write = (
      f"{jmeta_fr}    for (int d = 0; d < {NV}; ++d) Jw[rowidx * {NV} + d] = 0.0f;\n    Jw[rowidx * {NV} + dof] = 1.0f;"
    )
    lim_write = (
      f"{jmeta_lim}    for (int d = 0; d < {NV}; ++d) Jw[rowidx * {NV} + d] = 0.0f;\n    Jw[rowidx * {NV} + dof] = sgn;"
    )
    pt_meta = f"{jmeta_pt}        for (int d = 0; d < {NV}; ++d) Jw[(rowidx + dim) * {NV} + d] = 0.0f;\n        flw[rowidx + dim] = 0.0f;"
    pt_open = ""
    pt_val = f"          Jw[(rowidx + dim) * {NV} + dof] = val;"
    pt_close = ""
    rows_comment = "  // rows stream straight to global memory (no per-thread row arrays)"

  # SDF narrowphase (op 6): halton initpoints in the AABB intersection (g1
  # frame) + mjwarp's gradient descent; the deepest point wins. Port of
  # mjwarp _sdf_narrowphase + gradient_descent, single point per pair.
  sdf_branch = f"""    else if (op == 6) {{
      int k1s = geom_sdf_kind[g1];
      int k2s = geom_sdf_kind[g2];
      float r1sf = geom_sdf_attr[g1 * 3];
      float r2sf = geom_sdf_attr[g2 * 3];
      int t1sf = geom_type_i[g1];
      float3 sz1 = float3(geom_size[g1 * 3 + 0], geom_size[g1 * 3 + 1], geom_size[g1 * 3 + 2]);
      float3 sz2 = float3(geom_size[g2 * 3 + 0], geom_size[g2 * 3 + 1], geom_size[g2 * 3 + 2]);
      // rel_mat = gm1^T * gm2 (maps g2-local -> g1-local), rel_pos likewise
      thread float relmat[9];
      relmat[0] = gm[g1 * 9 + 0] * gm[g2 * 9 + 0] + gm[g1 * 9 + 3] * gm[g2 * 9 + 3] + gm[g1 * 9 + 6] * gm[g2 * 9 + 6];
      relmat[1] = gm[g1 * 9 + 0] * gm[g2 * 9 + 1] + gm[g1 * 9 + 3] * gm[g2 * 9 + 4] + gm[g1 * 9 + 6] * gm[g2 * 9 + 7];
      relmat[2] = gm[g1 * 9 + 0] * gm[g2 * 9 + 2] + gm[g1 * 9 + 3] * gm[g2 * 9 + 5] + gm[g1 * 9 + 6] * gm[g2 * 9 + 8];
      relmat[3] = gm[g1 * 9 + 1] * gm[g2 * 9 + 0] + gm[g1 * 9 + 4] * gm[g2 * 9 + 3] + gm[g1 * 9 + 7] * gm[g2 * 9 + 6];
      relmat[4] = gm[g1 * 9 + 1] * gm[g2 * 9 + 1] + gm[g1 * 9 + 4] * gm[g2 * 9 + 4] + gm[g1 * 9 + 7] * gm[g2 * 9 + 7];
      relmat[5] = gm[g1 * 9 + 1] * gm[g2 * 9 + 2] + gm[g1 * 9 + 4] * gm[g2 * 9 + 5] + gm[g1 * 9 + 7] * gm[g2 * 9 + 8];
      relmat[6] = gm[g1 * 9 + 2] * gm[g2 * 9 + 0] + gm[g1 * 9 + 5] * gm[g2 * 9 + 3] + gm[g1 * 9 + 8] * gm[g2 * 9 + 6];
      relmat[7] = gm[g1 * 9 + 2] * gm[g2 * 9 + 1] + gm[g1 * 9 + 5] * gm[g2 * 9 + 4] + gm[g1 * 9 + 8] * gm[g2 * 9 + 7];
      relmat[8] = gm[g1 * 9 + 2] * gm[g2 * 9 + 2] + gm[g1 * 9 + 5] * gm[g2 * 9 + 5] + gm[g1 * 9 + 8] * gm[g2 * 9 + 8];
      float3 dpos = xa2 - xa1;
      float3 relpos = float3(
        gm[g1 * 9 + 0] * dpos.x + gm[g1 * 9 + 3] * dpos.y + gm[g1 * 9 + 6] * dpos.z,
        gm[g1 * 9 + 1] * dpos.x + gm[g1 * 9 + 4] * dpos.y + gm[g1 * 9 + 7] * dpos.z,
        gm[g1 * 9 + 2] * dpos.x + gm[g1 * 9 + 5] * dpos.y + gm[g1 * 9 + 8] * dpos.z);
      float3 c1 = float3(geom_aabb[g1 * 6 + 0], geom_aabb[g1 * 6 + 1], geom_aabb[g1 * 6 + 2]);
      float3 h1 = float3(geom_aabb[g1 * 6 + 3], geom_aabb[g1 * 6 + 4], geom_aabb[g1 * 6 + 5]);
      float3 c2 = float3(geom_aabb[g2 * 6 + 0], geom_aabb[g2 * 6 + 1], geom_aabb[g2 * 6 + 2]);
      float3 h2 = float3(geom_aabb[g2 * 6 + 3], geom_aabb[g2 * 6 + 4], geom_aabb[g2 * 6 + 5]);
      // box2 in g1 frame: exact AABB of the 8 transformed corners (matches
      // mjwarp transform_aabb, which the halton initpoints depend on)
      float3 c2g_min = float3(1e30f);
      float3 c2g_max = float3(-1e30f);
      for (int jj = 0; jj < 8; ++jj) {{
        float3 v = float3(
          h2.x * ((jj & 1) != 0 ? 1.0f : -1.0f),
          h2.y * ((jj & 2) != 0 ? 1.0f : -1.0f),
          h2.z * ((jj & 4) != 0 ? 1.0f : -1.0f));
        float3 fv = g_mv(relmat, v + c2) + relpos;
        c2g_min = metal::min(c2g_min, fv);
        c2g_max = metal::max(c2g_max, fv);
      }}
      float3 bmin = metal::max(c1 - h1, c2g_min);
      float3 bmax = metal::min(c1 + h1, c2g_max);
      // keep the 6 smallest-distinct candidates (warp writes every initpoint
      // result; the row filter below drops the non-contacting ones)
      thread float cand_d[6];
      thread float cand_x[18];
      for (int ii = 0; ii < 6; ++ii) cand_d[ii] = 1e9f;
      if (bmax.x >= bmin.x && bmax.y >= bmin.y && bmax.z >= bmin.z) {{
        float3 bext = bmax - bmin;
        for (int ii = 0; ii < {SDFPTS}; ++ii) {{
          float3 xg1 = bmin + bext * float3(g_halton(ii, 2), g_halton(ii, 3), g_halton(ii, 5));
          float3 xw = g_mv(&gm[g1 * 9], xg1) + xa1;
          float3 dd = xw - xa2;
          thread float xx[3];
          xx[0] = gm[g2 * 9 + 0] * dd.x + gm[g2 * 9 + 3] * dd.y + gm[g2 * 9 + 6] * dd.z;
          xx[1] = gm[g2 * 9 + 1] * dd.x + gm[g2 * 9 + 4] * dd.y + gm[g2 * 9 + 7] * dd.z;
          xx[2] = gm[g2 * 9 + 2] * dd.x + gm[g2 * 9 + 5] * dd.y + gm[g2 * 9 + 8] * dd.z;
          thread float dout[1];
          thread float xout[3];
          dout[0] = 1e9f;
          xout[0] = xx[0]; xout[1] = xx[1]; xout[2] = xx[2];
          g_sdf_step(t1sf, xx, sz1, sz2, k1s, k2s, r1sf, r2sf, relmat, relpos, {SDFIT}, false, dout, xout);
          g_sdf_step(t1sf, xout, sz1, sz2, k1s, k2s, r1sf, r2sf, relmat, relpos, 1, true, dout, xout);
          if (dout[0] < cand_d[5]) {{
            int slot = 5;
            while (slot > 0 && cand_d[slot - 1] > dout[0]) {{
              cand_d[slot] = cand_d[slot - 1];
              cand_x[slot * 3 + 0] = cand_x[(slot - 1) * 3 + 0];
              cand_x[slot * 3 + 1] = cand_x[(slot - 1) * 3 + 1];
              cand_x[slot * 3 + 2] = cand_x[(slot - 1) * 3 + 2];
              slot -= 1;
            }}
            cand_d[slot] = dout[0];
            cand_x[slot * 3 + 0] = xout[0];
            cand_x[slot * 3 + 1] = xout[1];
            cand_x[slot * 3 + 2] = xout[2];
          }}
        }}
      }}
      int nc = 0;
      for (int ii = 0; ii < 6 && nc < 6; ++ii) {{
        if (cand_d[ii] > 1e8f) break;
        float3 xv = float3(cand_x[ii * 3 + 0], cand_x[ii * 3 + 1], cand_x[ii * 3 + 2]);
        float3 xv1 = g_mv(relmat, xv) + relpos;
        float3 gr1 = g_sdf_grad(t1sf, xv1, sz1, k1s, r1sf);
        float3 g1t = g_normalize3(float3(
          relmat[0] * gr1.x + relmat[3] * gr1.y + relmat[6] * gr1.z,
          relmat[1] * gr1.x + relmat[4] * gr1.y + relmat[7] * gr1.z,
          relmat[2] * gr1.x + relmat[5] * gr1.y + relmat[8] * gr1.z));
        float3 g2n = g_normalize3(g_sdf_grad(8, xv, sz2, k2s, r2sf));
        float3 nloc = g_normalize3(g1t - g2n);
        float3 posw = g_mv(&gm[g2 * 9], xv) + xa2;
        float3 nw = g_mv(&gm[g2 * 9], nloc);
        ds[nc] = cand_d[ii];
        psd[nc] = posw - nw * (0.5f * cand_d[ii]);
        g_make_frame(nw, frms[nc]);
        nc += 1;
      }}
      npts = nc;
    }}"""

  body = f"""
  int w = (int)thread_position_in_grid.x;
  if (w >= nworld_buf[0]) return;

  // mesh vertex read (macro: expands after packed-table name substitution, so
  // it never depends on the buffer's Metal address space)

  #define G_MESH_VERT(vadr, idx) float3( \
    mesh_vert[((vadr) + (idx)) * 3 + 0], \
    mesh_vert[((vadr) + (idx)) * 3 + 1], \
    mesh_vert[((vadr) + (idx)) * 3 + 2])

  thread float qpos[{NQ}];
  for (int k = 0; k < {NQ}; ++k) qpos[k] = qpos_in[w * {NQ} + k];
  thread float v[{NV}];
  for (int k = 0; k < {NV}; ++k) v[k] = qvel_in[w * {NV} + k];
  const float timestep = opt_buf[0];

  thread float xp[{NB} * 3];
  thread float xq[{NB} * 4];
  thread float xa[{NJ} * 3];
  thread float xs[{NJ} * 3];
  thread float xm[{NB} * 9];
  thread float xip[{NB} * 3];
  thread float subcom[{NB} * 3];
  thread float cin[{NB} * 10];
  thread float crb[{NB} * 10];
  thread float cdof[{NV} * 6];
  thread float cvel[{NB} * 6];
  thread float cdofdot[{NV} * 6];
  thread float cacc[{NB} * 6];
  thread float cfrc[{NB} * 6];

  // ==================================================== FK (jtype: 0 free, 2 slide, 3 hinge)
  for (int b = 0; b < {NB}; ++b) {{
    int ja = body_jntadr[b];
    int jn = body_jntnum[b];
    if (jn == 1 && jnt_type[ja] == 0) {{
      int qa = jnt_qposadr[ja];
      xp[b * 3 + 0] = qpos[qa + 0];
      xp[b * 3 + 1] = qpos[qa + 1];
      xp[b * 3 + 2] = qpos[qa + 2];
      float4 quat = g_normalize4(float4(qpos[qa + 3], qpos[qa + 4], qpos[qa + 5], qpos[qa + 6]));
      xq[b * 4 + 0] = quat.x; xq[b * 4 + 1] = quat.y; xq[b * 4 + 2] = quat.z; xq[b * 4 + 3] = quat.w;
      for (int k = 0; k < 3; ++k) {{
        xa[ja * 3 + k] = xp[b * 3 + k];
        xs[ja * 3 + k] = jnt_axis[ja * 3 + k];
      }}
      continue;
    }}
    float3 bpos = float3(body_pos[b * 3 + 0], body_pos[b * 3 + 1], body_pos[b * 3 + 2]);
    float4 bquat = float4(body_quat[b * 4 + 0], body_quat[b * 4 + 1], body_quat[b * 4 + 2], body_quat[b * 4 + 3]);
    int pid = body_parentid[b];
    if (pid >= 0 && pid != b) {{
      float3 pn = float3(xp[pid * 3 + 0], xp[pid * 3 + 1], xp[pid * 3 + 2]);
      float4 pq = float4(xq[pid * 4 + 0], xq[pid * 4 + 1], xq[pid * 4 + 2], xq[pid * 4 + 3]);
      bpos = g_qrot(pq, bpos) + pn;
      bquat = g_qmul(pq, bquat);
    }}
    for (int k = 0; k < jn; ++k) {{
      int j = ja + k;
      int qa = jnt_qposadr[j];
      float3 ax = float3(jnt_axis[j * 3 + 0], jnt_axis[j * 3 + 1], jnt_axis[j * 3 + 2]);
      float3 jpos = float3(jnt_pos[j * 3 + 0], jnt_pos[j * 3 + 1], jnt_pos[j * 3 + 2]);
      float3 anchor = g_qrot(bquat, jpos) + bpos;
      float3 gaxis = g_qrot(bquat, ax);
      int jt = jnt_type[j];
      if (jt == 2) {{
        bpos += gaxis * (qpos[qa] - qpos0[qa]);
      }} else if (jt == 3) {{
        float angle = qpos[qa] - qpos0[qa];
        float4 qloc = g_axis_angle(ax, angle);
        bquat = g_qmul(bquat, qloc);
        bpos = anchor - g_qrot(bquat, jpos);
      }}
      xa[j * 3 + 0] = anchor.x; xa[j * 3 + 1] = anchor.y; xa[j * 3 + 2] = anchor.z;
      xs[j * 3 + 0] = gaxis.x; xs[j * 3 + 1] = gaxis.y; xs[j * 3 + 2] = gaxis.z;
    }}
    float4 nq4 = g_normalize4(bquat);
    xp[b * 3 + 0] = bpos.x; xp[b * 3 + 1] = bpos.y; xp[b * 3 + 2] = bpos.z;
    xq[b * 4 + 0] = nq4.x; xq[b * 4 + 1] = nq4.y; xq[b * 4 + 2] = nq4.z; xq[b * 4 + 3] = nq4.w;
  }}
  for (int b = 0; b < {NB}; ++b) {{
    float4 q4 = float4(xq[b * 4 + 0], xq[b * 4 + 1], xq[b * 4 + 2], xq[b * 4 + 3]);
    g_qmat(q4, &xm[b * 9]);
    float3 ipos = float3(body_ipos[b * 3 + 0], body_ipos[b * 3 + 1], body_ipos[b * 3 + 2]);
    float3 bpos3 = float3(xp[b * 3 + 0], xp[b * 3 + 1], xp[b * 3 + 2]);
    float3 gip = bpos3 + g_qrot(q4, ipos);
    xip[b * 3 + 0] = gip.x; xip[b * 3 + 1] = gip.y; xip[b * 3 + 2] = gip.z;
  }}

  // local geom transforms (world frame), kept in registers
  thread float gp[{NGE} * 3];
  thread float gm[{NGE} * 9];
  for (int g = 0; g < {NGE}; ++g) {{
    int b = geom_bodyid_i[g];
    float3 gp3 = float3(geom_pos[g * 3 + 0], geom_pos[g * 3 + 1], geom_pos[g * 3 + 2]);
    float4 gq4 = float4(geom_quat[g * 4 + 0], geom_quat[g * 4 + 1], geom_quat[g * 4 + 2], geom_quat[g * 4 + 3]);
    float3 bp3 = float3(xp[b * 3 + 0], xp[b * 3 + 1], xp[b * 3 + 2]);
    float4 bq4 = float4(xq[b * 4 + 0], xq[b * 4 + 1], xq[b * 4 + 2], xq[b * 4 + 3]);
    float3 xgp = bp3 + g_qrot(bq4, gp3);
    gp[g * 3 + 0] = xgp.x; gp[g * 3 + 1] = xgp.y; gp[g * 3 + 2] = xgp.z;
    thread float gmat[9];
    g_qmat(g_qmul(bq4, gq4), gmat);
    for (int k = 0; k < 9; ++k) gm[g * 9 + k] = gmat[k];
  }}
{kin_geom}
  // ==================================================== subtree com
  for (int b = 0; b < {NB}; ++b) {{
    float mss = body_mass[b];
    subcom[b * 3 + 0] = xip[b * 3 + 0] * mss;
    subcom[b * 3 + 1] = xip[b * 3 + 1] * mss;
    subcom[b * 3 + 2] = xip[b * 3 + 2] * mss;
  }}
  for (int b = {NB} - 1; b > 0; --b) {{
    int pid = body_parentid[b];
    if (pid != b) {{
      subcom[pid * 3 + 0] += subcom[b * 3 + 0];
      subcom[pid * 3 + 1] += subcom[b * 3 + 1];
      subcom[pid * 3 + 2] += subcom[b * 3 + 2];
    }}
  }}
  for (int b = 0; b < {NB}; ++b) {{
    float mss = body_subtreemass[b];
    float inv = mss > 1e-30f ? 1.0f / mss : 0.0f;
    subcom[b * 3 + 0] *= inv;
    subcom[b * 3 + 1] *= inv;
    subcom[b * 3 + 2] *= inv;
  }}
{kin_sub}
  // ==================================================== cinert
  for (int b = 0; b < {NB}; ++b) {{
    int root = body_rootid[b];
    float3 dif = float3(
      xip[b * 3 + 0] - subcom[root * 3 + 0],
      xip[b * 3 + 1] - subcom[root * 3 + 1],
      xip[b * 3 + 2] - subcom[root * 3 + 2]);
    float i0 = body_inertia[b * 3 + 0];
    float i1 = body_inertia[b * 3 + 1];
    float i2 = body_inertia[b * 3 + 2];
    float4 q4 = float4(xq[b * 4 + 0], xq[b * 4 + 1], xq[b * 4 + 2], xq[b * 4 + 3]);
    float4 iq4 = float4(body_iquat[b * 4 + 0], body_iquat[b * 4 + 1], body_iquat[b * 4 + 2], body_iquat[b * 4 + 3]);
    thread float im[9];
    g_qmat(g_qmul(q4, iq4), im);
    float t00 = im[0] * i0 * im[0] + im[1] * i1 * im[1] + im[2] * i2 * im[2];
    float t01 = im[0] * i0 * im[3] + im[1] * i1 * im[4] + im[2] * i2 * im[5];
    float t02 = im[0] * i0 * im[6] + im[1] * i1 * im[7] + im[2] * i2 * im[8];
    float t11 = im[3] * i0 * im[3] + im[4] * i1 * im[4] + im[5] * i2 * im[5];
    float t12 = im[3] * i0 * im[6] + im[4] * i1 * im[7] + im[5] * i2 * im[8];
    float t22 = im[6] * i0 * im[6] + im[7] * i1 * im[7] + im[8] * i2 * im[8];
    float mss = body_mass[b];
    thread float* dst = &cin[b * 10];
    dst[0] = t00 + mss * (dif.y * dif.y + dif.z * dif.z);
    dst[1] = t11 + mss * (dif.x * dif.x + dif.z * dif.z);
    dst[2] = t22 + mss * (dif.x * dif.x + dif.y * dif.y);
    dst[3] = t01 - mss * dif.x * dif.y;
    dst[4] = t02 - mss * dif.x * dif.z;
    dst[5] = t12 - mss * dif.y * dif.z;
    dst[6] = mss * dif.x;
    dst[7] = mss * dif.y;
    dst[8] = mss * dif.z;
    dst[9] = mss;
  }}

  // ==================================================== cdof
  for (int i = 0; i < {NV} * 6; ++i) cdof[i] = 0.0f;
  for (int j = 0; j < {NJ}; ++j) {{
    int b = jnt_bodyid[j];
    int dof = jnt_dofadr[j];
    int jt = jnt_type[j];
    int root = body_rootid[b];
    float3 com3 = float3(subcom[root * 3 + 0], subcom[root * 3 + 1], subcom[root * 3 + 2]);
    float3 anchor = float3(xa[j * 3 + 0], xa[j * 3 + 1], xa[j * 3 + 2]);
    float3 axis = float3(xs[j * 3 + 0], xs[j * 3 + 1], xs[j * 3 + 2]);
    float3 offset = com3 - anchor;
    if (jt == 0) {{
      for (int k = 0; k < 3; ++k) cdof[(dof + k) * 6 + 3 + k] = 1.0f;
      for (int k = 0; k < 3; ++k) {{
        float3 colk = float3(xm[b * 9 + 0 + k], xm[b * 9 + 3 + k], xm[b * 9 + 6 + k]);
        float3 cr = g_cross(colk, offset);
        cdof[(dof + 3 + k) * 6 + 0] = colk.x;
        cdof[(dof + 3 + k) * 6 + 1] = colk.y;
        cdof[(dof + 3 + k) * 6 + 2] = colk.z;
        cdof[(dof + 3 + k) * 6 + 3] = cr.x;
        cdof[(dof + 3 + k) * 6 + 4] = cr.y;
        cdof[(dof + 3 + k) * 6 + 5] = cr.z;
      }}
    }} else if (jt == 2) {{
      cdof[dof * 6 + 3] = axis.x;
      cdof[dof * 6 + 4] = axis.y;
      cdof[dof * 6 + 5] = axis.z;
    }} else if (jt == 3) {{
      float3 cr = g_cross(axis, offset);
      cdof[dof * 6 + 0] = axis.x;
      cdof[dof * 6 + 1] = axis.y;
      cdof[dof * 6 + 2] = axis.z;
      cdof[dof * 6 + 3] = cr.x;
      cdof[dof * 6 + 4] = cr.y;
      cdof[dof * 6 + 5] = cr.z;
    }}
  }}
{kin_cdof}
  // ==================================================== cvel + cdof_dot
  for (int i = 0; i < {NV} * 6; ++i) cdofdot[i] = 0.0f;
  for (int k = 0; k < 6; ++k) cvel[k] = 0.0f;
  for (int b = 1; b < {NB}; ++b) {{
    int pid = body_parentid[b];
    float cv[6];
    for (int k = 0; k < 6; ++k) cv[k] = cvel[pid * 6 + k];
    int ja = body_jntadr[b];
    int jn = body_jntnum[b];
    for (int jj = 0; jj < jn; ++jj) {{
      int j = ja + jj;
      int jt = jnt_type[j];
      int dof = jnt_dofadr[j];
      if (jt == 0) {{
        for (int k = 0; k < 3; ++k)
          for (int s = 0; s < 6; ++s)
            cv[s] += cdof[(dof + k) * 6 + s] * v[dof + k];
        for (int k = 3; k < 6; ++k) {{
          float md[6];
          g_motion_cross(cv, &cdof[(dof + k) * 6], md);
          for (int s = 0; s < 6; ++s) cdofdot[(dof + k) * 6 + s] = md[s];
        }}
        for (int k = 3; k < 6; ++k)
          for (int s = 0; s < 6; ++s)
            cv[s] += cdof[(dof + k) * 6 + s] * v[dof + k];
      }} else {{
        float md[6];
        g_motion_cross(cv, &cdof[dof * 6], md);
        for (int s = 0; s < 6; ++s) cdofdot[dof * 6 + s] = md[s];
        for (int s = 0; s < 6; ++s) cv[s] += cdof[dof * 6 + s] * v[dof];
      }}
    }}
    for (int k = 0; k < 6; ++k) cvel[b * 6 + k] = cv[k];
  }}
{kin_body}
  // ==================================================== crb
  for (int i = 0; i < {NB} * 10; ++i) crb[i] = cin[i];
  for (int b = {NB} - 1; b > 0; --b) {{
    int pid = body_parentid[b];
    if (pid != 0 && pid != b) {{
      for (int k = 0; k < 10; ++k) crb[pid * 10 + k] += crb[b * 10 + k];
    }}
  }}

  // ==================================================== rne: cacc
  cacc[0] = 0.0f; cacc[1] = 0.0f; cacc[2] = 0.0f;
  cacc[3] = -gravity[0]; cacc[4] = -gravity[1]; cacc[5] = -gravity[2];
  for (int b = 1; b < {NB}; ++b) {{
    int pid = body_parentid[b];
    float loc[6];
    for (int k = 0; k < 6; ++k) loc[k] = cacc[pid * 6 + k];
    int da = body_dofadr[b];
    int dn = body_dofnum[b];
    for (int d = da; d < da + dn; ++d) {{
      for (int s = 0; s < 6; ++s) loc[s] += cdofdot[d * 6 + s] * v[d];
    }}
    for (int k = 0; k < 6; ++k) cacc[b * 6 + k] = loc[k];
  }}

  // ==================================================== rne: cfrc + qfrc_bias
  thread float qbias[{NV}];
  for (int i = 0; i < {NB} * 6; ++i) cfrc[i] = 0.0f;
  for (int b = 1; b < {NB}; ++b) {{
    thread float f[6];
    g_inert_vec(&cin[b * 10], &cacc[b * 6], f);
    thread float ivb[6];
    g_inert_vec(&cin[b * 10], &cvel[b * 6], ivb);
    float mcf[6];
    g_motion_cross_force(&cvel[b * 6], ivb, mcf);
    for (int k = 0; k < 6; ++k) cfrc[b * 6 + k] = f[k] + mcf[k];
  }}
  for (int b = {NB} - 1; b > 0; --b) {{
    int pid = body_parentid[b];
    if (pid != b) {{
      for (int k = 0; k < 6; ++k) cfrc[pid * 6 + k] += cfrc[b * 6 + k];
    }}
  }}
  for (int d = 0; d < {NV}; ++d) {{
    int b = dof_bodyid[d];
    float s = 0.0f;
    for (int k = 0; k < 6; ++k) s += cdof[d * 6 + k] * cfrc[b * 6 + k];
    qbias[d] = s;
    bias_out[w * {NV} + d] = s;
  }}

{m_build}

  // ==================================================== qfrc_smooth = -bias+passive+actuation
  thread float smoothf[{NV}];
  for (int i = 0; i < {NV}; ++i) smoothf[i] = -qbias[i] - dof_damping[i] * v[i];
  for (int j = 0; j < {NJ}; ++j) {{
    int jt = jnt_type[j];
    if (jt == 2 || jt == 3) {{
      float kst = jnt_stiffness[j];
      if (kst != 0.0f) {{
        int dof = jnt_dofadr[j];
        int qa = jnt_qposadr[j];
        smoothf[dof] += -(qpos[qa] - qpos_spring[qa]) * kst;
      }}
    }}
  }}
  for (int u = 0; u < {NU}; ++u) {{
    int j = act_trnid[u];
    int dof = jnt_dofadr[j];
    int qa = jnt_qposadr[j];
    float gear = act_gear[u];
    float length = qpos[qa] * gear;
    float vel = gear * v[dof];
    float c = ctrl_in[w * {NU} + u];
    if (act_ctrllimited[u] != 0) {{
      c = metal::min(metal::max(c, act_ctrlrange[u * 2 + 0]), act_ctrlrange[u * 2 + 1]);
    }}
    float force = act_gainprm[u * 6 + 0] * c
      + (act_biasprm[u * 6 + 0] + act_biasprm[u * 6 + 1] * length + act_biasprm[u * 6 + 2] * vel);
    if (act_forcelimited[u] != 0) {{
      force = metal::min(metal::max(force, act_forcerange[u * 2 + 0]), act_forcerange[u * 2 + 1]);
    }}
    smoothf[dof] += gear * force;
  }}
  for (int i = 0; i < {NV}; ++i) {{
    smooth_out[w * {NV} + i] = smoothf[i];
  }}

  // ==================================================== constraint rows
{rows_comment}
{jptr}
  device float* Dw = D_out + ((long)w) * {RC};
  device float* arefw = aref_out + ((long)w) * {RC};
  device float* flw = fl_out + ((long)w) * {RC};
  int rowidx = 0;
  int conct = 0;
  int ovr = 0;

  // ---- friction-dof rows
  for (int k = 0; k < {NFR}; ++k) {{
    int dof = fr_dofs[k];
{fr_write}
    Dw[rowidx] = fr_D[k];
    arefw[rowidx] = -fr_b[k] * v[dof];
    flw[rowidx] = fr_fl[k];
    rowidx += 1;
  }}

  // ---- hinge limit rows
  for (int k = 0; k < {NLIM}; ++k) {{
    int j = lim_jnts[k];
    int qa = jnt_qposadr[j];
    int dof = jnt_dofadr[j];
    float lo = jnt_range[j * 2 + 0];
    float hi = jnt_range[j * 2 + 1];
    float mg = jnt_margin[j];
    float dmin = qpos[qa] - lo;
    float dmax = hi - qpos[qa];
    float pos_lim = metal::min(dmin, dmax) - mg;
    if (pos_lim >= 0.0f) continue;
    float sgn = dmin < dmax ? 1.0f : -1.0f;
{lim_write}
    flw[rowidx] = 0.0f;
    thread float kb[3];
    g_kbimp(&jnt_solref[j * 2], &jnt_solimp[j * 5], pos_lim, timestep, kb);
    float iw = dof_invweight0[dof];
    Dw[rowidx] = 1.0f / metal::max(iw * (1.0f - kb[2]) / kb[2], 1e-15f);
    float velr = sgn * v[dof];
    arefw[rowidx] = -kb[0] * kb[2] * pos_lim - kb[1] * velr;
    rowidx += 1;
  }}

  // ---- contact rows
{kin_pconct_init}
  for (int p = 0; p < {NP}; ++p) {{
    if (ovr != 0) break;
    int g1 = pair_g1[p];
    int g2 = pair_g2[p];
    float rb1 = geom_rbound[g1];
    float rb2 = geom_rbound[g2];
    float fm1 = geom_mg[g1];
    float fm2 = geom_mg[g2];
    float3 xa1 = float3(gp[g1 * 3 + 0], gp[g1 * 3 + 1], gp[g1 * 3 + 2]);
    float3 xa2 = float3(gp[g2 * 3 + 0], gp[g2 * 3 + 1], gp[g2 * 3 + 2]);
    float margin_c = pair_margin[p];

    bool pass = true;
    if (rb1 == 0.0f || rb2 == 0.0f) {{
      if ((BF & 1) != 0) {{
        int pgid = rb1 == 0.0f ? g1 : g2;
        int ogid = rb1 == 0.0f ? g2 : g1;
        float3 pnorm = float3(gm[pgid * 9 + 2], gm[pgid * 9 + 5], gm[pgid * 9 + 8]);
        float3 ppos = (pgid == g1) ? xa1 : xa2;
        float3 opos = (pgid == g1) ? xa2 : xa1;
        float dist_p = g_dot3(opos - ppos, pnorm);
        pass = dist_p <= geom_rbound[ogid] + fm1 + fm2;
      }}
    }} else {{
      if ((BF & 2) != 0) {{
        float bound = rb1 + rb2 + fm1 + fm2;
        float3 dvec = xa2 - xa1;
        if (g_dot3(dvec, dvec) > bound * bound) pass = false;
      }}
      if (pass && (BF & 8) != 0) {{
        float3 c1 = float3(geom_aabb[g1 * 6 + 0], geom_aabb[g1 * 6 + 1], geom_aabb[g1 * 6 + 2]);
        float3 s1 = float3(geom_aabb[g1 * 6 + 3], geom_aabb[g1 * 6 + 4], geom_aabb[g1 * 6 + 5]);
        float3 c2 = float3(geom_aabb[g2 * 6 + 0], geom_aabb[g2 * 6 + 1], geom_aabb[g2 * 6 + 2]);
        float3 s2 = float3(geom_aabb[g2 * 6 + 3], geom_aabb[g2 * 6 + 4], geom_aabb[g2 * 6 + 5]);
        float3 w1 = xa1 + g_mv(&gm[g1 * 9], c1);
        float3 w2 = xa2 + g_mv(&gm[g2 * 9], c2);
        const thread float* m1p = &gm[g1 * 9];
        const thread float* m2p = &gm[g2 * 9];
        float mobs = fm1 + fm2;
        for (int jx = 0; jx < 2 && pass; ++jx) {{
          const thread float* mp = jx == 0 ? m1p : m2p;
          for (int kx = 0; kx < 3; ++kx) {{
            float3 axv = float3(mp[0 + kx], mp[3 + kx], mp[6 + kx]);
            float proj1 = g_dot3(w1, axv);
            float proj2 = g_dot3(w2, axv);
            float rad1 = metal::abs(s1.x * g_dot3(float3(m1p[0], m1p[3], m1p[6]), axv))
              + metal::abs(s1.y * g_dot3(float3(m1p[1], m1p[4], m1p[7]), axv))
              + metal::abs(s1.z * g_dot3(float3(m1p[2], m1p[5], m1p[8]), axv));
            float rad2 = metal::abs(s2.x * g_dot3(float3(m2p[0], m2p[3], m2p[6]), axv))
              + metal::abs(s2.y * g_dot3(float3(m2p[1], m2p[4], m2p[7]), axv))
              + metal::abs(s2.z * g_dot3(float3(m2p[2], m2p[5], m2p[8]), axv));
            if (rad1 + rad2 + mobs < metal::abs(proj1 - proj2)) {{
              pass = false;
              break;
            }}
          }}
        }}
      }}
    }}
    if (!pass) continue;

    int op = pair_op[p];
    float r1s = geom_size[g1 * 3 + 0];
    float r2s = geom_size[g2 * 3 + 0];
    float hl1 = geom_size[g1 * 3 + 1];
    float hl2 = geom_size[g2 * 3 + 1];
    float ds[6];
    float3 psd[6];
    float frms[6][9];
    int npts = 1;
    for (int i = 0; i < 6; ++i) ds[i] = 1e6f;

    if (op == 0) {{
      float3 nrm = float3(gm[g1 * 9 + 2], gm[g1 * 9 + 5], gm[g1 * 9 + 8]);
      float d = g_dot3(xa2 - xa1, nrm) - r2s;
      float3 cp = xa2 - nrm * (r2s + 0.5f * d);
      ds[0] = d; psd[0] = cp;
      g_make_frame(nrm, frms[0]);
    }} else if (op == 1) {{
      npts = 2;
      float3 nrm = float3(gm[g1 * 9 + 2], gm[g1 * 9 + 5], gm[g1 * 9 + 8]);
      float3 ax = float3(gm[g2 * 9 + 2], gm[g2 * 9 + 5], gm[g2 * 9 + 8]);
      float3 bvec = ax - nrm * g_dot3(nrm, ax);
      float bnorm = metal::sqrt(g_dot3(bvec, bvec));
      if (bnorm < 0.5f) {{
        if (-0.5f < nrm.y && nrm.y < 0.5f) bvec = float3(0.0f, 1.0f, 0.0f);
        else bvec = float3(0.0f, 0.0f, 1.0f);
        bvec = bvec - nrm * g_dot3(nrm, bvec);
        bvec = g_normalize3(bvec);
      }} else {{
        bvec = bvec / bnorm;
      }}
      float3 cvec = g_cross(nrm, bvec);
      for (int kk = 0; kk < 3; ++kk) {{
        frms[0][kk] = nrm[kk]; frms[0][3 + kk] = bvec[kk]; frms[0][6 + kk] = cvec[kk];
        frms[1][kk] = nrm[kk]; frms[1][3 + kk] = bvec[kk]; frms[1][6 + kk] = cvec[kk];
      }}
      float3 seg = ax * hl2;
      for (int i = 0; i < 2; ++i) {{
        float3 sp = i == 0 ? xa2 + seg : xa2 - seg;
        float d = g_dot3(sp - xa1, nrm) - r2s;
        float3 cp = sp - nrm * (r2s + 0.5f * d);
        ds[i] = d; psd[i] = cp;
      }}
    }} else if (op == 2) {{
      float3 dif = xa2 - xa1;
      float dl = metal::sqrt(g_dot3(dif, dif));
      float3 n = dl == 0.0f ? float3(1.0f, 0.0f, 0.0f) : dif / dl;
      float d = dl - (r1s + r2s);
      float3 cp = xa1 + n * (r1s + 0.5f * d);
      ds[0] = d; psd[0] = cp;
      g_make_frame(n, frms[0]);
    }} else if (op == 3) {{
      float3 ax = float3(gm[g2 * 9 + 2], gm[g2 * 9 + 5], gm[g2 * 9 + 8]);
      float3 a = xa2 - ax * hl2;
      float3 bpt = xa2 + ax * hl2;
      float3 ab = bpt - a;
      float t = metal::clamp(g_dot3(xa1 - a, ab) / (g_dot3(ab, ab) + 1e-6f), 0.0f, 1.0f);
      float3 cpt = a + t * ab;
      float3 dif = cpt - xa1;
      float dl = metal::sqrt(g_dot3(dif, dif));
      float3 n = dl == 0.0f ? float3(1.0f, 0.0f, 0.0f) : dif / dl;
      float d = dl - (r1s + r2s);
      float3 cp = xa1 + n * (r1s + 0.5f * d);
      ds[0] = d; psd[0] = cp;
      g_make_frame(n, frms[0]);
    }} else if (op == 4) {{
      npts = 2;
      float3 ax1 = float3(gm[g1 * 9 + 2], gm[g1 * 9 + 5], gm[g1 * 9 + 8]);
      float3 ax2 = float3(gm[g2 * 9 + 2], gm[g2 * 9 + 5], gm[g2 * 9 + 8]);
      float3 sv1 = ax1 * hl1;
      float3 sv2 = ax2 * hl2;
      float3 dif = xa1 - xa2;
      float ma = g_dot3(sv1, sv1);
      float mb = -g_dot3(sv1, sv2);
      float mc = g_dot3(sv2, sv2);
      float ma_safe = metal::max(ma, 1e-15f);
      float mc_safe = metal::max(mc, 1e-15f);
      float uu = -g_dot3(sv1, dif);
      float vv = g_dot3(sv2, dif);
      // Gram determinant as |sv1 x sv2|^2: equal to ma*mc - mb*mb but exact
      // for parallel axes (fma contraction of the difference is not)
      float3 crs = g_cross(sv1, sv2);
      float det = g_dot3(crs, crs);
      float mrg = margin_c;
      float cdist[2];
      float3 cpos[2];
      float3 cnor[2];
      int cnt = 0;
      cdist[0] = 1e6f; cdist[1] = 1e6f;
      if (det >= 1e-15f) {{
        float inv_det = 1.0f / det;
        float x1 = (mc * uu - mb * vv) * inv_det;
        float x2 = (ma * vv - mb * uu) * inv_det;
        if (x1 > 1.0f) {{ x1 = 1.0f; x2 = (vv - mb) / mc_safe; }}
        else if (x1 < -1.0f) {{ x1 = -1.0f; x2 = (vv + mb) / mc_safe; }}
        if (x2 > 1.0f) {{
          x2 = 1.0f;
          float xx = (uu - mb) / ma_safe;
          x1 = metal::clamp(xx, -1.0f, 1.0f);
        }} else if (x2 < -1.0f) {{
          x2 = -1.0f;
          float xx = (uu + mb) / ma_safe;
          x1 = metal::clamp(xx, -1.0f, 1.0f);
        }}
        float3 vec1 = xa1 + sv1 * x1;
        float3 vec2 = xa2 + sv2 * x2;
        float3 dfaa = vec2 - vec1;
        float dlc = metal::sqrt(g_dot3(dfaa, dfaa));
        float3 nrmv = dlc == 0.0f ? float3(1.0f, 0.0f, 0.0f) : dfaa / dlc;
        float dcl = dlc - (r1s + r2s);
        float3 cpc = vec1 + nrmv * (r1s + 0.5f * dcl);
        if (dcl <= mrg) {{
          cdist[0] = dcl; cpos[0] = cpc; cnor[0] = nrmv;
          cnt = 1;
        }}
      }} else {{
        {{
          float3 v1 = xa1 + sv1;
          float xx = (vv - mb) / mc_safe;
          float x2c = metal::clamp(xx, -1.0f, 1.0f);
          float3 v2 = xa2 + sv2 * x2c;
          float3 dfaa = v2 - v1;
          float dlk = metal::sqrt(g_dot3(dfaa, dfaa));
          float3 nrmv = dlk == 0.0f ? float3(1.0f, 0.0f, 0.0f) : dfaa / dlk;
          float dcl = dlk - (r1s + r2s);
          if (dcl <= mrg) {{
            cdist[cnt] = dcl; cpos[cnt] = v1 + nrmv * (r1s + 0.5f * dcl); cnor[cnt] = nrmv;
            cnt++;
          }}
        }}
        {{
          float3 v1 = xa1 - sv1;
          float xx = (vv + mb) / mc_safe;
          float x2c = metal::clamp(xx, -1.0f, 1.0f);
          float3 v2 = xa2 + sv2 * x2c;
          float3 dfaa = v2 - v1;
          float dlk = metal::sqrt(g_dot3(dfaa, dfaa));
          float3 nrmv = dlk == 0.0f ? float3(1.0f, 0.0f, 0.0f) : dfaa / dlk;
          float dcl = dlk - (r1s + r2s);
          if (dcl <= mrg) {{
            cdist[cnt] = dcl; cpos[cnt] = v1 + nrmv * (r1s + 0.5f * dcl); cnor[cnt] = nrmv;
            cnt++;
          }}
        }}
        if (cnt < 2) {{
          float3 v2 = xa2 + sv2;
          float xx = (uu - mb) / ma_safe;
          float x1c = metal::clamp(xx, -1.0f, 1.0f);
          float3 v1 = xa1 + sv1 * x1c;
          float3 dfaa = v2 - v1;
          float dlk = metal::sqrt(g_dot3(dfaa, dfaa));
          float3 nrmv = dlk == 0.0f ? float3(1.0f, 0.0f, 0.0f) : dfaa / dlk;
          float dcl = dlk - (r1s + r2s);
          if (dcl <= mrg) {{
            cdist[cnt] = dcl; cpos[cnt] = v1 + nrmv * (r1s + 0.5f * dcl); cnor[cnt] = nrmv;
            cnt++;
          }}
        }}
        if (cnt < 2) {{
          float3 v2 = xa2 - sv2;
          float xx = (uu + mb) / ma_safe;
          float x1c = metal::clamp(xx, -1.0f, 1.0f);
          float3 v1 = xa1 + sv1 * x1c;
          float3 dfaa = v2 - v1;
          float dlk = metal::sqrt(g_dot3(dfaa, dfaa));
          float3 nrmv = dlk == 0.0f ? float3(1.0f, 0.0f, 0.0f) : dfaa / dlk;
          float dcl = dlk - (r1s + r2s);
          if (dcl <= mrg) {{
            cdist[cnt] = dcl; cpos[cnt] = v1 + nrmv * (r1s + 0.5f * dcl); cnor[cnt] = nrmv;
            cnt++;
          }}
        }}
      }}
      for (int i = 0; i < cnt; ++i) {{
        ds[i] = cdist[i]; psd[i] = cpos[i];
        g_make_frame(cnor[i], frms[i]);
      }}
    }} else if (op == 5) {{
      // plane (g1) x convex mesh (g2): port of warp plane_convex (graph climb
      // or exhaustive), mesh vertex/graph tables indexed by per-pair offsets
      const float huge = 1e6f;
      float3 pnrm = float3(gm[g1 * 9 + 2], gm[g1 * 9 + 5], gm[g1 * 9 + 8]);
      const thread float* mr = &gm[g2 * 9];
      float3 dv = xa1 - xa2;
      float3 pl = float3(
        mr[0] * dv.x + mr[3] * dv.y + mr[6] * dv.z,
        mr[1] * dv.x + mr[4] * dv.y + mr[7] * dv.z,
        mr[2] * dv.x + mr[5] * dv.y + mr[8] * dv.z);
      float3 nn = float3(
        mr[0] * pnrm.x + mr[3] * pnrm.y + mr[6] * pnrm.z,
        mr[1] * pnrm.x + mr[4] * pnrm.y + mr[7] * pnrm.z,
        mr[2] * pnrm.x + mr[5] * pnrm.y + mr[8] * pnrm.z);
      int vadr = pair_vadr[p];
      int gadr = pair_gadr[p];
      // graph offsets stride by the hull's vertex count (mesh_graph[graphadr])
      int vnum = (pair_usegraph[p] != 0) ? mesh_graph[gadr] : pair_vertnum[p];
      // support point: deepest mesh vertex (graph climb, else exhaustive),
      // then MuJoCo's mjc_PlaneConvex manifold: the max-area quad of the face
      // incident to the deepest vertex whose normal is most anti-aligned with
      // the plane normal, anchored at the deepest vertex

      int lv = -1;
      float max_support = -huge;

      if (pair_usegraph[p] == 0) {{
        for (int i = 0; i < vnum; ++i) {{
          float sup = g_dot3(pl - G_MESH_VERT(vadr, i), nn);
          if (sup > max_support) {{ max_support = sup; lv = i; }}
        }}
      }} else {{
        int prev = -1; int imax = 0;
        while (true) {{
          prev = imax;
          int ii = mesh_graph[gadr + 2 + imax];
          while (mesh_graph[gadr + 2 + 2 * vnum + ii] >= 0) {{
            int subidx = mesh_graph[gadr + 2 + 2 * vnum + ii];
            float sup = g_dot3(pl - G_MESH_VERT(vadr, mesh_graph[gadr + 2 + vnum + subidx]), nn);
            if (sup > max_support) {{ max_support = sup; imax = subidx; }}
            ii += 1;
          }}
          if (imax == prev) break;
        }}
        lv = mesh_graph[gadr + 2 + vnum + imax];
      }}

      if (lv < 0 || max_support <= -margin_c) continue;

      float d0 = -max_support;
      g_make_frame(pnrm, frms[0]);
      ds[0] = d0;
      psd[0] = (xa2 + g_mv(mr, G_MESH_VERT(vadr, lv))) - 0.5f * d0 * pnrm;
      int nc = 1;

      if (pair_polyadr[p] >= 0) {{
        int pmapadr = mesh_polymapadr[vadr + lv];
        int pmapnum = mesh_polymapnum[vadr + lv];
        int best_poly = -1;
        float best_dot = 1.0f;

        for (int i = 0; i < pmapnum; ++i) {{
          int pidx = mesh_polymap[pmapadr + i];
          int pn_adr = (pair_polyadr[p] + pidx) * 3;
          float ndot = mesh_polynormal[pn_adr + 0] * nn.x + mesh_polynormal[pn_adr + 1] * nn.y + mesh_polynormal[pn_adr + 2] * nn.z;
          if (ndot < best_dot) {{ best_dot = ndot; best_poly = pidx; }}
        }}

        if (best_poly >= 0) {{
          int fvadr = mesh_polyvertadr[pair_polyadr[p] + best_poly];
          int fvnum = mesh_polyvertnum[pair_polyadr[p] + best_poly];
          int a = 0;

          for (int i = 0; i < fvnum; ++i) {{
            if (mesh_polyvert[fvadr + i] == lv) {{ a = i; break; }}
          }}

          int ia = a, ib = (a + 1) % fvnum, ic = (a + 2) % fvnum, idd = (a + 3) % fvnum;
          int nf = fvnum <= 4 ? fvnum : 4;

          if (fvnum > 4) {{
            float m = g_area4(G_MESH_VERT(vadr, mesh_polyvert[fvadr + ia]), G_MESH_VERT(vadr, mesh_polyvert[fvadr + ib]), G_MESH_VERT(vadr, mesh_polyvert[fvadr + ic]), G_MESH_VERT(vadr, mesh_polyvert[fvadr + idd]));
            while (true) {{
              int dn = (idd + 1) % fvnum;
              float mn = g_area4(G_MESH_VERT(vadr, mesh_polyvert[fvadr + ia]), G_MESH_VERT(vadr, mesh_polyvert[fvadr + ib]), G_MESH_VERT(vadr, mesh_polyvert[fvadr + ic]), G_MESH_VERT(vadr, mesh_polyvert[fvadr + dn]));
              if (mn <= m) break;
              idd = dn; m = mn;
              while (true) {{
                int cn = (ic + 1) % fvnum;
                mn = g_area4(G_MESH_VERT(vadr, mesh_polyvert[fvadr + ia]), G_MESH_VERT(vadr, mesh_polyvert[fvadr + ib]), G_MESH_VERT(vadr, mesh_polyvert[fvadr + cn]), G_MESH_VERT(vadr, mesh_polyvert[fvadr + idd]));
                if (mn <= m) break;
                ic = cn; m = mn;
              }}
              while (true) {{
                int bn = (ib + 1) % fvnum;
                mn = g_area4(G_MESH_VERT(vadr, mesh_polyvert[fvadr + ia]), G_MESH_VERT(vadr, mesh_polyvert[fvadr + bn]), G_MESH_VERT(vadr, mesh_polyvert[fvadr + ic]), G_MESH_VERT(vadr, mesh_polyvert[fvadr + idd]));
                if (mn <= m) break;
                ib = bn; m = mn;
              }}
            }}
          }}

          for (int i = 1; i < nf && nc < 4; ++i) {{
            int fj = (i == 1) ? ib : ((i == 2) ? ic : idd);
            float3 wp = xa2 + g_mv(mr, G_MESH_VERT(vadr, mesh_polyvert[fvadr + fj]));
            float vdist = g_dot3(pnrm, wp - xa1);
            if (vdist > margin_c) continue;
            if (g_dot3(pnrm, wp - xa2) > 0.0f) continue;
            g_make_frame(pnrm, frms[nc]);
            ds[nc] = vdist;
            psd[nc] = wp - 0.5f * vdist * pnrm;
            nc += 1;
          }}
        }}
      }}

      npts = nc;
    }}
{sdf_branch}

    float fric[5];
    for (int k = 0; k < 5; ++k) fric[k] = pair_friction[p * 5 + k];
    float invw = pair_invw[p];
    int nd = pair_ndim[p];
    int b1 = pair_body1[p];
    int b2 = pair_body2[p];
    int rt1 = body_rootid[b1];
    int rt2 = body_rootid[b2];
    float3 com1 = float3(subcom[rt1 * 3 + 0], subcom[rt1 * 3 + 1], subcom[rt1 * 3 + 2]);
    float3 com2 = float3(subcom[rt2 * 3 + 0], subcom[rt2 * 3 + 1], subcom[rt2 * 3 + 2]);

    int pct = 0;
    for (int i = 0; i < npts; ++i) {{
      float pos_c = ds[i] - margin_c;
      if (pos_c >= 0.0f) continue;
      if (rowidx + nd > {RC}) {{ ovr = 1; break; }}
      for (int dim = 0; dim < nd; ++dim) {{
{pt_meta}
      }}
      thread float kb[3];
      g_kbimp(&pair_solref[p * 2], &pair_solimp[p * 5], pos_c, timestep, kb);
      float Dv = 1.0f / metal::max(invw * (1.0f - kb[2]) / kb[2], 1e-15f);
      float3 cpt = psd[i];
      float3 off1 = cpt - com1;
      float3 off2 = cpt - com2;
      const thread float* frmi = frms[i];
      float qv[10];
      for (int dim = 0; dim < nd; ++dim) qv[dim] = 0.0f;
{pt_open}      for (int dof = 0; dof < {NV}; ++dof) {{
        int a1 = dof_anc[b1 * {NV} + dof];
        int a2 = dof_anc[b2 * {NV} + dof];
        if ((a1 | a2) == 0) continue;
        const thread float* cdof6 = &cdof[dof * 6];
        float3 ang = float3(cdof6[0], cdof6[1], cdof6[2]);
        float3 lin = float3(cdof6[3], cdof6[4], cdof6[5]);
        float3 jp1 = a1 != 0 ? lin + g_cross(ang, off1) : float3(0.0f);
        float3 jp2 = a2 != 0 ? lin + g_cross(ang, off2) : float3(0.0f);
        float3 jpd = jp2 - jp1;
        float3 jrd = (a2 != 0 ? ang : float3(0.0f)) - (a1 != 0 ? ang : float3(0.0f));
        float base0 = g_dot3(jpd, float3(frmi[0], frmi[1], frmi[2]));
        float e1 = (nd > 1) ? g_dot3(jpd, float3(frmi[3], frmi[4], frmi[5])) : 0.0f;
        float e2 = (nd > 2) ? g_dot3(jpd, float3(frmi[6], frmi[7], frmi[8])) : 0.0f;
        float e3 = (nd > 4) ? g_dot3(jrd, float3(frmi[0], frmi[1], frmi[2])) : 0.0f;
        float e4 = (nd > 6) ? g_dot3(jrd, float3(frmi[3], frmi[4], frmi[5])) : 0.0f;
        float e5 = (nd > 8) ? g_dot3(jrd, float3(frmi[6], frmi[7], frmi[8])) : 0.0f;
        for (int dim = 0; dim < nd; ++dim) {{
          int dimd2 = dim / 2 + 1;
          float fri = fric[dimd2 - 1];
          float sgnd = fri * (1.0f - 2.0f * (float)(dim & 1));
          float extra = 0.0f;
          if (dimd2 == 1) extra = e1;
          else if (dimd2 == 2) extra = e2;
          else if (dimd2 == 3) extra = e3;
          else if (dimd2 == 4) extra = e4;
          else extra = e5;
          float val = base0 + (nd > 1 ? sgnd * extra : 0.0f);
{pt_val}
          qv[dim] += val * v[dof];
        }}
{pt_close}      }}
      for (int dim = 0; dim < nd; ++dim) {{
        Dw[rowidx + dim] = Dv;
        arefw[rowidx + dim] = -kb[0] * kb[2] * pos_c - kb[1] * qv[dim];
      }}
      rowidx += nd;
      conct += 1;
      pct += 1;
    }}
{kin_pconct_write}
  }}

  nefc_out[w] = rowidx;
  ncon_out[w] = conct;
  overflow_out[w] = ovr;
"""
  for name, off in sorted((adr_s or {}).items(), key=lambda x: -len(x[0])):
    body = body.replace(f"{name}[", f"mbs[{off} + ")
  for name, off in sorted(adr_f.items(), key=lambda x: -len(x[0])):
    body = body.replace(f"{name}[", f"mbf[w * {MBS} + {off} + ")
  for name, off in sorted(adr_i.items(), key=lambda x: -len(x[0])):
    body = body.replace(f"{name}[", f"mif[{off} + ")
  return body, _MSL_HELPERS + _MSL_SDF


# --------------------------------------------------------------------- K2 solve+integrate
def solve_source(t: dict, adr_i: dict[str, int] | None = None, packed_m: bool = False) -> tuple[str, str]:
  """K2 source. Templates: NJNT NQ NV RC MS ITER LSITER WARMSTART EULERDAMP NCH RCH.

  `packed_m=True` consumes the ancestor-chain mass matrix (`NV * NCH` floats,
  `dof_pad`/`rev_dof` index tables) and applies it by walking the chain plus its
  transpose instead of a dense `NV * NV` block.
  """
  NJ, NQ, NV = t["NJNT"], t["NQ"], t["NV"]
  RC, ITER, LSITER = t["RC"], t["ITER"], t["LSITER"]
  MS = t["MS"]
  WS = t["WARMSTART"]
  ED = t["EULERDAMP"]
  NCH, RCH = t["NCH"], t["RCH"]

  # packed-M fragments (the dense path keeps the original inline code verbatim)
  m_load = (
    f"""
  // packed mass matrix: row i holds its ancestor chain (dof_pad), self first
  thread float M[{NV} * {NCH}];
  for (int i = 0; i < {NV} * {NCH}; ++i) M[i] = M_in[w * {NV} * {NCH} + i];
"""
    if packed_m
    else f"""
  thread float M[{NV} * {NV}];
  for (int i = 0; i < {NV} * {NV}; ++i) M[i] = M_in[w * {NV} * {NV} + i];
"""
  )

  def _matvec(vec: str, out: str, ind: str) -> str:
    """`out = M * vec` in the packed layout: row chain, then its transpose."""
    return (
      f"{ind}for (int i = 0; i < {NV}; ++i) {{\n"
      + f"{ind}  float ss = 0.0f;\n"
      + f"{ind}  for (int c = 0; c < {NCH}; ++c) {{ int j = dof_pad[i * {NCH} + c]; if (j < 0) break; ss += M[i * {NCH} + c] * {vec}[j]; }}\n"
      + f"{ind}  for (int t = 0; t < {RCH}; ++t) {{ int e = rev_dof[i * {RCH} + t]; if (e < 0) break; ss += M[(e >> 8) * {NCH} + (e & 255)] * {vec}[e >> 8]; }}\n"
      + f"{ind}  {out}[i] = ss;\n"
      + f"{ind}}}\n"
    )

  def _h_init_flat(ind: str) -> str:
    """`h = M` packed, statement-per-line form (used outside the REBUILD macro)."""
    return (
      f"{ind}for (int i = 0; i < {NV}; ++i) for (int j = 0; j <= i; ++j) h[i * {NV} + j] = 0.0f;\n"
      + f"{ind}for (int i = 0; i < {NV}; ++i) {{\n"
      + f"{ind}  for (int c = 0; c < {NCH}; ++c) {{ int j = dof_pad[i * {NCH} + c]; if (j < 0) break; h[i * {NV} + j] = M[i * {NCH} + c]; }}\n"
      + f"{ind}}}\n"
    )

  body = f"""
  int w = (int)thread_position_in_grid.x;
  if (w >= nworld_buf[0]) return;

  const float dt = opt_buf[0];
  const float tol = opt_buf[1];
  const float meaninertia = opt_buf[2];
  const float ls_tol = opt_buf[3];
{m_load}
  thread float smooth[{NV}];
  for (int i = 0; i < {NV}; ++i) smooth[i] = smooth_in[w * {NV} + i];
  int nefc = nefc_in[w];
  if (nefc > {RC}) nefc = {RC};

  thread float acc[{NV}];
  thread float h[{NV} * {NV}];
  int nit = 0;

  if (nefc == 0) {{
    for (int i = 0; i < {NV} * {NV}; ++i) h[i] = M[i];
    g_chol_inplace(h, {NV});
    g_chol_solve_fact(h, smooth, acc, {NV});
    for (int d = 0; d < {NV}; ++d) {{
      warm_out[w * {NV} + d] = acc[d];
      qacc_out[w * {NV} + d] = acc[d];
    }}
    if ({ED} != 0) {{
      for (int i = 0; i < {NV} * {NV}; ++i) h[i] = M[i];
      for (int i = 0; i < {NV}; ++i) h[i * {NV} + i] += dt * dof_damping[i];
      g_chol_inplace(h, {NV});
      g_chol_solve_fact(h, smooth, acc, {NV});
    }}
  }} else {{
    if ({WS} != 0) {{
      for (int i = 0; i < {NV}; ++i) acc[i] = warm_in[w * {NV} + i];
    }} else {{
      for (int i = 0; i < {NV} * {NV}; ++i) h[i] = M[i];
      g_chol_inplace(h, {NV});
      g_chol_solve_fact(h, smooth, acc, {NV});
    }}

    // hoist per-row state (constant across the Newton iterations)
    thread float Dd[{RC}];
    thread float fld[{RC}];
    thread float Jaref[{RC}];
    for (int r = 0; r < nefc; ++r) {{
      Dd[r] = D_in[w * {RC} + r];
      fld[r] = fl_in[w * {RC} + r];
      int meta = jmeta_in[w * {RC} + r];
      int num = sup_num[meta];
      int base = meta * {MS};
      float ss = 0.0f;
      // direct index: small inputs may be in the constant address space
      for (int k = 0; k < num; ++k) ss += J_in[((long)w * {RC} + r) * {MS} + k] * acc[sup_dofs[base + k]];
      Jaref[r] = ss - aref_in[w * {RC} + r];
    }}
    thread float Ma[{NV}];
    for (int i = 0; i < {NV}; ++i) {{
      float ss = 0.0f;
      for (int j = 0; j < {NV}; ++j) ss += M[i * {NV} + j] * acc[j];
      Ma[i] = ss;
    }}

    // row state (force + quadratic-active flag)
    thread float force[{RC}];
    thread float squad[{RC}];
    for (int r = 0; r < nefc; ++r) {{
      float c2, g2, h2;
      g_row_cost(Jaref[r], Dd[r], fld[r], &c2, &g2, &h2);
      force[r] = -g2;
      squad[r] = (h2 > 0.0f) ? 1.0f : 0.0f;
    }}

    thread float qc[{NV}];
    thread float grad[{NV}];
    for (int i = 0; i < {NV}; ++i) qc[i] = 0.0f;

    // lower-triangle Hessian rebuild (h is never read above the diagonal); deferred
    // past the convergence check so a converged last Newton iteration skips it;
    // dofs ascend within each support, so di >= dj keeps the lower triangle
#define REBUILD_LH() \\
    {{ \\
      for (int i = 0; i < {NV}; ++i) \\
        for (int j = 0; j <= i; ++j) h[i * {NV} + j] = M[i * {NV} + j]; \\
      for (int r = 0; r < nefc; ++r) {{ \\
        float dqr = squad[r] * Dd[r]; \\
        if (dqr == 0.0f) continue; \\
        int meta = jmeta_in[w * {RC} + r]; \\
        int num = sup_num[meta]; \\
        int base = meta * {MS}; \\
        const long jb = ((long)w * {RC} + r) * {MS}; \\
        for (int i = 0; i < num; ++i) {{ \\
          float dq2 = dqr * J_in[jb + i]; \\
          if (dq2 == 0.0f) continue; \\
          int di = sup_dofs[base + i]; \\
          for (int j = 0; j <= i; ++j) h[di * {NV} + sup_dofs[base + j]] += dq2 * J_in[jb + j]; \\
        }} \\
      }} \\
    }}
    REBUILD_LH();
    g_chol_inplace(h, {NV});
    for (int r = 0; r < nefc; ++r) {{
      int meta = jmeta_in[w * {RC} + r];
      float fr = force[r];
      for (int k = 0; k < sup_num[meta]; ++k) {{
        int d = sup_dofs[meta * {MS} + k];
        qc[d] += fr * J_in[((long)w * {RC} + r) * {MS} + k];
      }}
    }}
    for (int i = 0; i < {NV}; ++i) grad[i] = Ma[i] - smooth[i] - qc[i];

    int done = 0;
    thread float search[{NV}];
    thread float mv[{NV}];
    thread float jv[{RC}];
    for (int it = 0; it < {ITER}; ++it) {{
      nit = it + 1;
      thread float neggrad[{NV}];
      for (int i = 0; i < {NV}; ++i) neggrad[i] = -grad[i];
      g_chol_solve_fact(h, neggrad, search, {NV});

      for (int i = 0; i < {NV}; ++i) {{
        float sss = 0.0f;
        for (int j = 0; j < {NV}; ++j) sss += M[i * {NV} + j] * search[j];
        mv[i] = sss;
      }}
      float newton_dec = 0.0f;
      for (int i = 0; i < {NV}; ++i) newton_dec -= grad[i] * search[i];
      for (int r = 0; r < nefc; ++r) {{
        int meta = jmeta_in[w * {RC} + r];
        int num = sup_num[meta];
        int base = meta * {MS};
        float ssr = 0.0f;
        for (int k = 0; k < num; ++k) ssr += J_in[((long)w * {RC} + r) * {MS} + k] * search[sup_dofs[base + k]];
        jv[r] = ssr;
      }}

      float g_sm0 = 0.0f;
      for (int i = 0; i < {NV}; ++i) g_sm0 += (Ma[i] - smooth[i]) * search[i];
      float hs_pre = 0.0f;
      for (int i = 0; i < {NV}; ++i) hs_pre += search[i] * mv[i];

#define PHI(alpha, pc, pg, phs) \\
      {{ \\
        float arv = alpha; \\
        float cc = arv * g_sm0 + 0.5f * arv * arv * hs_pre; \\
        float gg = g_sm0 + arv * hs_pre; \\
        float hh = hs_pre; \\
        for (int r = 0; r < nefc; ++r) {{ \\
          float c2r, g2r, h2r; \\
          g_row_cost_shifted(Jaref[r], arv * jv[r], Dd[r], fld[r], &c2r, &g2r, &h2r); \\
          cc += c2r; \\
          gg += g2r * jv[r]; \\
          hh += h2r * jv[r] * jv[r]; \\
        }} \\
        *pc = cc; *pg = gg; *phs = hh; \\
      }}

#define PHI3(a0, a1, a2, c0p, g0p, h0p, c1p, g1p, h1p, c2p, g2p, h2p) \\
      {{ \\
        float av0 = a0, av1 = a1, av2 = a2; \\
        float cc0 = av0 * g_sm0 + 0.5f * av0 * av0 * hs_pre; \\
        float gg0 = g_sm0 + av0 * hs_pre; \\
        float hh0 = hs_pre; \\
        float cc1 = av1 * g_sm0 + 0.5f * av1 * av1 * hs_pre; \\
        float gg1 = g_sm0 + av1 * hs_pre; \\
        float hh1 = hs_pre; \\
        float cc2 = av2 * g_sm0 + 0.5f * av2 * av2 * hs_pre; \\
        float gg2 = g_sm0 + av2 * hs_pre; \\
        float hh2 = hs_pre; \\
        for (int r = 0; r < nefc; ++r) {{ \\
          float jar = Jaref[r], jvr = jv[r], ddr = Dd[r], flr = fld[r]; \\
          float jvr2 = jvr * jvr; \\
          float c2r, g2r, h2r; \\
          g_row_cost_shifted(jar, av0 * jvr, ddr, flr, &c2r, &g2r, &h2r); \\
          cc0 += c2r; gg0 += g2r * jvr; hh0 += h2r * jvr2; \\
          g_row_cost_shifted(jar, av1 * jvr, ddr, flr, &c2r, &g2r, &h2r); \\
          cc1 += c2r; gg1 += g2r * jvr; hh1 += h2r * jvr2; \\
          g_row_cost_shifted(jar, av2 * jvr, ddr, flr, &c2r, &g2r, &h2r); \\
          cc2 += c2r; gg2 += g2r * jvr; hh2 += h2r * jvr2; \\
        }} \\
        *c0p = cc0; *g0p = gg0; *h0p = hh0; \\
        *c1p = cc1; *g1p = gg1; *h1p = hh1; \\
        *c2p = cc2; *g2p = gg2; *h2p = hh2; \\
      }}

      const float c0 = 0.0f;
      float g0 = g_sm0;
      float hs0 = hs_pre;
      for (int r = 0; r < nefc; ++r) {{
        float jvr = jv[r];
        g0 -= force[r] * jvr;
        hs0 += squad[r] * Dd[r] * jvr * jvr;
      }}

      // warp-iterative line search (costs compared against c0 = shifted cost 0)
      float snorm = 0.0f;
      for (int i = 0; i < {NV}; ++i) snorm += search[i] * search[i];
      snorm = metal::sqrt(snorm);
      float gtol = metal::max(tol * ls_tol * snorm * (meaninertia * (float){NV}), 1e-6f);

      float lo_alpha_in = -g0 / metal::max(hs0, 1e-30f);
      float ic, ig, ih;
      PHI(lo_alpha_in, &ic, &ig, &ih);

      float al = 0.0f;
      float improvement = 0.0f;
      if (metal::abs(ig) < gtol && ic < c0) {{
        al = lo_alpha_in;
        improvement = c0 - ic;
      }} else {{
        int lo_less = ig < g0;
        float loc = lo_less ? ic : c0;
        float log_ = lo_less ? ig : g0;
        float loh = lo_less ? ih : hs0;
        float loa = lo_less ? lo_alpha_in : 0.0f;
        float hic = lo_less ? c0 : ic;
        float hig = lo_less ? g0 : ig;
        float hih = lo_less ? hs0 : ih;
        float hia = lo_less ? 0.0f : lo_alpha_in;
        for (int lsi = 0; lsi < {LSITER}; ++lsi) {{
          float lna = loa - log_ / metal::max(loh, 1e-30f);
          float hna = hia - hig / metal::max(hih, 1e-30f);
          float mna = 0.5f * (loa + hia);
          float lnc, lng, lnh, hnc, hng, hnh, mnc, mng, mnh;
          PHI3(lna, hna, mna, &lnc, &lng, &lnh, &hnc, &hng, &hnh, &mnc, &mng, &mnh);
          int conv_lo = (metal::abs(lng) < gtol) & (lnc < c0);
          int conv_hi = (metal::abs(hng) < gtol) & (hnc < c0);
          int conv_mid = (metal::abs(mng) < gtol) & (mnc < c0);
          int converged = conv_lo | conv_hi | conv_mid;
          int swap_lo = 0;
          int swap_hi = 0;
          if (converged != 0) {{
            float bcp = 1e30f; float bcd = 0.0f; float bch = 0.0f; float bca = 0.0f;
            if (conv_lo && lnc < bcp) {{ bcp = lnc; bcd = lng; bch = lnh; bca = lna; }}
            if (conv_hi && hnc < bcp) {{ bcp = hnc; bcd = hng; bch = hnh; bca = hna; }}
            if (conv_mid && mnc < bcp) {{ bcp = mnc; bcd = mng; bch = mnh; bca = mna; }}
            loc = bcp; log_ = bcd; loh = bch; loa = bca;
            hic = bcp; hig = bcd; hih = bch; hia = bca;
          }} else {{
            int s1 = ((log_ < lng && lng < 0.0f) || (log_ > lng && lng > 0.0f)) ? 1 : 0;
            if (s1) {{ loc = lnc; log_ = lng; loh = lnh; loa = lna; }}
            int s2 = ((log_ < mng && mng < 0.0f) || (log_ > mng && mng > 0.0f)) ? 1 : 0;
            if (s2) {{ loc = mnc; log_ = mng; loh = mnh; loa = mna; }}
            int s3 = ((log_ < hng && hng < 0.0f) || (log_ > hng && hng > 0.0f)) ? 1 : 0;
            if (s3) {{ loc = hnc; log_ = hng; loh = hnh; loa = hna; }}
            swap_lo = s1 | s2 | s3;
            int t1 = (((hig < hng && hng < 0.0f) || (hig > hng && hng > 0.0f)) || (hig < 0.0f && hng > 0.0f)) ? 1 : 0;
            if (t1) {{ hic = hnc; hig = hng; hih = hnh; hia = hna; }}
            int t2 = ((hig < mng && mng < 0.0f) || (hig > mng && mng > 0.0f)) ? 1 : 0;
            if (t2) {{ hic = mnc; hig = mng; hih = mnh; hia = mna; }}
            int t3 = ((hig < lng && lng < 0.0f) || (hig > lng && lng > 0.0f)) ? 1 : 0;
            if (t3) {{ hic = lnc; hig = lng; hih = lnh; hia = lna; }}
            swap_hi = t1 | t2 | t3;
          }}
          int ls_done = (converged != 0)
            | ((swap_lo | swap_hi) == 0 ? 1 : 0)
            | ((loc < c0 && log_ < 0.0f && log_ > -gtol) ? 1 : 0)
            | ((hic < c0 && hig > 0.0f && hig < gtol) ? 1 : 0);
          int improved = (loc < c0) | (hic < c0);
          int lo_better = loc < hic;
          float ba = lo_better ? loa : hia;
          float bdd = lo_better ? loc : hic;
          if (improved != 0) {{
            al = ba;
            improvement = c0 - bdd;
          }}
          if (ls_done != 0) break;
        }}
      }}
      if (done) al = 0.0f;

      for (int i = 0; i < {NV}; ++i) acc[i] += al * search[i];
      for (int i = 0; i < {NV}; ++i) Ma[i] += al * mv[i];
      for (int r = 0; r < nefc; ++r) Jaref[r] += al * jv[r];

      int squad_changed = 0;
      for (int r = 0; r < nefc; ++r) {{
        float c2, g2, h2;
        g_row_cost(Jaref[r], Dd[r], fld[r], &c2, &g2, &h2);
        force[r] = -g2;
        float new_sq = (h2 > 0.0f) ? 1.0f : 0.0f;
        if (new_sq != squad[r]) squad_changed = 1;
        squad[r] = new_sq;
      }}
      for (int i = 0; i < {NV}; ++i) qc[i] = 0.0f;
      for (int r = 0; r < nefc; ++r) {{
        int meta = jmeta_in[w * {RC} + r];
        float fr = force[r];
        for (int k = 0; k < sup_num[meta]; ++k) {{
          int d = sup_dofs[meta * {MS} + k];
          qc[d] += fr * J_in[((long)w * {RC} + r) * {MS} + k];
        }}
      }}
      float grad_dot = 0.0f;
      for (int i = 0; i < {NV}; ++i) {{
        grad[i] = Ma[i] - smooth[i] - qc[i];
        grad_dot += grad[i] * grad[i];
      }}
      float rescale = meaninertia * (float){NV};
      int newly_done = (al == 0.0f)
        | ((improvement / rescale < tol) ? 1 : 0)
        | ((metal::sqrt(grad_dot) / rescale < tol) ? 1 : 0)
        | ((0.5f * newton_dec / rescale < tol) ? 1 : 0);
      done = done | newly_done;
      if (done != 0) break;
      if (squad_changed != 0) {{
        REBUILD_LH();
        g_chol_inplace(h, {NV});
      }}
    }}

    for (int d = 0; d < {NV}; ++d) {{
      warm_out[w * {NV} + d] = acc[d];
      qacc_out[w * {NV} + d] = acc[d];
    }}

    if ({ED} != 0) {{
      float Ma2[{NV}];
      for (int i = 0; i < {NV}; ++i) {{
        float sss = 0.0f;
        for (int j = 0; j < {NV}; ++j) sss += M[i * {NV} + j] * acc[j];
        Ma2[i] = sss;
      }}
      for (int i = 0; i < {NV} * {NV}; ++i) h[i] = M[i];
      for (int i = 0; i < {NV}; ++i) h[i * {NV} + i] += dt * dof_damping[i];
      g_chol_inplace(h, {NV});
      for (int i = 0; i < {NV}; ++i) acc[i] = 0.0f;
      g_chol_solve_fact(h, Ma2, acc, {NV});
    }}
  }}

  thread float qpos[{NQ}];
  for (int k = 0; k < {NQ}; ++k) qpos[k] = qpos_in[w * {NQ} + k];
  thread float qvel[{NV}];
  for (int d = 0; d < {NV}; ++d) qvel[d] = qvel_in[w * {NV} + d] + dt * acc[d];

  for (int j = 0; j < {NJ}; ++j) {{  // advance positions (jtype: 0 free, 2 slide, 3 hinge)
    int jt = jnt_type[j];
    int qa = jnt_qposadr[j];
    int dof = jnt_dofadr[j];
    if (jt == 0) {{
      qpos[qa + 0] += dt * qvel[dof + 0];
      qpos[qa + 1] += dt * qvel[dof + 1];
      qpos[qa + 2] += dt * qvel[dof + 2];
      float4 quat = g_normalize4(float4(qpos[qa + 3], qpos[qa + 4], qpos[qa + 5], qpos[qa + 6]));
      float ax3 = dt * qvel[dof + 3];
      float ay3 = dt * qvel[dof + 4];
      float az3 = dt * qvel[dof + 5];
      float nrm = metal::sqrt(ax3 * ax3 + ay3 * ay3 + az3 * az3);
      float3 axis = nrm > 0.0f ? float3(ax3, ay3, az3) / nrm : float3(1.0f, 0.0f, 0.0f);
      float halfang = 0.5f * nrm;
      float4 qr = float4(metal::cos(halfang), axis * metal::sin(halfang));
      float4 resv = g_qmul(quat, qr);
      resv = g_normalize4(resv);
      qpos[qa + 3] = resv.x; qpos[qa + 4] = resv.y; qpos[qa + 5] = resv.z; qpos[qa + 6] = resv.w;
    }} else if (jt == 2 || jt == 3) {{
      qpos[qa] += dt * qvel[dof];
    }}
  }}

  for (int k = 0; k < {NQ}; ++k) qpos_out[w * {NQ} + k] = qpos[k];
  for (int d = 0; d < {NV}; ++d) {{
    qvel_out[w * {NV} + d] = qvel[d];
  }}
  niter_out[w] = nit;
"""
  if packed_m:
    body = _pack_m_body(body, NV, NCH, RCH, _matvec, _h_init_flat)
  for name, off in sorted((adr_i or {}).items(), key=lambda x: -len(x[0])):
    body = body.replace(f"{name}[", f"mif[{off} + ")
  return body, _MSL_HELPERS


def _pack_m_body(body: str, nv: int, nch: int, rch: int, matvec, h_init_flat) -> str:
  """Rewrites the solo solve's dense-M idioms into chain-packed form.

  The dense kernel reads `M[i * nv + j]` for the full square; the packed kernel
  stores `M[i * nch + c]` (chain `dof_pad`) and walks the transpose (`rev_dof`)
  for the descendant half of each matvec.  Arithmetic order per output element
  changes only in the matvecs (documented; parity validated numerically).
  """
  subs = []

  # h = M (whole-matrix copy sites: nefc == 0 branch, warmstart-else, euler damping)
  # anchored on the leading newline so the 4-space pattern cannot match inside the
  # 6-space lines; every occurrence must be rewritten (each has its own h init)
  subs.append(
    (
      f"\n    for (int i = 0; i < {nv} * {nv}; ++i) h[i] = M[i];",
      "\n" + h_init_flat("    ").rstrip("\n"),
    )
  )
  subs.append(
    (
      f"\n      for (int i = 0; i < {nv} * {nv}; ++i) h[i] = M[i];",
      "\n" + h_init_flat("      ").rstrip("\n"),
    )
  )
  # REBUILD_LH init (macro continuation lines end with a backslash)
  subs.append(
    (
      f"      for (int i = 0; i < {nv}; ++i) \\\n        for (int j = 0; j <= i; ++j) h[i * {nv} + j] = M[i * {nv} + j]; \\\n",
      f"      for (int i = 0; i < {nv}; ++i) \\\n"
      f"        for (int j = 0; j <= i; ++j) h[i * {nv} + j] = 0.0f; \\\n"
      f"      for (int i = 0; i < {nv}; ++i) {{ \\\n"
      f"        for (int c = 0; c < {nch}; ++c) {{ int j = dof_pad[i * {nch} + c]; if (j < 0) break; h[i * {nv} + j] = M[i * {nch} + c]; }} \\\n"
      f"      }} \\\n",
    )
  )
  # matvecs: Ma = M acc, mv = M search, Ma2 = M acc (euler damping)
  subs.append(
    (
      f"    for (int i = 0; i < {nv}; ++i) {{\n"
      f"      float ss = 0.0f;\n"
      f"      for (int j = 0; j < {nv}; ++j) ss += M[i * {nv} + j] * acc[j];\n"
      f"      Ma[i] = ss;\n"
      f"    }}\n",
      matvec("acc", "Ma", "    "),
    )
  )
  subs.append(
    (
      f"      for (int i = 0; i < {nv}; ++i) {{\n"
      f"        float sss = 0.0f;\n"
      f"        for (int j = 0; j < {nv}; ++j) sss += M[i * {nv} + j] * search[j];\n"
      f"        mv[i] = sss;\n"
      f"      }}\n",
      matvec("search", "mv", "      "),
    )
  )
  subs.append(
    (
      f"      float Ma2[{nv}];\n"
      f"      for (int i = 0; i < {nv}; ++i) {{\n"
      f"        float sss = 0.0f;\n"
      f"        for (int j = 0; j < {nv}; ++j) sss += M[i * {nv} + j] * acc[j];\n"
      f"        Ma2[i] = sss;\n"
      f"      }}\n",
      f"      float Ma2[{nv}];\n" + matvec("acc", "Ma2", "      "),
    )
  )
  for old, new in subs:
    assert old in body, f"packed-M rewrite missed a fragment:\n{old}"
    body = body.replace(old, new)
  # every dense-M read must be gone (the chain form uses M[i * nch + c])
  assert f"M[i * {nv} + j]" not in body, "packed-M rewrite left a dense M read"
  assert "h[i] = M[i]" not in body, "packed-M rewrite left a dense M copy"
  return body


# --------------------------------------------------------------------- K2 cooperative
def coop_v2_body(src: str, nv: int, tg: int, rc: int, ms: int, off: dict[str, int]) -> str:
  """Rewrites the cooperative solve into the balanced (coop-v2) form.

  * factor: the v1 factorization is row-owned - for column k, lane i updates its
    own row's trailing entries, so the warp issues `max_i(i - k)` divergent
    iterations and the last row (i = nv - 1) dominates the critical path.  v2
    walks the trailing triangle as a flat pair list (`tri_pairs`), so every lane
    handles the same ceil(count / TG) iterations with all lanes active; each
    element still receives its column-k update once, in ascending k order, so
    the arithmetic is unchanged.
  * rebuild: v1 loops the full `nv`-wide rows (each lane owns one matrix row and
    scans all `nv` columns, including the zeros outside the row's support); v2
    walks the support pairs (`tri_full` over `sup_dofs`) with all lanes sharing
    the work, reading the row's dense values at the support dofs only.

  `off` carries the packed int-table offsets (tri_pairs, tri_full, sup_dofs,
  sup_num); the caller passes them because the rendered source has already
  substituted table names to `mif` offsets.
  """
  assert tg == 32, "coop-v2 assumes a 32-lane threadgroup"
  pat = re.compile(r"(#define PAR_CHOL\(A\) \\\n  \{ \\\n    if \(" + str(tg) + r" == 32\) \{ \\\n).*?(\n    \} else \{ \\)", re.S)
  m = pat.search(src)
  assert m, "coop-v2: PAR_CHOL 32-lane branch not found"
  tri_off = off["tri_pairs"]
  body = (
    f"      for (int k = 0; k < {nv}; ++k) {{ \\\n"
    f"        float dk_loc = (lane == (k & 31)) ? metal::sqrt(metal::max((A)[k * {nv} + k], 1e-30f)) : 0.0f; \\\n"
    f"        float dk = metal::simd_broadcast(dk_loc, (ushort)(k & 31)); \\\n"
    f"        if (lane == (k & 31)) (A)[k * {nv} + k] = dk; \\\n"
    f"        for (int i = k + 1 + lane; i < {nv}; i += 32) (A)[i * {nv} + k] /= dk; \\\n"
    f"        threadgroup_barrier(mem_flags::mem_threadgroup); \\\n"
    f"        const int _m = {nv} - k; \\\n"
    f"        const int _npair = (_m - 1) * _m / 2; \\\n"
    f"        for (int t = lane; t < _npair; t += 32) {{ \\\n"
    f"          int _ii = mif[{tri_off} + 2 * t]; \\\n"
    f"          int _jj = mif[{tri_off} + 2 * t + 1]; \\\n"
    f"          (A)[(k + _ii) * {nv} + (k + _jj)] -= (A)[(k + _ii) * {nv} + k] * (A)[(k + _jj) * {nv} + k]; \\\n"
    f"        }} \\\n"
    f"      }} \\\n"
    f"      threadgroup_barrier(mem_flags::mem_threadgroup); \\\n"
  )
  out = src[: m.end(1)] + body + src[m.start(2) + 1 :]
  # a blank line inside a backslash-continued macro terminates it early
  start = out.index("#define PAR_CHOL")
  end = out.index("    } else { \\", start)
  assert "\n\n" not in out[start:end], "coop-v2: blank line inside the PAR_CHOL macro"

  # ---- support-pair Hessian rebuild
  pat_re = re.compile(r"(#define REBUILD_LH\(\) \\\n).*?(\n    \}\n)", re.S)
  m = pat_re.search(out)
  assert m, "coop-v2: REBUILD_LH macro not found"
  tf, sd, sn = off["tri_full"], off["sup_dofs"], off["sup_num"]
  rebuild = (
    f"    {{ \\\n"
    f"      for (int i = lane; i < {nv}; i += {tg}) \\\n"
    f"        for (int j = 0; j <= i; ++j) sh_h[i * {nv} + j] = sh_M[i * {nv} + j]; \\\n"
    f"      for (int r = 0; r < nefc; ++r) {{ \\\n"
    f"        float dqr = sh_squad[r] * sh_Dd[r]; \\\n"
    f"        if (dqr == 0.0f) continue; \\\n"
    f"        int _meta = jmeta_in[w * {rc} + r]; \\\n"
    f"        int _num = mif[{sn} + _meta]; \\\n"
    f"        int _base = _meta * {ms}; \\\n"
    f"        const long _jb = ((long)w * {rc} + r) * {nv}; \\\n"
    f"        const int _np = _num * (_num + 1) / 2; \\\n"
    f"        for (int t = lane; t < _np; t += {tg}) {{ \\\n"
    f"          int _da = mif[{sd} + _base + mif[{tf} + 2 * t]]; \\\n"
    f"          int _db = mif[{sd} + _base + mif[{tf} + 2 * t + 1]]; \\\n"
    f"          sh_h[_da * {nv} + _db] += dqr * J_in[_jb + _da] * J_in[_jb + _db]; \\\n"
    f"        }} \\\n"
    f"      }} \\\n"
    f"      threadgroup_barrier(mem_flags::mem_threadgroup); \\\n"
    f"    }}\n"
  )
  out = out[: m.end(1)] + rebuild + out[m.end(2) :]
  return out


def solve_source_coop(t: dict, adr_i: dict[str, int] | None = None, tg: int = 32) -> tuple[str, str]:
  """K2 source, cooperative: one threadgroup (TG threads) per world.

  Templates: NJNT NQ NV RC MS ITER LSITER WARMSTART EULERDAMP TG.
  Matrices and vectors live in threadgroup memory; row loops are lane-strided.
  """
  subs = {
    "@NV@": str(t["NV"]),
    "@NQ@": str(t["NQ"]),
    "@NJ@": str(t["NJNT"]),
    "@RC@": str(t["RC"]),
    "@MS@": str(t["MS"]),
    "@ITER@": str(t["ITER"]),
    "@LSITER@": str(t["LSITER"]),
    "@WS@": str(t["WARMSTART"]),
    "@ED@": str(t["EULERDAMP"]),
    "@TG@": str(tg),
  }
  body = r"""
  int lane = (int)thread_position_in_threadgroup.x;
  int w = (int)threadgroup_position_in_grid.x;
  if (w >= nworld_buf[0]) return;

  const float dt = opt_buf[0];
  const float tol = opt_buf[1];
  const float meaninertia = opt_buf[2];
  const float ls_tol = opt_buf[3];

  threadgroup float sh_M[@NV@ * @NV@];
  threadgroup float sh_h[@NV@ * @NV@];
  threadgroup float sh_smooth[@NV@], sh_acc[@NV@], sh_Ma[@NV@], sh_qc[@NV@], sh_grad[@NV@];
  threadgroup float sh_search[@NV@], sh_mv[@NV@], sh_tmp[@NV@];
  threadgroup float sh_qpos[@NQ@], sh_qvel[@NV@];
  threadgroup float sh_Dd[@RC@], sh_fld[@RC@], sh_Jaref[@RC@], sh_force[@RC@], sh_squad[@RC@], sh_jv[@RC@];
  threadgroup float sh_red[3 * @TG@];

  // right-looking lower Cholesky on a threadgroup matrix; lane owns rows i%TG
#define PAR_CHOL(A) \
  { \
    if (@TG@ == 32) { \
      for (int k = 0; k < @NV@; ++k) { \
        float dk_loc = (lane == (k & 31)) ? metal::sqrt(metal::max((A)[k * @NV@ + k], 1e-30f)) : 0.0f; \
        float dk = metal::simd_broadcast(dk_loc, (ushort)(k & 31)); \
        if (lane == (k & 31)) (A)[k * @NV@ + k] = dk; \
        for (int i = lane; i < @NV@; i += 32) { \
          if (i > k) (A)[i * @NV@ + k] /= dk; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (int i = lane; i < @NV@; i += 32) { \
          if (i > k) { \
            float lik = (A)[i * @NV@ + k]; \
            for (int j = k + 1; j <= i; ++j) (A)[i * @NV@ + j] -= lik * (A)[j * @NV@ + k]; \
          } \
        } \
      } \
      threadgroup_barrier(mem_flags::mem_threadgroup); \
    } else { \
      for (int k = 0; k < @NV@; ++k) { \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        if (lane == (k % @TG@)) (A)[k * @NV@ + k] = metal::sqrt(metal::max((A)[k * @NV@ + k], 1e-30f)); \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        float dk = (A)[k * @NV@ + k]; \
        for (int i = lane; i < @NV@; i += @TG@) { \
          if (i > k) (A)[i * @NV@ + k] = (A)[i * @NV@ + k] / dk; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        for (int i = lane; i < @NV@; i += @TG@) { \
          if (i > k) { \
            float lik = (A)[i * @NV@ + k]; \
            for (int j = k + 1; j <= i; ++j) (A)[i * @NV@ + j] -= lik * (A)[j * @NV@ + k]; \
          } \
        } \
      } \
      threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
  }

  // forward/backward substitution with a factored threadgroup matrix (in-place safe)
#define PAR_SOLVE(A, b, x) \
  { \
    if (@TG@ == 32 && @NV@ <= 32) { \
      float yi = (lane < @NV@) ? (b)[lane] : 0.0f; \
      for (int k = 0; k < @NV@; ++k) { \
        if (lane == k) yi /= (A)[k * @NV@ + k]; \
        float yk = metal::simd_broadcast(yi, (ushort)k); \
        if (lane > k && lane < @NV@) yi -= (A)[lane * @NV@ + k] * yk; \
      } \
      for (int k = @NV@ - 1; k >= 0; --k) { \
        if (lane == k) yi /= (A)[k * @NV@ + k]; \
        float xk = metal::simd_broadcast(yi, (ushort)k); \
        if (lane < k) yi -= (A)[k * @NV@ + lane] * xk; \
      } \
      if (lane < @NV@) (x)[lane] = yi; \
      threadgroup_barrier(mem_flags::mem_threadgroup); \
    } else if (@TG@ == 32) { \
      for (int i = 0; i < @NV@; ++i) { \
        float ps = 0.0f; \
        for (int j = lane; j < i; j += 32) ps += (A)[i * @NV@ + j] * (x)[j]; \
        float tot = metal::simd_sum(ps); \
        if (lane == 0) (x)[i] = ((b)[i] - tot) / (A)[i * @NV@ + i]; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
      } \
      for (int i = @NV@ - 1; i >= 0; --i) { \
        float ps = 0.0f; \
        for (int j = i + 1 + lane; j < @NV@; j += 32) ps += (A)[j * @NV@ + i] * (x)[j]; \
        float tot = metal::simd_sum(ps); \
        if (lane == 0) (x)[i] = ((x)[i] - tot) / (A)[i * @NV@ + i]; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
      } \
    } else { \
      for (int i = 0; i < @NV@; ++i) { \
        if (lane == (i % @TG@)) { \
          float s = (b)[i]; \
          for (int j = 0; j < i; ++j) s -= (A)[i * @NV@ + j] * (x)[j]; \
          (x)[i] = s / (A)[i * @NV@ + i]; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
      } \
      for (int i = @NV@ - 1; i >= 0; --i) { \
        if (lane == (i % @TG@)) { \
          float s = (x)[i]; \
          for (int j = i + 1; j < @NV@; ++j) s -= (A)[j * @NV@ + i] * (x)[j]; \
          (x)[i] = s / (A)[i * @NV@ + i]; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
      } \
    } \
  }

#define REDUCE3(a, b, c, r0, r1, r2) \
  { \
    if (@TG@ == 32) { \
      (r0) = metal::simd_sum(a); \
      (r1) = metal::simd_sum(b); \
      (r2) = metal::simd_sum(c); \
    } else { \
      sh_red[0 * @TG@ + lane] = (a); sh_red[1 * @TG@ + lane] = (b); sh_red[2 * @TG@ + lane] = (c); \
      threadgroup_barrier(mem_flags::mem_threadgroup); \
      for (int s = @TG@ / 2; s > 0; s >>= 1) { \
        if (lane < s) { \
          sh_red[0 * @TG@ + lane] += sh_red[0 * @TG@ + lane + s]; \
          sh_red[1 * @TG@ + lane] += sh_red[1 * @TG@ + lane + s]; \
          sh_red[2 * @TG@ + lane] += sh_red[2 * @TG@ + lane + s]; \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
      } \
      (r0) = sh_red[0 * @TG@]; (r1) = sh_red[1 * @TG@]; (r2) = sh_red[2 * @TG@]; \
      threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
  }

#define REDUCE1(a, r0) \
  { \
    if (@TG@ == 32) { \
      (r0) = metal::simd_sum(a); \
    } else { \
      sh_red[lane] = (a); \
      threadgroup_barrier(mem_flags::mem_threadgroup); \
      for (int s = @TG@ / 2; s > 0; s >>= 1) { \
        if (lane < s) sh_red[lane] += sh_red[lane + s]; \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
      } \
      (r0) = sh_red[0]; \
      threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
  }

  // ==================================================== load + acc start
  for (int i = lane; i < @NV@ * @NV@; i += @TG@) { sh_M[i] = M_in[w * @NV@ * @NV@ + i]; sh_h[i] = sh_M[i]; }
  for (int i = lane; i < @NV@; i += @TG@) sh_smooth[i] = smooth_in[w * @NV@ + i];
  for (int i = lane; i < @NQ@; i += @TG@) sh_qpos[i] = qpos_in[w * @NQ@ + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int nefc = nefc_in[w];
  if (nefc > @RC@) nefc = @RC@;
  int nit = 0;

  if (nefc == 0) {
    PAR_CHOL(sh_h);
    PAR_SOLVE(sh_h, sh_smooth, sh_acc);
    for (int d = lane; d < @NV@; d += @TG@) {
      warm_out[w * @NV@ + d] = sh_acc[d];
      qacc_out[w * @NV@ + d] = sh_acc[d];
    }
    if (@ED@ != 0) {
      for (int i = lane; i < @NV@ * @NV@; i += @TG@) sh_h[i] = sh_M[i];
      for (int i = lane; i < @NV@; i += @TG@) sh_h[i * @NV@ + i] += dt * dof_damping[i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      PAR_CHOL(sh_h);
      PAR_SOLVE(sh_h, sh_smooth, sh_acc);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  } else {
    if (@WS@ != 0) {
      for (int i = lane; i < @NV@; i += @TG@) sh_acc[i] = warm_in[w * @NV@ + i];
    } else {
      PAR_CHOL(sh_h);
      PAR_SOLVE(sh_h, sh_smooth, sh_acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ==================================================== hoist rows + Ma
    for (int r = lane; r < nefc; r += @TG@) {
      sh_Dd[r] = D_in[w * @RC@ + r];
      sh_fld[r] = fl_in[w * @RC@ + r];
      float ss = 0.0f;
      for (int i = 0; i < @NV@; ++i) ss += J_in[((long)w * @RC@ + r) * @NV@ + i] * sh_acc[i];
      sh_Jaref[r] = ss - aref_in[w * @RC@ + r];
    }
    for (int i = lane; i < @NV@; i += @TG@) {
      float ss = 0.0f;
      for (int j = 0; j < @NV@; ++j) ss += sh_M[i * @NV@ + j] * sh_acc[j];
      sh_Ma[i] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int r = lane; r < nefc; r += @TG@) {
      float c2, g2, h2;
      g_row_cost(sh_Jaref[r], sh_Dd[r], sh_fld[r], &c2, &g2, &h2);
      sh_force[r] = -g2;
      sh_squad[r] = (h2 > 0.0f) ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = lane; i < @NV@; i += @TG@) {
      float ss = 0.0f;
      for (int r = 0; r < nefc; ++r) ss += sh_force[r] * J_in[((long)w * @RC@ + r) * @NV@ + i];
      sh_qc[i] = ss;
      sh_grad[i] = sh_Ma[i] - sh_smooth[i] - ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // lower-triangle Hessian; each lane rebuilds only its own rows directly from J_in
#define REBUILD_LH() \
    { \
      for (int i = lane; i < @NV@; i += @TG@) \
        for (int j = 0; j <= i; ++j) sh_h[i * @NV@ + j] = sh_M[i * @NV@ + j]; \
      for (int r = 0; r < nefc; ++r) { \
        float dqr = sh_squad[r] * sh_Dd[r]; \
        if (dqr == 0.0f) continue; \
        const long jb = ((long)w * @RC@ + r) * @NV@; \
        for (int i = lane; i < @NV@; i += @TG@) { \
          float dq2 = dqr * J_in[jb + i]; \
          if (dq2 == 0.0f) continue; \
          for (int j = 0; j <= i; ++j) sh_h[i * @NV@ + j] += dq2 * J_in[jb + j]; \
        } \
      } \
      threadgroup_barrier(mem_flags::mem_threadgroup); \
    }
    REBUILD_LH();
    PAR_CHOL(sh_h);

    // ==================================================== Newton
    int done = 0;
    for (int it = 0; it < @ITER@; ++it) {
      nit = it + 1;
      for (int i = lane; i < @NV@; i += @TG@) sh_search[i] = -sh_grad[i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      PAR_SOLVE(sh_h, sh_search, sh_search);
      for (int i = lane; i < @NV@; i += @TG@) {
        float sss = 0.0f;
        for (int j = 0; j < @NV@; ++j) sss += sh_M[i * @NV@ + j] * sh_search[j];
        sh_mv[i] = sss;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      float nd_loc = 0.0f;
      for (int i = lane; i < @NV@; i += @TG@) nd_loc -= sh_grad[i] * sh_search[i];
      float newton_dec;
      REDUCE1(nd_loc, newton_dec);

      for (int r = lane; r < nefc; r += @TG@) {
        float ssr = 0.0f;
        for (int i = 0; i < @NV@; ++i) ssr += J_in[((long)w * @RC@ + r) * @NV@ + i] * sh_search[i];
        sh_jv[r] = ssr;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      float gs = 0.0f, hs = 0.0f, snorm_loc = 0.0f;
      for (int i = lane; i < @NV@; i += @TG@) {
        float mai = sh_Ma[i];
        float smi = sh_smooth[i];
        float si = sh_search[i];
        gs += (mai - smi) * si;
        hs += si * sh_mv[i];
        snorm_loc += si * si;
      }
      float g_sm0, hs_pre, snorm;
      REDUCE3(gs, hs, snorm_loc, g_sm0, hs_pre, snorm);
      snorm = metal::sqrt(snorm);
      float gtol = metal::max(tol * ls_tol * snorm * (meaninertia * (float)@NV@), 1e-6f);

#define PHI(alpha, pc, pg, phs) \
      { \
        float arv = alpha; \
        float cc = 0.0f; \
        float gg = 0.0f; \
        float hh = 0.0f; \
        for (int r = lane; r < nefc; r += @TG@) { \
          float jvr = sh_jv[r]; \
          float c2r, g2r, h2r; \
          g_row_cost_shifted(sh_Jaref[r], arv * jvr, sh_Dd[r], sh_fld[r], &c2r, &g2r, &h2r); \
          cc += c2r; \
          gg += g2r * jvr; \
          hh += h2r * jvr * jvr; \
        } \
        float cc_r, gg_r, hh_r; \
        REDUCE3(cc, gg, hh, cc_r, gg_r, hh_r); \
        *pc = arv * g_sm0 + 0.5f * arv * arv * hs_pre + cc_r; \
        *pg = g_sm0 + arv * hs_pre + gg_r; \
        *phs = hs_pre + hh_r; \
      }

#define PHI3(a0, a1, a2, c0p, g0p, h0p, c1p, g1p, h1p, c2p, g2p, h2p) \
      { \
        float av0 = a0, av1 = a1, av2 = a2; \
        float cc0 = 0.0f, gg0 = 0.0f, hh0 = 0.0f; \
        float cc1 = 0.0f, gg1 = 0.0f, hh1 = 0.0f; \
        float cc2 = 0.0f, gg2 = 0.0f, hh2 = 0.0f; \
        for (int r = lane; r < nefc; r += @TG@) { \
          float jar = sh_Jaref[r], jvr = sh_jv[r], ddr = sh_Dd[r], flr = sh_fld[r]; \
          float jvr2 = jvr * jvr; \
          float c2r, g2r, h2r; \
          g_row_cost_shifted(jar, av0 * jvr, ddr, flr, &c2r, &g2r, &h2r); \
          cc0 += c2r; gg0 += g2r * jvr; hh0 += h2r * jvr2; \
          g_row_cost_shifted(jar, av1 * jvr, ddr, flr, &c2r, &g2r, &h2r); \
          cc1 += c2r; gg1 += g2r * jvr; hh1 += h2r * jvr2; \
          g_row_cost_shifted(jar, av2 * jvr, ddr, flr, &c2r, &g2r, &h2r); \
          cc2 += c2r; gg2 += g2r * jvr; hh2 += h2r * jvr2; \
        } \
        float cc0_r, gg0_r, hh0_r, cc1_r, gg1_r, hh1_r, cc2_r, gg2_r, hh2_r; \
        REDUCE3(cc0, gg0, hh0, cc0_r, gg0_r, hh0_r); \
        REDUCE3(cc1, gg1, hh1, cc1_r, gg1_r, hh1_r); \
        REDUCE3(cc2, gg2, hh2, cc2_r, gg2_r, hh2_r); \
        *c0p = av0 * g_sm0 + 0.5f * av0 * av0 * hs_pre + cc0_r; \
        *g0p = g_sm0 + av0 * hs_pre + gg0_r; \
        *h0p = hs_pre + hh0_r; \
        *c1p = av1 * g_sm0 + 0.5f * av1 * av1 * hs_pre + cc1_r; \
        *g1p = g_sm0 + av1 * hs_pre + gg1_r; \
        *h1p = hs_pre + hh1_r; \
        *c2p = av2 * g_sm0 + 0.5f * av2 * av2 * hs_pre + cc2_r; \
        *g2p = g_sm0 + av2 * hs_pre + gg2_r; \
        *h2p = hs_pre + hh2_r; \
      }

      const float c0 = 0.0f;
      float g0_loc = 0.0f, hs0_loc = 0.0f, dummy_loc = 0.0f;
      for (int r = lane; r < nefc; r += @TG@) {
        float jvr = sh_jv[r];
        g0_loc -= sh_force[r] * jvr;
        hs0_loc += sh_squad[r] * sh_Dd[r] * jvr * jvr;
      }
      float g0_r, hs0_r, dummy_r;
      REDUCE3(g0_loc, hs0_loc, dummy_loc, g0_r, hs0_r, dummy_r);
      float g0 = g_sm0 + g0_r;
      float hs0 = hs_pre + hs0_r;

      float lo_alpha_in = -g0 / metal::max(hs0, 1e-30f);
      float ic, ig, ih;
      PHI(lo_alpha_in, &ic, &ig, &ih);

      float al = 0.0f;
      float improvement = 0.0f;
      if (metal::abs(ig) < gtol && ic < c0) {
        al = lo_alpha_in;
        improvement = c0 - ic;
      } else {
        int lo_less = ig < g0;
        float loc = lo_less ? ic : c0;
        float log_ = lo_less ? ig : g0;
        float loh = lo_less ? ih : hs0;
        float loa = lo_less ? lo_alpha_in : 0.0f;
        float hic = lo_less ? c0 : ic;
        float hig = lo_less ? g0 : ig;
        float hih = lo_less ? hs0 : ih;
        float hia = lo_less ? 0.0f : lo_alpha_in;
        for (int lsi = 0; lsi < @LSITER@; ++lsi) {
          float lna = loa - log_ / metal::max(loh, 1e-30f);
          float hna = hia - hig / metal::max(hih, 1e-30f);
          float mna = 0.5f * (loa + hia);
          float lnc, lng, lnh, hnc, hng, hnh, mnc, mng, mnh;
          PHI3(lna, hna, mna, &lnc, &lng, &lnh, &hnc, &hng, &hnh, &mnc, &mng, &mnh);
          int conv_lo = (metal::abs(lng) < gtol) & (lnc < c0);
          int conv_hi = (metal::abs(hng) < gtol) & (hnc < c0);
          int conv_mid = (metal::abs(mng) < gtol) & (mnc < c0);
          int converged = conv_lo | conv_hi | conv_mid;
          int swap_lo = 0;
          int swap_hi = 0;
          if (converged != 0) {
            float bcp = 1e30f; float bcd = 0.0f; float bch = 0.0f; float bca = 0.0f;
            if (conv_lo && lnc < bcp) { bcp = lnc; bcd = lng; bch = lnh; bca = lna; }
            if (conv_hi && hnc < bcp) { bcp = hnc; bcd = hng; bch = hnh; bca = hna; }
            if (conv_mid && mnc < bcp) { bcp = mnc; bcd = mng; bch = mnh; bca = mna; }
            loc = bcp; log_ = bcd; loh = bch; loa = bca;
            hic = bcp; hig = bcd; hih = bch; hia = bca;
          } else {
            int s1 = ((log_ < lng && lng < 0.0f) || (log_ > lng && lng > 0.0f)) ? 1 : 0;
            if (s1) { loc = lnc; log_ = lng; loh = lnh; loa = lna; }
            int s2 = ((log_ < mng && mng < 0.0f) || (log_ > mng && mng > 0.0f)) ? 1 : 0;
            if (s2) { loc = mnc; log_ = mng; loh = mnh; loa = mna; }
            int s3 = ((log_ < hng && hng < 0.0f) || (log_ > hng && hng > 0.0f)) ? 1 : 0;
            if (s3) { loc = hnc; log_ = hng; loh = hnh; loa = hna; }
            swap_lo = s1 | s2 | s3;
            int t1 = (((hig < hng && hng < 0.0f) || (hig > hng && hng > 0.0f)) || (hig < 0.0f && hng > 0.0f)) ? 1 : 0;
            if (t1) { hic = hnc; hig = hng; hih = hnh; hia = hna; }
            int t2 = ((hig < mng && mng < 0.0f) || (hig > mng && mng > 0.0f)) ? 1 : 0;
            if (t2) { hic = mnc; hig = mng; hih = mnh; hia = mna; }
            int t3 = ((hig < lng && lng < 0.0f) || (hig > lng && lng > 0.0f)) ? 1 : 0;
            if (t3) { hic = lnc; hig = lng; hih = lnh; hia = lna; }
            swap_hi = t1 | t2 | t3;
          }
          int ls_done = (converged != 0)
            | ((swap_lo | swap_hi) == 0 ? 1 : 0)
            | ((loc < c0 && log_ < 0.0f && log_ > -gtol) ? 1 : 0)
            | ((hic < c0 && hig > 0.0f && hig < gtol) ? 1 : 0);
          int improved = (loc < c0) | (hic < c0);
          int lo_better = loc < hic;
          float ba = lo_better ? loa : hia;
          float bdd = lo_better ? loc : hic;
          if (improved != 0) {
            al = ba;
            improvement = c0 - bdd;
          }
          if (ls_done != 0) break;
        }
      }
      if (done) al = 0.0f;

      for (int i = lane; i < @NV@; i += @TG@) {
        sh_acc[i] += al * sh_search[i];
        sh_Ma[i] += al * sh_mv[i];
      }
      for (int r = lane; r < nefc; r += @TG@) sh_Jaref[r] += al * sh_jv[r];
      threadgroup_barrier(mem_flags::mem_threadgroup);

      float sq_chg_loc = 0.0f;
      for (int r = lane; r < nefc; r += @TG@) {
        float c2, g2, h2;
        g_row_cost(sh_Jaref[r], sh_Dd[r], sh_fld[r], &c2, &g2, &h2);
        sh_force[r] = -g2;
        float new_sq = (h2 > 0.0f) ? 1.0f : 0.0f;
        if (new_sq != sh_squad[r]) sq_chg_loc = 1.0f;
        sh_squad[r] = new_sq;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      float gd_loc = 0.0f;
      for (int i = lane; i < @NV@; i += @TG@) {
        float ss = 0.0f;
        for (int r = 0; r < nefc; ++r) ss += sh_force[r] * J_in[((long)w * @RC@ + r) * @NV@ + i];
        sh_qc[i] = ss;
        sh_grad[i] = sh_Ma[i] - sh_smooth[i] - ss;
        gd_loc += sh_grad[i] * sh_grad[i];
      }
      float grad_dot, sq_chg_tot, dummy_chg;
      REDUCE3(gd_loc, sq_chg_loc, 0.0f, grad_dot, sq_chg_tot, dummy_chg);

      float rescale = meaninertia * (float)@NV@;
      int newly_done = (al == 0.0f)
        | ((improvement / rescale < tol) ? 1 : 0)
        | ((metal::sqrt(grad_dot) / rescale < tol) ? 1 : 0)
        | ((0.5f * newton_dec / rescale < tol) ? 1 : 0);
      done = done | newly_done;
      if (done != 0) break;
      if (sq_chg_tot > 0.0f) {
        REBUILD_LH();
        PAR_CHOL(sh_h);
      }
    }

    for (int d = lane; d < @NV@; d += @TG@) {
      warm_out[w * @NV@ + d] = sh_acc[d];
      qacc_out[w * @NV@ + d] = sh_acc[d];
    }

    // ==================================================== euler damping
    if (@ED@ != 0) {
      for (int i = lane; i < @NV@; i += @TG@) {
        float sss = 0.0f;
        for (int j = 0; j < @NV@; ++j) sss += sh_M[i * @NV@ + j] * sh_acc[j];
        sh_tmp[i] = sss;
      }
      for (int i = lane; i < @NV@ * @NV@; i += @TG@) sh_h[i] = sh_M[i];
      for (int i = lane; i < @NV@; i += @TG@) sh_h[i * @NV@ + i] += dt * dof_damping[i];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      PAR_CHOL(sh_h);
      PAR_SOLVE(sh_h, sh_tmp, sh_acc);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // ==================================================== integrate + write
  for (int d = lane; d < @NV@; d += @TG@) sh_qvel[d] = qvel_in[w * @NV@ + d] + dt * sh_acc[d];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int j = lane; j < @NJ@; j += @TG@) {
    int jt = jnt_type[j];
    int qa = jnt_qposadr[j];
    int dof = jnt_dofadr[j];
    if (jt == 0) {
      sh_qpos[qa + 0] += dt * sh_qvel[dof + 0];
      sh_qpos[qa + 1] += dt * sh_qvel[dof + 1];
      sh_qpos[qa + 2] += dt * sh_qvel[dof + 2];
      float4 quat = g_normalize4(float4(sh_qpos[qa + 3], sh_qpos[qa + 4], sh_qpos[qa + 5], sh_qpos[qa + 6]));
      float ax3 = dt * sh_qvel[dof + 3];
      float ay3 = dt * sh_qvel[dof + 4];
      float az3 = dt * sh_qvel[dof + 5];
      float nrm = metal::sqrt(ax3 * ax3 + ay3 * ay3 + az3 * az3);
      float3 axis = nrm > 0.0f ? float3(ax3, ay3, az3) / nrm : float3(1.0f, 0.0f, 0.0f);
      float halfang = 0.5f * nrm;
      float4 qr = float4(metal::cos(halfang), axis * metal::sin(halfang));
      float4 resv = g_normalize4(g_qmul(quat, qr));
      sh_qpos[qa + 3] = resv.x; sh_qpos[qa + 4] = resv.y; sh_qpos[qa + 5] = resv.z; sh_qpos[qa + 6] = resv.w;
    } else if (jt == 2 || jt == 3) {
      sh_qpos[qa] += dt * sh_qvel[dof];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int k = lane; k < @NQ@; k += @TG@) qpos_out[w * @NQ@ + k] = sh_qpos[k];
  for (int d = lane; d < @NV@; d += @TG@) {
    qvel_out[w * @NV@ + d] = sh_qvel[d];
  }
  if (lane == 0) niter_out[w] = nit;
"""
  for key, val in subs.items():
    body = body.replace(key, val)
  for name, off in sorted((adr_i or {}).items(), key=lambda x: -len(x[0])):
    body = body.replace(f"{name}[", f"mif[{off} + ")
  return body, _MSL_HELPERS

