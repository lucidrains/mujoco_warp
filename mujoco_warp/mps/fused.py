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

static inline float3 g_vertex(const device float* verts, int vadr, int idx) {
  return float3(verts[(vadr + idx) * 3 + 0], verts[(vadr + idx) * 3 + 1], verts[(vadr + idx) * 3 + 2]);
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
"""


# --------------------------------------------------------------------- K1 dyn+rows
def dyn_source(t: dict, adr_f: dict[str, int], adr_i: dict[str, int], kin: bool = False) -> tuple[str, str]:
  """Returns (source, header) for the K1 dyn kernel (source is the kernel body).

  Templates: NBODY NJNT NQ NV NGEOM NCHAIN NU NP RC NFR NLIM BF.

  `adr_f`/`adr_i` give the packed-buffer offsets of the static model tables
  (fewer device buffers than Metal's per-kernel limit). `kin=True` adds
  intermediates for validation (cdof, subtree_com, geom poses).
  """
  NB, NJ, NQ, NV = t["NBODY"], t["NJNT"], t["NQ"], t["NV"]
  NGE, NCH, NU = t["NGEOM"], t["NCHAIN"], t["NU"]
  NP, RC = t["NP"], t["RC"]
  NFR, NLIM, BF = t["NFR"], t["NLIM"], t["BF"]

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

  body = f"""
  int w = (int)thread_position_in_grid.x;
  if (w >= nworld_buf[0]) return;

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
  thread float xim[{NB} * 9];
  thread float subcom[{NB} * 3];
  thread float cin[{NB} * 10];
  thread float crb[{NB} * 10];
  thread float cdof[{NV} * 6];
  thread float cvel[{NB} * 6];
  thread float cdofdot[{NV} * 6];
  thread float cacc[{NB} * 6];
  thread float cfrc[{NB} * 6];
  thread float M[{NV} * {NV}];

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
    float4 iq4 = float4(body_iquat[b * 4 + 0], body_iquat[b * 4 + 1], body_iquat[b * 4 + 2], body_iquat[b * 4 + 3]);
    float3 ipos = float3(body_ipos[b * 3 + 0], body_ipos[b * 3 + 1], body_ipos[b * 3 + 2]);
    float3 bpos3 = float3(xp[b * 3 + 0], xp[b * 3 + 1], xp[b * 3 + 2]);
    float3 gip = bpos3 + g_qrot(q4, ipos);
    xip[b * 3 + 0] = gip.x; xip[b * 3 + 1] = gip.y; xip[b * 3 + 2] = gip.z;
    g_qmat(g_qmul(q4, iq4), &xim[b * 9]);
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
    const thread float* im = &xim[b * 9];
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

  // ==================================================== mass matrix
  for (int i = 0; i < {NV} * {NV}; ++i) M[i] = 0.0f;
  for (int i = 0; i < {NV}; ++i) {{
    int b = dof_bodyid[i];
    thread float buf[6];
    g_inert_vec(&crb[b * 10], &cdof[i * 6], buf);
    for (int c = 0; c < {NCH}; ++c) {{
      int j = dof_pad[i * {NCH} + c];
      if (j < 0) break;
      float val = 0.0f;
      for (int s = 0; s < 6; ++s) val += cdof[j * 6 + s] * buf[s];
      M[i * {NV} + j] += val;
      if (j != i) M[j * {NV} + i] += val;
    }}
    M[i * {NV} + i] += dof_armature[i];
  }}

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
    for (int k = 0; k < {NV}; ++k) M_out[(w * {NV} + i) * {NV} + k] = M[i * {NV} + k];
  }}

  // ==================================================== constraint rows
  // rows stream straight to global memory (no per-thread row arrays)
  device float* Jw = J_out + ((long)w) * {RC} * {NV};
  device float* Dw = D_out + ((long)w) * {RC};
  device float* arefw = aref_out + ((long)w) * {RC};
  device float* flw = fl_out + ((long)w) * {RC};
  int rowidx = 0;
  int conct = 0;
  int ovr = 0;

  // ---- friction-dof rows
  for (int k = 0; k < {NFR}; ++k) {{
    int dof = fr_dofs[k];
    for (int d = 0; d < {NV}; ++d) Jw[rowidx * {NV} + d] = 0.0f;
    Jw[rowidx * {NV} + dof] = 1.0f;
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
    for (int d = 0; d < {NV}; ++d) Jw[rowidx * {NV} + d] = 0.0f;
    Jw[rowidx * {NV} + dof] = sgn;
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
    float ds[4];
    float3 psd[4];
    float frms[4][9];
    int npts = 1;
    for (int i = 0; i < 4; ++i) ds[i] = 1e6f;

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
        if (x1 > 1.0f) {{ x1 = 1.0f; x2 = (vv - mb) / mc; }}
        else if (x1 < -1.0f) {{ x1 = -1.0f; x2 = (vv + mb) / mc; }}
        if (x2 > 1.0f) {{
          x2 = 1.0f;
          float xx = (uu - mb) / ma;
          x1 = metal::clamp(xx, -1.0f, 1.0f);
        }} else if (x2 < -1.0f) {{
          x2 = -1.0f;
          float xx = (uu + mb) / ma;
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
          float xx = (vv - mb) / mc;
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
          float xx = (vv + mb) / mc;
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
          float xx = (uu - mb) / ma;
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
          float xx = (uu + mb) / ma;
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
      const device float* mv = &mesh_vert[0];
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
      int idxs[4];
      idxs[0] = -1; idxs[1] = -1; idxs[2] = -1; idxs[3] = -1;
      float max_support = -huge;

      if (pair_usegraph[p] == 0) {{
        int aidx = -1;
        for (int i = 0; i < vnum; ++i) {{
          float3 vv = g_vertex(mv, vadr, i);
          float sup = g_dot3(pl - vv, nn);
          if (sup > max_support) {{ max_support = sup; aidx = i; }}
        }}
        if (max_support < 0.0f) continue;
        float threshold = max_support - 1e-3f;
        float3 av = g_vertex(mv, vadr, aidx);

        int bidx = -1; float bdist = -huge;
        for (int i = 0; i < vnum; ++i) {{
          float3 vv = g_vertex(mv, vadr, i);
          float sup = g_dot3(pl - vv, nn);
          float msk = (sup > threshold) ? 0.0f : -huge;
          float dd = g_dot3(av - vv, av - vv) + msk;
          if (dd > bdist) {{ bdist = dd; bidx = i; }}
        }}
        float3 bv = g_vertex(mv, vadr, bidx);
        float3 ab = g_cross(nn, av - bv);

        int cidx = -1; float cdist = -huge;
        for (int i = 0; i < vnum; ++i) {{
          float3 vv = g_vertex(mv, vadr, i);
          float sup = g_dot3(pl - vv, nn);
          float msk = (sup > threshold) ? 0.0f : -huge;
          float dd = metal::abs(g_dot3(av - vv, ab)) + msk;
          if (dd > cdist) {{ cdist = dd; cidx = i; }}
        }}
        float3 cv = g_vertex(mv, vadr, cidx);
        float3 ac = g_cross(nn, av - cv);
        float3 bc = g_cross(nn, bv - cv);

        int didx = -1; float ddist = -huge;
        for (int i = 0; i < vnum; ++i) {{
          float3 vv = g_vertex(mv, vadr, i);
          float sup = g_dot3(pl - vv, nn);
          float msk = (sup > threshold) ? 0.0f : -huge;
          float dd = metal::abs(g_dot3(av - vv, ac)) + metal::abs(g_dot3(bv - vv, bc)) + msk;
          if (dd > ddist) {{ ddist = dd; didx = i; }}
        }}
        idxs[0] = aidx; idxs[1] = bidx; idxs[2] = cidx; idxs[3] = didx;
      }} else {{
        int prev = -1; int imax = 0;
        while (true) {{
          prev = imax;
          int ii = mesh_graph[gadr + 2 + imax];
          while (mesh_graph[gadr + 2 + 2 * vnum + ii] >= 0) {{
            int subidx = mesh_graph[gadr + 2 + 2 * vnum + ii];
            int vidx = mesh_graph[gadr + 2 + vnum + subidx];
            float3 vv = g_vertex(mv, vadr, vidx);
            float sup = g_dot3(pl - vv, nn);
            if (sup > max_support) {{ max_support = sup; imax = subidx; }}
            ii += 1;
          }}
          if (imax == prev) break;
        }}
        if (max_support < 0.0f) continue;
        float threshold = metal::max(0.0f, max_support - 1e-3f);

        float adist = -huge;
        while (true) {{
          prev = imax;
          int ii = mesh_graph[gadr + 2 + imax];
          while (mesh_graph[gadr + 2 + 2 * vnum + ii] >= 0) {{
            int subidx = mesh_graph[gadr + 2 + 2 * vnum + ii];
            int vidx = mesh_graph[gadr + 2 + vnum + subidx];
            float3 vv = g_vertex(mv, vadr, vidx);
            float sup = g_dot3(pl - vv, nn);
            float dd = (sup > threshold) ? sup : -huge;
            if (dd > adist) {{ adist = dd; imax = subidx; }}
            ii += 1;
          }}
          if (imax == prev) break;
        }}
        int ag = mesh_graph[gadr + 2 + vnum + imax];
        float3 av = g_vertex(mv, vadr, ag);
        idxs[0] = ag;

        float bdist = -huge;
        while (true) {{
          prev = imax;
          int ii = mesh_graph[gadr + 2 + imax];
          while (mesh_graph[gadr + 2 + 2 * vnum + ii] >= 0) {{
            int subidx = mesh_graph[gadr + 2 + 2 * vnum + ii];
            int vidx = mesh_graph[gadr + 2 + vnum + subidx];
            float3 vv = g_vertex(mv, vadr, vidx);
            float sup = g_dot3(pl - vv, nn);
            float msk = (sup > threshold) ? 0.0f : -huge;
            float dd = g_dot3(av - vv, av - vv) + msk;
            if (dd > bdist) {{ bdist = dd; imax = subidx; }}
            ii += 1;
          }}
          if (imax == prev) break;
        }}
        int bg = mesh_graph[gadr + 2 + vnum + imax];
        float3 bv = g_vertex(mv, vadr, bg);
        idxs[1] = bg;
        float3 ab = g_cross(nn, av - bv);

        float cdist = -huge;
        while (true) {{
          prev = imax;
          int ii = mesh_graph[gadr + 2 + imax];
          while (mesh_graph[gadr + 2 + 2 * vnum + ii] >= 0) {{
            int subidx = mesh_graph[gadr + 2 + 2 * vnum + ii];
            int vidx = mesh_graph[gadr + 2 + vnum + subidx];
            float3 vv = g_vertex(mv, vadr, vidx);
            float sup = g_dot3(pl - vv, nn);
            float msk = (sup > threshold) ? 0.0f : -huge;
            float dd = metal::abs(g_dot3(av - vv, ab)) + msk;
            if (dd > cdist) {{ cdist = dd; imax = subidx; }}
            ii += 1;
          }}
          if (imax == prev) break;
        }}
        int cg = mesh_graph[gadr + 2 + vnum + imax];
        float3 cv = g_vertex(mv, vadr, cg);
        idxs[2] = cg;
        float3 ac = g_cross(nn, av - cv);
        float3 bc = g_cross(nn, bv - cv);

        float ddist = -huge;
        while (true) {{
          prev = imax;
          int ii = mesh_graph[gadr + 2 + imax];
          while (mesh_graph[gadr + 2 + 2 * vnum + ii] >= 0) {{
            int subidx = mesh_graph[gadr + 2 + 2 * vnum + ii];
            int vidx = mesh_graph[gadr + 2 + vnum + subidx];
            float3 vv = g_vertex(mv, vadr, vidx);
            float sup = g_dot3(pl - vv, nn);
            float msk = (sup > threshold) ? 0.0f : -huge;
            float dd = metal::abs(g_dot3(av - vv, ac)) + metal::abs(g_dot3(bv - vv, bc)) + msk;
            if (dd > ddist) {{ ddist = dd; imax = subidx; }}
            ii += 1;
          }}
          if (imax == prev) break;
        }}
        int dg = mesh_graph[gadr + 2 + vnum + imax];
        idxs[3] = dg;
      }}

      // emit unique indices in slot order, transformed to the world frame
      int slot = 0;
      for (int ci = 3; ci >= 0; --ci) {{
        int idx = idxs[ci];
        int count = 0;
        for (int cj = 0; cj <= ci; ++cj) if (idxs[cj] == idx) count += 1;
        if (count != 1) continue;
        float3 lv = g_vertex(mv, vadr, idx);
        float3 wp = xa2 + float3(
          mr[0] * lv.x + mr[1] * lv.y + mr[2] * lv.z,
          mr[3] * lv.x + mr[4] * lv.y + mr[5] * lv.z,
          mr[6] * lv.x + mr[7] * lv.y + mr[8] * lv.z);
        float sup = g_dot3(pl - lv, nn);
        float dist = -sup;
        wp = wp - 0.5f * dist * pnrm;
        ds[slot] = dist;
        psd[slot] = wp;
        g_make_frame(pnrm, frms[slot]);
        slot += 1;
      }}
      npts = slot;
    }}

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
        for (int d = 0; d < {NV}; ++d) Jw[(rowidx + dim) * {NV} + d] = 0.0f;
        flw[rowidx + dim] = 0.0f;
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
      for (int dof = 0; dof < {NV}; ++dof) {{
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
          Jw[(rowidx + dim) * {NV} + dof] = val;
          qv[dim] += val * v[dof];
        }}
      }}
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
  for name, off in sorted(adr_f.items(), key=lambda x: -len(x[0])):
    body = body.replace(f"{name}[", f"mbf[{off} + ")
  for name, off in sorted(adr_i.items(), key=lambda x: -len(x[0])):
    body = body.replace(f"{name}[", f"mif[{off} + ")
  return body, _MSL_HELPERS


# --------------------------------------------------------------------- K2 solve+integrate
def solve_source(t: dict, adr_i: dict[str, int] | None = None) -> tuple[str, str]:
  """K2 source. Templates: NJNT NQ NV RC ITER LSITER WARMSTART EULERDAMP."""
  NJ, NQ, NV = t["NJNT"], t["NQ"], t["NV"]
  RC, ITER, LSITER = t["RC"], t["ITER"], t["LSITER"]
  WS = t["WARMSTART"]
  ED = t["EULERDAMP"]
  body = f"""
  int w = (int)thread_position_in_grid.x;
  if (w >= nworld_buf[0]) return;

  const float dt = opt_buf[0];
  const float tol = opt_buf[1];
  const float meaninertia = opt_buf[2];
  const float ls_tol = opt_buf[3];

  thread float M[{NV} * {NV}];
  for (int i = 0; i < {NV} * {NV}; ++i) M[i] = M_in[w * {NV} * {NV} + i];

  thread float acc[{NV}];
  if ({WS} != 0) {{
    for (int i = 0; i < {NV}; ++i) acc[i] = warm_in[w * {NV} + i];
  }} else {{
    // factorization lives only on the cold path: with warmstart the smoothed
    // acceleration is never read
    thread float L[{NV} * {NV}];
    for (int i = 0; i < {NV} * {NV}; ++i) L[i] = M[i];
    g_chol_inplace(L, {NV});
    thread float smooth0[{NV}];
    for (int i = 0; i < {NV}; ++i) smooth0[i] = smooth_in[w * {NV} + i];
    thread float qacc_smooth[{NV}];
    g_chol_solve_fact(L, smooth0, qacc_smooth, {NV});
    for (int i = 0; i < {NV}; ++i) acc[i] = qacc_smooth[i];
  }}
  thread float smooth[{NV}];
  for (int i = 0; i < {NV}; ++i) smooth[i] = smooth_in[w * {NV} + i];
  int nefc = nefc_in[w];
  if (nefc > {RC}) nefc = {RC};

  // hoist per-row state (constant across the Newton iterations)
  thread float Dd[{RC}];
  thread float fld[{RC}];
  thread float Jaref[{RC}];
  for (int r = 0; r < nefc; ++r) {{
    Dd[r] = D_in[w * {RC} + r];
    fld[r] = fl_in[w * {RC} + r];
    float ss = 0.0f;
    // direct index: small inputs may be in the constant address space
    for (int i = 0; i < {NV}; ++i) ss += J_in[((long)w * {RC} + r) * {NV} + i] * acc[i];
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
  thread float h[{NV} * {NV}];
  for (int i = 0; i < {NV}; ++i) qc[i] = 0.0f;

  // lower-triangle Hessian rebuild (h is never read above the diagonal); deferred
  // past the convergence check so a converged last Newton iteration skips it
#define REBUILD_LH() \\
  {{ \\
    for (int i = 0; i < {NV}; ++i) \\
      for (int j = 0; j <= i; ++j) h[i * {NV} + j] = M[i * {NV} + j]; \\
    for (int r = 0; r < nefc; ++r) {{ \\
      float dqr = squad[r] * Dd[r]; \\
      if (dqr == 0.0f) continue; \\
      const long jb = ((long)w * {RC} + r) * {NV}; \\
      thread float jr[{NV}]; \\
      for (int j = 0; j < {NV}; ++j) jr[j] = J_in[jb + j]; \\
      for (int i = 0; i < {NV}; ++i) {{ \\
        float dq2 = dqr * jr[i]; \\
        for (int j = 0; j <= i; ++j) h[i * {NV} + j] += dq2 * jr[j]; \\
      }} \\
    }} \\
  }}
  REBUILD_LH();
  for (int r = 0; r < nefc; ++r) {{
    const long jbase = ((long)w * {RC} + r) * {NV};
    float fr = force[r];
    for (int i = 0; i < {NV}; ++i) qc[i] += fr * J_in[jbase + i];
  }}
  for (int i = 0; i < {NV}; ++i) grad[i] = Ma[i] - smooth[i] - qc[i];

  int done = 0;
  int nit = 0;
  thread float search[{NV}];
  thread float mv[{NV}];
  thread float jv[{RC}];
  for (int it = 0; it < {ITER}; ++it) {{
    nit = it + 1;
    g_chol_inplace(h, {NV});
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
      float ssr = 0.0f;
      for (int i = 0; i < {NV}; ++i) ssr += J_in[((long)w * {RC} + r) * {NV} + i] * search[i];
      jv[r] = ssr;
    }}

    float c_sm0 = 0.0f;
    for (int i = 0; i < {NV}; ++i) c_sm0 += 0.5f * acc[i] * Ma[i] - acc[i] * smooth[i];
    float g_sm0 = 0.0f;
    for (int i = 0; i < {NV}; ++i) g_sm0 += (Ma[i] - smooth[i]) * search[i];
    float hs_pre = 0.0f;
    for (int i = 0; i < {NV}; ++i) hs_pre += search[i] * mv[i];

#define PHI(alpha, pc, pg, phs) \\
  {{ \\
    float arv = alpha; \\
    float cc = c_sm0 + arv * g_sm0 + 0.5f * arv * arv * hs_pre; \\
    float gg = g_sm0 + arv * hs_pre; \\
    float hh = hs_pre; \\
    for (int r = 0; r < nefc; ++r) {{ \\
      float xrc = Jaref[r] + arv * jv[r]; \\
      float c2r, g2r, h2r; \\
      g_row_cost(xrc, Dd[r], fld[r], &c2r, &g2r, &h2r); \\
      cc += c2r; \\
      gg += g2r * jv[r]; \\
      hh += h2r * jv[r] * jv[r]; \\
    }} \\
    *pc = cc; *pg = gg; *phs = hh; \\
  }}

    float c0, g0, hs0;
    PHI(0.0f, &c0, &g0, &hs0);

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
        PHI(lna, &lnc, &lng, &lnh);
        PHI(hna, &hnc, &hng, &hnh);
        PHI(mna, &mnc, &mng, &mnh);
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

    for (int r = 0; r < nefc; ++r) {{
      float c2, g2, h2;
      g_row_cost(Jaref[r], Dd[r], fld[r], &c2, &g2, &h2);
      force[r] = -g2;
      squad[r] = (h2 > 0.0f) ? 1.0f : 0.0f;
    }}
    for (int i = 0; i < {NV}; ++i) qc[i] = 0.0f;
    for (int r = 0; r < nefc; ++r) {{
      const long jbase = ((long)w * {RC} + r) * {NV};
      float fr = force[r];
      for (int i = 0; i < {NV}; ++i) qc[i] += fr * J_in[jbase + i];
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
    REBUILD_LH();
  }}

  if ({ED} != 0) {{
    float Ma2[{NV}];
    for (int i = 0; i < {NV}; ++i) {{
      float sss = 0.0f;
      for (int j = 0; j < {NV}; ++j) sss += M[i * {NV} + j] * acc[j];
      Ma2[i] = sss;
    }}
    float Md[{NV} * {NV}];
    for (int i = 0; i < {NV} * {NV}; ++i) Md[i] = M[i];
    for (int i = 0; i < {NV}; ++i) Md[i * {NV} + i] += dt * dof_damping[i];
    g_chol_inplace(Md, {NV});
    for (int i = 0; i < {NV}; ++i) acc[i] = 0.0f;
    g_chol_solve_fact(Md, Ma2, acc, {NV});
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
    warm_out[w * {NV} + d] = acc[d];
    qacc_out[w * {NV} + d] = acc[d];
  }}
  niter_out[w] = nit;
"""
  for name, off in sorted((adr_i or {}).items(), key=lambda x: -len(x[0])):
    body = body.replace(f"{name}[", f"mif[{off} + ")
  return body, _MSL_HELPERS


# --------------------------------------------------------------------- K2 cooperative
def solve_source_coop(t: dict, adr_i: dict[str, int] | None = None, tg: int = 32) -> tuple[str, str]:
  """K2 source, cooperative: one threadgroup (TG threads) per world.

  Templates: NJNT NQ NV RC ITER LSITER WARMSTART EULERDAMP TG.
  Matrices and vectors live in threadgroup memory; row loops are lane-strided.
  """
  subs = {
    "@NV@": str(t["NV"]),
    "@NQ@": str(t["NQ"]),
    "@NJ@": str(t["NJNT"]),
    "@RC@": str(t["RC"]),
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
  }

  // forward/backward substitution with a factored threadgroup matrix (in-place safe)
#define PAR_SOLVE(A, b, x) \
  { \
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
  }

#define REDUCE3(a, b, c, r0, r1, r2) \
  { \
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
  }

#define REDUCE1(a, r0) \
  { \
    sh_red[lane] = (a); \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    for (int s = @TG@ / 2; s > 0; s >>= 1) { \
      if (lane < s) sh_red[lane] += sh_red[lane + s]; \
      threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    (r0) = sh_red[0]; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
  }

  // ==================================================== load + acc start
  for (int i = lane; i < @NV@ * @NV@; i += @TG@) { sh_M[i] = M_in[w * @NV@ * @NV@ + i]; sh_h[i] = sh_M[i]; }
  for (int i = lane; i < @NV@; i += @TG@) sh_smooth[i] = smooth_in[w * @NV@ + i];
  for (int i = lane; i < @NQ@; i += @TG@) sh_qpos[i] = qpos_in[w * @NQ@ + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (@WS@ != 0) {
    for (int i = lane; i < @NV@; i += @TG@) sh_acc[i] = warm_in[w * @NV@ + i];
  } else {
    PAR_CHOL(sh_h);
    PAR_SOLVE(sh_h, sh_smooth, sh_acc);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int nefc = nefc_in[w];
  if (nefc > @RC@) nefc = @RC@;

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

  threadgroup float sh_Jr[@NV@];

  // lower-triangle Hessian; each lane rebuilds only its own rows
#define REBUILD_LH() \
  { \
    for (int i = lane; i < @NV@; i += @TG@) \
      for (int j = 0; j <= i; ++j) sh_h[i * @NV@ + j] = sh_M[i * @NV@ + j]; \
    for (int r = 0; r < nefc; ++r) { \
      float dqr = sh_squad[r] * sh_Dd[r]; \
      if (dqr == 0.0f) continue; \
      const long jb = ((long)w * @RC@ + r) * @NV@; \
      for (int k = lane; k < @NV@; k += @TG@) sh_Jr[k] = J_in[jb + k]; \
      threadgroup_barrier(mem_flags::mem_threadgroup); \
      for (int i = lane; i < @NV@; i += @TG@) { \
        float dq2 = dqr * sh_Jr[i]; \
        for (int j = 0; j <= i; ++j) sh_h[i * @NV@ + j] += dq2 * sh_Jr[j]; \
      } \
      threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
  }
  REBUILD_LH();

  // ==================================================== Newton
  int done = 0;
  int nit = 0;
  for (int it = 0; it < @ITER@; ++it) {
    nit = it + 1;
    PAR_CHOL(sh_h);
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

    float cs = 0.0f, gs = 0.0f, hs = 0.0f;
    for (int i = lane; i < @NV@; i += @TG@) {
      float mai = sh_Ma[i];
      float smi = sh_smooth[i];
      cs += 0.5f * sh_acc[i] * mai - sh_acc[i] * smi;
      gs += (mai - smi) * sh_search[i];
      hs += sh_search[i] * sh_mv[i];
    }
    float c_sm0, g_sm0, hs_pre;
    REDUCE3(cs, gs, hs, c_sm0, g_sm0, hs_pre);

#define PHI(alpha, pc, pg, phs) \
  { \
    float arv = alpha; \
    float cc = 0.0f; \
    float gg = 0.0f; \
    float hh = 0.0f; \
    for (int r = lane; r < nefc; r += @TG@) { \
      float xrc = sh_Jaref[r] + arv * sh_jv[r]; \
      float c2r, g2r, h2r; \
      g_row_cost(xrc, sh_Dd[r], sh_fld[r], &c2r, &g2r, &h2r); \
      cc += c2r; \
      gg += g2r * sh_jv[r]; \
      hh += h2r * sh_jv[r] * sh_jv[r]; \
    } \
    float cc_r, gg_r, hh_r; \
    REDUCE3(cc, gg, hh, cc_r, gg_r, hh_r); \
    *pc = c_sm0 + arv * g_sm0 + 0.5f * arv * arv * hs_pre + cc_r; \
    *pg = g_sm0 + arv * hs_pre + gg_r; \
    *phs = hs_pre + hh_r; \
  }

    float c0, g0, hs0;
    PHI(0.0f, &c0, &g0, &hs0);

    float snorm_loc = 0.0f;
    for (int i = lane; i < @NV@; i += @TG@) snorm_loc += sh_search[i] * sh_search[i];
    float snorm;
    REDUCE1(snorm_loc, snorm);
    snorm = metal::sqrt(snorm);
    float gtol = metal::max(tol * ls_tol * snorm * (meaninertia * (float)@NV@), 1e-6f);

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
        PHI(lna, &lnc, &lng, &lnh);
        PHI(hna, &hnc, &hng, &hnh);
        PHI(mna, &mnc, &mng, &mnh);
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

    for (int r = lane; r < nefc; r += @TG@) {
      float c2, g2, h2;
      g_row_cost(sh_Jaref[r], sh_Dd[r], sh_fld[r], &c2, &g2, &h2);
      sh_force[r] = -g2;
      sh_squad[r] = (h2 > 0.0f) ? 1.0f : 0.0f;
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
    float grad_dot;
    REDUCE1(gd_loc, grad_dot);

    float rescale = meaninertia * (float)@NV@;
    int newly_done = (al == 0.0f)
      | ((improvement / rescale < tol) ? 1 : 0)
      | ((metal::sqrt(grad_dot) / rescale < tol) ? 1 : 0)
      | ((0.5f * newton_dec / rescale < tol) ? 1 : 0);
    done = done | newly_done;
    if (done != 0) break;
    REBUILD_LH();
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
    warm_out[w * @NV@ + d] = sh_acc[d];
    qacc_out[w * @NV@ + d] = sh_acc[d];
  }
  if (lane == 0) niter_out[w] = nit;
"""
  for key, val in subs.items():
    body = body.replace(key, val)
  for name, off in sorted((adr_i or {}).items(), key=lambda x: -len(x[0])):
    body = body.replace(f"{name}[", f"mif[{off} + ")
  return body, _MSL_HELPERS
