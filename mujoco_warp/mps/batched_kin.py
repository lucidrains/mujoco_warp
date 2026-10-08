"""Fused forward-kinematics tree pass: one MSL thread per world."""

from __future__ import annotations

import numpy as np
import mlx.core as mx

_MSL = r"""
    uint w = thread_position_in_grid.x;
    if (w >= NWORLD[0]) return;

    float xp[NBODY * 3];
    float xq[NBODY * 4];
    float xa[NJNT * 3];
    float xs[NJNT * 3];

    for (int b = 0; b < NBODY; ++b) {
        int ja = jntadr[b];
        int jn = jntnum[b];
        if (jn == 1 && jtype[ja] == 0) {
            int qa = jqposadr[ja];
            for (int k = 0; k < 3; ++k) xp[b * 3 + k] = qpos[w * NQ + qa + k];
            float q0 = qpos[w * NQ + qa + 3];
            float q1 = qpos[w * NQ + qa + 4];
            float q2 = qpos[w * NQ + qa + 5];
            float q3 = qpos[w * NQ + qa + 6];
            float nrm = metal::sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
            xq[b * 4 + 0] = q0 / nrm;
            xq[b * 4 + 1] = q1 / nrm;
            xq[b * 4 + 2] = q2 / nrm;
            xq[b * 4 + 3] = q3 / nrm;
            for (int k = 0; k < 3; ++k) { xa[ja * 3 + k] = xp[b * 3 + k]; xs[ja * 3 + k] = jaxis[ja * 3 + k]; }
            continue;
        }
        float px = bpos[b * 3 + 0];
        float py = bpos[b * 3 + 1];
        float pz = bpos[b * 3 + 2];
        float qw = bquat[b * 4 + 0];
        float qx = bquat[b * 4 + 1];
        float qy = bquat[b * 4 + 2];
        float qz = bquat[b * 4 + 3];
        int pid = parent[b];
        if (pid >= 0 && pid != b) {
            float rw = xq[pid * 4 + 0], rx = xq[pid * 4 + 1], ry = xq[pid * 4 + 2], rz = xq[pid * 4 + 3];
            float tx = 2.0f * (ry * pz - rz * py) + (rw * rw - rx * rx - ry * ry - rz * rz) * px + 2.0f * rx * (rx * px + ry * py + rz * pz);
            float ty = 2.0f * (rz * px - rx * pz) + (rw * rw - rx * rx - ry * ry - rz * rz) * py + 2.0f * ry * (rx * px + ry * py + rz * pz);
            float tz = 2.0f * (rx * py - ry * px) + (rw * rw - rx * rx - ry * ry - rz * rz) * pz + 2.0f * rz * (rx * px + ry * py + rz * pz);
            px = tx + xp[pid * 3 + 0];
            py = ty + xp[pid * 3 + 1];
            pz = tz + xp[pid * 3 + 2];
            float nw = rw * qw - rx * qx - ry * qy - rz * qz;
            float nx = rw * qx + rx * qw + ry * qz - rz * qy;
            float ny = rw * qy - rx * qz + ry * qw + rz * qx;
            float nz = rw * qz + rx * qy - ry * qx + rz * qw;
            qw = nw; qx = nx; qy = ny; qz = nz;
        }
        for (int k = 0; k < jn; ++k) {
            int j = ja + k;
            int qa = jqposadr[j];
            float ax = jaxis[j * 3 + 0], ay = jaxis[j * 3 + 1], az = jaxis[j * 3 + 2];
            float jpx = jpos[j * 3 + 0], jpy = jpos[j * 3 + 1], jpz = jpos[j * 3 + 2];
            // anchor = rot(jpos, xquat) + (px,py,pz)
            float dotu = qx * jpx + qy * jpy + qz * jpz;
            float sx = 2.0f * (qy * jpz - qz * jpy);
            float sy = 2.0f * (qz * jpx - qx * jpz);
            float sz = 2.0f * (qx * jpy - qy * jpx);
            float ax_ = jpx + qw * sx + (qw * qw - qx * qx - qy * qy - qz * qz) * 0.0f;
            // (use standard rotate: v + 2*qw*(u x v) + 2*u x (u x v))
            float uxv_x = qy * jpz - qz * jpy;
            float uxv_y = qz * jpx - qx * jpz;
            float uxv_z = qx * jpy - qy * jpx;
            float uuxv_x = qy * uxv_z - qz * uxv_y;
            float uuxv_y = qz * uxv_x - qx * uxv_z;
            float uuxv_z = qx * uxv_y - qy * uxv_x;
            float rx_ = jpx + 2.0f * (qw * uxv_x + uuxv_x);
            float ry_ = jpy + 2.0f * (qw * uxv_y + uuxv_y);
            float rz_ = jpz + 2.0f * (qw * uxv_z + uuxv_z);
            float anchor_x = rx_ + px;
            float anchor_y = ry_ + py;
            float anchor_z = rz_ + pz;
            float uxa_x = qy * az - qz * ay;
            float uxa_y = qz * ax - qx * az;
            float uxa_z = qx * ay - qy * ax;
            float uuxa_x = qy * uxa_z - qz * uxa_y;
            float uuxa_y = qz * uxa_x - qx * uxa_z;
            float uuxa_z = qx * uxa_y - qy * uxa_x;
            float gax = ax + 2.0f * (qw * uxa_x + uuxa_x);
            float gay = ay + 2.0f * (qw * uxa_y + uuxa_y);
            float gaz = az + 2.0f * (qw * uxa_z + uuxa_z);
            if (jtype[j] == 3) {
                float angle = qpos[w * NQ + qa] - qpos0[qa];
                float s = metal::sin(angle * 0.5f);
                float c = metal::cos(angle * 0.5f);
                float lw = c, lx = ax * s, ly = ay * s, lz = az * s;
                float nw = qw * lw - qx * lx - qy * ly - qz * lz;
                float nx = qw * lx + qx * lw + qy * lz - qz * ly;
                float ny = qw * ly - qx * lz + qy * lw + qz * lx;
                float nz = qw * lz + qx * ly - qy * lx + qz * lw;
                qw = nw; qx = nx; qy = ny; qz = nz;
                // xpos = anchor - rot(jpos, xquat)
                float vxu_x = qy * jpz - qz * jpy;
                float vxu_y = qz * jpx - qx * jpz;
                float vxu_z = qx * jpy - qy * jpx;
                float vvx_x = qy * vxu_z - qz * vxu_y;
                float vvx_y = qz * vxu_x - qx * vxu_z;
                float vvx_z = qx * vxu_y - qy * vxu_x;
                px = anchor_x - (jpx + 2.0f * (qw * vxu_x + vvx_x));
                py = anchor_y - (jpy + 2.0f * (qw * vxu_y + vvx_y));
                pz = anchor_z - (jpz + 2.0f * (qw * vxu_z + vvx_z));
            }
            xa[j * 3 + 0] = anchor_x; xa[j * 3 + 1] = anchor_y; xa[j * 3 + 2] = anchor_z;
            xs[j * 3 + 0] = gax; xs[j * 3 + 1] = gay; xs[j * 3 + 2] = gaz;
        }
        float nrm = metal::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        qw /= nrm; qx /= nrm; qy /= nrm; qz /= nrm;
        xp[b * 3 + 0] = px; xp[b * 3 + 1] = py; xp[b * 3 + 2] = pz;
        xq[b * 4 + 0] = qw; xq[b * 4 + 1] = qx; xq[b * 4 + 2] = qy; xq[b * 4 + 3] = qz;
    }

    for (int b = 0; b < NBODY; ++b) {
        float qw = xq[b * 4 + 0], qx = xq[b * 4 + 1], qy = xq[b * 4 + 2], qz = xq[b * 4 + 3];
        float q00 = qw * qw, q01 = qw * qx, q02 = qw * qy, q03 = qw * qz;
        float q11 = qx * qx, q12 = qx * qy, q13 = qx * qz;
        float q22 = qy * qy, q23 = qy * qz, q33 = qz * qz;
        XMAT[w * NBODY * 9 + b * 9 + 0] = q00 + q11 - q22 - q33;
        XMAT[w * NBODY * 9 + b * 9 + 1] = 2.0f * (q12 - q03);
        XMAT[w * NBODY * 9 + b * 9 + 2] = 2.0f * (q13 + q02);
        XMAT[w * NBODY * 9 + b * 9 + 3] = 2.0f * (q12 + q03);
        XMAT[w * NBODY * 9 + b * 9 + 4] = q00 - q11 + q22 - q33;
        XMAT[w * NBODY * 9 + b * 9 + 5] = 2.0f * (q23 - q01);
        XMAT[w * NBODY * 9 + b * 9 + 6] = 2.0f * (q13 - q02);
        XMAT[w * NBODY * 9 + b * 9 + 7] = 2.0f * (q23 + q01);
        XMAT[w * NBODY * 9 + b * 9 + 8] = q00 - q11 - q22 + q33;
        // xipos
        float ix = bipos[b * 3 + 0], iy = bipos[b * 3 + 1], iz = bipos[b * 3 + 2];
        float uxv_x = qy * iz - qz * iy;
        float uxv_y = qz * ix - qx * iz;
        float uxv_z = qx * iy - qy * ix;
        float uuxv_x = qy * uxv_z - qz * uxv_y;
        float uuxv_y = qz * uxv_x - qx * uxv_z;
        float uuxv_z = qx * uxv_y - qy * uxv_x;
        XIPOS[w * NBODY * 3 + b * 3 + 0] = xp[b * 3 + 0] + ix + 2.0f * (qw * uxv_x + uuxv_x);
        XIPOS[w * NBODY * 3 + b * 3 + 1] = xp[b * 3 + 1] + iy + 2.0f * (qw * uxv_y + uuxv_y);
        XIPOS[w * NBODY * 3 + b * 3 + 2] = xp[b * 3 + 2] + iz + 2.0f * (qw * uxv_z + uuxv_z);
        // ximat = quat_to_mat(xquat * iquat)
        float iw = biquat[b * 4 + 0], xx = biquat[b * 4 + 1], xy = biquat[b * 4 + 2], xz = biquat[b * 4 + 3];
        float nw = qw * iw - qx * xx - qy * xy - qz * xz;
        float nx = qw * xx + qx * iw + qy * xz - qz * xy;
        float ny = qw * xy - qx * xz + qy * iw + qz * xx;
        float nz = qw * xz + qx * xy - qy * xx + qz * iw;
        float r00 = nw * nw, r01 = nw * nx, r02 = nw * ny, r03 = nw * nz;
        float r11 = nx * nx, r12 = nx * ny, r13 = nx * nz;
        float r22 = ny * ny, r23 = ny * nz, r33 = nz * nz;
        XIMAT[w * NBODY * 9 + b * 9 + 0] = r00 + r11 - r22 - r33;
        XIMAT[w * NBODY * 9 + b * 9 + 1] = 2.0f * (r12 - r03);
        XIMAT[w * NBODY * 9 + b * 9 + 2] = 2.0f * (r13 + r02);
        XIMAT[w * NBODY * 9 + b * 9 + 3] = 2.0f * (r12 + r03);
        XIMAT[w * NBODY * 9 + b * 9 + 4] = r00 - r11 + r22 - r33;
        XIMAT[w * NBODY * 9 + b * 9 + 5] = 2.0f * (r23 - r01);
        XIMAT[w * NBODY * 9 + b * 9 + 6] = 2.0f * (r13 - r02);
        XIMAT[w * NBODY * 9 + b * 9 + 7] = 2.0f * (r23 + r01);
        XIMAT[w * NBODY * 9 + b * 9 + 8] = r00 - r11 - r22 + r33;
    }
    for (int b = 0; b < NBODY; ++b)
        for (int k = 0; k < 4; ++k)
            XQUAT[w * NBODY * 4 + b * 4 + k] = xq[b * 4 + k];
    for (int j = 0; j < NJNT; ++j)
        for (int k = 0; k < 3; ++k) {
            XANCHOR[w * NJNT * 3 + j * 3 + k] = xa[j * 3 + k];
            XAXIS[w * NJNT * 3 + j * 3 + k] = xs[j * 3 + k];
        }
"""

_MSL = _MSL.replace("""            float ax_ = jpx + qw * sx + (qw * qw - qx * qx - qy * qy - qz * qz) * 0.0f;
            // (use standard rotate: v + 2*qw*(u x v) + 2*u x (u x v))
""", "")

_CACHE: dict = {}


def kinematics_gpu(sim, m):
  nw, nbody, njnt, nq = sim.nworld, int(m.nbody), int(m.njnt), int(m.nq)
  key = (nw, nbody, njnt, nq)
  k = _CACHE.get(key)
  if k is None:
    a = lambda x, dt: mx.array(np.asarray(x), dtype=dt)
    static = [
      a(m.body_parentid, mx.int32), a(m.body_jntadr, mx.int32), a(m.body_jntnum, mx.int32),
      a(m.body_pos, mx.float32), a(m.body_quat, mx.float32), a(m.body_ipos, mx.float32), a(m.body_iquat, mx.float32),
      a(m.jnt_type, mx.int32), a(m.jnt_qposadr, mx.int32), a(m.jnt_axis, mx.float32), a(m.jnt_pos, mx.float32),
      a(m.qpos0, mx.float32),
    ]
    k = mx.fast.metal_kernel(
      name=f"kinematics_{nbody}_{njnt}_{nq}",
      input_names=["qpos", "parent", "jntadr", "jntnum", "bpos", "bquat", "bipos", "biquat",
                   "jtype", "jqposadr", "jaxis", "jpos", "qpos0", "NWORLD"],
      output_names=["XPOS", "XQUAT", "XMAT", "XANCHOR", "XAXIS", "XIPOS", "XIMAT"],
      source=_MSL,
    )
    _CACHE[key] = (k, static)
  k, static = _CACHE[key]
  nwr = mx.array(np.array([nw], np.int32))
  outs = k(
    inputs=[sim.qpos] + static + [nwr],
    template=[("NBODY", nbody), ("NJNT", njnt), ("NQ", nq)],
    grid=((nw + 31) // 32 * 32, 1, 1),
    threadgroup=(32, 1, 1),
    output_shapes=[(nw * nbody * 3,), (nw * nbody * 4,), (nw * nbody * 9,), (nw * njnt * 3,), (nw * njnt * 3,), (nw * nbody * 3,), (nw * nbody * 9,)],
    output_dtypes=[mx.float32] * 7,
  )
  sim.xpos = mx.reshape(outs[0], (nw, nbody, 3))
  sim.xquat = mx.reshape(outs[1], (nw, nbody, 4))
  sim.xmat = mx.reshape(outs[2], (nw, nbody, 3, 3))
  sim.xanchor = mx.reshape(outs[3], (nw, njnt, 3))
  sim.xaxis = mx.reshape(outs[4], (nw, njnt, 3))
  sim.xipos = mx.reshape(outs[5], (nw, nbody, 3))
  sim.ximat = mx.reshape(outs[6], (nw, nbody, 3, 3))
