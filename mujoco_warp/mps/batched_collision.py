"""World-batched plane/convex collision as a single Metal kernel.

Ports mujoco_warp's `plane_convex` (mesh convex-graph hill climb, exhaustive
fallback) verbatim into MSL, one GPU thread per world. This keeps the exact
selection/tie behaviour of the validated single-world engine while turning
collision into one dispatch for any nworld.
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

_MSL = r"""
    uint w = thread_position_in_grid.x;
    if (w >= NWORLD[0]) return;

    const float _HUGE = 1e6f;

    float3 PL = float3(pl[w * 3 + 0], pl[w * 3 + 1], pl[w * 3 + 2]);
    float3 NN = float3(nrm[w * 3 + 0], nrm[w * 3 + 1], nrm[w * 3 + 2]);
    float3 PP = float3(pos[w * 3 + 0], pos[w * 3 + 1], pos[w * 3 + 2]);
    float3 PN = float3(pn[w * 3 + 0], pn[w * 3 + 1], pn[w * 3 + 2]);

    int indices[4] = {-1, -1, -1, -1};

    if (USE_GRAPH == 0) {
        // exhaustive vertex path
        float max_support = -_HUGE;
        int a_idx = -1;
        float3 a_v = float3(0.0f);
        for (int i = 0; i < VERTNUM; ++i) {
            float3 v = v3at(verts, i);
            float sup = dot(PL - v, NN);
            if (sup > max_support) { max_support = sup; a_idx = i; a_v = v; }
        }
        if (max_support < 0.0f) {
            out_dist[w * 4 + 0] = _HUGE; out_dist[w * 4 + 1] = _HUGE;
            out_dist[w * 4 + 2] = _HUGE; out_dist[w * 4 + 3] = _HUGE;
            return;
        }
        float threshold = max_support - 1e-3f;

        float b_dist = -_HUGE; int b_idx = -1; float3 b_v = float3(0.0f);
        for (int i = 0; i < VERTNUM; ++i) {
            float3 v = v3at(verts, i);
            float sup = dot(PL - v, NN);
            float mask = (sup > threshold) ? 0.0f : -_HUGE;
            float d = dot(a_v - v, a_v - v) + mask;
            if (d > b_dist) { b_dist = d; b_idx = i; b_v = v; }
        }
        float3 ab = cross(NN, a_v - b_v);

        float c_dist = -_HUGE; int c_idx = -1; float3 c_v = float3(0.0f);
        for (int i = 0; i < VERTNUM; ++i) {
            float3 v = v3at(verts, i);
            float sup = dot(PL - v, NN);
            float mask = (sup > threshold) ? 0.0f : -_HUGE;
            float d = abs(dot(a_v - v, ab)) + mask;
            if (d > c_dist) { c_dist = d; c_idx = i; c_v = v; }
        }
        float3 ac = cross(NN, a_v - c_v);
        float3 bc = cross(NN, b_v - c_v);

        float d_dist = -_HUGE; int d_idx = -1;
        for (int i = 0; i < VERTNUM; ++i) {
            float3 v = v3at(verts, i);
            float sup = dot(PL - v, NN);
            float mask = (sup > threshold) ? 0.0f : -_HUGE;
            float d = abs(dot(a_v - v, ac)) + abs(dot(b_v - v, bc)) + mask;
            if (d > d_dist) { d_dist = d; d_idx = i; }
        }
        indices[0] = a_idx; indices[1] = b_idx; indices[2] = c_idx; indices[3] = d_idx;
    } else {
        int numvert = graph[GRAPHADR];
        int vert_edgeadr = GRAPHADR + 2;
        int vert_globalid = GRAPHADR + 2 + numvert;
        int edge_localid = GRAPHADR + 2 + 2 * numvert;

        float max_support = -_HUGE;
        int prev = -1; int imax = 0;
        while (true) {
            prev = imax;
            int i = graph[vert_edgeadr + imax];
            while (graph[edge_localid + i] >= 0) {
                int subidx = graph[edge_localid + i];
                int idx = graph[vert_globalid + subidx];
                float3 v = v3at(verts, idx);
                float sup = dot(PL - v, NN);
                if (sup > max_support) { max_support = sup; imax = subidx; }
                i += 1;
            }
            if (imax == prev) break;
        }
        if (max_support < 0.0f) {
            for (int k = 0; k < 4; ++k) out_dist[w * 4 + k] = _HUGE;
            return;
        }
        float threshold = metal::max(0.0f, max_support - 1e-3f);

        // a
        float a_dist = -_HUGE;
        while (true) {
            prev = imax;
            int i = graph[vert_edgeadr + imax];
            while (graph[edge_localid + i] >= 0) {
                int subidx = graph[edge_localid + i];
                int idx = graph[vert_globalid + subidx];
                float3 v = v3at(verts, idx);
                float sup = dot(PL - v, NN);
                float d = (sup > threshold) ? sup : -_HUGE;
                if (d > a_dist) { a_dist = d; imax = subidx; }
                i += 1;
            }
            if (imax == prev) break;
        }
        int a_g = graph[vert_globalid + imax];
        float3 a_v = v3at(verts, a_g);
        indices[0] = a_g;

        // b
        float b_dist = -_HUGE;
        while (true) {
            prev = imax;
            int i = graph[vert_edgeadr + imax];
            while (graph[edge_localid + i] >= 0) {
                int subidx = graph[edge_localid + i];
                int idx = graph[vert_globalid + subidx];
                float3 v = v3at(verts, idx);
                float sup = dot(PL - v, NN);
                float mask = (sup > threshold) ? 0.0f : -_HUGE;
                float d = dot(a_v - v, a_v - v) + mask;
                if (d > b_dist) { b_dist = d; imax = subidx; }
                i += 1;
            }
            if (imax == prev) break;
        }
        int b_g = graph[vert_globalid + imax];
        float3 b_v = v3at(verts, b_g);
        indices[1] = b_g;
        float3 ab = cross(NN, a_v - b_v);

        // c
        float c_dist = -_HUGE;
        while (true) {
            prev = imax;
            int i = graph[vert_edgeadr + imax];
            while (graph[edge_localid + i] >= 0) {
                int subidx = graph[edge_localid + i];
                int idx = graph[vert_globalid + subidx];
                float3 v = v3at(verts, idx);
                float sup = dot(PL - v, NN);
                float mask = (sup > threshold) ? 0.0f : -_HUGE;
                float d = abs(dot(a_v - v, ab)) + mask;
                if (d > c_dist) { c_dist = d; imax = subidx; }
                i += 1;
            }
            if (imax == prev) break;
        }
        int c_g = graph[vert_globalid + imax];
        float3 c_v = v3at(verts, c_g);
        indices[2] = c_g;
        float3 ac = cross(NN, a_v - c_v);
        float3 bc = cross(NN, b_v - c_v);

        // d
        float d_dist = -_HUGE;
        while (true) {
            prev = imax;
            int i = graph[vert_edgeadr + imax];
            while (graph[edge_localid + i] >= 0) {
                int subidx = graph[edge_localid + i];
                int idx = graph[vert_globalid + subidx];
                float3 v = v3at(verts, idx);
                float sup = dot(PL - v, NN);
                float mask = (sup > threshold) ? 0.0f : -_HUGE;
                float ap = abs(dot(a_v - v, ac));
                float bp = abs(dot(b_v - v, bc));
                float d = ap + bp + mask;
                if (d > d_dist) { d_dist = d; imax = subidx; }
                i += 1;
            }
            if (imax == prev) break;
        }
        int d_g = graph[vert_globalid + imax];
        indices[3] = d_g;
    }

    // emit unique indices, order 3..0, transformed to world frame
    int slot = 0;
    for (int k = 0; k < 4; ++k) { out_dist[w * 4 + k] = _HUGE; }
    for (int ci = 3; ci >= 0; --ci) {
        int idx = indices[ci];
        int count = 0;
        for (int cj = 0; cj <= ci; ++cj) if (indices[cj] == idx) count += 1;
        if (count != 1) continue;
        float3 lv = v3at(verts, idx);
        float3 wp = PP + float3(
            rot[w * 9 + 0] * lv.x + rot[w * 9 + 1] * lv.y + rot[w * 9 + 2] * lv.z,
            rot[w * 9 + 3] * lv.x + rot[w * 9 + 4] * lv.y + rot[w * 9 + 5] * lv.z,
            rot[w * 9 + 6] * lv.x + rot[w * 9 + 7] * lv.y + rot[w * 9 + 8] * lv.z);
        float sup = dot(PL - lv, NN);
        float dist = -sup;
        wp = wp - 0.5f * dist * PN;
        out_dist[w * 4 + slot] = dist;
        out_pos[(w * 4 + slot) * 3 + 0] = wp.x;
        out_pos[(w * 4 + slot) * 3 + 1] = wp.y;
        out_pos[(w * 4 + slot) * 3 + 2] = wp.z;
        slot += 1;
    }
"""

_cache: dict = {}


def prepare_mesh(m, geom: int):
  """Returns (verts mx.array, graph mx.array, graphadr, vertnum, use_graph)."""
  mid = int(m.geom_dataid[geom])
  va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
  verts = np.asarray(m.mesh_vert[va : va + vn], np.float32)
  graph = np.asarray(m.mesh_graph, np.int32)
  graphadr = int(m.mesh_graphadr[mid])
  use_graph = not (graphadr == -1 or vn < 10)
  return mx.array(verts), mx.array(graph), graphadr, vn, use_graph


def build_kernel(name: str, vertnum: int, graphadr: int, use_graph: bool, nworld: int):
  key = (name, vertnum, graphadr, use_graph)
  if key in _cache:
    return _cache[key]
  k = mx.fast.metal_kernel(
    name=name,
    input_names=["verts", "graph", "pl", "nrm", "rot", "pos", "pn", "NWORLD"],
    output_names=["out_dist", "out_pos"],
    source=_MSL,
    header="static inline float3 v3at(const device float* verts, int idx) { return float3(verts[idx*3], verts[idx*3+1], verts[idx*3+2]); }",
  )
  # NWORLD as an input buffer keeps the graph shape-independent
  k = _wrap(k, vertnum, graphadr, use_graph)
  _cache[key] = k
  return k


def _wrap(k, vertnum, graphadr, use_graph):
  def call(verts, graph, pl, nrm, rot, pos, pn, nworld):
    nw = mx.array(np.array([nworld], np.int32))
    return k(
      inputs=[verts, graph, pl, nrm, rot, pos, pn, nw],
      template=[
        ("VERTNUM", vertnum),
        ("GRAPHADR", graphadr),
        ("USE_GRAPH", 1 if use_graph else 0),
      ],
      grid=((nworld + 31) // 32 * 32, 1, 1),
      threadgroup=(32, 1, 1),
      output_shapes=[(nworld * 4,), (nworld * 4 * 3,)],
      output_dtypes=[mx.float32, mx.float32],
    )

  return call


def contacts_batched(m, sim, nworld: int, plane_normal=(0.0, 0.0, 1.0), plane_pos=(0.0, 0.0, 0.0)):
  """Returns (dist (nw, 2, 4), pos (nw, 2, 4, 3)) for foot geoms in model order."""
  feet = [g for g in range(int(m.ngeom)) if int(m.geom_type[g]) == 7 and (int(m.geom_contype[g]) & 1) and (int(m.geom_conaffinity[g]) & 1)]
  pn = mx.broadcast_to(mx.array(np.array(plane_normal, np.float32)), (nworld, 3))
  pp = mx.broadcast_to(mx.array(np.array(plane_pos, np.float32)), (nworld, 3))
  dists, poss = [], []
  for g in feet:
    mid = int(m.geom_dataid[g])
    va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
    verts = mx.array(np.asarray(m.mesh_vert[va : va + vn], np.float32))
    graph = mx.array(np.asarray(m.mesh_graph, np.int32))
    graphadr = int(m.mesh_graphadr[mid])
    use_graph = not (graphadr == -1 or vn < 10)
    body = int(m.geom_bodyid[g])
    rot_b = sim.xmat[:, body]  # (nw,3,3)
    pos_b = sim.xpos[:, body]
    gmat = _quat_to_mat(np.asarray(m.geom_quat[g], np.float64))
    groot = rot_b @ mx.array(gmat.astype(np.float32))[None]
    gpos = pos_b + mx.matmul(rot_b, mx.array(np.asarray(m.geom_pos[g], np.float32))[None, :, None])[:, :, 0]
    nrm = mx.matmul(mx.swapaxes(groot, -1, -2), pn[:, :, None])[:, :, 0]
    pl = mx.matmul(mx.swapaxes(groot, -1, -2), (pp - gpos)[:, :, None])[:, :, 0]
    call = build_kernel(f"plane_convex_v{vn}_{graphadr if use_graph else -1}", vn, graphadr if use_graph else -1, use_graph, nworld)
    d, p = call(verts, graph, pl, nrm, mx.reshape(groot, (nworld, 9)), gpos, pn, nworld)
    dists.append(mx.reshape(d, (nworld, 4)))
    poss.append(mx.reshape(p, (nworld, 4, 3)))
  return mx.stack(dists, axis=1), mx.stack(poss, axis=1)


def _quat_to_mat(q):
  w, x, y, z = q
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ],
    dtype=np.float32,
  )
