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

"""Optional Nim CPU fast paths for the dense solver kernels.

Warp's CPU backend executes ``wp.tile_*`` kernels with a single lane, so the
dense linear algebra kernels dominate the CPU step.  When ``nimporter_plus``
(and a Nim compiler) are available, the same computations run from compiled Nim
loops partitioned across cores.  Import is deferred and best-effort: any missing
dependency or compile failure falls back to the stock Warp kernels.
"""

import importlib.util
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import warp as wp

_OPS = None
_TRIED = False
_SKIP = set(filter(None, os.environ.get("MUJOCO_WARP_NIM_SKIP", "").split(",")))
_TREE_TABLES: dict[tuple, tuple[int, ...]] = {}
_TREE_ARRAYS: dict[tuple, tuple] = {}


def enabled(name: str) -> bool:
  """Whether the given op is not disabled via MUJOCO_WARP_NIM_SKIP."""
  return name not in _SKIP


def dense_worthy(nv_pad: int) -> bool:
  """Whether the dense solver is large enough to beat Warp's CPU tile emulation.

  Warp emulates ``wp.tile_*`` kernels with a single lane per block on CPU.  For
  sub-tile problems the per-call thread wakeup outweighs the kernel win, so the
  Nim path only engages once the dense blocks span a full 16-wide tile.
  """
  return nv_pad >= 16


def ptr(a) -> int:
  """Raw address of a warp array; empty arrays carry a null pointer."""
  return int(a.ptr or 0)


def is_contiguous(a) -> bool:
  """Whether a warp array is densely packed in row-major order."""
  if a is None:
    return False
  expected = wp.types.type_size_in_bytes(a.dtype)
  for dim, stride in zip(reversed(a.shape), reversed(a.strides)):
    if dim != 1 and stride != expected:
      return False
    expected *= dim
  return True


def layout(a) -> tuple[int, int]:
  """(leading-dim count, row stride in 4-byte scalars) for a contiguous warp table.

  Vector and matrix types are counted per component, so a row of vec3 entries is three
  scalars wide.  Scalars and 1D tables report (1, len).
  """
  if a.ndim < 2:
    return (1, int(a.shape[0]))
  return (int(a.shape[0]), int(a.strides[0]) // 4)


def _import_nim_ops():
  from mujoco_warp._src import nim_cpu_ops

  return nim_cpu_ops


def _compile_nim_ops():
  """Compile nim_cpu_ops explicitly when nimporter's import hook cannot find it.

  nimporter-plus reconstructs the source path from ``sys.path``, which fails for
  editable installs imported from outside the repository root.  Its exporter can
  be pointed at the package root (the directory containing ``mujoco_warp/``).
  """
  from nimporter_plus import nimport

  fullname = "mujoco_warp._src.nim_cpu_ops"
  root = str(Path(__file__).resolve().parents[2])
  spec = nimport(fullname, [root], library=False)
  if spec is None:
    raise ImportError(f"nimporter-plus could not locate {fullname}.nim under {root}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[fullname] = module
  try:
    spec.loader.exec_module(module)
  except Exception:
    sys.modules.pop(fullname, None)
    raise
  return module


def get_ops():
  """Return the compiled Nim ops module, or None when unavailable."""
  global _OPS, _TRIED
  if not wp.get_device().is_cpu:
    return None
  if os.environ.get("MUJOCO_WARP_DISABLE_NIM", ""):
    return None
  if getattr(getattr(wp._src.context, "runtime", None), "tape", None) is not None:
    return None
  if _TRIED:
    return _OPS
  _TRIED = True
  try:
    import nimporter_plus  # noqa: F401
  except ImportError:
    return None
  try:
    _OPS = _import_nim_ops()
  except Exception as import_err:
    try:
      _OPS = _compile_nim_ops()
    except Exception as err:
      warnings.warn(
        f"mujoco_warp: Nim CPU fast path disabled: {err} (import: {import_err})", stacklevel=2
      )
  return _OPS


def _enabled(m, attr: str, env: str) -> bool:
  value = getattr(m, attr, None) if m is not None else None
  if value is not None:
    return bool(value)
  return os.environ.get(env, "") not in ("", "0")


def tree_enabled(m=None) -> bool:
  """Whether the opt-in tree-sparse Newton factorization is requested."""
  return _enabled(m, "_nim_tree", "MUJOCO_WARP_NIM_TREE")


def sparse_enabled(m=None) -> bool:
  """Whether the opt-in Nim sparse-Jacobian Newton solver is requested."""
  return _enabled(m, "_nim_sparse", "MUJOCO_WARP_NIM_SPARSE")


def tree_tables(m) -> tuple[int, ...]:
  """Device addresses of (parent, descendant CSR, ancestor mask, depth)."""
  if not tree_enabled(m):
    return (0, 0, 0, 0, 0)
  parent = m.dof_parentid.numpy()
  key = (int(m.nv), int(m.dof_parentid.ptr))
  cached = _TREE_TABLES.get(key)
  if cached is not None:
    return cached
  nv = int(m.nv)
  desc = [[] for _ in range(nv)]
  depth = np.zeros(nv, np.int32)
  anc = np.zeros((nv, nv), np.int8)
  for q in range(nv):
    p = int(parent[q])
    while p >= 0:
      desc[p].append(q)
      depth[q] += 1
      anc[q, p] = 1
      p = int(parent[p])
    anc[q, q] = 1
  ptr = np.zeros(nv + 1, np.int32)
  idx: list[int] = []
  for i, row in enumerate(desc):
    idx.extend(row)
    ptr[i + 1] = len(idx)
  ptr_arr = wp.array(ptr, dtype=int, device="cpu")
  idx_arr = wp.array(np.asarray(idx, np.int32), dtype=int, device="cpu")
  anc_arr = wp.array(anc.reshape(-1), dtype=wp.int8, device="cpu")
  dep_arr = wp.array(depth, dtype=int, device="cpu")
  _TREE_ARRAYS[key] = (ptr_arr, idx_arr, anc_arr, dep_arr)
  _TREE_TABLES[key] = (
    int(m.dof_parentid.ptr),
    int(ptr_arr.ptr),
    int(idx_arr.ptr),
    int(anc_arr.ptr),
    int(dep_arr.ptr),
  )
  return _TREE_TABLES[key]
