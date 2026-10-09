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

import os
import warnings

import warp as wp

_OPS = None
_TRIED = False
_SKIP = set(filter(None, os.environ.get("MUJOCO_WARP_NIM_SKIP", "").split(",")))


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


def get_ops():
  """Return the compiled Nim ops module, or None when unavailable."""
  global _OPS, _TRIED
  if _TRIED:
    return _OPS
  _TRIED = True
  if os.environ.get("MUJOCO_WARP_DISABLE_NIM", ""):
    return None
  if not wp.get_device().is_cpu:
    return None
  try:
    import nimporter_plus  # noqa: F401
  except ImportError:
    return None
  try:
    from mujoco_warp._src import nim_cpu_ops

    _OPS = nim_cpu_ops
  except Exception as err:
    warnings.warn(f"mujoco_warp: Nim CPU fast path disabled: {err}", stacklevel=2)
  return _OPS
