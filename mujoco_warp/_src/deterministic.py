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
"""Bit-reproducible runs through Warp's deterministic execution modes.

`run_to_run` is bit-exact across repeats on one GPU architecture; `gpu_to_gpu` trades speed for
the same result across architectures; `off` is the default atomic path. Select before the first
kernel launch via `MJWARP_DETERMINISTIC` (read at import) or `set_deterministic`.

Slot loops need scatter headroom (`MJWARP_DETERMINISTIC_MAX_RECORDS` or `max_records=`, e.g. 64):
Warp raises on overflow, except inside CUDA graph capture where reservations are silently dropped.

Gaps: EnableBit.SLEEP wakes race on `tree_asleep`; convex/heightfield CCD needs upstream #1636;
the lowering is CUDA-only; tape gradients through consumed counters need mode `off`.
"""

import os
import sys

import warp as wp

ENV_MODE = "MJWARP_DETERMINISTIC"
ENV_MAX_RECORDS = "MJWARP_DETERMINISTIC_MAX_RECORDS"

_MODE_ALIASES = {
  "off": wp.DeterministicMode.NOT_GUARANTEED,
  "none": wp.DeterministicMode.NOT_GUARANTEED,
  "0": wp.DeterministicMode.NOT_GUARANTEED,
  "false": wp.DeterministicMode.NOT_GUARANTEED,
  "run_to_run": wp.DeterministicMode.RUN_TO_RUN,
  "run-to-run": wp.DeterministicMode.RUN_TO_RUN,
  "1": wp.DeterministicMode.RUN_TO_RUN,
  "true": wp.DeterministicMode.RUN_TO_RUN,
  "gpu_to_gpu": wp.DeterministicMode.GPU_TO_GPU,
  "gpu-to-gpu": wp.DeterministicMode.GPU_TO_GPU,
}


def _normalize(mode: "str | int | wp.DeterministicMode") -> wp.DeterministicMode:
  """Coerces a mode name, int, bool, or `wp.DeterministicMode` into a mode."""
  if isinstance(mode, wp.DeterministicMode):
    return mode
  if isinstance(mode, bool):
    return wp.DeterministicMode.RUN_TO_RUN if mode else wp.DeterministicMode.NOT_GUARANTEED
  if isinstance(mode, int):
    return wp.DeterministicMode(mode)
  if isinstance(mode, str):
    key = mode.strip().lower()
    if key in _MODE_ALIASES:
      return _MODE_ALIASES[key]
  raise ValueError(
    f"invalid deterministic mode {mode!r}; expected one of {sorted(set(_MODE_ALIASES))} or a wp.DeterministicMode"
  )


def get_deterministic() -> wp.DeterministicMode:
  """Current process-wide mode."""
  return wp.config.deterministic


def _kernel_modules() -> dict[int, wp.Module]:
  """Warp modules holding mujoco_warp kernels, `module="unique"` ones included."""
  modules = {}
  for name, module in list(sys.modules.items()):
    if module is None or (name != "mujoco_warp" and not name.startswith("mujoco_warp.")):
      continue
    for obj in vars(module).values():
      if isinstance(obj, wp.Kernel) and obj.module is not None:
        modules[id(obj.module)] = obj.module
  return modules


def set_deterministic(
  mode: "str | int | wp.DeterministicMode",
  *,
  max_records: "int | None" = None,
  force: bool = False,
) -> wp.DeterministicMode:
  """Sets the mode for every mujoco_warp kernel; must run before the first kernel launch.

  Args:
    mode: `"run_to_run"`, `"gpu_to_gpu"`, `"off"`, or a `wp.DeterministicMode`.
    max_records: per-target, per-thread scatter bound (0 = Warp's codegen bound).
    force: recompile loaded modules instead of raising.
  """
  resolved = _normalize(mode)
  if max_records is not None:
    if isinstance(max_records, bool) or int(max_records) < 0:
      raise ValueError(f"max_records must be a non-negative integer, got {max_records!r}")
    wp.config.deterministic_max_records = int(max_records)

  modules = _kernel_modules()
  loaded = sorted(m.name for m in modules.values() if m.execs)
  if loaded and not force:
    raise RuntimeError(
      "deterministic mode must be set before the first kernel launch so every kernel compiles with the same "
      f"lowering; already-loaded modules: {', '.join(loaded)}. Set {ENV_MODE} before importing mujoco_warp, "
      "or pass force=True to recompile (results after the switch point differ from earlier steps)."
    )

  wp.config.deterministic = resolved
  for module in modules.values():
    module.options["deterministic"] = resolved
    module.options["deterministic_max_records"] = wp.config.deterministic_max_records
    module.mark_modified()
  return resolved


def configure_from_env() -> "wp.DeterministicMode | None":
  """Applies `MJWARP_DETERMINISTIC` / `MJWARP_DETERMINISTIC_MAX_RECORDS` at import time."""
  raw = os.environ.get(ENV_MODE)
  if raw is None:
    return None
  resolved = _normalize(raw)
  wp.config.deterministic = resolved
  max_records = os.environ.get(ENV_MAX_RECORDS)
  if max_records is not None:
    wp.config.deterministic_max_records = int(max_records)
  return resolved
