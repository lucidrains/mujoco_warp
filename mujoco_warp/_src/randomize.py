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
"""Per-world domain randomization.

MJWarp Model fields are batched over worlds (`put_model(..., batch_sizes=...)`),
so randomization needs no scene duplication: one batched render or physics step
covers every randomized variant. These helpers jitter material and light colors
per world while preserving their mean brightness.
"""

import numpy as np

from mujoco_warp._src.types import Model


def _tint(rng: np.random.Generator, shape, amount: float) -> np.ndarray:
  tint = 1.0 + amount * (rng.random(shape).astype(np.float32) * 2.0 - 1.0)
  return tint / tint.mean(axis=-1, keepdims=True)


def randomize_colors(
  m: Model,
  seed: int = 0,
  material_tint: float = 0.4,
  light_tint: float = 0.4,
) -> None:
  """Randomizes per-world material and light colors in place.

  Requires the model to have been created with per-world batch sizes, e.g.
  `put_model(mjm, batch_sizes={"mat_rgba": nworld, "light_diffuse": nworld})`.

  Args:
    m: The model on device (modified in place).
    seed: RNG seed.
    material_tint: Maximum relative per-channel jitter of material colors.
    light_tint: Maximum relative per-channel jitter of light colors.

  Raises:
    ValueError: If the batched fields do not have a per-world row.
  """
  if material_tint < 0.0 or light_tint < 0.0:
    raise ValueError("tint amounts must be non-negative.")
  nworld = m.mat_rgba.shape[0]
  if nworld == 1:
    raise ValueError("mat_rgba has a single row; create the model with batch_sizes={'mat_rgba': nworld}.")
  if m.light_diffuse.shape[0] not in (1, nworld):
    raise ValueError(f"light_diffuse batch {m.light_diffuse.shape[0]} is neither 1 nor nworld ({nworld}).")

  rng = np.random.default_rng(seed)
  rgba = m.mat_rgba.numpy()
  rgba[..., :3] = np.clip(rgba[..., :3] * _tint(rng, (nworld, 1, 3), material_tint), 0.0, 1.0)
  m.mat_rgba.assign(rgba)

  if light_tint > 0.0 and m.light_diffuse.shape[0] == nworld:
    diffuse = m.light_diffuse.numpy()
    diffuse = np.clip(diffuse * _tint(rng, diffuse.shape, light_tint), 0.0, 1.0)
    m.light_diffuse.assign(diffuse)
