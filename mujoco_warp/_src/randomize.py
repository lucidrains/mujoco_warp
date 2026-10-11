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
per world while preserving their mean brightness, randomize physics parameters
per world on device (for sim-to-real robustness training), and reset a subset
of worlds in place.
"""

import dataclasses

import numpy as np
import warp as wp

from mujoco_warp._src.types import Data
from mujoco_warp._src.types import Model
from mujoco_warp._src.types import vec10
from mujoco_warp._src.warp_util import cache_kernel

wp.set_module_options({"enable_backward": False, "default_grid_stride": False})


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


# field -> (group members). Fields listed as "opt.x" live in Model.opt.
_PHYSICS_GROUPS: dict[str, tuple[str, ...]] = {
  "body": ("body_mass", "body_inertia"),
  "mass": ("body_mass",),
  "inertia": ("body_inertia",),
  "friction": ("geom_friction",),
  "solref": ("geom_solref",),
  "damping": ("dof_damping",),
  "armature": ("dof_armature",),
  "frictionloss": ("dof_frictionloss",),
  "stiffness": ("jnt_stiffness",),
  "gain": ("actuator_gainprm", "actuator_biasprm"),
  "gravity": ("opt.gravity",),
}

_DEFAULT_GROUPS = ("body", "friction", "damping", "armature", "frictionloss", "stiffness", "gain", "gravity")

# fields that must move together -> share one jitter draw (one RNG stream)
_SHARED_DRAW = {
  "body_mass": "body",
  "body_inertia": "body",
  "actuator_gainprm": "gain",
  "actuator_biasprm": "gain",
}


def physics_batch_sizes(nworld: int, groups=_DEFAULT_GROUPS) -> dict[str, int]:
  """Returns a `put_model(batch_sizes=...)` dict batching the given randomization groups.

  Args:
    nworld: Number of worlds.
    groups: Group names from `randomize.PhysicsRandomizer`.

  Returns:
    Dict mapping Model field names (including `opt.*`) to `nworld`.
  """
  fields: set[str] = set()
  for group in groups:
    try:
      fields.update(_PHYSICS_GROUPS[group])
    except KeyError:
      raise ValueError(f"unknown physics group {group!r}; expected one of {sorted(_PHYSICS_GROUPS)}.") from None
  return {name: nworld for name in sorted(fields)}


@cache_kernel
def _scale_gravity_kernel():
  @wp.kernel(module="unique")
  def kernel(
    # In:
    base: wp.array[wp.vec3],
    amount: float,
    alpha: wp.array[float],
    seed: int,
    epoch: int,
    stream: int,
    mask: wp.array[bool],
    # Out:
    arr: wp.array[wp.vec3],
  ):
    w = wp.tid()
    if not mask[w]:
      return
    rng = wp.rand_init(seed, (epoch * 73856093) ^ (stream * 19349663) ^ w)
    # one scalar per world -> direction preserved
    arr[w] = base[w] * (1.0 + amount * alpha[w] * (2.0 * wp.randf(rng) - 1.0))

  return kernel


@cache_kernel
def _scale_2d_kernel(kind: str):
  @wp.kernel(module="unique")
  def kernel_f(
    # In:
    base: wp.array2d[float],
    amount: float,
    alpha: wp.array[float],
    seed: int,
    epoch: int,
    stream: int,
    mask: wp.array[bool],
    # Out:
    arr: wp.array2d[float],
  ):
    w, j = wp.tid()
    if not mask[w]:
      return
    rng = wp.rand_init(seed, (epoch * 73856093) ^ (stream * 19349663) ^ (w * arr.shape[1] + j))
    arr[w, j] = base[w, j] * (1.0 + amount * alpha[w] * (2.0 * wp.randf(rng) - 1.0))

  @wp.kernel(module="unique")
  def kernel_v2(
    # In:
    base: wp.array2d[wp.vec2],
    amount: float,
    alpha: wp.array[float],
    seed: int,
    epoch: int,
    stream: int,
    mask: wp.array[bool],
    # Out:
    arr: wp.array2d[wp.vec2],
  ):
    w, j = wp.tid()
    if not mask[w]:
      return
    rng = wp.rand_init(seed, (epoch * 73856093) ^ (stream * 19349663) ^ (w * arr.shape[1] + j))
    arr[w, j] = base[w, j] * (1.0 + amount * alpha[w] * (2.0 * wp.randf(rng) - 1.0))

  @wp.kernel(module="unique")
  def kernel_v3(
    # In:
    base: wp.array2d[wp.vec3],
    amount: float,
    alpha: wp.array[float],
    seed: int,
    epoch: int,
    stream: int,
    mask: wp.array[bool],
    # Out:
    arr: wp.array2d[wp.vec3],
  ):
    w, j = wp.tid()
    if not mask[w]:
      return
    rng = wp.rand_init(seed, (epoch * 73856093) ^ (stream * 19349663) ^ (w * arr.shape[1] + j))
    arr[w, j] = base[w, j] * (1.0 + amount * alpha[w] * (2.0 * wp.randf(rng) - 1.0))

  return {"scalar": kernel_f, "vec2": kernel_v2, "vec3": kernel_v3}[kind]


@cache_kernel
def _scale_vec3_slot_kernel(slot: int):
  """Scales one component of each vec3 element; the other components are untouched."""

  @wp.kernel(module="unique")
  def kernel(
    # In:
    base: wp.array2d[wp.vec3],
    amount: float,
    alpha: wp.array[float],
    seed: int,
    epoch: int,
    stream: int,
    mask: wp.array[bool],
    # Out:
    arr: wp.array2d[wp.vec3],
  ):
    w, j = wp.tid()
    if not mask[w]:
      return
    rng = wp.rand_init(seed, (epoch * 73856093) ^ (stream * 19349663) ^ (w * arr.shape[1] + j))
    s = 1.0 + amount * alpha[w] * (2.0 * wp.randf(rng) - 1.0)
    v = arr[w, j]
    v[wp.static(slot)] = base[w, j][wp.static(slot)] * s
    arr[w, j] = v

  return kernel


@cache_kernel
def _scale_vec10_kernel(slot: int):
  @wp.kernel(module="unique")
  def kernel(
    # In:
    base: wp.array2d[vec10],
    amount: float,
    alpha: wp.array[float],
    seed: int,
    epoch: int,
    stream: int,
    mask: wp.array[bool],
    # Out:
    arr: wp.array2d[vec10],
  ):
    w, j = wp.tid()
    if not mask[w]:
      return
    rng = wp.rand_init(seed, (epoch * 73856093) ^ (stream * 19349663) ^ (w * arr.shape[1] + j))
    s = 1.0 + amount * alpha[w] * (2.0 * wp.randf(rng) - 1.0)  # keyed by element, shared across slots
    v = arr[w, j]
    b = base[w, j]
    if wp.static(slot == 0):
      v[0] = b[0] * s
    else:
      v[1] = b[1] * s
    arr[w, j] = v

  return kernel


@dataclasses.dataclass
class _Entry:
  name: str
  target: object
  nominal: object
  kind: str
  slot: int
  amount: float
  stream: int


class PhysicsRandomizer:
  """Per-world physics randomization anchored to nominal parameter values.

  Construct once (nominal values are snapshotted), then call `sample` at every
  episode reset. Each call draws fresh uniform jitter of relative half-width
  `amount` around the nominal value, so repeated sampling never random-walks.
  Parameters are written on device in a handful of kernels: no host round-trip,
  per-world RNG streams are deterministic in `(seed, epoch, world)`, and an
  optional `worlds` mask resamples only the worlds that just reset.

  Fields are per-world only if they were batched at model creation; pass
  `physics_batch_sizes(nworld, ...)` to `put_model`. A field with batch size 1
  is randomized once, shared by all worlds, and follows world 0's `worlds` mask.

  Group semantics (relative half-width of a uniform multiplicative jitter):
    body: body_mass and body_inertia together (uniform density scaling).
    mass, inertia: the two body quantities separately.
    friction: geom_friction slide component (spin/roll untouched).
    solref: geom_solref (both entries).
    damping, armature, frictionloss, stiffness: matching dof/joint fields.
    gain: actuator force scale: gainprm[0] and biasprm[1] (e.g. motor strength
      or position-actuator kp) to keep gain/bias consistent.
    gravity: gravity magnitude (direction preserved).

  Every world also carries a width multiplier `alpha` (default 1, shape
  (nworld,)); the sampled half-width is `amount * alpha[w]`. `adapt` updates it
  on device from a per-world score, giving automatic domain randomization
  without host round-trips: compute scores on device (a CUDA tensor can be
  wrapped zero-copy with `wp.from_torch`), `adapt(score, worlds=done)`, then
  `sample(seed, epoch, worlds=done)` for the worlds that just reset.
  """

  def __init__(self, m: Model, **amounts: float):
    unknown = set(amounts) - set(_PHYSICS_GROUPS)
    if unknown:
      raise ValueError(f"unknown groups {sorted(unknown)}; expected one of {sorted(_PHYSICS_GROUPS)}.")
    for group, amount in amounts.items():
      if amount < 0.0:
        raise ValueError(f"group {group!r} amount must be non-negative, got {amount}.")

    field_amount: dict[str, float] = {}
    for group, amount in amounts.items():
      if amount == 0.0:
        continue
      for field in _PHYSICS_GROUPS[group]:
        field_amount[field] = max(field_amount.get(field, 0.0), amount)

    # fields sharing a cluster draw the same jitter per (world, element)
    clusters = {_SHARED_DRAW.get(name, name) for name in field_amount}
    streams = {cluster: i for i, cluster in enumerate(sorted(clusters))}

    self.m = m
    self._entries: list[_Entry] = []
    for name, amount in sorted(field_amount.items()):
      stream = streams[_SHARED_DRAW.get(name, name)]
      if name == "opt.gravity":
        target = m.opt.gravity
        if target.ndim != 1:
          raise ValueError(f"{name} must be a 1-D array of vec3, got shape {target.shape}.")
        kind, slot = "gravity", 0
      elif name == "geom_friction":
        target = m.geom_friction
        kind, slot = "vec3_slot", 0
      elif name == "actuator_gainprm":
        target = m.actuator_gainprm
        kind, slot = "vec10", 0
      elif name == "actuator_biasprm":
        target = m.actuator_biasprm
        kind, slot = "vec10", 1
      else:
        target = getattr(m, name)
        if not isinstance(target, wp.array):
          raise ValueError(f"{name} is not an array field.")
        if target.dtype in (float, wp.float32):
          kind, slot = "scalar", 0
        elif target.dtype == wp.vec2:
          kind, slot = "vec2", 0
        elif target.dtype == wp.vec3 and target.ndim == 2:
          kind, slot = "vec3", 0
        else:
          raise ValueError(f"{name} has unsupported dtype/shape {target.dtype} / {target.shape}.")
      base = wp.clone(target)
      self._entries.append(_Entry(name, target, base, kind, slot, amount, stream))

    nw = max((e.target.shape[0] for e in self._entries), default=1)
    self.alpha = wp.ones(nw, dtype=float)

  def sample(self, seed: int = 0, epoch: int = 0, worlds: wp.array | None = None) -> None:
    """Draws fresh jitter around the nominal values (in place, on device).

    Args:
      seed: Base RNG seed (per stream and world).
      epoch: Episode/reset counter; different epochs draw different values.
      worlds: Optional bool mask of worlds to resample; `None` resamples all.
        The draw for a world does not depend on the mask, so a world's values are
        reproducible regardless of which other worlds were resampled. Each
        world's half-width is `group amount * alpha[world]`.
    """
    nw = max((e.target.shape[0] for e in self._entries), default=1)
    for entry in self._entries:
      if entry.target.shape[0] not in (1, nw):
        raise ValueError(
          f"{entry.name} has batch {entry.target.shape[0]}, expected 1 or nworld ({nw}); "
          "create the model with batch_sizes=physics_batch_sizes(nworld)."
        )
    if worlds is not None and len(worlds) != nw:
      raise ValueError(f"worlds mask has length {len(worlds)}, expected {nw}.")
    mask = worlds if worlds is not None else wp.ones(nw, dtype=bool)

    for entry in self._entries:
      args = [entry.nominal, entry.amount, self.alpha, seed, epoch, entry.stream, mask]
      if entry.kind == "gravity":
        wp.launch(_scale_gravity_kernel(), dim=entry.target.shape[0], inputs=args, outputs=[entry.target])
      elif entry.kind == "vec10":
        wp.launch(_scale_vec10_kernel(entry.slot), dim=entry.target.shape, inputs=args, outputs=[entry.target])
      elif entry.kind == "vec3_slot":
        wp.launch(_scale_vec3_slot_kernel(entry.slot), dim=entry.target.shape, inputs=args, outputs=[entry.target])
      else:
        wp.launch(_scale_2d_kernel(entry.kind), dim=entry.target.shape, inputs=args, outputs=[entry.target])

  def adapt(
    self,
    score,
    *,
    worlds: wp.array | None = None,
    expand: float = 1.05,
    shrink: float = 0.95,
    low: float = 0.5,
    high: float = 0.5,
    floor: float = 0.0,
    cap: float = 4.0,
  ) -> None:
    """Updates `alpha` from a per-world performance score, on device.

    A score above `high` multiplies the world's width by `expand` (widen the
    domain), below `low` by `shrink` (narrow it), and in between leaves it
    unchanged. Results are clamped to `[floor, cap]`.

    Args:
      score: Per-world scalar (e.g. success rate or normalized return), shape
        (nworld,); array-like or device array.
      worlds: Optional bool mask of worlds to adapt; `None` adapts all.
      expand: Width multiplier for scores above `high`.
      shrink: Width multiplier for scores below `low`.
      low: Lower score threshold.
      high: Upper score threshold.
      floor: Minimum width multiplier.
      cap: Maximum width multiplier.
    """
    nw = self.alpha.shape[0]
    if not isinstance(score, wp.array):
      score = wp.array(np.asarray(score, dtype=np.float32), dtype=float)
    if score.shape[0] != nw:
      raise ValueError(f"score has length {score.shape[0]}, expected {nw}.")
    if worlds is not None and len(worlds) != nw:
      raise ValueError(f"worlds mask has length {len(worlds)}, expected {nw}.")
    if expand <= 0.0 or shrink <= 0.0:
      raise ValueError(f"expand and shrink must be positive, got {expand}, {shrink}.")
    if low > high:
      raise ValueError(f"expected low <= high, got low={low}, high={high}.")
    if not 0.0 <= floor <= cap:
      raise ValueError(f"expected 0 <= floor <= cap, got floor={floor}, cap={cap}.")
    mask = worlds if worlds is not None else wp.ones(nw, dtype=bool)
    wp.launch(
      _adapt_kernel,
      dim=nw,
      inputs=[score, mask, expand, shrink, low, high, floor, cap],
      outputs=[self.alpha],
    )


@wp.kernel
def _adapt_kernel(
  # In:
  score: wp.array[float],
  mask: wp.array[bool],
  expand: float,
  shrink: float,
  low: float,
  high: float,
  floor: float,
  cap: float,
  # Out:
  alpha: wp.array[float],
):
  w = wp.tid()
  if not mask[w]:
    return
  s = score[w]
  a = alpha[w]
  if s > high:
    a *= expand
  elif s < low:
    a *= shrink
  alpha[w] = wp.min(wp.max(a, floor), cap)


@wp.kernel
def _reset_worlds_kernel(
  # In:
  done: wp.array[bool],
  qpos_init: wp.array2d[float],
  qvel_init: wp.array2d[float],
  ctrl_init: wp.array2d[float],
  # Out:
  time: wp.array[float],
  qpos: wp.array2d[float],
  qvel: wp.array2d[float],
  act: wp.array2d[float],
  ctrl: wp.array2d[float],
  qacc_warmstart: wp.array2d[float],
):
  w, j = wp.tid()
  if not done[w]:
    return
  if j == 0:
    time[w] = 0.0
  if j < qpos.shape[1]:
    qpos[w, j] = qpos_init[w, j]
  if j < qvel.shape[1]:
    qvel[w, j] = qvel_init[w, j]
    qacc_warmstart[w, j] = 0.0
  if j < act.shape[1]:
    act[w, j] = 0.0
  if j < ctrl.shape[1]:
    ctrl[w, j] = ctrl_init[w, j]


def reset_worlds(
  m: Model,
  d: Data,
  done: wp.array,
  qpos_init,
  qvel_init=None,
  ctrl_init=None,
) -> None:
  """Resets the state of the worlds where `done` is True, in place on device.

  Resets `time`, `qpos`, `qvel`, `act`, `ctrl` and `qacc_warmstart` of the
  selected worlds (like `mj_resetData`, with `qpos_init` as `qpos0`).

  Typical RL usage: after stepping, mark finished episodes in `done`, sample
  fresh initial states, call `reset_worlds`, and resample that subset's physics
  with `PhysicsRandomizer.sample(..., worlds=done)` — all without host sync.

  Args:
    m: The model on device.
    d: The data on device (modified in place).
    done: Boolean mask of worlds to reset, shape (nworld,).
    qpos_init: Per-world initial positions, shape (nworld, nq) (host array-like
      or device array).
    qvel_init: Optional per-world initial velocities, shape (nworld, nv).
    ctrl_init: Optional per-world initial controls, shape (nworld, nu).
  """
  nw = d.nworld

  def _as_array(x, shape):
    if x is None:
      return None
    if isinstance(x, wp.array):
      if x.shape != shape:
        raise ValueError(f"expected shape {shape}, got {x.shape}.")
      return x
    return wp.array(np.asarray(x, dtype=np.float32).reshape(shape), dtype=float)

  qpos_init = _as_array(qpos_init, (nw, m.nq))
  qvel_init = _as_array(qvel_init, (nw, m.nv)) if qvel_init is not None else wp.zeros((nw, m.nv), dtype=float)
  ctrl_init = _as_array(ctrl_init, (nw, m.nu)) if ctrl_init is not None else wp.zeros((nw, m.nu), dtype=float)

  max_cols = max(m.nq, m.nv, m.nu, m.na, 1)
  wp.launch(
    _reset_worlds_kernel,
    dim=(nw, max_cols),
    inputs=[done, qpos_init, qvel_init, ctrl_init],
    outputs=[d.time, d.qpos, d.qvel, d.act, d.ctrl, d.qacc_warmstart],
  )


@wp.kernel
def _latency_resample_kernel(
  # In:
  seed: int,
  epoch: int,
  max_delay: int,
  mask: wp.array[bool],
  # Out:
  delays: wp.array[int],
):
  w = wp.tid()
  if not mask[w]:
    return
  rng = wp.rand_init(seed, (epoch * 73856093) ^ (w * 19349663))
  delays[w] = wp.min(int(wp.randf(rng) * float(max_delay + 1)), max_delay)


@wp.kernel
def _latency_reset_kernel(
  # In:
  mask: wp.array[bool],
  # Out:
  age: wp.array[int],
  held: wp.array2d[float],
):
  w, j = wp.tid()
  if not mask[w]:
    return
  if j == 0:
    age[w] = 0
  held[w, j] = 0.0


@wp.kernel
def _latency_clear_kernel(
  # In:
  mask: wp.array[bool],
  # Out:
  buffer: wp.array3d[float],
):
  w, d, j = wp.tid()
  if mask[w]:
    buffer[w, d, j] = 0.0


@wp.kernel
def _latency_hold_kernel(
  # In:
  seed: int,
  step: int,
  max_hold: int,
  # Out:
  age: wp.array[int],
  fresh: wp.array[bool],
):
  w = wp.tid()
  if age[w] <= 0:
    rng = wp.rand_init(seed, (step * 73856093) ^ (w * 19349663))
    age[w] = int(wp.randf(rng) * float(max_hold + 1))  # 0 = fresh every step
    fresh[w] = True
  else:
    age[w] = age[w] - 1
    fresh[w] = False


@wp.kernel
def _latency_apply_kernel(
  # In:
  fresh: wp.array[bool],
  ctrl_in: wp.array2d[float],
  # Out:
  held: wp.array2d[float],
):
  w, j = wp.tid()
  if fresh[w]:
    held[w, j] = ctrl_in[w, j]


@wp.kernel
def _latency_push_kernel(
  # In:
  cursor: int,
  ctrl_in: wp.array2d[float],
  # Out:
  buffer: wp.array3d[float],
):
  w, j = wp.tid()
  buffer[w, cursor, j] = ctrl_in[w, j]


@wp.kernel
def _latency_read_kernel(
  # In:
  cursor: int,
  delays: wp.array[int],
  buffer: wp.array3d[float],
  # Out:
  ctrl_out: wp.array2d[float],
):
  w, j = wp.tid()
  depth = buffer.shape[1]
  idx = (cursor - delays[w] + depth) % depth
  ctrl_out[w, j] = buffer[w, idx, j]


class ActionLatency:
  """Per-world actuation delay and rate jitter, sampled per episode, on device.

  Physical robots apply commands late and with jitter; training without it is a
  classic sim-to-real gap. `apply` writes the command from `delay` steps ago
  into `d.ctrl` (per-world delay in `[0, max_delay]`), optionally holding each
  incoming command for a per-world random number of extra steps
  (`max_hold > 0`, a zero-order-hold rate mismatch). Latency-0/no-hold worlds
  see the command immediately. No host round-trips; deterministic in
  `(seed, epoch, world)`.
  """

  def __init__(self, m: Model, d: Data, max_delay: int = 3, max_hold: int = 0, seed: int = 0):
    """Allocates the ring buffer and initializes zero delay for every world."""
    if max_delay < 0:
      raise ValueError(f"max_delay must be non-negative, got {max_delay}.")
    if max_hold < 0:
      raise ValueError(f"max_hold must be non-negative, got {max_hold}.")
    self.m = m
    self.d = d
    self.max_delay = max_delay
    self.max_hold = max_hold
    self.seed = seed
    self._buffer = wp.zeros((d.nworld, max_delay + 1, m.nu), dtype=float)
    self._delays = wp.zeros(d.nworld, dtype=int)
    self._held = wp.zeros((d.nworld, m.nu), dtype=float)
    self._age = wp.zeros(d.nworld, dtype=int)
    self._fresh = wp.zeros(d.nworld, dtype=bool)
    self._cursor = 0
    self._step = 0

  @property
  def delays(self) -> wp.array:
    """Current per-world delay in steps, shape (nworld,)."""
    return self._delays

  def reset(self, worlds: wp.array | None = None) -> None:
    """Clears the delay pipeline, held commands and hold timers (call at episode reset)."""
    if worlds is not None and len(worlds) != self.d.nworld:
      raise ValueError(f"worlds mask has length {len(worlds)}, expected {self.d.nworld}.")
    mask = worlds if worlds is not None else wp.ones(self.d.nworld, dtype=bool)
    wp.launch(
      _latency_reset_kernel,
      dim=(self.d.nworld, self.m.nu),
      inputs=[mask],
      outputs=[self._age, self._held],
    )
    wp.launch(_latency_clear_kernel, dim=self._buffer.shape, inputs=[mask], outputs=[self._buffer])

  def resample(self, seed: int = 0, epoch: int = 0, worlds: wp.array | None = None) -> None:
    """Redraws per-world delays uniformly in `[0, max_delay]`.

    Args:
      seed: Base RNG seed.
      epoch: Episode/reset counter.
      worlds: Optional bool mask of worlds to resample; `None` resamples all.
    """
    if worlds is not None and len(worlds) != self.d.nworld:
      raise ValueError(f"worlds mask has length {len(worlds)}, expected {self.d.nworld}.")
    mask = worlds if worlds is not None else wp.ones(self.d.nworld, dtype=bool)
    wp.launch(
      _latency_resample_kernel,
      dim=self.d.nworld,
      inputs=[seed, epoch, self.max_delay, mask],
      outputs=[self._delays],
    )

  def apply(self, ctrl: wp.array) -> None:
    """Pushes `ctrl` into the pipeline and writes the delayed/held command to `d.ctrl`."""
    depth = self.max_delay + 1
    cursor = self._cursor % depth
    wp.launch(
      _latency_hold_kernel,
      dim=self.d.nworld,
      inputs=[self.seed, self._step, self.max_hold],
      outputs=[self._age, self._fresh],
    )
    wp.launch(
      _latency_apply_kernel,
      dim=(self.d.nworld, self.m.nu),
      inputs=[self._fresh, ctrl],
      outputs=[self._held],
    )
    wp.launch(_latency_push_kernel, dim=(self.d.nworld, self.m.nu), inputs=[cursor, self._held], outputs=[self._buffer])
    wp.launch(
      _latency_read_kernel,
      dim=(self.d.nworld, self.m.nu),
      inputs=[cursor, self._delays, self._buffer],
      outputs=[self.d.ctrl],
    )
    self._cursor += 1
    self._step += 1


@wp.kernel
def _obs_reset_kernel(
  # In:
  mask: wp.array[bool],
  # Out:
  age: wp.array[int],
  prev: wp.array2d[float],
):
  w, j = wp.tid()
  if not mask[w]:
    return
  if j == 0:
    age[w] = 0
  prev[w, j] = 0.0


@wp.kernel
def _obs_hold_kernel(
  # In:
  seed: int,
  step: int,
  max_hold: int,
  # Out:
  age: wp.array[int],
  fresh: wp.array[bool],
):
  w = wp.tid()
  if age[w] <= 0:
    rng = wp.rand_init(seed, (step * 73856093) ^ (w * 19349663))
    age[w] = int(wp.randf(rng) * float(max_hold + 1))  # 0 = fresh every step
    fresh[w] = True
  else:
    age[w] = age[w] - 1
    fresh[w] = False


@wp.kernel
def _obs_corrupt_kernel(
  # In:
  obs: wp.array2d[float],
  fresh: wp.array[bool],
  noise: float,
  quantum: float,
  dropout: float,
  seed: int,
  step: int,
  # Out:
  prev: wp.array2d[float],
  out: wp.array2d[float],
):
  w, j = wp.tid()
  rng = wp.rand_init(seed, (step * 73856093) ^ (w * 19349663) ^ (j * 83492791))
  if fresh[w] and wp.randf(rng) >= dropout:
    v = obs[w, j]
    if noise > 0.0:
      v = v + noise * wp.randn(rng)
    if quantum > 0.0:
      v = wp.round(v / quantum) * quantum
    prev[w, j] = v
  out[w, j] = prev[w, j]


class ObservationModel:
  """Per-world observation corruption: noise, quantization, dropout, stale holds.

  Deployment sensors are noisy, quantized, occasionally missing (dropout keeps
  the last reading), and sampled at a lower rate than the controller (stale
  holds). `corrupt` applies all of these on device in one pass; `reset` clears
  the stale buffer at episode reset. Noise and dropout draws are deterministic
  in `(seed, step, world, channel)`.
  """

  def __init__(
    self,
    nworld: int,
    dim: int,
    *,
    noise: float = 0.0,
    quantum: float = 0.0,
    dropout: float = 0.0,
    max_hold: int = 0,
    seed: int = 0,
  ):
    """Validates corruption settings and allocates state buffers.

    Args:
      nworld: Number of worlds.
      dim: Observation dimension.
      noise: Gaussian noise standard deviation added to fresh samples.
      quantum: Quantization step (e.g. an encoder resolution); 0 disables.
      dropout: Per-channel probability of keeping the previous reading instead.
      max_hold: Maximum extra steps a fresh sample is held (sampling rate
        mismatch); 0 means a new sample every step.
      seed: Base RNG seed.
    """
    if noise < 0.0 or quantum < 0.0:
      raise ValueError("noise and quantum must be non-negative.")
    if not 0.0 <= dropout <= 1.0:
      raise ValueError(f"dropout must be in [0, 1], got {dropout}.")
    if max_hold < 0:
      raise ValueError(f"max_hold must be non-negative, got {max_hold}.")
    self.nworld = nworld
    self.dim = dim
    self.noise = noise
    self.quantum = quantum
    self.dropout = dropout
    self.max_hold = max_hold
    self.seed = seed
    self._prev = wp.zeros((nworld, dim), dtype=float)
    self._out = wp.zeros((nworld, dim), dtype=float)
    self._age = wp.zeros(nworld, dtype=int)
    self._fresh = wp.zeros(nworld, dtype=bool)
    self._step = 0

  def reset(self, worlds: wp.array | None = None) -> None:
    """Clears stale readings and hold timers (call at episode reset)."""
    if worlds is not None and len(worlds) != self.nworld:
      raise ValueError(f"worlds mask has length {len(worlds)}, expected {self.nworld}.")
    mask = worlds if worlds is not None else wp.ones(self.nworld, dtype=bool)
    wp.launch(_obs_reset_kernel, dim=(self.nworld, self.dim), inputs=[mask], outputs=[self._age, self._prev])

  def corrupt(self, obs: wp.array) -> wp.array:
    """Corrupts `obs` (nworld, dim); returns the internal buffer, valid until the next call."""
    if obs.shape[0] != self.nworld or obs.shape[1] != self.dim:
      raise ValueError(f"obs has shape {obs.shape}, expected ({self.nworld}, {self.dim}).")
    wp.launch(
      _obs_hold_kernel,
      dim=self.nworld,
      inputs=[self.seed, self._step, self.max_hold],
      outputs=[self._age, self._fresh],
    )
    wp.launch(
      _obs_corrupt_kernel,
      dim=(self.nworld, self.dim),
      inputs=[obs, self._fresh, self.noise, self.quantum, self.dropout, self.seed, self._step],
      outputs=[self._prev, self._out],
    )
    self._step += 1
    return self._out
