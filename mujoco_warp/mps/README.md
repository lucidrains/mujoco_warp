# MLX Rigid-Body Engine (`mujoco_warp.mps`)

World-batched rigid-body simulation engine written directly on MLX (Metal) for Apple silicon, mirroring the `mujoco_warp` physics pipeline in float32.

## Install

```bash
pip install "mujoco_warp_lucidrains_fork[mlx,nim]"   # or: uv add ...
```

- `mlx` - the MLX (Metal) engine in `mujoco_warp.mps` (Apple silicon).
- `nim` - the Nim CPU fast paths (requires a Nim compiler; the package builds them on first
  import via `nimporter-plus` and falls back to the stock Warp kernels when unavailable).

Both extras are best-effort at runtime: without `mlx` the Warp backends still work, and
without `nim`/a compiler the CPU path uses the stock kernels.

Fused Metal kernels:

1. **K1 (`dyn`)**: Forward kinematics, local geom transforms, subtree CoM, composite rigid body inertia (`cinert`/`crb`), joint-space mass matrix $M$, spatial velocities and accelerations (`cvel`/`cacc`/`cdofdot`), recursive Newton-Euler bias forces (`qfrc_bias`), actuation, passive damping, and fused collision narrowphase + constraint row assembly (joint friction, hinge limits, pyramidal contacts).
2. **K2 (`solve`)**: In-thread or cooperative Cholesky factorization, mask-Newton solver with linesearch, and Euler integration.
3. **Cooperative solve (`coop`)**: One threadgroup (32 threads) cooperates per world using on-chip threadgroup memory (SRAM), with the balanced `v2` factorization. It is now the default through 16,384 worlds (or until the dense cooperative Jacobian would exceed 1 GiB); beyond that the one-thread-per-world kernel with packed rows scales to larger batches with less memory. In an interleaved A/B on the humanoid, v2 is 1.12x faster than v1 at 64 worlds, 1.19x at 1,024, 1.20x at 4,096, and 1.21x at 16,384.

4. **Tree solve (`tree=True`)**: when every active constraint row is supported on one root-to-leaf dof chain (ground contacts, the common locomotion case), `H = M + JᵀDJ` stays tree-sparse. The one-thread solve then stores H packed as `(nv, nch)` and factors it with a leaf-first LDLᵀ instead of dense Cholesky. On the 81-dof `n_humanoid`, the factor uses 24.7x fewer flops and is ~4x faster in wall time; the full step drops from ~34-44 to ~16-20 ms at 64-1024 worlds (2.1-2.3x), and from ~111 to ~65 ms at 4,096 worlds (1.7x, K1 collision dominates there). It is opt-in because a contact row coupling two branches (e.g. limb self-collision) is not represented by the tree pattern; the dense solve remains the default.

Supported scope: free, slide, and hinge joints; collision geoms (primitives, plane-convex meshes, and plugin SDFs incl. metric threaded nut/bolt); pyramidal contacts; Euler integrator; joint motor and position actuators.

Sparse constraint rows: the one-thread-per-world path stores each Jacobian row packed
by its true support (`J[r * MS + k]` for dof `sup_dofs[meta * MS + k]`, row kind in
`jmeta`) - the ancestor dofs of the contact pair, or the single friction/limit dof.
Row assembly in K1 and the `Jaref`/`Jv`/`J^T f`/`J^T D J` loops in K2 then walk 1-15
entries instead of the full `nv` (and skip the dense zero-fill). The cooperative
kernel keeps dense rows: its lane-strided loops are latency-bound, and per-entry dof
gathers measured slower there than the work they remove.

Supported scope: free, slide, and hinge joints; collision geoms (primitives, plane-convex meshes, and plugin SDFs incl. metric threaded nut/bolt); pyramidal contacts; Euler integrator; joint motor and position actuators.

Model tables: the fused dyn kernel reads float tables through a per-world buffer
(`mbf[w * MBS + offset]`, set via `BatchedEngine.set_world_tables`), which is how
per-world ramps (`geom_quat`/`geom_pos` of a plane), payloads (`body_mass`),
actuator gains, friction and gravity are expressed; large static tables
(`mesh_vert`, `mesh_polynormal`) stay shared across worlds in `mbs`, and int
tables live in `mif`.

CPU fast paths: on the Warp CPU backend every kernel runs on a single lane, so the
scalar step pipeline dominates. The Nim worker pool (`_src/nim_cpu_ops.nim`, loaded
best-effort via `nimporter-plus`) mirrors the hot kernels - dense solver, forward
kinematics, subtree CoM/inertias, body velocities, recursive Newton-Euler, NxN
broadphase, primitive narrowphase, contact rows, actuation, passive forces, applied
forces, integration - and reproduces the serial Warp ordering (the narrowphase
computes contacts per pair in parallel, then appends them in pair order, so contacts
are emitted world-major and the downstream efcid assignment is unchanged). The ported
kernels are bit-exact against their Warp counterparts on the humanoid and the full
test suite passes with the fast paths enabled (`MUJOCO_WARP_DISABLE_NIM=1` forces the
stock kernels).

Sparse-Jacobian Newton (`nim_sparse=True`): for models with `nv > 32`, Warp switches
to the sparse constraint layout, which bypasses the dense Nim solver. The opt-in
sparse path densifies each world's sparse J row block into per-thread scratch, then
runs the same Nim Newton/line-search pool, optionally with the tree Hessian factor
(`nim_tree=True`). Set them per model with
`mujoco_warp.put_model(mjm, nim_sparse=True, nim_tree=True)`; the
`MUJOCO_WARP_NIM_SPARSE` / `MUJOCO_WARP_NIM_TREE` env vars remain the fallback when
the constructor values are unset. On the 81-dof `n_humanoid` it is 1.8-2.5x faster
than the stock sparse solver at 64-1024 worlds (e.g. 64 worlds: 21.7 -> 12.1 ms,
256: 73.0 -> 31.9 ms, 1024: 252 -> 102 ms). It is opt-in because its summation order
differs from the stock kernels; the default CPU path remains bit-exact.

The primitive narrowphase fast path covers plane/sphere/capsule/ellipsoid/cylinder/box
pairs and capsule-capsule; a model that can emit any other primitive pair type (mesh
convex, box-box, capsule-box...) keeps the stock kernel, so partial coverage never
changes results.

Plane-convex contacts follow MuJoCo's `mjc_PlaneConvex`: the support point is the
first contact, and up to three more come from the max-area quad (`hull4f`) of the
mesh face incident to the deepest vertex whose normal is most anti-aligned with
the plane normal. Both the Metal kernel and the warp CPU `plane_convex` implement
this, so engine contacts match vanilla MuJoCo on curved meshes (e.g. the
microduck's multi-thousand-vertex sole meshes) rather than only on boxes.

---

## Python API

```python
from mujoco_warp.mps.robot import BatchedEngine

# Initialize batched engine for N worlds
eng = BatchedEngine(mjm, nworld=4096)

# Set state from NumPy arrays (N, nq / nv / nu)
eng.set_state(qpos, qvel, ctrl)

# Step simulation and retrieve NumPy state (evaluation handled internally)
qpos, qvel = eng.step_np(ctrl)
```

Direct stepping in the lazy MLX graph without host synchronization:

```python
eng.step()  # appends to pending MLX graph
```

---

## Validation

Run the whole check (engine tests in both solve layouts + numeric validation, ~12s):

```bash
just check
```

Validate numerical agreement against the `mujoco_warp` CPU backend:

```bash
uv run python -m mujoco_warp.mps.validate
```

Checks on the humanoid model across 100 simulation steps with random controls:
- Trajectory drift remains at float32 noise level (up to $\sim 5 \times 10^{-6}$ max abs error in `qpos`, $\sim 3 \times 10^{-4}$ in `qvel`).
- Active constraint count (`nefc`) and contact detections match bit-for-bit.

---

## Benchmark

Run the benchmark scoreboard:

```bash
uv run python -m mujoco_warp.mps.bench
```

For the high-DOF tree solve (ground-contact scenes):

```bash
uv run python -m mujoco_warp.mps.bench --scene mujoco_warp/test_data/humanoid/n_humanoid.xml --worlds 64,256,1024 --tree
```

The tree path changes the float summation order, so it matches the dense solver closely but is not bit-exact; chaotic trajectories can diverge over hundreds of steps.

### Fairness Methodology
- **Identical Initial State**: Both engines start from the exact same initial state (`qpos0`, zero velocities).
- **Matched Warmup**: Both engines run 5 warmup steps to compile/cache kernels and warm up solver caches, then reset to identical initial states before timing starts.
- **Synchronous Substep Evaluation**: MLX evaluates every substep (`eval_every=1`), ensuring GPU execution and synchronization complete for every step just as Warp CPU steps synchronously.

### Humanoid Benchmark Results (Apple M1 Pro, 10 CPU cores / 16 GPU cores)

Both columns are one (warm-laptop) session of `bench.py` (`--steps 50`), measured
back-to-back. Sustained GPU load throttles the M1 Pro by ~40%, so compare ratios
from runs taken back-to-back. The warp-CPU column runs the Nim fast paths (dense
solver, kinematics, collisions, contact rows, actuation), which are enabled
automatically when `nimporter-plus` and a Nim compiler are present.

```
 worlds | MLX ms/substep | warp-CPU ms/substep | speedup
--------+----------------+---------------------+--------
     64 |           1.31 |                3.21 |     2.4x
    256 |           1.65 |                4.95 |     3.0x
   1024 |           3.36 |               11.71 |     3.5x
   4096 |          10.14 |               41.08 |     4.1x
  16384 |          42.83 |                    - |        -
```

- At 4096 worlds, MLX runs at **~404,000 worlds/s** vs **~100,000 worlds/s** for `mujoco_warp` CPU (4.1x speedup); at 16,384 worlds it sustains **~383,000 worlds/s** (42.8 ms/substep).
- The CPU column improved ~2.1x over the pre-optimization Nim baseline (88.4 ms/substep at
  4096 worlds) by moving the remaining hot scalar kernels (broadphase, contact rows,
  kinematics, inertias, velocities, actuation, integration) into the same worker pool.
- Packed constraint rows speed up the one-thread-per-world path (now used only above
  16,384 worlds or when the cooperative Jacobian would exceed the memory guard):
  same-process interleaved A/B (bit-identical trajectories, min over runs) gives 1.1-1.2x
  at 4,096 worlds and 1.3-1.4x at 16,384 worlds, mostly in K2's row loops; K1 also drops
  its dense row zero-fill. The cooperative `v2` solve is 1.1-1.2x faster than this path
  over the full 1,024-16,384 world range, so it is the default there.
