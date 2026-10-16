# MLX Rigid-Body Engine (`mujoco_warp.mps`)

World-batched rigid-body simulation engine written directly on MLX (Metal) for Apple silicon, mirroring the `mujoco_warp` physics pipeline in float32.

Fused Metal kernels:

1. **K1 (`dyn`)**: Forward kinematics, local geom transforms, subtree CoM, composite rigid body inertia (`cinert`/`crb`), joint-space mass matrix $M$, spatial velocities and accelerations (`cvel`/`cacc`/`cdofdot`), recursive Newton-Euler bias forces (`qfrc_bias`), actuation, passive damping, and fused collision narrowphase + constraint row assembly (joint friction, hinge limits, pyramidal contacts).
2. **K2 (`solve`)**: In-thread or cooperative Cholesky factorization, mask-Newton solver with linesearch, and Euler integration.
3. **Cooperative solve (`coop`)**: For small-to-medium batches ($\le 2048$ worlds), one threadgroup (32 threads) cooperates per world using on-chip threadgroup memory (SRAM), reducing per-step latency ~4-5x. For larger batches ($> 2048$ worlds), the one-thread-per-world kernel maximizes parallel GPU throughput.

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
- Trajectory drift remains at float32 noise level ($\sim 2.6 \times 10^{-6}$ max abs error in `qpos`, $\sim 1.3 \times 10^{-4}$ in `qvel`).
- Active constraint count (`nefc`) and contact detections match bit-for-bit.

---

## Benchmark

Run the benchmark scoreboard:

```bash
uv run python -m mujoco_warp.mps.bench
```

### Fairness Methodology
- **Identical Initial State**: Both engines start from the exact same initial state (`qpos0`, zero velocities).
- **Matched Warmup**: Both engines run 5 warmup steps to compile/cache kernels and warm up solver caches, then reset to identical initial states before timing starts.
- **Synchronous Substep Evaluation**: MLX evaluates every substep (`eval_every=1`), ensuring GPU execution and synchronization complete for every step just as Warp CPU steps synchronously.

### Humanoid Benchmark Results (Apple M1 Pro, 10 CPU cores / 16 GPU cores)

The MLX column is one (warm-laptop) session of `bench.py` (`--steps 50`); the warp-CPU
column is the baseline from the same machine and is unaffected by engine changes. Sustained
GPU load throttles the M1 Pro by ~40%, so compare ratios from runs taken back-to-back.

```
 worlds | MLX ms/substep | warp-CPU ms/substep | speedup
--------+----------------+---------------------+--------
     64 |           1.93 |                9.38 |     4.9x
    256 |           1.98 |               24.09 |    12.2x
   1024 |           3.73 |               88.19 |    23.6x
   4096 |          10.68 |              336.25 |    31.5x
  16384 |          42.50 |                    - |        -
```

- At 4096 worlds, MLX runs at **~385,000 worlds/s** vs **~12,200 worlds/s** for `mujoco_warp` CPU (31.5x speedup); at 16,384 worlds it sustains **~385,000 worlds/s** (42.5 ms/substep).
- Packed constraint rows speed up the one-thread-per-world path (nworld > 2048):
  same-process interleaved A/B (bit-identical trajectories, min over runs) gives 1.1-1.2x
  at 4,096 worlds and 1.3-1.4x at 16,384 worlds, mostly in K2's row loops; K1 also drops
  its dense row zero-fill. The cooperative kernel (nworld <= 2048) is unchanged
  (bit-identical Metal sources).
