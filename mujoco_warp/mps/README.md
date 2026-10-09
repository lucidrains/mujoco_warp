# MLX Rigid-Body Engine (`mujoco_warp.mps`)

World-batched rigid-body simulation engine written directly on MLX (Metal) for Apple silicon, mirroring the `mujoco_warp` physics pipeline in float32.

Fused Metal kernels:

1. **K1 (`dyn`)**: Forward kinematics, local geom transforms, subtree CoM, composite rigid body inertia (`cinert`/`crb`), joint-space mass matrix $M$, spatial velocities and accelerations (`cvel`/`cacc`/`cdofdot`), recursive Newton-Euler bias forces (`qfrc_bias`), actuation, passive damping, and fused collision narrowphase + constraint row assembly (joint friction, hinge limits, pyramidal contacts).
2. **K2 (`solve`)**: In-thread or cooperative Cholesky factorization, mask-Newton solver with linesearch, and Euler integration.
3. **Cooperative solve (`coop`)**: For small-to-medium batches ($\le 2048$ worlds), one threadgroup (32 threads) cooperates per world using on-chip threadgroup memory (SRAM), reducing per-step latency ~4-5x. For larger batches ($> 2048$ worlds), the one-thread-per-world kernel maximizes parallel GPU throughput.

Supported scope: free, slide, and hinge joints; primitive collision geoms (plane, sphere, capsule); pyramidal contacts; Euler integrator; joint motor and position actuators.

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

```
 worlds | MLX ms/substep | warp-CPU ms/substep | speedup
--------+----------------+---------------------+--------
     64 |           1.73 |                9.38 |     5.4x
    256 |           2.30 |               24.09 |    10.5x
   1024 |           4.59 |               88.19 |    19.2x
   4096 |          12.85 |              336.25 |    26.2x
```

- At 4096 worlds, MLX runs at **~319,000 worlds/s** vs **~12,200 worlds/s** for `mujoco_warp` CPU (**26.2x speedup**).
- At 16,384 worlds, MLX scales to **~330,000 worlds/s** (49.7 ms/substep).
