# microduck on MLX

World-batched implementation of the microduck simulation loop, written directly on
MLX (Metal) and validated against `mujoco_warp` CPU and MuJoCo. It mirrors
mujoco_warp's algorithms, so trajectories track the CPU backends at float32 noise
while running thousands of independent worlds in one process and one GPU.

## Run it (one flag)

```bash
# batched MLX engine (Metal): all worlds in one pass
uv run --with mlx --with mujoco-warp python -m microduck_mlx.run --backend mlx --worlds 4096 --steps 50

# same batch on the mujoco_warp CPU backend
uv run --with mlx --with mujoco-warp python -m microduck_mlx.run --backend cpu --worlds 8 --steps 20

# plain MuJoCo, one world (reported as the world-sequential equivalent)
uv run --with mlx --with mujoco-warp python -m microduck_mlx.run --backend mujoco --worlds 1 --steps 200
```

The default backend can be selected with an environment variable:

```bash
MICRODUCK_BACKEND=mlx uv run --with mlx --with mujoco-warp python -m microduck_mlx.run
```

Full benchmark scoreboard (same machine, Apple silicon):

```bash
uv run --with mlx --with mujoco-warp python -m microduck_mlx.bench_massive
```

## Python API

```python
from microduck_mlx.model import convert
from microduck_mlx.batched_engine import BatchedEngine

eng = BatchedEngine(convert(mjm), nworld=4096)   # one engine, N worlds
eng.set_state(qpos, qvel, ctrl)                  # numpy in: (N, nq/nv/nu)
qpos, qvel = eng.step_np(ctrl)                   # numpy out (eval handled internally)
```

Also `eng.set_ctrl(ctrl)` and `eng.get_state()` if you only want to update one side.

MLX laziness is handled internally: the convenience methods take/return NumPy and
evaluate when needed, so user code never needs `mx.array` or `mx.eval`. Staying in
the lazy MLX graph (e.g. for custom fused kernels) is possible via the raw
`eng.sim.*` arrays, but that is optional.
Single-world, drop-in for `MicroduckGaitEnv`: `from microduck_mlx.env import MicroduckMlxEnv`; `obs = env.reset(seed)`; `obs, reward, term, trunc, info = env.step(action)`.

## Validity

`bstep` compares one full batched step against the validated single-world engine
(which itself matches `mujoco_warp` CPU to ~1e-6): batched vs single-world is
`dq ~= 5.8e-11`, `dv ~= 4.5e-8`.

```bash
uv run --with mlx --with mujoco-warp python /tmp/bstep.py   # see tests below
uv run --with mlx --with mujoco-warp python -m microduck_mlx.batched_validate
```

## Measured (Apple M-series, MLX 0.32, MuJoCo 3.12)

```
CPU reference  | plain MuJoCo, 1 world      :      22.3 us/substep/world
CPU reference  | mujoco_warp CPU, 1 world   :    5305.8 us/substep

  nworld | MLX ms/substep |  us/world |  worlds/s | vs MuJoCo CPU | vs warp CPU
------------------------------------------------------------------------------
      64 |          49.31 |    770.47 |      1298 |          0.0x |      7x
     512 |          53.44 |    104.38 |      9580 |          0.2x |     55x
    2048 |          61.15 |     29.86 |     33489 |          0.8x |    192x
    8192 |         107.90 |     13.17 |     75924 |          1.8x |    435x
   16384 |         195.01 |     11.90 |     84016 |          2.0x |    482x
```

Crossover vs a sequential MuJoCo process is around ~3k worlds; above that the
single MLX process wins per world and keeps scaling with batch. Small batches are
latency-bound by the fixed per-step dispatch floor (~49 ms), which is the target
of the remaining fusion work.

## Files

- `batched.py` — world-batched state/kinematics/com/CRB/velocities/RNE/M/actuation
- `batched_kin.py` — fused MSL forward-kinematics tree pass
- `batched_collision.py` — fused MSL plane/mesh convex-graph collision (exact warp port)
- `batched_contact.py` — fused MSL contact-row assembly (J, D, aref)
- `batched_engine.py` — constraints + mask-based Newton solver (MSL batched Cholesky) + Euler
- `run.py` — one-flag runner (this file's examples)
- `bench_massive.py` — CPU-vs-MLX scoreboard
- `env.py`, `engine.py`, `sim.py` — validated single-world engine and env (reference)
- `batched_validate.py`, `validate_stage1.py`, `validate_step.py` — numeric validation

## Known limits / next steps

- The batched path currently runs the physics loop; the reward/obs env wrapper is
  only wired for the single-world reference env. A batched env (reward/termination
  vectorized over worlds) is needed to drop into EPO training unchanged.
- Contacts cover plane vs convex mesh (feet); no mesh/mesh self-collision, SDF,
  heightfields, equality/tendon or gradients yet.
- Next perf step: block-specialize solver rows (friction + limits are single-DOF,
  only 32 contact rows need dense work) and fuse the solver iteration into MSL;
  expected ~3-6x over sequential MuJoCo at 16k worlds.


## Importing from mujoco_warp

```bash
uv sync --extra mlx
```

```python
from mujoco_warp.mps.batched_engine import BatchedEngine
from mujoco_warp.mps.model import convert

eng = BatchedEngine(convert(mjm), nworld=4096)
eng.set_state(qpos, qvel, ctrl)      # numpy
qpos, qvel = eng.step_np(ctrl)
```

This is an independent MLX engine (not a Warp device backend): Warp itself has no
Metal backend, so the MPS support lives here as a parallel implementation of the
rigid-body pipeline. Wiring it behind the `mujoco_warp` step API (backend switch)
is the next step; for now it is imported explicitly.
