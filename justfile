# MLX/Metal engine (`mujoco_warp.mps`) checks.
# Both solvers (cooperative and packed rows) are exercised by robot_test.py.

# default: full engine check (parallel tests + warp-CPU validation, ~12s)
default: check

# everything: tests in both solve layouts + numeric parity vs warp-CPU
check:
  uv run pytest mujoco_warp/mps/robot_test.py -q -n 8
  just validate

# engine test suite only (parallel)
test:
  uv run pytest mujoco_warp/mps/robot_test.py -q -n 8

# numeric parity vs the warp-CPU backend (one-step elements, 100-step trajectories)
validate:
  uv run python -m mujoco_warp.mps.validate

# humanoid scoreboard (MLX vs warp-CPU)
bench:
  uv run python -m mujoco_warp.mps.bench

# full repo suite before a PR (slow)
pr:
  uv run pytest -n 8
