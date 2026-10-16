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

"""Bit-exact equivalence between the Nim CPU fast paths and the pre-existing kernels.

The fast paths added here must not change results: with every one of them disabled
through ``nim_cpu.enabled`` (leaving the earlier solver fast paths in place), stepping
a scene has to produce bit-identical state.  The scenes below cover the paths that have
regressed in the past (contact overflow, composite inertias, adhesion, tendons,
prismatic/free joints); the test also asserts that the fast paths really engaged, so it
cannot pass by falling back to the stock kernels.  Run with ``MUJOCO_WARP_DISABLE_NIM=1``
the whole module skips.
"""

import unittest

import mujoco
import numpy as np
import warp as wp

import mujoco_warp
from mujoco_warp._src import nim_cpu

# every fast path added on top of the pre-existing solver kernels
FAST_PATHS = (
  "kinematics_dense",
  "smooth_com_pos_dense",
  "smooth_com_vel_dense",
  "nxn_broadphase_dense",
  "primitive_narrowphase_dense",
  "efc_contact_init_dense",
  "efc_contact_update_dense",
  "gravity_force_dense",
  "crb_dense",
  "mass_matrix_dense",
  "qfrc_smooth_dense",
  "advance_dense",
  "apply_ft_dense",
  "spring_damper_dense",
  "qfrc_passive_dense",
  "transmission_dense",
  "actuator_velocity_dense",
  "actuator_force_dense",
)

SCENES = (
  "mujoco_warp/test_data/humanoid/humanoid.xml",
  "mujoco_warp/test_data/humanoid/n_humanoid.xml",
  "mujoco_warp/test_data/constraints.xml",
  "mujoco_warp/test_data/actuation/adhesion.xml",
  "mujoco_warp/test_data/tendon/armature.xml",
  # narrowphase overflow: nacon exceeds naconmax and nefc exceeds njmax
  "mujoco_warp/test_data/convex_collision/box100.xml",
  "mujoco_warp/test_data/pendula.xml",
  "mujoco_warp/test_data/jdotv/connect_3d.xml",
)


def _run_scene(scene: str, nworld: int = 3, steps: int = 10, solver_only: bool = False):
  """Step a scene from a fixed random state; returns the state plus fast-path call counts."""
  mjm = mujoco.MjModel.from_xml_path(scene)
  m = mujoco_warp.put_model(mjm)
  d = mujoco_warp.make_data(mjm, nworld=nworld)

  rng = np.random.default_rng(1234)
  mujoco_warp.reset_data(m, d)
  qpos = d.qpos.numpy()
  np.add(qpos, rng.normal(0, 0.01, qpos.shape).astype(np.float32), out=qpos)
  if mjm.nv:
    np.add(d.qvel.numpy(), rng.normal(0, 0.1, d.qvel.numpy().shape).astype(np.float32), out=d.qvel.numpy())
  if mjm.nu:
    d.ctrl.numpy()[:] = rng.uniform(-1, 1, size=(nworld, mjm.nu)).astype(np.float32)

  ops = nim_cpu.get_ops()
  counts = {name: 0 for name in FAST_PATHS}
  originals = {}
  if ops is not None:
    for name in FAST_PATHS:
      fn = getattr(ops, name, None)
      if callable(fn):
        originals[name] = fn

        def make(fn=fn, name=name):
          def wrapper(*args, **kwargs):
            counts[name] += 1
            return fn(*args, **kwargs)

          return wrapper

        setattr(ops, name, make())

  enabled = nim_cpu.enabled
  if solver_only:
    nim_cpu.enabled = lambda name: name not in FAST_PATHS
  try:
    for _ in range(steps):
      mujoco_warp.step(m, d)
    out = (d.qpos.numpy().copy(), d.qvel.numpy().copy())
  finally:
    nim_cpu.enabled = enabled
    for name, fn in originals.items():
      setattr(ops, name, fn)

  return out, counts


class CpuFastPathEquivalenceTest(unittest.TestCase):
  def test_bit_exact_and_engaged(self):
    """Every scene matches the pre-existing kernels bitwise, with the fast paths engaged."""
    if not wp.get_device().is_cpu:
      self.skipTest("CPU backend only")
    if nim_cpu.get_ops() is None:
      self.skipTest("Nim fast paths unavailable")

    for scene in SCENES:
      with self.subTest(scene=scene):
        with_fast, counts = _run_scene(scene, solver_only=False)
        without_fast, _ = _run_scene(scene, solver_only=True)
        engaged = sum(counts.values())
        self.assertGreater(engaged, 0, f"{scene}: no fast path engaged")
        np.testing.assert_array_equal(with_fast[0], without_fast[0], err_msg=f"{scene}: qpos differs")
        np.testing.assert_array_equal(with_fast[1], without_fast[1], err_msg=f"{scene}: qvel differs")


if __name__ == "__main__":
  unittest.main()
