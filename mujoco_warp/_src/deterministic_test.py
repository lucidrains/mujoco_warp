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

"""Tests for deterministic execution mode."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import warp as wp
from absl.testing import absltest

import mujoco_warp as mjwarp
from mujoco_warp._src import sensor
from mujoco_warp._src import visuotactile
from mujoco_warp._src.types import MJ_MAXCONPAIR

_TACTILE_XML = """
<mujoco>
  <option>
    <flag multiccd="enable"/>
  </option>
  <asset>
    <mesh name="sensor_mesh" builtin="sphere" params="0"/>
  </asset>
  <worldbody>
    <body name="sensor_body" pos="0 0 1">
      <freejoint/>
      <geom name="sensor_geom" type="mesh" mesh="sensor_mesh"/>
    </body>
    <body>
      <geom type="box" size=".7 .7 .3"/>
    </body>
  </worldbody>
  <sensor>
    <tactile geom="sensor_geom" mesh="sensor_mesh"/>
  </sensor>
  <keyframe>
    <key qpos="0 0 1 1 0 0 0"/>
  </keyframe>
</mujoco>
"""


def _run(code: str, env: dict | None = None) -> subprocess.CompletedProcess:
  """Runs `code` in a fresh interpreter so Warp binds module options cleanly."""
  subprocess_env = os.environ.copy()
  subprocess_env.pop("MJWARP_DETERMINISTIC", None)
  subprocess_env.pop("MJWARP_DETERMINISTIC_MAX_RECORDS", None)
  if env:
    subprocess_env.update(env)
  return subprocess.run(
    [sys.executable, "-c", textwrap.dedent(code)],
    capture_output=True,
    text=True,
    env=subprocess_env,
    timeout=1800,
  )


def _digest_code(kind: str) -> str:
  """Script template: step a model under the requested deterministic mode and print a digest."""
  if kind == "humanoid":
    setup = 'test_data.fixture("humanoid/humanoid.xml", nworld=64)'
    steps = 60
    state = "d.qpos.numpy().tobytes() + d.qvel.numpy().tobytes()"
  elif kind == "tactile":
    setup = "test_data.fixture(xml=_XML, keyframe=0, nworld=16)"
    steps = 40
    state = "d.qpos.numpy().tobytes() + d.qvel.numpy().tobytes() + d.sensordata.numpy().tobytes()"
  else:
    raise ValueError(kind)

  return f"""
    import hashlib
    import warp as wp
    import mujoco_warp as mjw
    from mujoco_warp import test_data

    assert wp.get_device().is_cuda, "deterministic atomic lowering requires CUDA"
    assert wp.config.deterministic == wp.DeterministicMode.RUN_TO_RUN, wp.config.deterministic

    _XML = {_TACTILE_XML!r}
    _, _, m, d = {setup}
    for _ in range({steps}):
      mjw.step(m, d)
    print(hashlib.sha256({state}).hexdigest())
  """


class DeterministicModeTest(absltest.TestCase):
  """Configuration and mode plumbing."""

  def test_env_configures_before_import(self):
    """MJWARP_DETERMINISTIC is applied to shared and unique modules at import."""
    cp = _run(
      """
      import warp as wp
      import mujoco_warp as mjw
      import mujoco_warp._src.sensor as sensor

      assert wp.config.deterministic == wp.DeterministicMode.RUN_TO_RUN, wp.config.deterministic
      assert wp.config.deterministic_max_records == 64
      shared = wp.get_module("mujoco_warp._src.sensor")
      assert shared.options["deterministic"] == wp.DeterministicMode.RUN_TO_RUN
      assert shared.options["deterministic_max_records"] == 64
      unique = [v for v in vars(sensor).values() if isinstance(v, wp.Kernel)][0]
      assert unique.module.options["deterministic"] == wp.DeterministicMode.RUN_TO_RUN
      assert unique.module.options["deterministic_max_records"] == 64
      """,
      env={"MJWARP_DETERMINISTIC": "run_to_run", "MJWARP_DETERMINISTIC_MAX_RECORDS": "64"},
    )
    self.assertEqual(cp.returncode, 0, cp.stderr)

  def test_default_mode_is_not_guaranteed(self):
    cp = _run(
      """
      import warp as wp
      import mujoco_warp  # noqa: F401
      assert wp.config.deterministic == wp.DeterministicMode.NOT_GUARANTEED
      """
    )
    self.assertEqual(cp.returncode, 0, cp.stderr)

  def test_set_deterministic_before_launch(self):
    """Programmatic selection updates modules that were already imported."""
    cp = _run(
      """
      import warp as wp
      import mujoco_warp as mjw
      import mujoco_warp._src.solver as solver

      mjw.set_deterministic("gpu_to_gpu", max_records=16)
      shared = wp.get_module("mujoco_warp._src.solver")
      assert shared.options["deterministic"] == wp.DeterministicMode.GPU_TO_GPU
      assert shared.options["deterministic_max_records"] == 16
      unique = [v for v in vars(solver).values() if isinstance(v, wp.Kernel)][0]
      assert unique.module.options["deterministic"] == wp.DeterministicMode.GPU_TO_GPU
      assert unique.module.options["deterministic_max_records"] == 16
      """
    )
    self.assertEqual(cp.returncode, 0, cp.stderr)

  def test_set_deterministic_after_launch_raises(self):
    cp = _run(
      """
      import mujoco_warp as mjw
      from mujoco_warp import test_data

      _, _, m, d = test_data.fixture(
        xml='<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
      )
      mjw.step(m, d)
      try:
        mjw.set_deterministic("run_to_run")
      except RuntimeError as e:
        assert "before the first kernel launch" in str(e), e
      else:
        raise AssertionError("expected RuntimeError")
      """
    )
    self.assertEqual(cp.returncode, 0, cp.stderr)

  def test_invalid_mode_raises(self):
    with self.assertRaises(ValueError):
      mjwarp.set_deterministic("bogus")


class DeterministicContactListTest(absltest.TestCase):
  """Contact-list compaction is canonical, independent of atomic scheduling."""

  def _build_list(self, bits: "list[list[int]]"):
    _, list_kernel = sensor._preprocess_tactile_contacts_kernel(0)
    nworld, ntactile, nwords = 1, 1, len(bits)
    seen = wp.zeros((nworld, ntactile, nwords), dtype=wp.uint32)
    seen_np = seen.numpy()
    for word, geom_ids in enumerate(bits):
      for geom in geom_ids:
        seen_np[0, 0, word] |= np.uint32(1) << np.uint32(geom % 32)
    seen.assign(seen_np)

    count = wp.zeros((nworld, ntactile), dtype=int)
    geom_list = wp.full((nworld, ntactile, MJ_MAXCONPAIR), -1, dtype=int)
    overflow = wp.zeros(nworld, dtype=int)
    wp.launch(list_kernel, dim=(nworld, ntactile, nwords), inputs=[seen], outputs=[overflow, count, geom_list])
    return count.numpy(), geom_list.numpy(), overflow.numpy()

  def test_ascending_order(self):
    """Words and bits are scanned ascending, so the list is canonical."""
    bits = [[3, 31], [8]]
    count, geom_list, overflow = self._build_list(bits)
    np.testing.assert_array_equal(geom_list[0, 0, :3], [3, 31, 40])
    np.testing.assert_array_equal(geom_list[0, 0, 3:], -np.ones(MJ_MAXCONPAIR - 3, dtype=int))
    self.assertEqual(int(count[0, 0]), 3)
    self.assertEqual(int(overflow[0]), 0)

  def test_overflow(self):
    """More distinct geoms than MJ_MAXCONPAIR truncates the list and flags overflow."""
    geoms = list(range(MJ_MAXCONPAIR + 1))
    bits = [[g for g in geoms if g // 32 == word] for word in range((MJ_MAXCONPAIR + 33) // 32)]
    count, geom_list, overflow = self._build_list(bits)
    self.assertEqual(int(count[0, 0]), MJ_MAXCONPAIR + 1)
    np.testing.assert_array_equal(geom_list[0, 0], geoms[:MJ_MAXCONPAIR])
    self.assertNotEqual(int(overflow[0]), 0)

  def test_gather_list_ascending_order(self):
    """The visuotactile gather also compacts in ascending geom order."""
    seen = wp.zeros((1, 1, 2), dtype=wp.uint32)
    seen_np = seen.numpy()
    seen_np[0, 0, 0] = (np.uint32(1) << np.uint32(5)) | (np.uint32(1) << np.uint32(2))
    seen_np[0, 0, 1] = np.uint32(1) << np.uint32(7)
    seen.assign(seen_np)

    count = wp.zeros((1, 1), dtype=int)
    cand = wp.full((1, 1, 8), -1, dtype=int)
    wp.launch(visuotactile._contact_gather_list, dim=(1, 1, 2), inputs=[8, seen], outputs=[count, cand])
    np.testing.assert_array_equal(cand.numpy()[0, 0, :3], [2, 5, 39])
    self.assertEqual(int(count.numpy()[0, 0]), 3)


class DeterministicReproducibilityTest(absltest.TestCase):
  """Cross-process bitwise reproducibility (CUDA only)."""

  def _assert_processes_agree(self, kind: str):
    env = {"MJWARP_DETERMINISTIC": "run_to_run", "MJWARP_DETERMINISTIC_MAX_RECORDS": "64"}
    digests = []
    for _ in range(2):
      cp = _run(_digest_code(kind), env=env)
      self.assertEqual(cp.returncode, 0, cp.stderr)
      digests.append(cp.stdout.strip().splitlines()[-1])
    self.assertEqual(digests[0], digests[1], f"{kind} trajectories differ across processes: {digests}")

  @absltest.skipUnless(wp.is_cuda_available(), "requires CUDA")
  def test_humanoid_run_to_run(self):
    self._assert_processes_agree("humanoid")

  @absltest.skipUnless(wp.is_cuda_available(), "requires CUDA")
  def test_tactile_sensors_run_to_run(self):
    """Tactile (taxel) preprocessing compiles and reproduces under a deterministic guarantee."""
    self._assert_processes_agree("tactile")


if __name__ == "__main__":
  absltest.main()
