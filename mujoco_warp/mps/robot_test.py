"""Targeted regression and validation tests for the MLX batched engine."""

from __future__ import annotations

import subprocess
import sys

import mujoco
import numpy as np
import pytest

import mujoco_warp

mx = pytest.importorskip("mlx.core")
from mujoco_warp.mps.robot import BatchedEngine  # noqa: E402


def test_empty_limits_rig_init():
  """Verifies that models without joint limits initialize without wp.array TypeError."""
  xml = """
  <mujoco>
    <option timestep="0.005" integrator="Euler"/>
    <worldbody>
      <body name="body1" pos="0 0 0">
        <joint type="free"/>
        <geom type="sphere" size="0.05"/>
      </body>
    </worldbody>
  </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  eng = BatchedEngine(mjm, nworld=2)
  assert eng.rig.nlim == 0
  assert len(eng.rig.limited_joints) == 0


def test_capsule_capsule_contact():
  """Regression test for non-parallel capsule-capsule collisions and contact normals."""
  xml = """
  <mujoco>
    <option timestep="0.005" integrator="Euler"/>
    <worldbody>
      <body name="cap1" pos="0 0 0">
        <joint type="free"/>
        <geom type="capsule" size="0.05 0.2" zaxis="1 0 0"/>
      </body>
      <body name="cap2" pos="0 0 0.08">
        <joint type="free"/>
        <geom type="capsule" size="0.05 0.2" zaxis="0 1 0"/>
      </body>
    </worldbody>
  </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  m = mujoco_warp.put_model(mjm)
  d = mujoco_warp.make_data(mjm, nworld=1)
  mujoco_warp.forward(m, d)

  eng = BatchedEngine(mjm, nworld=1)
  eng.set_state(d.qpos.numpy(), d.qvel.numpy())
  eng._dyn_dispatch(kin=True)

  assert int(eng.ncon[0]) == int(d.nacon.numpy()[0]) == 1
  assert int(eng.nefc[0]) == int(d.nefc.numpy()[0]) == 4

  j_warp = d.efc.J.numpy()[0, : int(d.nefc.numpy()[0]), : mjm.nv]
  j_mlx = np.array(eng.J)[0, : int(eng.nefc[0]), : mjm.nv]
  np.testing.assert_allclose(j_mlx, j_warp, atol=1e-5)


def test_sphere_capsule_contact():
  """Regression test for sphere-capsule contact normal and Jacobian sign."""
  xml = """
  <mujoco>
    <option timestep="0.005" integrator="Euler"/>
    <worldbody>
      <body name="sph1" pos="0 0 0">
        <joint type="free"/>
        <geom type="sphere" size="0.05"/>
      </body>
      <body name="cap2" pos="0 0 0.08">
        <joint type="free"/>
        <geom type="capsule" size="0.05 0.2" zaxis="1 0 0"/>
      </body>
    </worldbody>
  </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  m = mujoco_warp.put_model(mjm)
  d = mujoco_warp.make_data(mjm, nworld=1)
  mujoco_warp.forward(m, d)

  eng = BatchedEngine(mjm, nworld=1)
  eng.set_state(d.qpos.numpy(), d.qvel.numpy())
  eng._dyn_dispatch(kin=True)

  assert int(eng.ncon[0]) == int(d.nacon.numpy()[0]) == 1
  assert int(eng.nefc[0]) == int(d.nefc.numpy()[0]) == 4

  j_warp = d.efc.J.numpy()[0, : int(d.nefc.numpy()[0]), : mjm.nv]
  j_mlx = np.array(eng.J)[0, : int(eng.nefc[0]), : mjm.nv]
  np.testing.assert_allclose(j_mlx, j_warp, atol=1e-5)


def test_condim1_frictionless_contact():
  """Regression test verifying condim=1 contact rows do not include tangential friction."""
  xml = """
  <mujoco>
    <option timestep="0.005" integrator="Euler"/>
    <worldbody>
      <body name="sph1" pos="0 0 0">
        <joint type="free"/>
        <geom type="sphere" size="0.05" condim="1"/>
      </body>
      <body name="cap2" pos="0 0 0.08">
        <joint type="free"/>
        <geom type="capsule" size="0.05 0.2" zaxis="1 0 0" condim="1"/>
      </body>
    </worldbody>
  </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  m = mujoco_warp.put_model(mjm)
  d = mujoco_warp.make_data(mjm, nworld=1)
  mujoco_warp.forward(m, d)

  eng = BatchedEngine(mjm, nworld=1)
  eng.set_state(d.qpos.numpy(), d.qvel.numpy())
  eng._dyn_dispatch(kin=True)

  assert int(eng.ncon[0]) == int(d.nacon.numpy()[0]) == 1
  assert int(eng.nefc[0]) == int(d.nefc.numpy()[0]) == 1

  j_warp = d.efc.J.numpy()[0, : int(d.nefc.numpy()[0]), : mjm.nv]
  j_mlx = np.array(eng.J)[0, : int(eng.nefc[0]), : mjm.nv]
  np.testing.assert_allclose(j_mlx, j_warp, atol=1e-5)


def test_humanoid_one_step_elements():
  """Verifies one-step dynamics elements match Warp CPU with active ground contacts."""
  scene = "mujoco_warp/test_data/humanoid/humanoid.xml"
  mjm = mujoco.MjModel.from_xml_path(scene)
  m = mujoco_warp.put_model(mjm)
  d = mujoco_warp.make_data(mjm, nworld=1)

  qpos = np.tile(np.array(mjm.qpos0, np.float32), (1, 1))
  qvel = np.zeros((1, mjm.nv), np.float32)
  ctrl = np.zeros((1, mjm.nu), np.float32)

  d.qpos.numpy()[:] = qpos
  d.qvel.numpy()[:] = qvel
  d.ctrl.numpy()[:] = ctrl
  mujoco_warp.forward(m, d)

  eng = BatchedEngine(mjm, nworld=1)
  eng.set_state(qpos, qvel, ctrl)
  eng.step()

  assert int(eng.nefc[0]) == int(d.nefc.numpy()[0]) == 32
  np.testing.assert_allclose(np.array(eng.smooth)[0], d.qfrc_smooth.numpy()[0], atol=1e-4)
  np.testing.assert_allclose(np.array(eng.qfrc_bias)[0], d.qfrc_bias.numpy()[0], atol=1e-4)
  np.testing.assert_allclose(np.array(eng.qacc)[0], d.qacc.numpy()[0], atol=1e-2)


def test_humanoid_contact_trajectory():
  """Verifies multi-step humanoid trajectory across ground contacts matches Warp CPU."""
  scene = "mujoco_warp/test_data/humanoid/humanoid.xml"
  mjm = mujoco.MjModel.from_xml_path(scene)
  m = mujoco_warp.put_model(mjm)
  d = mujoco_warp.make_data(mjm, nworld=1)

  rng = np.random.default_rng(0)
  qpos = np.tile(np.array(mjm.qpos0, np.float32), (1, 1))
  qpos[:, :3] += rng.normal(0, 0.005, (1, 3)).astype(np.float32)
  qvel = np.zeros((1, mjm.nv), np.float32)
  ctrl = rng.uniform(-0.3, 0.3, (1, mjm.nu)).astype(np.float32)

  d.qpos.numpy()[:] = qpos
  d.qvel.numpy()[:] = qvel
  d.ctrl.numpy()[:] = ctrl

  eng = BatchedEngine(mjm, nworld=1)
  eng.set_state(qpos, qvel, ctrl)

  for step in range(1, 51):
    mujoco_warp.step(m, d)
    eng.step()

  qm, vm = eng.get_state()
  err_q = float(np.abs(qm - d.qpos.numpy()).max())
  err_v = float(np.abs(vm - d.qvel.numpy()).max())

  assert err_q < 1e-4, f"qpos diverged at step 50: {err_q}"
  assert err_v < 1e-3, f"qvel diverged at step 50: {err_v}"


def _plane_box_mesh_model():
  """Box (inline mesh) free body on a plane; exercises the plane-convex narrowphase."""
  verts = " ".join(f"{x} {y} {z}" for x in (-0.05, 0.05) for y in (-0.05, 0.05) for z in (-0.05, 0.05))
  xml = f"""
  <mujoco>
    <option timestep="0.005" integrator="Euler"/>
    <asset><mesh name="box" vertex="{verts}"/></asset>
    <worldbody>
      <geom type="plane" size="1 1 0.1" pos="0 0 0"/>
      <body name="b" pos="0 0 0.049">
        <joint type="free"/>
        <geom type="mesh" mesh="box"/>
      </body>
    </worldbody>
  </mujoco>
  """
  return mujoco.MjModel.from_xml_string(xml)


def test_plane_mesh_contact_one_step():
  """Verifies plane-convex (mesh) contact rows match Warp CPU exactly."""
  mjm = _plane_box_mesh_model()
  m = mujoco_warp.put_model(mjm)
  d = mujoco_warp.make_data(mjm, nworld=1)
  qpos = np.array(mjm.qpos0, np.float32)[None]
  qvel = np.zeros((1, mjm.nv), np.float32)
  d.qpos.numpy()[:] = qpos
  d.qvel.numpy()[:] = qvel
  mujoco_warp.forward(m, d)

  eng = BatchedEngine(mjm, nworld=1)
  eng.set_state(qpos, qvel)
  eng._dyn_dispatch(kin=True)

  assert int(eng.ncon[0]) == int(d.nacon.numpy()[0]) == 4
  assert int(eng.nefc[0]) == int(d.nefc.numpy()[0]) == 16
  assert np.array(eng.pair_conct)[0, 0] == 4

  n = int(d.nefc.numpy()[0])
  np.testing.assert_allclose(np.array(eng.J)[0, :n], d.efc.J.numpy()[0, :n, : mjm.nv], atol=1e-6)
  np.testing.assert_allclose(np.array(eng.aref)[0, :n], d.efc.aref.numpy()[0, :n], atol=1e-6)
  np.testing.assert_allclose(np.array(eng.D)[0, :n], d.efc.D.numpy()[0, :n], atol=1e-6)


def test_plane_mesh_trajectory():
  """Verifies multi-step trajectory of a mesh box on a plane matches Warp CPU."""
  mjm = _plane_box_mesh_model()
  m = mujoco_warp.put_model(mjm)
  d = mujoco_warp.make_data(mjm, nworld=1)

  rng = np.random.default_rng(0)
  qpos = np.array(mjm.qpos0, np.float32)[None]
  qpos[:, 2] += 0.004
  qvel = rng.normal(0, 0.05, (1, mjm.nv)).astype(np.float32)
  d.qpos.numpy()[:] = qpos
  d.qvel.numpy()[:] = qvel

  eng = BatchedEngine(mjm, nworld=1)
  eng.set_state(qpos, qvel)
  for _ in range(50):
    mujoco_warp.step(m, d)
    eng.step()

  qm, vm = eng.get_state()
  assert np.abs(qm - d.qpos.numpy()).max() < 1e-5
  assert np.abs(vm - d.qvel.numpy()).max() < 1e-4


def test_mesh_mesh_pairs_skipped():
  """Verifies unsupported convex-convex pairs are skipped, not fatal."""
  verts = " ".join(f"{x} {y} {z}" for x in (-0.05, 0.05) for y in (-0.05, 0.05) for z in (-0.05, 0.05))
  xml = f"""
  <mujoco>
    <option timestep="0.005" integrator="Euler"/>
    <asset><mesh name="box" vertex="{verts}"/></asset>
    <worldbody>
      <body name="b1" pos="0 0 0.05"><joint type="free"/><geom type="mesh" mesh="box"/></body>
      <body name="b2" pos="0 0 0.14"><joint type="free"/><geom type="mesh" mesh="box"/></body>
    </worldbody>
  </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  eng = BatchedEngine(mjm, nworld=1)
  assert eng.rig.npairc == 0
  assert eng.rig.skipped_mesh_pairs == 1


def test_coop_matches_single_thread_solver():
  """Verifies the cooperative solver matches the single-thread solver over 50 steps."""
  scene = "mujoco_warp/test_data/humanoid/humanoid.xml"
  mjm = mujoco.MjModel.from_xml_path(scene)

  rng = np.random.default_rng(0)
  qpos = np.tile(np.array(mjm.qpos0, np.float32), (1, 1))
  qpos[:, :3] += rng.normal(0, 0.005, (1, 3)).astype(np.float32)
  qvel = rng.normal(0, 0.01, (1, mjm.nv)).astype(np.float32)
  ctrl = rng.uniform(-0.3, 0.3, (1, mjm.nu)).astype(np.float32)

  eng_coop = BatchedEngine(mjm, nworld=1, coop=True)
  eng_single = BatchedEngine(mjm, nworld=1, coop=False)
  assert eng_coop.coop and not eng_single.coop

  eng_coop.set_state(qpos, qvel, ctrl)
  eng_single.set_state(qpos, qvel, ctrl)
  for _ in range(50):
    eng_coop.step()
    eng_single.step()

  qc, vc = eng_coop.get_state()
  qs, vs = eng_single.get_state()
  assert np.abs(qc - qs).max() < 1e-4
  assert np.abs(vc - vs).max() < 1e-3


def test_coop_wide_model_lane_striding():
  """Regression test for cooperative solve with nv > threadgroup (rows handled in strides)."""
  chain = '<body pos="0.04 0 0"><joint type="hinge"/><geom type="capsule" size="0.02 0.03"/>'
  xml = '<mujoco><option integrator="Euler"/><worldbody><body><joint type="free"/><geom type="capsule" size="0.02 0.05"/>'
  xml += chain * 34 + "</body>" * 35 + "</worldbody></mujoco>"
  mjm = mujoco.MjModel.from_xml_string(xml)
  assert mjm.nv > 32

  rng = np.random.default_rng(0)
  qpos = np.tile(np.array(mjm.qpos0, np.float32), (2, 1))
  qvel = rng.normal(0, 0.02, (2, mjm.nv)).astype(np.float32)

  eng_coop = BatchedEngine(mjm, nworld=2, coop=True)
  eng_single = BatchedEngine(mjm, nworld=2, coop=False)
  assert eng_coop.coop
  eng_coop.set_state(qpos, qvel)
  eng_single.set_state(qpos, qvel)
  for _ in range(10):
    eng_coop.step()
    eng_single.step()

  qc, vc = eng_coop.get_state()
  qs, vs = eng_single.get_state()
  assert np.isfinite(qc).all() and np.isfinite(vc).all()
  assert np.abs(qc - qs).max() < 1e-3
  assert np.abs(vc - vs).max() < 1e-2


def test_parallel_capsule_capsule_contact():
  """Regression test for parallel-axis capsule-capsule collisions (two-point branch)."""
  xml = """
  <mujoco>
    <option timestep="0.005" integrator="Euler"/>
    <worldbody>
      <body name="cap1" pos="0 0 0">
        <joint type="free"/>
        <geom type="capsule" size="0.05 0.2" zaxis="0 0 1"/>
      </body>
      <body name="cap2" pos="0.06 0 0.05">
        <joint type="free"/>
        <geom type="capsule" size="0.05 0.2" zaxis="0 0 1"/>
      </body>
    </worldbody>
  </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  m = mujoco_warp.put_model(mjm)
  d = mujoco_warp.make_data(mjm, nworld=1)
  mujoco_warp.forward(m, d)

  eng = BatchedEngine(mjm, nworld=1)
  eng.set_state(d.qpos.numpy(), d.qvel.numpy())
  eng._dyn_dispatch(kin=True)

  assert int(eng.ncon[0]) == int(d.nacon.numpy()[0]) == 2
  assert int(eng.nefc[0]) == int(d.nefc.numpy()[0]) == 8

  j_warp = d.efc.J.numpy()[0, : int(d.nefc.numpy()[0]), : mjm.nv]
  j_mlx = np.array(eng.J)[0, : int(eng.nefc[0]), : mjm.nv]
  np.testing.assert_allclose(j_mlx, j_warp, atol=1e-5)


def test_explicit_pair_material_params():
  """Verifies explicit <pair> overrides and geom material mixing match Warp CPU."""
  xml = """
  <mujoco>
    <option timestep="0.005" integrator="Euler" cone="pyramidal"/>
    <worldbody>
      <body name="a" pos="0 0 0">
        <geom name="ga" type="sphere" size="0.1" margin="0.01" gap="0.02" solmix="0.3"
          solref="0.03 0.5" friction="1.2 0.02 0.001" priority="1"/>
      </body>
      <body name="b" pos="0 0 0.24">
        <joint type="free"/>
        <geom name="gb" type="capsule" size="0.05 0.1" margin="0.005" solmix="0.7"
          friction="0.8 0.01 0.0005"/>
      </body>
    </worldbody>
    <contact>
      <pair geom1="ga" geom2="gb" condim="3" friction="0.5 0.01 0.001" solref="0.02 0.9"
        solimp="0.8 0.9 0.001 0.5 2" margin="0.03"/>
    </contact>
  </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  m = mujoco_warp.put_model(mjm)
  d = mujoco_warp.make_data(mjm, nworld=1)
  mujoco_warp.forward(m, d)

  eng = BatchedEngine(mjm, nworld=1)
  eng.set_state(d.qpos.numpy(), d.qvel.numpy())
  eng._dyn_dispatch(kin=True)

  nefc = int(eng.nefc[0])
  assert nefc == int(d.nefc.numpy()[0]) == 4
  np.testing.assert_allclose(np.array(eng.D)[0, :nefc], d.efc.D.numpy()[0, :nefc], atol=1e-5)
  np.testing.assert_allclose(np.array(eng.aref)[0, :nefc], d.efc.aref.numpy()[0, :nefc], atol=1e-4)
  np.testing.assert_allclose(np.array(eng.J)[0, :nefc, : mjm.nv], d.efc.J.numpy()[0, :nefc, : mjm.nv], atol=1e-5)


_UNSUPPORTED_MODELS = {
  "gravcomp": "<worldbody><body gravcomp='1'><joint type='hinge'/><geom type='capsule' size='0.02 0.1'/></body></worldbody>",
  "stiffness_poly": (
    "<worldbody><body><joint type='hinge' stiffness='5 0.3 0.02'/><geom type='capsule' size='0.02 0.1'/></body></worldbody>"
  ),
  "damping_poly": (
    "<worldbody><body><joint type='hinge' damping='0.1 0.2 0.3'/><geom type='capsule' size='0.02 0.1'/></body></worldbody>"
  ),
  "actuator_damping": (
    "<worldbody><body><joint name='h' type='hinge'/><geom type='capsule' size='0.02 0.1'/></body></worldbody>"
    "<actuator><motor joint='h' damping='0.5'/></actuator>"
  ),
  "fluid": (
    "<worldbody><body><joint type='free'/>"
    "<geom type='ellipsoid' size='0.05 0.06 0.07' fluidshape='ellipsoid'/></body></worldbody>"
  ),
  "adhesion": "<worldbody><body><joint type='free'/><geom type='sphere' size='0.05' adhesion='0.1'/></body></worldbody>",
  "surfacevel": (
    "<worldbody><body><joint type='free'/><geom type='sphere' size='0.05' surfacevel='0.1 0 0'/></body></worldbody>"
  ),
  "sleep": (
    "<option><flag sleep='enable'/></option>"
    "<worldbody><body><joint type='hinge'/><geom type='capsule' size='0.02 0.1'/></body></worldbody>"
  ),
  "gravity_disable": (
    "<option><flag gravity='disable'/></option>"
    "<worldbody><body><joint type='hinge'/><geom type='capsule' size='0.02 0.1'/></body></worldbody>"
  ),
  "solver_cg": (
    "<option solver='CG'/><worldbody><body><joint type='hinge'/><geom type='capsule' size='0.02 0.1'/></body></worldbody>"
  ),
}


@pytest.mark.parametrize("name", sorted(_UNSUPPORTED_MODELS))
def test_unsupported_features_raise(name):
  """Verifies unsupported model features raise at construction instead of silently diverging."""
  mjm = mujoco.MjModel.from_xml_string("<mujoco>" + _UNSUPPORTED_MODELS[name] + "</mujoco>")
  with pytest.raises(ValueError, match="unsupported model feature"):
    BatchedEngine(mjm, nworld=1)


def test_tiny_model_small_batch_solve():
  """Regression test for small J buffers passed by MLX in the constant address space.

  A one-world, one-dof model produces a 1-element J buffer; taking a device pointer into it
  failed to compile, so the solve kernel must index the input directly.
  """
  code = (
    "import numpy as np, mujoco\n"
    "from mujoco_warp.mps.robot import BatchedEngine\n"
    "xml = '''<mujoco><option integrator=\"Euler\"/><worldbody><body>"
    '<joint type="hinge"/><geom type="capsule" size="0.02 0.1"/></body></worldbody></mujoco>\'\'\'\n'
    "mjm = mujoco.MjModel.from_xml_string(xml)\n"
    "eng = BatchedEngine(mjm, nworld=1)\n"
    "eng.set_state(np.zeros((1, mjm.nq), np.float32), np.zeros((1, mjm.nv), np.float32))\n"
    "eng.step()\n"
    "q, v = eng.get_state()\n"
    "assert np.isfinite(q).all() and np.isfinite(v).all()\n"
  )
  subprocess.run([sys.executable, "-c", code], check=True, timeout=300)


@pytest.mark.parametrize("condim,expected_rows", [(4, 6), (6, 10)])
def test_higher_condim_contact_rows(condim: int, expected_rows: int):
  """Regression test for torsional (condim=4) and rolling (condim=6) pyramidal contacts."""
  xml = f"""
  <mujoco>
    <option timestep="0.005" integrator="Euler" cone="pyramidal"/>
    <worldbody>
      <body name="sph1" pos="0 0 0">
        <joint type="free"/>
        <geom type="sphere" size="0.05" condim="{condim}" friction="0.8 0.05 0.01"/>
      </body>
      <body name="cap2" pos="0 0 0.08">
        <joint type="free"/>
        <geom type="capsule" size="0.05 0.2" zaxis="1 0 0" condim="{condim}" friction="0.8 0.05 0.01"/>
      </body>
    </worldbody>
  </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  m = mujoco_warp.put_model(mjm)
  d = mujoco_warp.make_data(mjm, nworld=1)
  mujoco_warp.forward(m, d)

  eng = BatchedEngine(mjm, nworld=1)
  eng.set_state(d.qpos.numpy(), d.qvel.numpy())
  eng._dyn_dispatch(kin=True)

  assert int(eng.nefc[0]) == int(d.nefc.numpy()[0]) == expected_rows
  j_warp = d.efc.J.numpy()[0, :expected_rows, : mjm.nv]
  j_mlx = np.array(eng.J)[0, :expected_rows, : mjm.nv]
  np.testing.assert_allclose(j_mlx, j_warp, atol=1e-5)
  np.testing.assert_allclose(np.array(eng.D)[0, :expected_rows], d.efc.D.numpy()[0, :expected_rows], atol=1e-5)
  np.testing.assert_allclose(np.array(eng.aref)[0, :expected_rows], d.efc.aref.numpy()[0, :expected_rows], atol=1e-4)


def test_priority_negative_solref_mixing():
  """Verifies high-priority geom solref overrides lower-priority geom even when solref <= 0."""
  xml = """
  <mujoco>
    <option timestep="0.005" integrator="Euler"/>
    <worldbody>
      <body name="b1" pos="0 0 0">
        <joint type="free"/>
        <geom name="g1" type="sphere" size="0.1" priority="1" solref="0.05 1.2"/>
      </body>
      <body name="b2" pos="0 0 0.15">
        <joint type="free"/>
        <geom name="g2" type="sphere" size="0.1" priority="0" solref="-1000 -100"/>
      </body>
    </worldbody>
  </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  eng = BatchedEngine(mjm, nworld=1)
  assert eng.rig.npairc == 1
  np.testing.assert_allclose(eng.rig.pair_solref[0], [0.05, 1.2], atol=1e-6)


def test_collision_sensor_pair_excluded_from_contacts():
  """Verifies broadphase pairs solely for collision sensors do not generate contact rows."""
  xml = """
  <mujoco>
    <option timestep="0.005" integrator="Euler"/>
    <worldbody>
      <body name="b1" pos="0 0 0">
        <joint type="free"/>
        <geom name="g1" type="sphere" size="0.1" contype="0" conaffinity="0"/>
      </body>
      <body name="b2" pos="0 0 0.15">
        <joint type="free"/>
        <geom name="g2" type="sphere" size="0.1" contype="0" conaffinity="0"/>
      </body>
    </worldbody>
    <sensor>
      <distance name="dist" geom1="g1" geom2="g2"/>
    </sensor>
  </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  eng = BatchedEngine(mjm, nworld=1)
  assert eng.rig.npairc == 0
