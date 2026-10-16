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
  # no candidate pairs: the support tables stay empty and the stride floors at 1
  assert eng.rig.nfr == 0 and eng.rig.npairc == 0
  assert eng.rig.sup_num.size == 0 and eng.rig.maxsup == 1


@pytest.mark.parametrize("coop", [True, False])
def test_capsule_capsule_contact(coop: bool):
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

  eng = BatchedEngine(mjm, nworld=1, coop=coop)
  eng.set_state(d.qpos.numpy(), d.qvel.numpy())
  eng._dyn_dispatch(kin=True)

  assert int(eng.ncon[0]) == int(d.nacon.numpy()[0]) == 1
  assert int(eng.nefc[0]) == int(d.nefc.numpy()[0]) == 4

  j_warp = d.efc.J.numpy()[0, : int(d.nefc.numpy()[0]), : mjm.nv]
  j_mlx = eng.dense_J()[0, : int(eng.nefc[0]), : mjm.nv]
  np.testing.assert_allclose(j_mlx, j_warp, atol=1e-5)


@pytest.mark.parametrize("coop", [True, False])
def test_sphere_capsule_contact(coop: bool):
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

  eng = BatchedEngine(mjm, nworld=1, coop=coop)
  eng.set_state(d.qpos.numpy(), d.qvel.numpy())
  eng._dyn_dispatch(kin=True)

  assert int(eng.ncon[0]) == int(d.nacon.numpy()[0]) == 1
  assert int(eng.nefc[0]) == int(d.nefc.numpy()[0]) == 4

  j_warp = d.efc.J.numpy()[0, : int(d.nefc.numpy()[0]), : mjm.nv]
  j_mlx = eng.dense_J()[0, : int(eng.nefc[0]), : mjm.nv]
  np.testing.assert_allclose(j_mlx, j_warp, atol=1e-5)


@pytest.mark.parametrize("coop", [True, False])
def test_condim1_frictionless_contact(coop: bool):
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

  eng = BatchedEngine(mjm, nworld=1, coop=coop)
  eng.set_state(d.qpos.numpy(), d.qvel.numpy())
  eng._dyn_dispatch(kin=True)

  assert int(eng.ncon[0]) == int(d.nacon.numpy()[0]) == 1
  assert int(eng.nefc[0]) == int(d.nefc.numpy()[0]) == 1

  j_warp = d.efc.J.numpy()[0, : int(d.nefc.numpy()[0]), : mjm.nv]
  j_mlx = eng.dense_J()[0, : int(eng.nefc[0]), : mjm.nv]
  np.testing.assert_allclose(j_mlx, j_warp, atol=1e-5)


@pytest.mark.parametrize("coop", [True, False])
def test_friction_limit_contact_rows(coop: bool):
  """Verifies friction/limit/contact row kinds (singleton supports) match Warp CPU."""
  xml = """
  <mujoco>
    <option timestep="0.005" integrator="Euler"/>
    <worldbody>
      <body name="arm" pos="0 0 0.06">
        <joint name="j1" type="hinge" axis="0 1 0" ref="0.3" range="-0.1 0.1" frictionloss="0.3"/>
        <geom type="capsule" size="0.03 0.05"/>
      </body>
      <geom name="gnd" type="plane" size="1 1 0.1"/>
    </worldbody>
  </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  m = mujoco_warp.put_model(mjm)
  d = mujoco_warp.make_data(mjm, nworld=1)
  mujoco_warp.forward(m, d)

  eng = BatchedEngine(mjm, nworld=1, coop=coop)
  assert eng.rig.nfr == 1 and eng.rig.nlim == 1
  eng.set_state(d.qpos.numpy(), d.qvel.numpy())
  eng._dyn_dispatch(kin=True)

  # row order: friction dof, limit joint, then contact
  n = int(d.nefc.numpy()[0])
  assert int(eng.nefc[0]) == n
  types = [int(t) for t in d.efc.type.numpy()[0, :n]]
  assert types[:2] == [1, 3] and all(t == 6 for t in types[2:])
  j_warp = d.efc.J.numpy()[0, :n, : mjm.nv]
  np.testing.assert_allclose(eng.dense_J()[0, :n], j_warp, atol=1e-5)
  np.testing.assert_allclose(np.array(eng.D)[0, :n], d.efc.D.numpy()[0, :n], atol=1e-5)
  np.testing.assert_allclose(np.array(eng.aref)[0, :n], d.efc.aref.numpy()[0, :n], atol=1e-5)


@pytest.mark.parametrize("coop", [True, False])
def test_humanoid_one_step_elements(coop: bool):
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

  eng = BatchedEngine(mjm, nworld=1, coop=coop)
  assert eng.coop == coop
  eng.set_state(qpos, qvel, ctrl)
  eng.step()

  assert int(eng.nefc[0]) == int(d.nefc.numpy()[0]) == 32
  n = int(d.nefc.numpy()[0])
  np.testing.assert_allclose(eng.dense_J()[0, :n], d.efc.J.numpy()[0, :n, : mjm.nv], atol=1e-5)
  np.testing.assert_allclose(np.array(eng.D)[0, :n], d.efc.D.numpy()[0, :n], atol=1e-5)
  np.testing.assert_allclose(np.array(eng.aref)[0, :n], d.efc.aref.numpy()[0, :n], atol=1e-5)
  np.testing.assert_allclose(np.array(eng.smooth)[0], d.qfrc_smooth.numpy()[0], atol=1e-4)
  np.testing.assert_allclose(np.array(eng.qfrc_bias)[0], d.qfrc_bias.numpy()[0], atol=1e-4)
  np.testing.assert_allclose(np.array(eng.qacc)[0], d.qacc.numpy()[0], atol=1e-2)


@pytest.mark.parametrize("coop", [True, False])
def test_humanoid_contact_trajectory(coop: bool):
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

  eng = BatchedEngine(mjm, nworld=1, coop=coop)
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


@pytest.mark.parametrize("coop", [True, False])
def test_plane_mesh_contact_one_step(coop: bool):
  """Verifies plane-convex (mesh) contact rows match Warp CPU exactly."""
  mjm = _plane_box_mesh_model()
  m = mujoco_warp.put_model(mjm)
  d = mujoco_warp.make_data(mjm, nworld=1)
  qpos = np.array(mjm.qpos0, np.float32)[None]
  qvel = np.zeros((1, mjm.nv), np.float32)
  d.qpos.numpy()[:] = qpos
  d.qvel.numpy()[:] = qvel
  mujoco_warp.forward(m, d)

  eng = BatchedEngine(mjm, nworld=1, coop=coop)
  eng.set_state(qpos, qvel)
  eng._dyn_dispatch(kin=True)

  assert int(eng.ncon[0]) == int(d.nacon.numpy()[0]) == 4
  assert int(eng.nefc[0]) == int(d.nefc.numpy()[0]) == 16
  assert np.array(eng.pair_conct)[0, 0] == 4

  n = int(d.nefc.numpy()[0])
  np.testing.assert_allclose(eng.dense_J()[0, :n], d.efc.J.numpy()[0, :n, : mjm.nv], atol=1e-6)
  np.testing.assert_allclose(np.array(eng.aref)[0, :n], d.efc.aref.numpy()[0, :n], atol=1e-6)
  np.testing.assert_allclose(np.array(eng.D)[0, :n], d.efc.D.numpy()[0, :n], atol=1e-6)


@pytest.mark.parametrize("coop", [True, False])
def test_plane_mesh_trajectory(coop: bool):
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

  eng = BatchedEngine(mjm, nworld=1, coop=coop)
  eng.set_state(qpos, qvel)
  for _ in range(50):
    mujoco_warp.step(m, d)
    eng.step()

  qm, vm = eng.get_state()
  assert np.abs(qm - d.qpos.numpy()).max() < 1e-5
  assert np.abs(vm - d.qvel.numpy()).max() < 1e-4


@pytest.mark.parametrize("coop", [True, False])
def test_mesh_mesh_pairs_skipped(coop: bool):
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
  eng = BatchedEngine(mjm, nworld=1, coop=coop)
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


@pytest.mark.parametrize("coop", [True, False])
def test_parallel_capsule_capsule_contact(coop: bool):
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

  eng = BatchedEngine(mjm, nworld=1, coop=coop)
  eng.set_state(d.qpos.numpy(), d.qvel.numpy())
  eng._dyn_dispatch(kin=True)

  assert int(eng.ncon[0]) == int(d.nacon.numpy()[0]) == 2
  assert int(eng.nefc[0]) == int(d.nefc.numpy()[0]) == 8

  j_warp = d.efc.J.numpy()[0, : int(d.nefc.numpy()[0]), : mjm.nv]
  j_mlx = eng.dense_J()[0, : int(eng.nefc[0]), : mjm.nv]
  np.testing.assert_allclose(j_mlx, j_warp, atol=1e-5)


@pytest.mark.parametrize("coop", [True, False])
def test_explicit_pair_material_params(coop: bool):
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

  eng = BatchedEngine(mjm, nworld=1, coop=coop)
  eng.set_state(d.qpos.numpy(), d.qvel.numpy())
  eng._dyn_dispatch(kin=True)

  nefc = int(eng.nefc[0])
  assert nefc == int(d.nefc.numpy()[0]) == 4
  np.testing.assert_allclose(np.array(eng.D)[0, :nefc], d.efc.D.numpy()[0, :nefc], atol=1e-5)
  np.testing.assert_allclose(np.array(eng.aref)[0, :nefc], d.efc.aref.numpy()[0, :nefc], atol=1e-4)
  np.testing.assert_allclose(eng.dense_J()[0, :nefc, : mjm.nv], d.efc.J.numpy()[0, :nefc, : mjm.nv], atol=1e-5)


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
@pytest.mark.parametrize("coop", [True, False])
def test_higher_condim_contact_rows(condim: int, expected_rows: int, coop: bool):
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

  eng = BatchedEngine(mjm, nworld=1, coop=coop)
  eng.set_state(d.qpos.numpy(), d.qvel.numpy())
  eng._dyn_dispatch(kin=True)

  assert int(eng.nefc[0]) == int(d.nefc.numpy()[0]) == expected_rows
  j_warp = d.efc.J.numpy()[0, :expected_rows, : mjm.nv]
  j_mlx = eng.dense_J()[0, :expected_rows, : mjm.nv]
  np.testing.assert_allclose(j_mlx, j_warp, atol=1e-5)
  np.testing.assert_allclose(np.array(eng.D)[0, :expected_rows], d.efc.D.numpy()[0, :expected_rows], atol=1e-5)
  np.testing.assert_allclose(np.array(eng.aref)[0, :expected_rows], d.efc.aref.numpy()[0, :expected_rows], atol=1e-4)


@pytest.mark.parametrize("coop", [True, False])
def test_priority_negative_solref_mixing(coop: bool):
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
  eng = BatchedEngine(mjm, nworld=1, coop=coop)
  assert eng.rig.npairc == 1
  np.testing.assert_allclose(eng.rig.pair_solref[0], [0.05, 1.2], atol=1e-6)


@pytest.mark.parametrize("coop", [True, False])
def test_collision_sensor_pair_excluded_from_contacts(coop: bool):
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
  eng = BatchedEngine(mjm, nworld=1, coop=coop)
  assert eng.rig.npairc == 0


def test_reset_worlds_and_mlx_inputs():
  """Verifies zero-copy MLX array inputs and selective world resets on GPU."""
  xml = """
  <mujoco>
    <option timestep="0.005" integrator="Euler"/>
    <worldbody>
      <body name="b1" pos="0 0 0.5">
        <joint name="j1" type="hinge"/>
        <geom type="capsule" size="0.02 0.1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="j1"/>
    </actuator>
  </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  eng = BatchedEngine(mjm, nworld=4)
  qpos = mx.array([[0.1], [0.2], [0.3], [0.4]], dtype=mx.float32)
  qvel = mx.array([[1.0], [2.0], [3.0], [4.0]], dtype=mx.float32)
  ctrl = mx.array([[0.5], [-0.5], [0.2], [-0.2]], dtype=mx.float32)
  eng.set_state(qpos, qvel, ctrl)
  eng.step(ctrl=ctrl)

  mask = mx.array([True, False, True, False])
  eng.reset_worlds(mask)
  q_out, v_out = eng.get_state()
  np.testing.assert_allclose(q_out[0], mjm.qpos0, atol=1e-6)
  np.testing.assert_allclose(q_out[2], mjm.qpos0, atol=1e-6)
  np.testing.assert_allclose(v_out[0], 0.0, atol=1e-6)
  np.testing.assert_allclose(v_out[2], 0.0, atol=1e-6)
  assert abs(float(q_out[1, 0]) - 0.2) > 1e-4
  assert abs(float(q_out[3, 0]) - 0.4) > 1e-4


@pytest.mark.parametrize("coop", [False, True])
def test_unconstrained_fast_path(coop: bool):
  """Verifies nefc == 0 uses direct Cholesky solve with niter == 0 and matches Warp CPU."""
  xml = """
  <mujoco>
    <option timestep="0.005" integrator="Euler"/>
    <worldbody>
      <body name="b1" pos="0 0 1.0">
        <joint name="j1" type="hinge" damping="0.5"/>
        <geom type="capsule" size="0.02 0.1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="j1"/>
    </actuator>
  </mujoco>
  """
  mjm = mujoco.MjModel.from_xml_string(xml)
  m = mujoco_warp.put_model(mjm)
  d = mujoco_warp.make_data(mjm, nworld=2)
  qpos = np.array([[0.3], [-0.4]], np.float32)
  qvel = np.array([[1.2], [-0.8]], np.float32)
  ctrl = np.array([[0.5], [-0.2]], np.float32)
  d.qpos.numpy()[:] = qpos
  d.qvel.numpy()[:] = qvel
  d.ctrl.numpy()[:] = ctrl
  mujoco_warp.step(m, d)

  eng = BatchedEngine(mjm, nworld=2, coop=coop)
  eng.set_state(qpos, qvel, ctrl)
  eng.step()

  assert np.all(np.array(eng.nefc) == 0)
  assert np.all(np.array(eng.niter) == 0)
  np.testing.assert_allclose(np.array(eng.qacc), d.qacc.numpy(), rtol=1e-5, atol=1e-5)
  qm, vm = eng.get_state()
  np.testing.assert_allclose(qm, d.qpos.numpy(), rtol=1e-5, atol=1e-6)
  np.testing.assert_allclose(vm, d.qvel.numpy(), rtol=1e-5, atol=1e-6)



@pytest.mark.parametrize("eulerdamp", [False, True])
def test_packed_m_parity(eulerdamp: bool):
  """Dense and packed mass-matrix routes agree bitwise on M and keep trajectories in step.

  The packed route stores one ancestor chain per dof (`dof_pad`) and applies it
  through the chain plus its transpose, so the matvec accumulation order - and
  only that - differs from the dense block.  `eulerdamp` flips the scene's
  integrator flag so both factor-solve sites (Newton and implicit damping) run.
  """
  scene = "mujoco_warp/test_data/humanoid/humanoid.xml"
  with open(scene) as f:
    mjm = mujoco.MjModel.from_xml_string(f.read())
  if eulerdamp:
    mjm.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)
  else:
    mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EULERDAMP)

  nworld = 4
  rng = np.random.default_rng(7)
  qpos = np.tile(np.array(mjm.qpos0, np.float32), (nworld, 1))
  qpos[:, :3] += rng.normal(0, 0.005, (nworld, 3)).astype(np.float32)
  qvel = rng.normal(0, 0.02, (nworld, mjm.nv)).astype(np.float32)
  ctrl = rng.uniform(-0.3, 0.3, (nworld, mjm.nu)).astype(np.float32)

  dense = BatchedEngine(mjm, nworld=nworld, coop=False, m_layout="dense")
  packed = BatchedEngine(mjm, nworld=nworld, coop=False, m_layout="packed")
  assert dense.packed_m is False and packed.packed_m is True
  for eng in (dense, packed):
    eng.set_state(qpos, qvel, ctrl)

  dense.step()
  packed.step()
  np.testing.assert_array_equal(dense.dense_M(), packed.dense_M())

  for _ in range(50):
    dense.step()
    packed.step()
  qd, vd = dense.get_state()
  qp, vp = packed.get_state()
  assert np.isfinite(qp).all() and np.isfinite(vp).all()
  np.testing.assert_allclose(qp, qd, rtol=0, atol=1e-5)
  np.testing.assert_allclose(vp, vd, rtol=0, atol=2e-4)


def test_packed_m_env_switch(monkeypatch):
  """`MUJOCO_WARP_MPS_M=packed` selects the packed layout; the constructor argument wins."""
  mjm = mujoco.MjModel.from_xml_path("mujoco_warp/test_data/humanoid/humanoid.xml")
  monkeypatch.setenv("MUJOCO_WARP_MPS_M", "packed")
  eng = BatchedEngine(mjm, nworld=4, coop=False)
  assert eng.m_layout == "packed"
  eng_dense = BatchedEngine(mjm, nworld=4, coop=False, m_layout="dense")
  assert eng_dense.m_layout == "dense"
  # cooperative solves keep the dense block (per-lane M path, chains save little)
  coop = BatchedEngine(mjm, nworld=4, coop=True)
  assert coop.m_layout == "dense"


def test_coop_v2_parity():
  """The balanced (v2) cooperative solve is bit-exact with the row-owned v1 solve.

  v2 replaces the factor's divergent row-owned updates with a flat trailing
  triangle and the rebuild's full-width row scan with a support-pair walk; both
  keep each element's accumulation order, so trajectories must match bitwise.
  """
  mjm = mujoco.MjModel.from_xml_path("mujoco_warp/test_data/humanoid/humanoid.xml")
  nworld = 4
  rng = np.random.default_rng(3)
  qpos = np.tile(np.array(mjm.qpos0, np.float32), (nworld, 1))
  qpos[:, :3] += rng.normal(0, 0.005, (nworld, 3)).astype(np.float32)
  qvel = rng.normal(0, 0.02, (nworld, mjm.nv)).astype(np.float32)
  ctrl = rng.uniform(-0.3, 0.3, (nworld, mjm.nu)).astype(np.float32)

  v1 = BatchedEngine(mjm, nworld=nworld, coop_solver="v1")
  v2 = BatchedEngine(mjm, nworld=nworld, coop_solver="v2")
  assert v1.coop and v2.coop
  assert v1.coop_solver == "v1" and v2.coop_solver == "v2"
  # v2 keeps dense rows but needs the per-row meta table
  assert v2.packed is False and v2.want_jmeta is True
  for eng in (v1, v2):
    eng.set_state(qpos, qvel, ctrl)

  for _ in range(50):
    v1.step()
    v2.step()
  q1, u1 = v1.get_state()
  q2, u2 = v2.get_state()
  assert np.isfinite(q2).all() and np.isfinite(u2).all()
  np.testing.assert_array_equal(q1, q2)
  np.testing.assert_array_equal(u1, u2)


def test_coop_v2_env_switch(monkeypatch):
  """`MUJOCO_WARP_MPS_COOP=v2` selects the balanced solve; it is inert off the coop path."""
  mjm = mujoco.MjModel.from_xml_path("mujoco_warp/test_data/humanoid/humanoid.xml")
  monkeypatch.setenv("MUJOCO_WARP_MPS_COOP", "v2")
  coop = BatchedEngine(mjm, nworld=4)
  assert coop.coop and coop.coop_solver == "v2"
  solo = BatchedEngine(mjm, nworld=4, coop=False)
  assert not solo.coop and solo.coop_solver == "v1"
