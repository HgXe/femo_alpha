import numpy as np
import pytest
from pathlib import Path

dolfinx = pytest.importorskip("dolfinx")
pytest.importorskip("petsc4py")
pytest.importorskip("ufl")
meshio = pytest.importorskip("meshio")

import csdl_alpha as csdl
from mpi4py import MPI
from dolfinx.fem import Function

from femo_alpha.rm_shell.rm_shell_model import RMShellModel


RECORDS_ROOT = Path(__file__).resolve().parent / "_records"
MESH_FILE = RECORDS_ROOT / "rmshell_test_mesh.xdmf"


def _clamped_boundary(x):
    return np.less(x[0], 3e-11)


def _build_surface_mesh(nx=4, ny=2):
    RECORDS_ROOT.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(0.0, 10.0, nx + 1)
    ys = np.linspace(0.0, 2.0, ny + 1)
    points = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float64)

    cells = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n3 = n0 + (nx + 1)
            n2 = n3 + 1
            cells.append([n0, n1, n2, n3])
    cells = np.array(cells, dtype=np.int64)

    mesh_file = RECORDS_ROOT / f"rmshell_test_mesh_{nx}x{ny}.xdmf"
    meshio.write(mesh_file, meshio.Mesh(points, [("quad", cells)]))
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, str(mesh_file), "r") as xdmf:
        return xdmf.read_mesh(name="Grid")


def _build_shell_model(recorder_path, nx=4, ny=2, record=True, geometry_input_mode="coordinates"):
    mesh = _build_surface_mesh(nx=nx, ny=ny)
    return RMShellModel(
        mesh,
        shell_bc_func=_clamped_boundary,
        element_wise_material=False,
        record=record,
        solve_direct=True,
        recorder_path=recorder_path,
        geometry_input_mode=geometry_input_mode,
    )


def _fenics_ordered_inputs(shell_model):
    nn = shell_model.nn

    thickness = 0.1 * np.ones(nn)
    E = 1.0e8 * np.ones(nn)
    nu = 0.3 * np.ones(nn)
    density = 10.0 * np.ones(nn)

    nodal_pressure = np.zeros((nn, 3))
    nodal_pressure[:, 2] = 5.0
    node_disp = np.zeros((nn, 3))

    return dict(
        thickness=thickness,
        E=E,
        nu=nu,
        density=density,
        M_solid=np.zeros(shell_model.fea.inputs_dict["M_solid"]["shape"]),
        load_vector=shell_model.assemble_generalized_load_vector(
            nodal_pressure=nodal_pressure,
            node_disp=node_disp,
        ),
        nodal_pressure=nodal_pressure,
        node_disp=node_disp,
    )


def _fenics_ordered_inputs_with_moments(shell_model):
    vals = _fenics_ordered_inputs(shell_model)
    nn = shell_model.nn

    nodal_moments = np.zeros((nn, 3))
    nodal_moments[:, 1] = np.linspace(0.1, 0.4, nn)

    vals["M_solid"] = nodal_moments.reshape(-1)
    vals["load_vector"] = shell_model.assemble_generalized_load_vector(
        nodal_pressure=vals["nodal_pressure"],
        nodal_moments=nodal_moments,
        node_disp=vals["node_disp"],
    )
    vals["nodal_moments"] = nodal_moments
    return vals


def _assemble_load_direction(shell_model, field_direction):
    f_func = Function(shell_model.shell_pde.VF)
    m_func = Function(shell_model.shell_pde.VF)
    f_func.x.array[:] = field_direction
    m_func.x.array[:] = 0.0
    return shell_model.shell_pde.assemble_generalized_load_vector(
        f=f_func,
        m=m_func,
    )


def _assemble_combined_load_direction(shell_model, force_direction, moment_direction):
    f_func = Function(shell_model.shell_pde.VF)
    m_func = Function(shell_model.shell_pde.VF)
    f_func.x.array[:] = force_direction
    m_func.x.array[:] = moment_direction
    return shell_model.shell_pde.assemble_generalized_load_vector(
        f=f_func,
        m=m_func,
    )


def _expected_compliance_from_load_vector(load_vector, disp_solid):
    return np.asarray(load_vector).reshape(-1) @ np.asarray(disp_solid).reshape(-1)


def _build_isotropic_material(shell_model, vals, suffix):
    return shell_model.material_inputs.from_isotropic(
        thickness=csdl.Variable(value=vals["thickness"], name=f"thickness_{suffix}"),
        E=csdl.Variable(value=vals["E"], name=f"E_{suffix}"),
        nu=csdl.Variable(value=vals["nu"], name=f"nu_{suffix}"),
        density=csdl.Variable(value=vals["density"], name=f"density_{suffix}"),
    )


def _build_field_loads(shell_model, vals, suffix, include_moments=False):
    kwargs = {
        "nodal_pressure": csdl.Variable(value=vals["nodal_pressure"], name=f"nodal_pressure_{suffix}"),
    }
    if include_moments:
        kwargs["nodal_moments"] = csdl.Variable(value=vals["nodal_moments"], name=f"nodal_moments_{suffix}")
    return shell_model.load_inputs.from_fields(**kwargs)


def _build_vector_loads(shell_model, vals, suffix):
    return shell_model.load_inputs.from_vector(
        load_vector=csdl.Variable(value=vals["load_vector"], name=f"direct_load_vector_{suffix}"),
    )


def _build_node_disp(vals, suffix):
    return csdl.Variable(value=vals["node_disp"], name=f"node_disp_{suffix}")


def test_rmshell_solve_does_not_build_postprocessor_outputs():
    shell_model = _build_shell_model(
        str(RECORDS_ROOT / "rmshell_solve_only"),
    )
    vals = _fenics_ordered_inputs(shell_model)

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    material = _build_isotropic_material(shell_model, vals, "solve_only")
    loads = _build_field_loads(shell_model, vals, "solve_only")
    state = shell_model.solve(material, loads, node_disp=_build_node_disp(vals, "solve_only"))

    assert shell_model._post_fea is None
    assert state.disp_solid is not None

    outputs = shell_model.post.evaluate(state=state)
    assert shell_model._post_fea is not None
    assert outputs.compliance is not None


def test_rmshell_direct_load_vector_matches_field_loads_and_derivatives():
    field_shell = _build_shell_model(
        str(RECORDS_ROOT / "rmshell_field_load"),
    )
    direct_shell = _build_shell_model(
        str(RECORDS_ROOT / "rmshell_direct_load"),
    )

    field_vals = _fenics_ordered_inputs(field_shell)
    direct_vals = _fenics_ordered_inputs(direct_shell)

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    field_material = _build_isotropic_material(field_shell, field_vals, "field")
    direct_material = _build_isotropic_material(direct_shell, direct_vals, "direct")
    field_loads = _build_field_loads(field_shell, field_vals, "field")
    direct_loads = _build_vector_loads(direct_shell, direct_vals, "direct")

    field_state = field_shell.solve(field_material, field_loads, node_disp=_build_node_disp(field_vals, "field"))
    field_outputs = field_shell.post.evaluate(state=field_state)
    direct_state = direct_shell.solve(direct_material, direct_loads, node_disp=_build_node_disp(direct_vals, "direct"))
    direct_outputs = direct_shell.post.evaluate(state=direct_state)

    np.testing.assert_allclose(
        direct_outputs.disp_solid.value,
        field_outputs.disp_solid.value,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        direct_outputs.compliance.value,
        field_outputs.compliance.value,
        rtol=1e-10,
        atol=1e-10,
    )

    q_field = field_outputs.compliance + 1e-6 * csdl.sum(field_outputs.disp_solid)
    q_direct = direct_outputs.compliance + 1e-6 * csdl.sum(direct_outputs.disp_solid)

    sim = csdl.experimental.PySimulator(recorder)
    sim.check_totals(
        [q_field, q_direct],
        [field_loads.nodal_pressure, direct_loads.load_vector],
        step_size=1e-6,
        print_results=False,
        raise_on_error=True,
    )

    dq_dp = csdl.derivative(q_field, [field_loads.nodal_pressure])[field_loads.nodal_pressure]
    dq_dL = csdl.derivative(q_direct, [direct_loads.load_vector])[direct_loads.load_vector]

    pressure_direction = np.zeros_like(field_vals["nodal_pressure"])
    pressure_direction[:, 2] = np.linspace(0.25, 1.25, field_shell.nn)
    field_direction = pressure_direction[
        np.asarray(field_shell.shell_pde.mesh.geometry.input_global_indices)
    ].reshape(-1)
    load_direction = _assemble_load_direction(field_shell, field_direction)

    directional_field = dq_dp.value.reshape(-1) @ pressure_direction.reshape(-1)
    directional_direct = dq_dL.value.reshape(-1) @ load_direction

    np.testing.assert_allclose(
        directional_direct,
        directional_field,
        rtol=1e-8,
        atol=1e-8,
    )

    recorder.stop()


def test_rmshell_post_field_recording_defaults_and_overrides():
    shell_recording = _build_shell_model(
        str(RECORDS_ROOT / "rmshell_post_recording_default"),
        nx=2,
        ny=1,
        record=True,
    )
    assert shell_recording.post_fea.outputs_dict == {}
    assert shell_recording.post_fea.outputs_field_dict == {}
    shell_recording.post.clear().strains().kinematics().stress()
    for output in shell_recording.post._outputs:
        if hasattr(output, "record"):
            assert output.record is True

    shell_quiet = _build_shell_model(
        str(RECORDS_ROOT / "rmshell_post_recording_quiet"),
        nx=2,
        ny=1,
        record=False,
    )
    shell_quiet.post.clear().strains().kinematics().stress()
    for output in shell_quiet.post._outputs:
        if hasattr(output, "record"):
            assert output.record is False

    shell_quiet.post.add_field_form(
        "quiet_override",
        lambda context: context.post._displacement_components(context)[0],
        ["disp_solid"],
        element=("DG", 0),
        record=True,
    )
    assert shell_quiet.post._outputs[-1].record is True


def test_rmshell_direct_load_vector_matches_combined_force_and_moment_loads():
    field_shell = _build_shell_model(
        str(RECORDS_ROOT / "rmshell_field_force_moment"),
    )
    direct_shell = _build_shell_model(
        str(RECORDS_ROOT / "rmshell_direct_force_moment"),
    )

    vals = _fenics_ordered_inputs_with_moments(field_shell)
    direct_vals = _fenics_ordered_inputs_with_moments(direct_shell)

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    field_material = _build_isotropic_material(field_shell, vals, "field_fm")
    direct_material = _build_isotropic_material(direct_shell, direct_vals, "direct_fm")
    field_loads = _build_field_loads(field_shell, vals, "field_fm", include_moments=True)
    direct_loads = _build_vector_loads(direct_shell, direct_vals, "direct_fm")

    field_outputs = field_shell.post.evaluate(
        state=field_shell.solve(field_material, field_loads, node_disp=_build_node_disp(vals, "field_fm"))
    )
    direct_outputs = direct_shell.post.evaluate(
        state=direct_shell.solve(direct_material, direct_loads, node_disp=_build_node_disp(direct_vals, "direct_fm"))
    )

    np.testing.assert_allclose(
        direct_outputs.disp_solid.value,
        field_outputs.disp_solid.value,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        direct_outputs.compliance.value,
        field_outputs.compliance.value,
        rtol=1e-10,
        atol=1e-10,
    )

    q_field = field_outputs.compliance + 1e-6 * csdl.sum(field_outputs.disp_solid)
    q_direct = direct_outputs.compliance + 1e-6 * csdl.sum(direct_outputs.disp_solid)

    sim = csdl.experimental.PySimulator(recorder)
    sim.check_totals(
        [q_field, q_direct],
        [field_loads.nodal_pressure, field_loads.nodal_moments, direct_loads.load_vector],
        step_size=1e-6,
        print_results=False,
        raise_on_error=True,
    )

    dq_dp = csdl.derivative(q_field, [field_loads.nodal_pressure])[field_loads.nodal_pressure]
    dq_dm = csdl.derivative(q_field, [field_loads.nodal_moments])[field_loads.nodal_moments]
    dq_dL = csdl.derivative(q_direct, [direct_loads.load_vector])[direct_loads.load_vector]

    pressure_direction = np.zeros_like(vals["nodal_pressure"])
    pressure_direction[:, 2] = np.linspace(0.2, 0.6, field_shell.nn)
    moment_direction = np.zeros_like(vals["nodal_moments"])
    moment_direction[:, 1] = np.linspace(-0.1, 0.3, field_shell.nn)

    mesh_indices = np.asarray(field_shell.shell_pde.mesh.geometry.input_global_indices)
    force_field_direction = pressure_direction[mesh_indices].reshape(-1)
    moment_field_direction = moment_direction[mesh_indices].reshape(-1)
    load_direction = _assemble_combined_load_direction(
        field_shell,
        force_field_direction,
        moment_field_direction,
    )

    directional_field = (
        dq_dp.value.reshape(-1) @ pressure_direction.reshape(-1)
        + dq_dm.value.reshape(-1) @ moment_direction.reshape(-1)
    )
    directional_direct = dq_dL.value.reshape(-1) @ load_direction

    np.testing.assert_allclose(
        directional_direct,
        directional_field,
        rtol=1e-8,
        atol=1e-8,
    )

    recorder.stop()


def test_rmshell_uniform_pressure_matches_cantilever_beam_tip_deflection():
    shell_model = _build_shell_model(
        str(RECORDS_ROOT / "rmshell_textbook_deflection"),
        nx=8,
        ny=2,
    )
    vals = _fenics_ordered_inputs(shell_model)

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    material = _build_isotropic_material(shell_model, vals, "tb")
    loads = _build_field_loads(shell_model, vals, "tb")
    outputs = shell_model.post.evaluate(state=shell_model.solve(material, loads, node_disp=_build_node_disp(vals, "tb")))

    tip_deflection_fe = np.max(outputs.disp_extracted.value[:, 2])
    length = 10.0
    width = 2.0
    pressure = 5.0
    thickness_val = 0.1
    youngs_modulus = 1.0e8
    line_load = pressure * width
    inertia = width * thickness_val**3 / 12.0
    tip_deflection_eb = line_load * length**4 / (8.0 * youngs_modulus * inertia)

    np.testing.assert_allclose(
        tip_deflection_fe,
        tip_deflection_eb,
        rtol=0.12,
        atol=1e-3,
    )

    recorder.stop()


def test_rmshell_compliance_matches_generalized_load_work():
    shell_model = _build_shell_model(
        str(RECORDS_ROOT / "rmshell_compliance_check"),
        nx=4,
        ny=2,
    )
    vals = _fenics_ordered_inputs(shell_model)

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    material = _build_isotropic_material(shell_model, vals, "comp")
    loads = _build_field_loads(shell_model, vals, "comp")
    state = shell_model.solve(material, loads, node_disp=_build_node_disp(vals, "comp"))
    outputs = shell_model.post.evaluate(state=state)

    expected_field_compliance = _expected_compliance_from_load_vector(
        vals["load_vector"],
        outputs.disp_solid.value,
    )

    np.testing.assert_allclose(
        outputs.compliance.value,
        expected_field_compliance,
        rtol=1e-10,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        2.0 * outputs.elastic_energy.value,
        outputs.compliance.value,
        rtol=1e-8,
        atol=1e-8,
    )

    recorder.stop()


def test_rmshell_compliance_matches_total_work_for_direct_and_mixed_loads():
    direct_shell = _build_shell_model(
        str(RECORDS_ROOT / "rmshell_compliance_direct"),
        nx=4,
        ny=2,
    )
    mixed_shell = _build_shell_model(
        str(RECORDS_ROOT / "rmshell_compliance_mixed"),
        nx=4,
        ny=2,
    )

    direct_vals = _fenics_ordered_inputs_with_moments(direct_shell)
    mixed_vals = _fenics_ordered_inputs_with_moments(mixed_shell)

    extra_direct = np.zeros_like(direct_vals["load_vector"])
    extra_direct[2::6] = np.linspace(0.05, 0.25, extra_direct[2::6].size)

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    direct_material = _build_isotropic_material(direct_shell, direct_vals, "comp_mix_direct")
    direct_loads = _build_vector_loads(direct_shell, direct_vals, "comp_mix_direct")
    direct_outputs = direct_shell.post.evaluate(
        state=direct_shell.solve(direct_material, direct_loads, node_disp=_build_node_disp(direct_vals, "comp_mix_direct"))
    )

    expected_direct_compliance = _expected_compliance_from_load_vector(
        direct_vals["load_vector"],
        direct_outputs.disp_solid.value,
    )
    np.testing.assert_allclose(
        direct_outputs.compliance.value,
        expected_direct_compliance,
        rtol=1e-10,
        atol=1e-10,
    )

    mixed_material = _build_isotropic_material(mixed_shell, mixed_vals, "comp_mix_mixed")
    mixed_field_loads = _build_field_loads(mixed_shell, mixed_vals, "comp_mix_mixed", include_moments=True)
    mixed_direct_loads = mixed_shell.load_inputs.from_vector(
        load_vector=csdl.Variable(value=extra_direct, name="mixed_direct_load_vector"),
    )
    mixed_loads = mixed_shell.load_inputs.combine(mixed_field_loads, mixed_direct_loads)
    mixed_outputs = mixed_shell.post.evaluate(
        state=mixed_shell.solve(mixed_material, mixed_loads, node_disp=_build_node_disp(mixed_vals, "comp_mix_mixed"))
    )

    total_load_vector = mixed_vals["load_vector"] + extra_direct
    expected_mixed_compliance = _expected_compliance_from_load_vector(
        total_load_vector,
        mixed_outputs.disp_solid.value,
    )
    np.testing.assert_allclose(
        mixed_outputs.compliance.value,
        expected_mixed_compliance,
        rtol=1e-10,
        atol=1e-10,
    )

    recorder.stop()


def test_rmshell_post_evaluate_matches_solved_state_for_external_displacement():
    shell_model = _build_shell_model(
        str(RECORDS_ROOT / "rmshell_external_postprocess"),
        nx=4,
        ny=2,
    )
    vals = _fenics_ordered_inputs_with_moments(shell_model)

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    material = _build_isotropic_material(shell_model, vals, "external")
    loads = _build_field_loads(shell_model, vals, "external", include_moments=True)
    state = shell_model.solve(material, loads, node_disp=_build_node_disp(vals, "external"))
    solved_outputs = shell_model.post.evaluate(state=state)
    external_outputs = shell_model.post.evaluate(
        material=material,
        loads=loads,
        displacement=state.disp_solid,
        node_disp=_build_node_disp(vals, "external_post"),
    )

    np.testing.assert_allclose(
        external_outputs.compliance.value,
        solved_outputs.compliance.value,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        external_outputs.disp_extracted.value,
        solved_outputs.disp_extracted.value,
        rtol=1e-12,
        atol=1e-12,
    )

    recorder.stop()


def test_rmshell_legacy_evaluate_matches_solve_and_post_evaluate():
    shell_model = _build_shell_model(
        str(RECORDS_ROOT / "rmshell_legacy_evaluate"),
        nx=4,
        ny=2,
    )
    vals = _fenics_ordered_inputs(shell_model)

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    material = _build_isotropic_material(shell_model, vals, "legacy")
    loads = _build_field_loads(shell_model, vals, "legacy")
    outputs_new = shell_model.post.evaluate(state=shell_model.solve(material, loads, node_disp=_build_node_disp(vals, "legacy")))

    outputs_old = shell_model.evaluate(
        vals["nodal_pressure"],
        vals["thickness"],
        vals["E"],
        vals["nu"],
        vals["density"],
        vals["node_disp"],
        is_pressure=True,
    )

    np.testing.assert_allclose(
        outputs_old.compliance.value,
        outputs_new.compliance.value,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        outputs_old.disp_extracted.value,
        outputs_new.disp_extracted.value,
        rtol=1e-8,
        atol=1e-8,
    )

    recorder.stop()


def test_rmshell_post_registry_supports_groups_and_custom_outputs():
    shell_model = _build_shell_model(
        str(RECORDS_ROOT / "rmshell_post_registry"),
        nx=4,
        ny=2,
    )
    vals = _fenics_ordered_inputs(shell_model)

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    material = _build_isotropic_material(shell_model, vals, "registry")
    loads = _build_field_loads(shell_model, vals, "registry")
    state = shell_model.solve(material, loads, node_disp=_build_node_disp(vals, "registry"))

    assert shell_model.post._outputs == []
    with pytest.raises(ValueError, match="No RMShell postprocessing outputs"):
        shell_model.post.compute(state=state)

    shell_model.post.mass_properties().average_strain(
        "avg_eps_x",
        strain_type="mid",
        component="xx",
        region=None,
    )
    shell_model.post.pnorm_stress_custom(
        "custom_pnorm",
        rho=shell_model.rho,
        m=shell_model.m,
        region=None,
    ).pnorm_stress().stress().stress_top().stress_mid().stress_bottom()

    outputs = shell_model.post.compute(state=state)

    assert outputs.avg_eps_x.shape == (1,)
    assert outputs.stress.shape == outputs.stress_top.shape
    assert outputs.stress_mid.shape == outputs.stress.shape
    assert outputs.stress_bottom.shape == outputs.stress.shape
    assert not hasattr(outputs, "elastic_energy")
    assert outputs.cg.shape == (3,)
    np.testing.assert_allclose(
        outputs.stress.value,
        outputs.stress_top.value,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        outputs.custom_pnorm.value,
        outputs.pnorm_stress.value,
        rtol=1e-10,
        atol=1e-10,
    )

    recorder.stop()


def test_rmshell_accepts_direct_mesh_coordinates_and_differentiates_them():
    shell_model = _build_shell_model(
        str(RECORDS_ROOT / "rmshell_direct_mesh_coordinates"),
        nx=2,
        ny=1,
        record=False,
        geometry_input_mode="coordinates",
    )
    vals = _fenics_ordered_inputs(shell_model)

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    material = _build_isotropic_material(shell_model, vals, "direct_coordinates")
    loads = _build_vector_loads(shell_model, vals, "direct_coordinates")
    mesh_nodes = csdl.Variable(
        value=shell_model.reference_mesh_nodes.copy(),
        name="direct_mesh_coordinates",
    )
    state = shell_model.solve(
        material,
        loads,
        mesh_nodes=mesh_nodes,
    )
    outputs = shell_model.post.evaluate(state=state)

    assert state.mesh_nodes is mesh_nodes
    assert state.node_disp is None
    np.testing.assert_allclose(state.raw_outputs.disp_solid.value, outputs.disp_solid.value)

    derivative = csdl.derivative(outputs.compliance, [mesh_nodes])[mesh_nodes]
    assert derivative.shape == (1, int(np.prod(mesh_nodes.shape)))
    assert np.all(np.isfinite(derivative.value))
    assert np.linalg.norm(derivative.value) > 0.0

    recorder.stop()


def test_rmshell_mesh_coordinate_state_and_post_derivatives_match_finite_difference():
    shell_model = _build_shell_model(
        str(RECORDS_ROOT / "rmshell_mesh_coordinate_fd"),
        nx=2,
        ny=1,
        record=False,
    )
    vals = _fenics_ordered_inputs(shell_model)

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    material = _build_isotropic_material(shell_model, vals, "coordinate_fd")
    loads = _build_vector_loads(shell_model, vals, "coordinate_fd")

    reference = shell_model.reference_mesh_nodes.copy()
    direction = np.zeros_like(reference)
    direction[:, 0] = 0.03 * reference[:, 0] / np.max(reference[:, 0])
    direction[:, 1] = 0.02 * reference[:, 1] / np.max(reference[:, 1])
    direction[:, 2] = (
        0.01
        * reference[:, 0]
        / np.max(reference[:, 0])
        * (reference[:, 1] - np.mean(reference[:, 1]))
    )
    geometry_control = csdl.Variable(value=np.array([0.0]), name="geometry_control")
    mesh_nodes = (
        csdl.Variable(value=reference, name="reference_mesh_nodes")
        + geometry_control * csdl.Variable(value=direction, name="mesh_direction")
    )

    state = shell_model.solve(material, loads, mesh_nodes=mesh_nodes)
    shell_model.post.compliance().mass().cg().elastic_energy().mid_strain().shear_strain().curvature()
    outputs = shell_model.post.compute(state=state)

    checked_outputs = [
        outputs.compliance,
        outputs.mass,
        outputs.cg[0],
        outputs.cg[1],
        outputs.elastic_energy,
        csdl.sum(outputs.mid_strain),
        csdl.sum(outputs.shear_strain),
        csdl.sum(outputs.curvature),
    ]
    sim = csdl.experimental.PySimulator(recorder)
    totals = sim.check_totals(
        checked_outputs,
        [geometry_control],
        step_size=2e-6,
        print_results=False,
        raise_on_error=True,
    )

    for result in totals.values():
        assert np.all(np.isfinite(result["value"]))
        assert np.all(np.isfinite(result["fd_value"]))

    recorder.stop()
