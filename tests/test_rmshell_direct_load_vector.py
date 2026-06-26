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
from dolfinx.fem import assemble_scalar, form

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


def _build_shell_model(recorder_path, nx=4, ny=2):
    mesh = _build_surface_mesh(nx=nx, ny=ny)
    shell_model = RMShellModel(
        mesh,
        shell_bc_func=_clamped_boundary,
        element_wise_material=False,
        record=True,
        solve_direct=True,
        recorder_path=recorder_path,
    )

    shell_model.fea.outputs_dict = {
        "compliance": shell_model.fea.outputs_dict["compliance"],
        "pnorm_stress": shell_model.fea.outputs_dict["compliance"],
        "elastic_energy": shell_model.fea.outputs_dict["elastic_energy"],
        "mass": shell_model.fea.outputs_dict["mass"],
        "cgx_num": shell_model.fea.outputs_dict["cgx_num"],
        "cgy_num": shell_model.fea.outputs_dict["cgy_num"],
        "cgz_num": shell_model.fea.outputs_dict["cgz_num"],
    }
    shell_model.fea.outputs_field_dict = {}
    return shell_model


def _fenics_ordered_inputs(shell_model):
    nn = shell_model.nn
    mesh_indices = np.asarray(shell_model.shell_pde.mesh.geometry.input_global_indices)

    thickness = 0.1 * np.ones(nn)
    E = 1.0e8 * np.ones(nn)
    nu = 0.3 * np.ones(nn)
    density = 10.0 * np.ones(nn)

    nodal_pressure = np.zeros((nn, 3))
    nodal_pressure[:, 2] = 5.0
    node_disp = np.zeros((nn, 3))

    return dict(
        thickness=thickness[mesh_indices],
        E=E[mesh_indices],
        nu=nu[mesh_indices],
        density=density[mesh_indices],
        F_solid=nodal_pressure[mesh_indices].reshape(-1),
        M_solid=np.zeros(shell_model.fea.inputs_dict["M_solid"]["shape"]),
        load_vector=shell_model.assemble_generalized_load_vector(
            nodal_pressure=nodal_pressure,
            node_disp=node_disp,
        ),
        uhat=node_disp[mesh_indices].reshape(-1),
        nodal_pressure=nodal_pressure,
        node_disp=node_disp,
    )


def _fenics_ordered_inputs_with_moments(shell_model):
    vals = _fenics_ordered_inputs(shell_model)
    nn = shell_model.nn
    mesh_indices = np.asarray(shell_model.shell_pde.mesh.geometry.input_global_indices)

    nodal_moments = np.zeros((nn, 3))
    nodal_moments[:, 1] = np.linspace(0.1, 0.4, nn)

    vals["M_solid"] = nodal_moments[mesh_indices].reshape(-1)
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
    uhat_func = Function(shell_model.shell_pde.VU)
    f_func.x.array[:] = field_direction
    m_func.x.array[:] = 0.0
    uhat_func.x.array[:] = 0.0
    return shell_model.shell_pde.assemble_generalized_load_vector(
        uhat=uhat_func,
        f=f_func,
        m=m_func,
    )


def _assemble_combined_load_direction(shell_model, force_direction, moment_direction):
    f_func = Function(shell_model.shell_pde.VF)
    m_func = Function(shell_model.shell_pde.VF)
    uhat_func = Function(shell_model.shell_pde.VU)
    f_func.x.array[:] = force_direction
    m_func.x.array[:] = moment_direction
    uhat_func.x.array[:] = 0.0
    return shell_model.shell_pde.assemble_generalized_load_vector(
        uhat=uhat_func,
        f=f_func,
        m=m_func,
    )


def _assembled_compliance_from_current_fea(shell_model):
    fea = shell_model.fea
    return assemble_scalar(form(fea.outputs_dict["compliance"]["form"]))


def _expected_compliance_from_load_vector(load_vector, disp_solid):
    return np.asarray(load_vector).reshape(-1) @ np.asarray(disp_solid).reshape(-1)


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

    thickness = csdl.Variable(value=field_vals["thickness"], name="thickness")
    E = csdl.Variable(value=field_vals["E"], name="E")
    nu = csdl.Variable(value=field_vals["nu"], name="nu")
    density = csdl.Variable(value=field_vals["density"], name="density")
    nodal_pressure = csdl.Variable(value=field_vals["nodal_pressure"], name="nodal_pressure")
    node_disp = csdl.Variable(value=field_vals["node_disp"], name="node_disp")
    direct_load_vector = csdl.Variable(
        value=direct_vals["load_vector"],
        name="direct_load_vector",
    )

    field_outputs = field_shell.evaluate(
        thickness=thickness,
        E=E,
        nu=nu,
        density=density,
        nodal_pressure=nodal_pressure,
        node_disp=node_disp,
    )
    direct_outputs = direct_shell.evaluate(
        thickness=thickness,
        E=E,
        nu=nu,
        density=density,
        load_vector=direct_load_vector,
        node_disp=node_disp,
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
        [nodal_pressure, direct_load_vector],
        step_size=1e-6,
        print_results=False,
        raise_on_error=True,
    )

    dq_dp = csdl.derivative(q_field, [nodal_pressure])[nodal_pressure]
    dq_dL = csdl.derivative(q_direct, [direct_load_vector])[direct_load_vector]

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

    thickness = csdl.Variable(value=vals["thickness"], name="thickness_fm")
    E = csdl.Variable(value=vals["E"], name="E_fm")
    nu = csdl.Variable(value=vals["nu"], name="nu_fm")
    density = csdl.Variable(value=vals["density"], name="density_fm")
    nodal_pressure = csdl.Variable(value=vals["nodal_pressure"], name="nodal_pressure_fm")
    nodal_moments = csdl.Variable(value=vals["nodal_moments"], name="nodal_moments_fm")
    node_disp = csdl.Variable(value=vals["node_disp"], name="node_disp_fm")
    direct_load_vector = csdl.Variable(
        value=direct_vals["load_vector"],
        name="direct_load_vector_fm",
    )

    field_outputs = field_shell.evaluate(
        thickness=thickness,
        E=E,
        nu=nu,
        density=density,
        nodal_pressure=nodal_pressure,
        nodal_moments=nodal_moments,
        node_disp=node_disp,
    )
    direct_outputs = direct_shell.evaluate(
        thickness=thickness,
        E=E,
        nu=nu,
        density=density,
        load_vector=direct_load_vector,
        node_disp=node_disp,
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
        [nodal_pressure, nodal_moments, direct_load_vector],
        step_size=1e-6,
        print_results=False,
        raise_on_error=True,
    )

    dq_dp = csdl.derivative(q_field, [nodal_pressure])[nodal_pressure]
    dq_dm = csdl.derivative(q_field, [nodal_moments])[nodal_moments]
    dq_dL = csdl.derivative(q_direct, [direct_load_vector])[direct_load_vector]

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

    thickness = csdl.Variable(value=vals["thickness"], name="thickness_tb")
    E = csdl.Variable(value=vals["E"], name="E_tb")
    nu = csdl.Variable(value=vals["nu"], name="nu_tb")
    density = csdl.Variable(value=vals["density"], name="density_tb")
    nodal_pressure = csdl.Variable(value=vals["nodal_pressure"], name="nodal_pressure_tb")
    node_disp = csdl.Variable(value=vals["node_disp"], name="node_disp_tb")

    outputs = shell_model.evaluate(
        thickness=thickness,
        E=E,
        nu=nu,
        density=density,
        nodal_pressure=nodal_pressure,
        node_disp=node_disp,
    )

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


def test_rmshell_compliance_matches_assembled_output_form():
    shell_model = _build_shell_model(
        str(RECORDS_ROOT / "rmshell_compliance_check"),
        nx=4,
        ny=2,
    )
    vals = _fenics_ordered_inputs(shell_model)

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    thickness = csdl.Variable(value=vals["thickness"], name="thickness_comp")
    E = csdl.Variable(value=vals["E"], name="E_comp")
    nu = csdl.Variable(value=vals["nu"], name="nu_comp")
    density = csdl.Variable(value=vals["density"], name="density_comp")
    nodal_pressure = csdl.Variable(value=vals["nodal_pressure"], name="nodal_pressure_comp")
    node_disp = csdl.Variable(value=vals["node_disp"], name="node_disp_comp")

    outputs = shell_model.evaluate(
        thickness=thickness,
        E=E,
        nu=nu,
        density=density,
        nodal_pressure=nodal_pressure,
        node_disp=node_disp,
    )

    assembled_field_compliance = _assembled_compliance_from_current_fea(shell_model)
    expected_field_compliance = _expected_compliance_from_load_vector(
        vals["load_vector"],
        outputs.disp_solid.value,
    )

    np.testing.assert_allclose(
        assembled_field_compliance,
        expected_field_compliance,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        outputs.compliance.value,
        expected_field_compliance,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        outputs.compliance.value,
        2.0 * outputs.elastic_energy.value,
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

    thickness = csdl.Variable(value=direct_vals["thickness"], name="thickness_comp_mix")
    E = csdl.Variable(value=direct_vals["E"], name="E_comp_mix")
    nu = csdl.Variable(value=direct_vals["nu"], name="nu_comp_mix")
    density = csdl.Variable(value=direct_vals["density"], name="density_comp_mix")
    node_disp = csdl.Variable(value=direct_vals["node_disp"], name="node_disp_comp_mix")
    direct_load_vector = csdl.Variable(
        value=direct_vals["load_vector"],
        name="direct_only_load_vector",
    )

    direct_outputs = direct_shell.evaluate(
        thickness=thickness,
        E=E,
        nu=nu,
        density=density,
        load_vector=direct_load_vector,
        node_disp=node_disp,
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
    np.testing.assert_allclose(
        direct_outputs.compliance.value,
        2.0 * direct_outputs.elastic_energy.value,
        rtol=1e-8,
        atol=1e-8,
    )

    nodal_pressure = csdl.Variable(
        value=mixed_vals["nodal_pressure"],
        name="nodal_pressure_comp_mix",
    )
    nodal_moments = csdl.Variable(
        value=mixed_vals["nodal_moments"],
        name="nodal_moments_comp_mix",
    )
    mixed_direct_load = csdl.Variable(
        value=extra_direct,
        name="mixed_direct_load_vector",
    )

    mixed_outputs = mixed_shell.evaluate(
        thickness=thickness,
        E=E,
        nu=nu,
        density=density,
        nodal_pressure=nodal_pressure,
        nodal_moments=nodal_moments,
        load_vector=mixed_direct_load,
        node_disp=node_disp,
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
    np.testing.assert_allclose(
        mixed_outputs.compliance.value,
        2.0 * mixed_outputs.elastic_energy.value,
        rtol=1e-8,
        atol=1e-8,
    )

    recorder.stop()
