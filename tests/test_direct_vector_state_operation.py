import numpy as np
import pytest

dolfinx = pytest.importorskip("dolfinx")
pytest.importorskip("petsc4py")
pytest.importorskip("ufl")
pytest.importorskip("scipy")

import csdl_alpha as csdl
import ufl
from mpi4py import MPI
from scipy.sparse import csr_matrix
from dolfinx.fem import Function, FunctionSpace, form, locate_dofs_geometrical
from dolfinx.fem.petsc import assemble_vector
from dolfinx.mesh import create_interval

from femo_alpha.csdl_alpha_opt.fea_model import FEAModel
from femo_alpha.csdl_alpha_opt.state_operation import StateOperation
from femo_alpha.fea.fea_dolfinx import FEA, getFuncArray


def _vec_array(vec):
    return vec.array if hasattr(vec, "array") else vec.getArray()


def _boundary_dofs(V):
    left = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    right = locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 1.0))
    return np.unique(np.concatenate([left, right])).astype(int)


def _make_dirichlet_zero_bc(V):
    ubc = Function(V)
    ubc.vector.set(0.0)
    return ubc


def _build_function_load_problem():
    mesh = create_interval(MPI.COMM_WORLD, 8, [0.0, 1.0])
    V = FunctionSpace(mesh, ("CG", 1))

    fea = FEA(mesh)
    state = Function(V)
    test = ufl.TestFunction(V)
    load = Function(V)

    residual_form = ufl.inner(ufl.grad(state), ufl.grad(test)) * ufl.dx - ufl.inner(load, test) * ufl.dx

    fea.add_input("load_function", load, init_val=0.0)
    load.interpolate(lambda x: 1.0 + 0.25 * x[0])
    fea.add_state(
        name="state",
        function=state,
        residual_form=residual_form,
        arguments=["load_function"],
    )
    fea.add_strong_bc(_make_dirichlet_zero_bc(V), _boundary_dofs(V))

    return fea, load


def _build_direct_vector_problem(scatter=None):
    mesh = create_interval(MPI.COMM_WORLD, 8, [0.0, 1.0])
    V = FunctionSpace(mesh, ("CG", 1))

    fea = FEA(mesh)
    state = Function(V)
    test = ufl.TestFunction(V)
    residual_form = ufl.inner(ufl.grad(state), ufl.grad(test)) * ufl.dx

    constrained_dofs = _boundary_dofs(V)
    state_size = len(getFuncArray(state))
    input_size = state_size if scatter is None else scatter.shape[1]

    fea.add_direct_vector_input(
        "load_vector",
        shape=input_size,
        sign=-1.0,
        scatter=scatter,
        constrained_dofs=constrained_dofs,
    )
    fea.add_state(
        name="state",
        function=state,
        residual_form=residual_form,
        arguments=["load_vector"],
    )
    fea.add_strong_bc(_make_dirichlet_zero_bc(V), constrained_dofs)

    return fea, constrained_dofs


def _assemble_load_vector(load_function, state_function):
    test = ufl.TestFunction(state_function.function_space)
    load_form = form(ufl.inner(load_function, test) * ufl.dx)
    return _vec_array(assemble_vector(load_form)).copy()


def _solve_state(operation, input_name, input_value):
    outputs = {}
    operation.solve_residual_equations({input_name: input_value}, outputs)
    return outputs["state"].copy()


def test_direct_vector_matches_function_load_solution():
    fea_function, load_function = _build_function_load_problem()
    function_op = StateOperation(fea_function, ["load_function"], "state")
    function_solution = _solve_state(
        function_op,
        "load_function",
        getFuncArray(load_function),
    )

    direct_load = _assemble_load_vector(
        load_function,
        fea_function.states_dict["state"]["function"],
    )

    fea_direct, _ = _build_direct_vector_problem()
    direct_op = StateOperation(fea_direct, ["load_vector"], "state")
    direct_solution = _solve_state(direct_op, "load_vector", direct_load)

    np.testing.assert_allclose(
        direct_solution,
        function_solution,
        rtol=1e-10,
        atol=1e-12,
    )


def test_direct_vector_jacvec_forward_and_reverse_with_bc_masking():
    fea, constrained_dofs = _build_direct_vector_problem()
    operation = StateOperation(fea, ["load_vector"], "state")
    base_load = np.zeros(fea.states_dict["state"]["shape"])
    outputs = {}
    operation.solve_residual_equations({"load_vector": base_load}, outputs)

    dF = np.linspace(0.5, 1.5, base_load.size)
    d_residuals = {"state": np.zeros_like(base_load)}
    operation.compute_jacvec_product(
        {"load_vector": base_load},
        outputs,
        {"load_vector": dF.copy()},
        {},
        d_residuals,
        mode="fwd",
    )

    expected_fwd = -dF.copy()
    expected_fwd[constrained_dofs] = 0.0
    np.testing.assert_allclose(
        d_residuals["state"],
        expected_fwd,
        rtol=1e-12,
        atol=1e-12,
    )

    psi = np.linspace(-1.0, 2.0, base_load.size)
    d_inputs = {"load_vector": np.zeros_like(base_load)}
    operation.compute_jacvec_product(
        {"load_vector": base_load},
        outputs,
        d_inputs,
        {},
        {"state": psi.copy()},
        mode="rev",
    )

    expected_rev = -psi.copy()
    expected_rev[constrained_dofs] = 0.0
    np.testing.assert_allclose(
        d_inputs["load_vector"],
        expected_rev,
        rtol=1e-12,
        atol=1e-12,
    )

    lhs = psi @ d_residuals["state"]
    rhs = d_inputs["load_vector"] @ dF
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_direct_vector_scatter_products_and_dot_consistency():
    fea_full, constrained_dofs = _build_direct_vector_problem()
    state_size = fea_full.states_dict["state"]["shape"]
    free_dofs = np.setdiff1d(np.arange(state_size), constrained_dofs)
    selected = free_dofs[: max(1, len(free_dofs) // 2)]
    scatter = csr_matrix(
        (
            np.ones(len(selected)),
            (selected, np.arange(len(selected))),
        ),
        shape=(state_size, len(selected)),
    )

    fea, _ = _build_direct_vector_problem(scatter=scatter)
    operation = StateOperation(fea, ["load_vector"], "state")
    base_load = np.zeros(scatter.shape[1])
    outputs = {}
    operation.solve_residual_equations({"load_vector": base_load}, outputs)

    dF_small = np.linspace(1.0, 2.0, scatter.shape[1])
    d_residuals = {"state": np.zeros(state_size)}
    operation.compute_jacvec_product(
        {"load_vector": base_load},
        outputs,
        {"load_vector": dF_small.copy()},
        {},
        d_residuals,
        mode="fwd",
    )
    expected_fwd = -(scatter @ dF_small)
    np.testing.assert_allclose(
        d_residuals["state"],
        expected_fwd,
        rtol=1e-12,
        atol=1e-12,
    )

    psi = np.linspace(-0.25, 1.75, state_size)
    psi_masked = psi.copy()
    psi_masked[constrained_dofs] = 0.0
    d_inputs = {"load_vector": np.zeros(scatter.shape[1])}
    operation.compute_jacvec_product(
        {"load_vector": base_load},
        outputs,
        d_inputs,
        {},
        {"state": psi.copy()},
        mode="rev",
    )
    expected_rev = -(scatter.T @ psi_masked)
    np.testing.assert_allclose(
        d_inputs["load_vector"],
        expected_rev,
        rtol=1e-12,
        atol=1e-12,
    )

    lhs = psi @ d_residuals["state"]
    rhs = d_inputs["load_vector"] @ dF_small
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_constrained_direct_load_does_not_change_solution():
    fea, constrained_dofs = _build_direct_vector_problem()
    operation = StateOperation(fea, ["load_vector"], "state")
    zero_load = np.zeros(fea.states_dict["state"]["shape"])
    zero_solution = _solve_state(operation, "load_vector", zero_load)

    constrained_load = np.zeros_like(zero_load)
    constrained_load[constrained_dofs] = np.linspace(1.0, 2.0, len(constrained_dofs))
    constrained_solution = _solve_state(operation, "load_vector", constrained_load)

    np.testing.assert_allclose(
        constrained_solution,
        zero_solution,
        rtol=1e-12,
        atol=1e-12,
    )


def test_total_derivative_through_implicit_direct_vector_matches_fd():
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    fea, _ = _build_direct_vector_problem()
    fea_model = FEAModel(fea=[fea])
    load_size = fea.states_dict["state"]["shape"]
    base_load = np.linspace(0.1, 0.9, load_size)
    perturbation = np.linspace(-0.5, 0.5, load_size)

    inputs = csdl.VariableGroup()
    load_vector = csdl.Variable(value=base_load, name="load_vector")
    inputs.load_vector = load_vector

    fea_outputs = fea_model.evaluate(inputs)
    q = csdl.sum(fea_outputs.state)
    dq_dF = csdl.derivative(q, [load_vector])[load_vector]

    directional_adj = dq_dF.value.reshape(-1) @ perturbation

    operation = StateOperation(fea, ["load_vector"], "state")
    q_plus = np.sum(_solve_state(operation, "load_vector", base_load + 1e-6 * perturbation))
    q_minus = np.sum(_solve_state(operation, "load_vector", base_load - 1e-6 * perturbation))
    directional_fd = (q_plus - q_minus) / (2e-6)

    np.testing.assert_allclose(
        directional_adj,
        directional_fd,
        rtol=1e-6,
        atol=1e-8,
    )
