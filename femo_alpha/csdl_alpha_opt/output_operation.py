from femo_alpha.fea.fea_dolfinx import (
    FEA,
    update,
    assemble,
    computePartials,
    getFuncArray,
    assembleMatrix,
    setUpKSP_MUMPS,
)
import csdl_alpha as csdl
import numpy as np
from ufl import TestFunction, TrialFunction, inner, dx
import ufl
from dolfinx.fem import Function


def _is_mesh_coordinates(arg):
    return arg.get("type", "function") == "mesh_coordinates"


def _update_input(arg, values):
    if not _is_mesh_coordinates(arg):
        update(arg["function"], values)
        return
    values = np.asarray(values).reshape(arg["shape"])
    local_values = values[arg["input_global_indices"]]
    if local_values.shape != arg["mesh"].geometry.x.shape:
        raise ValueError(
            f"Mesh coordinates map to {local_values.shape}; "
            f"expected {arg['mesh'].geometry.x.shape}."
        )
    arg["mesh"].geometry.x[:] = local_values


def _coordinate_function(arg, external_values):
    direction = Function(arg["function_space"])
    values = np.asarray(external_values).reshape(arg["shape"])
    update(direction, values[arg["input_global_indices"]].reshape(-1))
    return direction


def _external_coordinate_values(arg, local_values):
    local_values = np.asarray(local_values).reshape((-1, arg["mesh"].geometry.dim))
    external_values = np.zeros(arg["shape"], dtype=local_values.dtype)
    np.add.at(external_values, arg["input_global_indices"], local_values)
    return external_values


class OutputOperation(csdl.CustomExplicitOperation):
    """
    input: input/state variables
    output: output
    """

    def __init__(self, fea, args_name_list, output_name):
        super().__init__()

        # define any checks for the parameters
        csdl.check_parameter(fea, "fea", types=FEA)
        csdl.check_parameter(args_name_list, "args_name_list", types=list)
        csdl.check_parameter(output_name, "output_name", types=str)

        args_dict = dict()
        for arg_name in args_name_list:
            if arg_name in fea.inputs_dict:
                args_dict[arg_name] = fea.inputs_dict[arg_name]
            elif arg_name in fea.states_dict:
                args_dict[arg_name] = fea.states_dict[arg_name]

        # assign parameters to the class
        self.fea = fea
        self.args_dict = args_dict
        self.output_name = output_name
        self.fea_output = fea.outputs_dict[output_name]
        self.output_dim = 0 # for scalar outputs
        
    def evaluate(self, inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        for arg_name in self.args_dict:
            if getattr(inputs, arg_name) is not None:
                self.declare_input(arg_name, getattr(inputs, arg_name))
            else:
                raise ValueError(f"Variable {arg_name} not found in the FEA model.")

        # declare output variables
        output = self.create_output(self.output_name, (1,))
        output.add_name(self.output_name)

        # declare any derivative parameters
        self.declare_derivative_parameters(self.output_name, '*', dependent=True)

        return output

    def compute(self, input_vals, output_vals):
        for arg_name in input_vals:
            arg = self.args_dict[arg_name]
            _update_input(arg, input_vals[arg_name])

        output_vals[self.output_name] = assemble(self.fea_output["form"])

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        for arg_name in input_vals:
            arg = self.args_dict[arg_name]
            _update_input(arg, input_vals[arg_name])

        for arg_name in input_vals:
            arg = self.args_dict[arg_name]
            if _is_mesh_coordinates(arg):
                coordinate_test = TestFunction(arg["function_space"])
                coordinate_derivative = ufl.derivative(
                    self.fea_output["form"],
                    arg["coordinate"],
                    coordinate_test,
                )
                local_gradient = assemble(coordinate_derivative, dim=1)
                external_gradient = _external_coordinate_values(arg, local_gradient)
                derivatives[self.output_name, arg_name] = external_gradient.reshape((1, -1))
                continue
            derivatives[self.output_name, arg_name] = assemble(
                computePartials(
                    self.fea_output["form"], arg["function"]
                ),
                dim=self.output_dim + 1,
            )


class OutputFieldOperation(csdl.CustomExplicitOperation):
    """
    input: input/state variables
    output: output
    """

    def __init__(self, fea, args_name_list, output_name):
        super().__init__()

        # define any checks for the parameters
        csdl.check_parameter(fea, "fea", types=FEA)
        csdl.check_parameter(args_name_list, "args_name_list", types=list)
        csdl.check_parameter(output_name, "output_name", types=str)

        args_dict = dict()
        for arg_name in args_name_list:
            if arg_name in fea.inputs_dict:
                args_dict[arg_name] = fea.inputs_dict[arg_name]
            elif arg_name in fea.states_dict:
                args_dict[arg_name] = fea.states_dict[arg_name]

        # assign parameters to the class
        self.fea = fea
        self.args_dict = args_dict
        self.output_name = output_name
        self.fea_output = fea.outputs_field_dict[output_name]
        self.output_dim = 1 # for field outputs

        # Lazily initialized projection operators used by jacobian-vector products.
        self._projection_rhs_form = None
        self._projection_mass_matrix = None
        self._projection_mass_form = None
        self._projection_ksp = None

    def evaluate(self, inputs: csdl.VariableGroup):
        # assign method inputs to input dictionary
        for arg_name in self.args_dict:
            if getattr(inputs, arg_name) is not None:
                self.declare_input(arg_name, getattr(inputs, arg_name))
            else:
                raise ValueError(f"Variable {arg_name} not found in the FEA model.")

        # declare output variables
        output = self.create_output(self.output_name, (self.fea_output['shape'],))
        output.add_name(self.output_name)

        # declare any derivative parameters
        self.declare_derivative_parameters(self.output_name, '*', dependent=True)
        return output

    def compute(self, input_vals, output_vals):
        for arg_name in input_vals:
            arg = self.args_dict[arg_name]
            _update_input(arg, input_vals[arg_name])
        self.fea.projectFieldOutput(self.fea_output['form'],self.fea_output['function'])
        output_vals[self.output_name] = getFuncArray(self.fea_output['function'])

        # record the function values in XDMF files
        if self.fea_output['record']:
            self.fea.opt_iter += 1
            self.fea_output['recorder'].write_function(
                self.fea_output['function'], self.fea.opt_iter
            )

    def _initialize_projection_operators(self):
        output_space = self.fea_output['function'].function_space
        test_func = TestFunction(output_space)
        trial_func = TrialFunction(output_space)

        # Projection equation: M * y = b where
        # b_i = int(inner(v(x), w_i) dx)
        self._projection_rhs_form = inner(self.fea_output['form'], test_func) * dx
        self._projection_mass_form = inner(trial_func, test_func) * dx
        self._projection_mass_matrix = assembleMatrix(self._projection_mass_form)
        self._projection_ksp = setUpKSP_MUMPS(self._projection_mass_matrix)

    def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, mode):
        for arg_name in input_vals:
            arg = self.args_dict[arg_name]
            _update_input(arg, input_vals[arg_name])
        update(self.fea_output["function"], output_vals[self.output_name])

        if self.output_name not in d_outputs:
            return

        self._initialize_projection_operators()

        if mode == 'fwd':
            rhs = self._projection_mass_matrix.createVecRight()
            rhs.set(0.0)

            for arg_name in self.args_dict:
                if arg_name in d_inputs:
                    d_arg = d_inputs[arg_name]
                    if d_arg is None:
                        continue
                    arg = self.args_dict[arg_name]
                    if _is_mesh_coordinates(arg):
                        direction = _coordinate_function(arg, d_arg)
                        d_rhs_form = ufl.derivative(
                            self._projection_rhs_form,
                            arg["coordinate"],
                            direction,
                        )
                        rhs.array[:] += assemble(d_rhs_form, dim=1)
                        d_mass_form = ufl.derivative(
                            self._projection_mass_form,
                            arg["coordinate"],
                            direction,
                        )
                        d_mass_matrix = assembleMatrix(d_mass_form)
                        current_output_vec = d_mass_matrix.createVecRight()
                        current_output_vec.array[:] = output_vals[self.output_name].reshape(-1)
                        current_output_vec.assemble()
                        mass_contrib = d_mass_matrix.createVecLeft()
                        d_mass_matrix.mult(current_output_vec, mass_contrib)
                        rhs.array[:] -= mass_contrib.getArray()
                        continue
                    d_rhs_form = computePartials(
                        self._projection_rhs_form,
                        arg['function'],
                    )
                    d_rhs_mat = assembleMatrix(d_rhs_form)

                    d_arg_vec = d_rhs_mat.createVecRight()
                    d_arg_vec.array[:] = d_arg.reshape(-1)
                    d_arg_vec.assemble()

                    rhs_contrib = d_rhs_mat.createVecLeft()
                    d_rhs_mat.mult(d_arg_vec, rhs_contrib)
                    rhs += rhs_contrib

            d_output_vec = self._projection_mass_matrix.createVecRight()
            self._projection_ksp.solve(rhs, d_output_vec)
            d_outputs[self.output_name] += d_output_vec.getArray()

        elif mode == 'rev':
            # Solve M^T * lambda_tilde = lambda.
            # For projection mass matrix M, this is equivalent to M * lambda_tilde = lambda.
            output_cot = d_outputs[self.output_name]
            output_cot_vec = self._projection_mass_matrix.createVecRight()
            output_cot_vec.array[:] = output_cot.reshape(-1)
            output_cot_vec.assemble()

            adjoint_rhs = self._projection_mass_matrix.createVecRight()
            self._projection_ksp.solve(output_cot_vec, adjoint_rhs)

            for arg_name in self.args_dict:
                if arg_name in d_inputs:
                    arg = self.args_dict[arg_name]
                    if _is_mesh_coordinates(arg):
                        coordinate_trial = TrialFunction(arg["function_space"])
                        d_rhs_form = ufl.derivative(
                            self._projection_rhs_form,
                            arg["coordinate"],
                            coordinate_trial,
                        )
                        d_rhs_matrix = assembleMatrix(d_rhs_form)
                        mass_action = ufl.action(
                            self._projection_mass_form,
                            self.fea_output["function"],
                        )
                        d_mass_action = ufl.derivative(
                            mass_action,
                            arg["coordinate"],
                            coordinate_trial,
                        )
                        d_mass_matrix = assembleMatrix(d_mass_action)
                        local_contrib = d_rhs_matrix.createVecRight()
                        d_rhs_matrix.multTranspose(adjoint_rhs, local_contrib)
                        mass_contrib = d_mass_matrix.createVecRight()
                        d_mass_matrix.multTranspose(adjoint_rhs, mass_contrib)
                        local_contrib.array[:] -= mass_contrib.getArray()
                        external = _external_coordinate_values(arg, local_contrib.getArray())
                        d_inputs[arg_name] += external
                        continue
                    d_rhs_form = computePartials(
                        self._projection_rhs_form,
                        self.args_dict[arg_name]['function'],
                    )
                    d_rhs_mat = assembleMatrix(d_rhs_form)
                    input_contrib = d_rhs_mat.createVecRight()
                    d_rhs_mat.multTranspose(adjoint_rhs, input_contrib)
                    d_inputs[arg_name] += input_contrib.getArray().reshape(
                        d_inputs[arg_name].shape
                    )

        else:
            raise ValueError("mode must be either 'fwd' or 'rev'.")

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        raise NotImplementedError(
            "OutputFieldOperation uses compute_jacvec_product for derivatives. "
            "Full Jacobian assembly is intentionally disabled for field outputs."
        )
