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
            update(arg["function"], input_vals[arg_name])

        output_vals[self.output_name] = assemble(self.fea_output["form"])

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        for arg_name in input_vals:
            arg = self.args_dict[arg_name]
            update(arg["function"], input_vals[arg_name])

        for arg_name in input_vals:
            derivatives[self.output_name, arg_name] = assemble(
                computePartials(
                    self.fea_output["form"], self.args_dict[arg_name]["function"]
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
            update(arg['function'], input_vals[arg_name])
        self.fea.projectFieldOutput(self.fea_output['form'],self.fea_output['function'])
        output_vals[self.output_name] = getFuncArray(self.fea_output['function'])

        # record the function values in XDMF files
        if self.fea_output['record']:
            self.fea.opt_iter += 1
            self.fea_output['recorder'].write_function(
                self.fea_output['function'], self.fea.opt_iter
            )

    def _initialize_projection_operators(self):
        if self._projection_mass_matrix is not None:
            return

        output_space = self.fea_output['function'].function_space
        test_func = TestFunction(output_space)
        trial_func = TrialFunction(output_space)

        # Projection equation: M * y = b where
        # b_i = int(inner(v(x), w_i) dx)
        self._projection_rhs_form = inner(self.fea_output['form'], test_func) * dx
        mass_form = inner(trial_func, test_func) * dx
        self._projection_mass_matrix = assembleMatrix(mass_form)
        self._projection_ksp = setUpKSP_MUMPS(self._projection_mass_matrix)

    def compute_jacvec_product(self, input_vals, output_vals, d_inputs, d_outputs, mode):
        for arg_name in input_vals:
            arg = self.args_dict[arg_name]
            update(arg['function'], input_vals[arg_name])

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
                    d_rhs_form = computePartials(
                        self._projection_rhs_form,
                        self.args_dict[arg_name]['function'],
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
