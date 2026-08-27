import dolfinx
from dolfinx.fem import Function
import csdl_alpha as csdl
import ufl
import numpy as np
import os
import copy

from femo_alpha.fea.fea_dolfinx import FEA, getFuncArray
from femo_alpha.fea.utils_dolfinx import createCustomMeasure
from femo_alpha.rm_shell.rm_shell_pde_composite import RMShellPDE as CompositeShellPDE
from femo_alpha.csdl_alpha_opt.fea_model import FEAModel
from femo_alpha.rm_shell.linear_shell_fenicsx.linear_shell_model import custom_solve_direct


class ShellState:
    """Container for the solved shell state and the inputs used to obtain it."""

    def __init__(self, material, loads, node_disp, disp_solid, raw_outputs, mesh_nodes):
        self.material = material
        self.loads = loads
        self.node_disp = node_disp
        self.disp_solid = disp_solid
        self.mesh_nodes = mesh_nodes
        self.raw_outputs = raw_outputs


class MaterialInputFactory:
    """Factory for creating canonical material input groups for RMShell."""

    def __init__(self, shell_model):
        self.shell_model = shell_model

    def from_isotropic(self, E, nu, thickness, density):
        """Create canonical shell material inputs from isotropic material data."""
        thickness = self.shell_model._ensure_variable(thickness, "thickness")
        ref_size = thickness.shape[0]
        E = self.shell_model._broadcast_variable(E, (ref_size,), "E")
        nu = self.shell_model._broadcast_variable(nu, (ref_size,), "nu")
        density = self.shell_model._broadcast_variable(density, (ref_size,), "density")

        prefactor = E / (1.0 - nu * nu)
        c11 = prefactor
        c12 = prefactor * nu
        c33 = prefactor * 0.5 * (1.0 - nu)
        shear = 0.833 * (E / (2.0 * (1.0 + nu))) * thickness

        zeros = 0.0 * thickness

        A = self.shell_model._assemble_tensor33(
            thickness * c11,
            thickness * c12,
            zeros,
            thickness * c12,
            thickness * c11,
            zeros,
            zeros,
            zeros,
            thickness * c33,
        )
        B = csdl.Variable(value=np.zeros((ref_size, 3, 3)))
        D = self.shell_model._assemble_tensor33(
            thickness**3 / 12.0 * c11,
            thickness**3 / 12.0 * c12,
            zeros,
            thickness**3 / 12.0 * c12,
            thickness**3 / 12.0 * c11,
            zeros,
            zeros,
            zeros,
            thickness**3 / 12.0 * c33,
        )
        As = self.shell_model._assemble_tensor22(
            shear,
            zeros,
            zeros,
            shear,
        )

        material = csdl.VariableGroup()
        material.kind = "abd"
        material.is_isotropic = True
        material.thickness = thickness
        material.E = E
        material.nu = nu
        material.A = A
        material.B = B
        material.D = D
        material.As = As
        material.density = density
        return material

    def from_abd(self, A, B, D, As, thickness, density):
        """Create canonical shell material inputs directly from A/B/D/As matrices."""
        material = csdl.VariableGroup()
        material.kind = "abd"
        material.is_isotropic = False
        material.thickness = self.shell_model._ensure_variable(thickness, "thickness")
        material.A = self.shell_model._ensure_variable(A, "A")
        material.B = self.shell_model._ensure_variable(B, "B")
        material.D = self.shell_model._ensure_variable(D, "D")
        material.As = self.shell_model._ensure_variable(As, "As")
        material.density = self.shell_model._ensure_variable(density, "density")
        return material


class LoadInputFactory:
    """Factory for building canonical shell load input groups."""

    def __init__(self, shell_model):
        self.shell_model = shell_model

    def from_fields(self, nodal_forces=None, nodal_pressure=None, nodal_moments=None):
        """Create a load group from nodal force, pressure, and moment fields."""
        loads = csdl.VariableGroup()
        loads.nodal_forces = self.shell_model._maybe_variable(nodal_forces, "nodal_forces")
        loads.nodal_pressure = self.shell_model._maybe_variable(nodal_pressure, "nodal_pressure")
        loads.nodal_moments = self.shell_model._maybe_variable(nodal_moments, "nodal_moments")
        loads.load_vector = None
        return loads

    def from_vector(self, load_vector):
        """Create a load group from a direct generalized shell load vector."""
        loads = csdl.VariableGroup()
        loads.nodal_forces = None
        loads.nodal_pressure = None
        loads.nodal_moments = None
        loads.load_vector = self.shell_model._ensure_variable(load_vector, "load_vector")
        return loads

    def combine(self, *load_groups):
        """Combine multiple load groups into a single canonical load description."""
        combined = self.from_fields()
        for group in load_groups:
            if group is None:
                continue
            for name in ("nodal_forces", "nodal_pressure", "nodal_moments", "load_vector"):
                val = getattr(group, name, None)
                if val is None:
                    continue
                current = getattr(combined, name, None)
                setattr(combined, name, val if current is None else current + val)
        return combined


class ScalarFormOutput:
    """Postprocessing output assembled from one scalar UFL form."""

    def __init__(self, name, form_fn, arguments):
        self.name = name
        self.form_fn = form_fn
        self.arguments = arguments

    def collect_forms(self, context, scalar_forms, field_forms):
        scalar_forms[self.name] = (self.form_fn(context), self.arguments)

    def finalize(self, context, form_outputs, outputs):
        setattr(outputs, self.name, form_outputs[self.name])


class FieldFormOutput:
    """Postprocessing output projected from one UFL field form."""

    def __init__(self, name, form_fn, arguments, element, reorder=None, record=None, vtk=False):
        self.name = name
        self.form_fn = form_fn
        self.arguments = arguments
        self.element = element
        self.reorder = reorder
        self.record = record
        self.vtk = vtk

    def collect_forms(self, context, scalar_forms, field_forms):
        field_forms[self.name] = (
            self.form_fn(context),
            self.arguments,
            self.element,
            self.reorder,
            self.record,
            self.vtk,
        )

    def finalize(self, context, form_outputs, outputs):
        setattr(outputs, self.name, form_outputs[self.name])


class RatioFormOutput:
    """Postprocessing output computed as the ratio of two scalar forms."""

    def __init__(self, name, numerator_fn, numerator_arguments, denominator_fn, denominator_arguments):
        self.name = name
        self.numerator_name = f"__{name}_numerator"
        self.denominator_name = f"__{name}_denominator"
        self.numerator_fn = numerator_fn
        self.numerator_arguments = numerator_arguments
        self.denominator_fn = denominator_fn
        self.denominator_arguments = denominator_arguments

    def collect_forms(self, context, scalar_forms, field_forms):
        scalar_forms[self.numerator_name] = (self.numerator_fn(context), self.numerator_arguments)
        scalar_forms[self.denominator_name] = (self.denominator_fn(context), self.denominator_arguments)

    def finalize(self, context, form_outputs, outputs):
        setattr(outputs, self.name, form_outputs[self.numerator_name] / form_outputs[self.denominator_name])


class CSDLOutput:
    """Postprocessing output computed from CSDL operations after form assembly."""

    def __init__(self, name, compute_fn, dependencies=()):
        self.name = name
        self.compute_fn = compute_fn
        self.dependencies = tuple(dependencies)

    def collect_forms(self, context, scalar_forms, field_forms):
        for dependency in self.dependencies:
            dependency.collect_forms(context, scalar_forms, field_forms)

    def finalize(self, context, form_outputs, outputs):
        for dependency in self.dependencies:
            if not hasattr(outputs, dependency.name):
                dependency.finalize(context, form_outputs, outputs)
        value = self.compute_fn(context, form_outputs, outputs)
        if hasattr(value, "add_name"):
            value.add_name(self.name)
        setattr(outputs, self.name, value)


class ShellPostContext:
    """Evaluation context for shell postprocessing outputs."""

    def __init__(self, post_processor, state=None, material=None, loads=None, displacement=None, node_disp=None, mesh_nodes=None, debug_mode=False):
        self.post = post_processor
        self.shell_model = post_processor.shell_model
        if state is not None:
            material = state.material
            loads = state.loads
            displacement = state.disp_solid
            node_disp = state.node_disp
            mesh_nodes = state.mesh_nodes
        else:
            if material is None or loads is None or displacement is None:
                raise ValueError("Provide either a ShellState or material, loads, and displacement.")
        self.state = state
        self.material = material
        self.loads = loads
        self.node_disp = node_disp
        self.displacement = displacement
        self.debug_mode = debug_mode
        self.mesh_nodes = mesh_nodes
        self._post_inputs = None

    @property
    def shell_pde(self):
        """Return the shell PDE object used by this context."""
        return self.shell_model.shell_pde

    @property
    def post_fea(self):
        """Return the postprocessing FEA object used by this context."""
        return self.shell_model.post_fea

    @property
    def solve_fea(self):
        """Return the solve FEA object used by this context."""
        return self.shell_model.fea

    def inputs(self):
        """Return the CSDL inputs used for postprocessing in FEniCS ordering."""
        if self._post_inputs is None:
            self._post_inputs = self.shell_model._prepare_solver_inputs(
                self.material,
                self.loads,
                node_disp=self.node_disp,
                mesh_nodes=self.mesh_nodes,
            )
            self._post_inputs.disp_solid = self.displacement
        return self._post_inputs

    def region_measure(self, region=None):
        """Return the appropriate UFL measure for a named or indexed region."""
        if region is None:
            return ufl.dx
        if not hasattr(self.shell_model, "association_table"):
            raise ValueError("No mesh tags are available on this shell model.")
        if isinstance(region, str):
            if region not in self.shell_model.association_table:
                raise KeyError(f"Unknown region '{region}'.")
            region = self.shell_model.association_table[region]
        return self.shell_model.dxx(region)

    def area_form(self, region=None):
        """Return the current-mesh area form over the requested region."""
        return self.shell_pde.area_subdomain(self.region_measure(region))

    def compute(self):
        """Compute the outputs currently requested on the postprocessor."""
        return self.post.compute(context=self)


class ShellPostProcessor:
    """Request-and-compute shell postprocessing interface."""

    _STRAIN_COMPONENTS = {
        "xx": (0, 0),
        "yy": (1, 1),
        "xy": (0, 1),
        "yx": (1, 0),
    }

    def __init__(self, shell_model):
        self.shell_model = shell_model
        self._outputs = []
        self._record_file_counter = 0

    def clear(self):
        """Remove all requested postprocessing outputs."""
        self._outputs.clear()
        return self

    def _add(self, output):
        self._outputs.append(output)
        return self

    def _displacement_components(self, context):
        return ufl.split(context.post_fea.inputs_dict["disp_solid"]["function"])

    def _compliance_form(self, context):
        u_mid, theta = self._displacement_components(context)
        return context.shell_pde.compliance(
            u_mid,
            theta,
            context.post_fea.inputs_dict["thickness"]["function"],
            context.post_fea.inputs_dict["F_solid"]["function"],
            context.post_fea.inputs_dict["M_solid"]["function"],
        )

    def _mass_and_cg_forms(self, context):
        return context.shell_pde.cg_form(
            context.post_fea.inputs_dict["thickness"]["function"],
            context.post_fea.inputs_dict["density"]["function"],
        )

    def _elastic_energy_form(self, context):
        return context.shell_pde.elastic_energy(
            context.post_fea.inputs_dict["disp_solid"]["function"],
            context.post_fea.inputs_dict["thickness"]["function"],
            None,
        )

    def _strain_element(self):
        return ufl.TensorElement("DG", self.shell_model.mesh.ufl_cell(), degree=0, shape=(2, 2))

    def _shear_strain_element(self):
        return ufl.VectorElement("DG", self.shell_model.mesh.ufl_cell(), degree=0, dim=2)

    def _vector_node_element(self):
        return ufl.VectorElement("CG", self.shell_model.mesh.ufl_cell(), degree=1, dim=3)

    def add_default_outputs(self):
        """Request the default shell output bundle."""
        return (
            self.compliance()
            .mass()
            .cg()
            .elastic_energy()
            .disp_extracted()
            .displacements()
            .rotations()
            .aggregated_stress()
        )

    def add_mass_properties(self):
        """Request mass and center-of-gravity outputs."""
        return self.mass().cg()

    def add_kinematics(self):
        """Request displacement and rotation kinematic outputs."""
        return self.disp_extracted().displacements().rotations()

    def add_strains(self):
        """Request strain-related field outputs."""
        return self.mid_strain().shear_strain().curvature()

    def add_stress_outputs(self, include_fields=False):
        """Request built-in stress outputs."""
        self.aggregated_stress()
        if include_fields:
            self.stress().stress_top().stress_mid().stress_bottom()
        return self

    def context(self, state=None, material=None, loads=None, displacement=None, node_disp=None, mesh_nodes=None, debug_mode=False):
        """Create a reusable postprocessing context for a solved or external displacement."""
        return ShellPostContext(
            self,
            state=state,
            material=material,
            loads=loads,
            displacement=displacement,
            node_disp=node_disp,
            debug_mode=debug_mode,
            mesh_nodes=mesh_nodes,
        )

    def add_scalar_form(self, name, form_fn, arguments):
        """Request a custom scalar output assembled from a UFL form."""
        return self._add(ScalarFormOutput(name, form_fn, arguments))

    def add_field_form(self, name, form_fn, arguments, element, reorder=None, record=None, vtk=False):
        """Request a custom projected field output assembled from a UFL form."""
        return self._add(FieldFormOutput(name, form_fn, arguments, element, reorder=reorder, record=record, vtk=vtk))

    def add_form_ratio(self, name, numerator_fn, numerator_arguments, denominator_fn, denominator_arguments):
        """Request a custom scalar output computed as numerator form / denominator form."""
        return self._add(RatioFormOutput(name, numerator_fn, numerator_arguments, denominator_fn, denominator_arguments))

    def add_derived_output(self, name, compute_fn):
        """Request a custom CSDL output computed after form assembly."""
        return self._add(CSDLOutput(name, compute_fn))

    def evaluate(self, state=None, material=None, loads=None, displacement=None, node_disp=None, mesh_nodes=None, debug_mode=False):
        """Compatibility wrapper for existing callers."""
        if not self._outputs:
            self.add_default_outputs()
        return self.compute(
            state=state,
            material=material,
            loads=loads,
            displacement=displacement,
            node_disp=node_disp,
            mesh_nodes=mesh_nodes,
            debug_mode=debug_mode,
        )

    def compute(self, state=None, material=None, loads=None, displacement=None, node_disp=None, mesh_nodes=None, debug_mode=False, context=None):
        """Compute all requested postprocessing outputs and return a VariableGroup."""
        if not self._outputs:
            raise ValueError("No RMShell postprocessing outputs have been requested.")
        if context is None:
            context = self.context(
                state=state,
                material=material,
                loads=loads,
                displacement=displacement,
                node_disp=node_disp,
                mesh_nodes=mesh_nodes,
                debug_mode=debug_mode,
            )
        scalar_forms = {}
        field_forms = {}
        for output in self._outputs:
            output.collect_forms(context, scalar_forms, field_forms)

        form_outputs = self._evaluate_forms(context, scalar_forms, field_forms)
        outputs = csdl.VariableGroup()
        for output in self._outputs:
            output.finalize(context, form_outputs, outputs)
        outputs.disp_solid = context.displacement
        outputs.mesh_nodes = context.inputs().mesh_nodes
        return outputs

    def mass_properties(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Request mass and center-of-gravity outputs."""
        return self.add_mass_properties()

    def kinematics(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Request displacement and rotation kinematic outputs."""
        return self.add_kinematics()

    def strains(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Request strain-related field outputs."""
        return self.add_strains()

    def compliance(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Request compliance."""
        raw_compliance = ScalarFormOutput(
            "__raw_compliance",
            self._compliance_form,
            ["disp_solid", "F_solid", "M_solid", "thickness", "mesh_nodes"],
        )

        def compute_compliance(context, form_outputs, outputs):
            return form_outputs["__raw_compliance"] + csdl.vdot(context.inputs().load_vector, context.displacement)

        raw_compliance.name = "__raw_compliance"
        return self._add(CSDLOutput("compliance", compute_compliance, dependencies=[raw_compliance]))

    def mass(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Request mass."""
        return self._add(ScalarFormOutput(
            "mass",
            lambda context: self._mass_and_cg_forms(context)[3],
            ["thickness", "density", "mesh_nodes"],
        ))

    def cg(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Request center of gravity."""
        cgx = ScalarFormOutput(
            "__cgx_num",
            lambda context: self._mass_and_cg_forms(context)[0],
            ["thickness", "density", "mesh_nodes"],
        )
        cgy = ScalarFormOutput(
            "__cgy_num",
            lambda context: self._mass_and_cg_forms(context)[1],
            ["thickness", "density", "mesh_nodes"],
        )
        cgz = ScalarFormOutput(
            "__cgz_num",
            lambda context: self._mass_and_cg_forms(context)[2],
            ["thickness", "density", "mesh_nodes"],
        )
        mass = ScalarFormOutput(
            "__cg_mass",
            lambda context: self._mass_and_cg_forms(context)[3],
            ["thickness", "density", "mesh_nodes"],
        )

        def compute_cg(context, form_outputs, outputs):
            return csdl.concatenate((
                form_outputs["__cgx_num"] / form_outputs["__cg_mass"],
                form_outputs["__cgy_num"] / form_outputs["__cg_mass"],
                form_outputs["__cgz_num"] / form_outputs["__cg_mass"],
            ))

        return self._add(CSDLOutput("cg", compute_cg, dependencies=[cgx, cgy, cgz, mass]))

    def disp_extracted(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Request extracted nodal displacements."""
        return self._add(CSDLOutput(
            "disp_extracted",
            lambda context, form_outputs, outputs: DisplacementExtractionModel(shell_pde=context.shell_pde).evaluate(context.displacement),
        ))

    def mid_strain(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Request mid-surface strain field."""
        return self._add(FieldFormOutput(
            "mid_strain",
            lambda context: context.shell_pde.elastic_model.eps,
            ["disp_solid", "mesh_nodes"],
            self._strain_element(),
            reorder="mid_strain",
            record=self.shell_model.record,
        ))

    def pnorm_stress(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Request the built-in p-norm stress output."""
        return self._add(self._pnorm_stress_output("pnorm_stress", rho=self.shell_model.rho, m=self.shell_model.m))

    def elastic_energy(self):
        return self._add(ScalarFormOutput(
            "elastic_energy",
            self._elastic_energy_form,
            ["thickness", "disp_solid", "mesh_nodes", "A", "B", "D", "As", "E"],
        ))

    def displacements(self):
        return self._add(FieldFormOutput(
            "displacements",
            lambda context: self._displacement_components(context)[0],
            ["disp_solid", "mesh_nodes"],
            self._vector_node_element(),
            reorder="displacement",
            record=self.shell_model.record,
        ))

    def rotations(self):
        return self._add(FieldFormOutput(
            "rotations",
            lambda context: self._displacement_components(context)[1],
            ["disp_solid", "mesh_nodes"],
            self._vector_node_element(),
            reorder="rotation",
            record=self.shell_model.record,
        ))

    def shear_strain(self):
        return self._add(FieldFormOutput(
            "shear_strain",
            lambda context: context.shell_pde.elastic_model.gamma,
            ["disp_solid", "mesh_nodes"],
            self._shear_strain_element(),
            reorder="shear_strain",
            record=self.shell_model.record,
        ))

    def curvature(self):
        return self._add(FieldFormOutput(
            "curvature",
            lambda context: context.shell_pde.elastic_model.kappa,
            ["disp_solid", "mesh_nodes"],
            self._strain_element(),
            reorder="curvature",
            record=self.shell_model.record,
        ))

    def stress(self):
        return self._add(self._von_mises_stress_output("stress", surface="top"))

    def stress_top(self):
        return self._add(self._von_mises_stress_output("stress_top", surface="top"))

    def stress_mid(self):
        return self._add(self._von_mises_stress_output("stress_mid", surface="mid"))

    def stress_bottom(self):
        return self._add(self._von_mises_stress_output("stress_bottom", surface="bottom"))

    def aggregated_stress(self):
        pnorm = self._pnorm_stress_output("__aggregated_pnorm_stress", rho=self.shell_model.rho, m=self.shell_model.m)

        def compute_aggregated(context, form_outputs, outputs):
            return AggregatedStressModel(m=context.shell_model.m, rho=context.shell_model.rho).evaluate(
                form_outputs["__aggregated_pnorm_stress"]
            )

        return self._add(CSDLOutput("aggregated_stress", compute_aggregated, dependencies=[pnorm]))

    def average_strain(self, name, strain_type="mid", component="xx", region=None):
        if component not in self._STRAIN_COMPONENTS:
            raise ValueError(f"Unsupported strain component '{component}'.")
        i, j = self._STRAIN_COMPONENTS[component]

        def strain_form(context):
            if strain_type == "mid":
                strain = context.shell_pde.elastic_model.eps
            elif strain_type == "curvature":
                strain = context.shell_pde.elastic_model.kappa
            else:
                raise ValueError(f"Unsupported strain_type '{strain_type}'.")
            return strain[i, j] * ufl.as_ufl(1.0) * context.region_measure(region)

        return self.add_form_ratio(
            name,
            strain_form,
            ["disp_solid", "mesh_nodes"],
            lambda context: context.area_form(region),
            ["mesh_nodes"],
        )

    def pnorm_stress_custom(self, name, rho=100, m=1e-6, region=None):
        return self._add(self._pnorm_stress_output(name, rho=rho, m=m, region=region))

    def _pnorm_stress_output(self, name, rho=100, m=1e-6, region=None):
        def form_fn(context):
            if not getattr(context.material, "is_isotropic", False):
                raise ValueError("pnorm_stress requires material created with from_isotropic(...).")
            dx_measure = ufl.Measure("dx", domain=context.shell_model.mesh, metadata={"quadrature_degree": 4})
            if region is not None:
                dx_measure = context.region_measure(region)
            return context.shell_pde.pnorm_stress(
                context.post_fea.inputs_dict["disp_solid"]["function"],
                context.post_fea.inputs_dict["thickness"]["function"],
                context.post_fea.inputs_dict["E"]["function"],
                context.post_fea.inputs_dict["nu"]["function"],
                dx_measure,
                m=m,
                rho=rho,
                alpha=None,
                regularization=False,
            )

        return ScalarFormOutput(name, form_fn, ["disp_solid", "thickness", "E", "nu", "mesh_nodes"])

    def _von_mises_stress_output(self, name, surface="top"):
        surface_map = {"top": "Top", "mid": "Mid", "middle": "Mid", "bottom": "Bot", "bot": "Bot"}
        surface_key = surface.lower()
        if surface_key not in surface_map:
            raise ValueError("surface must be one of 'top', 'mid', or 'bottom'.")

        def form_fn(context):
            if not getattr(context.material, "is_isotropic", False):
                raise ValueError("von_mises_stress requires material created with from_isotropic(...).")
            return context.shell_pde.von_Mises_stress(
                context.post_fea.inputs_dict["disp_solid"]["function"],
                context.post_fea.inputs_dict["thickness"]["function"],
                context.post_fea.inputs_dict["E"]["function"],
                context.post_fea.inputs_dict["nu"]["function"],
                surface=surface_map[surface_key],
            )

        return FieldFormOutput(
            name,
            form_fn,
            ["thickness", "disp_solid", "E", "nu", "mesh_nodes"],
            ("DG", 1),
            reorder="element_dofs",
            record=self.shell_model.record,
        )

    def _resolve_field_record(self, record):
        if record is None:
            record = self.shell_model.record
        return record

    def _next_record_name(self, name):
        self._record_file_counter += 1
        return f"{name}_{self._record_file_counter}"

    def _evaluate_forms(self, context, scalar_forms, field_forms):
        fea = copy.copy(context.post_fea)
        fea.outputs_dict = {}
        fea.outputs_field_dict = {}
        for name, (form, arguments) in scalar_forms.items():
            fea.outputs_dict[name] = dict(
                form=form,
                shape=1,
                arguments=arguments,
            )
        for name, (form, arguments, element, reorder, record, vtk) in field_forms.items():
            function_space = dolfinx.fem.FunctionSpace(context.shell_model.mesh, element)
            record = self._resolve_field_record(record)
            record_name = self._next_record_name(name) if record else name
            fea.outputs_field_dict[name] = dict(
                form=form,
                function=Function(function_space),
                shape=len(getFuncArray(Function(function_space))),
                arguments=arguments,
                record=record,
                recorder=None,
                vtk=vtk,
                reorder=reorder,
            )
            fea.outputs_field_dict[name]["recorder"] = fea.createRecorder(
                record_name,
                fea.outputs_field_dict[name]["record"],
                vtk=fea.outputs_field_dict[name]["vtk"],
            )
        outputs = FEAModel(fea=[fea], fea_name="rm_shell_post_subset").evaluate(
            context.inputs(),
            debug_mode=context.debug_mode,
        )
        result = {}
        for name in scalar_forms:
            result[name] = getattr(outputs, name)
        for name in field_forms:
            value = getattr(outputs, name)
            reorder = field_forms[name][3]
            if reorder == "mid_strain":
                value = context.shell_model._reorder_elements(value.reshape((-1, 2, 2)))
            elif reorder == "shear_strain":
                value = context.shell_model._reorder_elements(value.reshape((-1, 2)))
            elif reorder == "curvature":
                value = context.shell_model._reorder_elements(value.reshape((-1, 2, 2)))
            elif reorder == "displacement":
                value = context.shell_model._reorder_nodes(value.reshape((-1, 3)))
            elif reorder == "rotation":
                value = context.shell_model._reorder_nodes(value.reshape((-1, 3)))
            elif reorder == "element_dofs":
                # generic per-cell reorder for scalar fields with an
                # arbitrary (but uniform) number of DOFs per cell, e.g. a
                # DG1 field on quads (4 dofs/cell) -- unlike mid_strain/
                # curvature/etc., these fields have no fixed known trailing
                # shape, so infer dofs-per-cell from the array size instead
                # of hardcoding a reshape target
                num_elements = context.shell_model.mesh.topology.original_cell_index.shape[0]
                value = context.shell_model._reorder_elements(value.reshape((num_elements, -1)))
            result[name] = value
        return result


class RMShellModel:
    def __init__(
        self,
        mesh: dolfinx.mesh,
        shell_bc_func: callable = None,
        element_wise_material=False,
        rho=100,
        PENALTY_BC=True,
        additional_outputs=None,
        mesh_tags=None,
        record=True,
        elementwise_pressure=False,
        solve_direct=False,
        recorder_path="records",
        strain_heights=None,
        element="CG2CG1",
        geometry_input_mode="coordinates",
        bc_dof_mask=None,
        gauge_points=None,
    ):
        """
        ``bc_dof_mask``: optional 6-entry sequence (u_x, u_y, u_z, theta_x,
        theta_y, theta_z) selecting which DOF components ``shell_bc_func``'s
        region penalizes to zero. Defaults to ``None``, i.e. all six (today's
        full clamp) -- pass e.g. ``(0, 1, 0, 1, 0, 1)`` for a symmetry-plane
        condition normal to y that leaves u_x, u_z, theta_y free.

        ``gauge_points``: optional list of ``(point_predicate, components)``
        pairs, each adding strong Dirichlet BCs (independent of
        ``PENALTY_BC``) pinning the listed DOF components (same 0-5 indexing
        as ``bc_dof_mask``) at the single node ``point_predicate`` locates.
        Used to remove residual rigid-body modes left over after a symmetry
        ``bc_dof_mask`` (e.g. a statically-determinate nose/tail support),
        not to enforce a real physical boundary condition.
        """
        self.mesh = mesh
        self.mesh_tags = mesh_tags
        self.additional_outputs = additional_outputs
        self.shell_bc_func = shell_bc_func
        self.bc_dof_mask = tuple(bc_dof_mask) if bc_dof_mask is not None else None
        self.gauge_points = list(gauge_points) if gauge_points else []
        self.element_wise_material = element_wise_material
        self.record = record
        self.recorder_path = recorder_path
        os.makedirs(self.recorder_path, exist_ok=True)
        self.m, self.rho = 1e-6, rho
        self.PENALTY_BC = PENALTY_BC
        self.solve_direct = solve_direct
        self.strain_heights = strain_heights
        self.elementwise_pressure = elementwise_pressure
        self.element = element
        if geometry_input_mode not in {"ale", "coordinates"}:
            raise ValueError(
                "geometry_input_mode must be either 'ale' or 'coordinates'; "
                f"got {geometry_input_mode!r}."
            )
        # ``ale`` is retained as a constructor alias; all geometry is now represented by coordinates.
        self.geometry_input_mode = "coordinates"

        self.nel = mesh.topology.index_map(mesh.topology.dim).size_local
        self.nn = mesh.topology.index_map(0).size_local
        self._coordinate_input_indices = np.asarray(
            mesh.geometry.input_global_indices, dtype=np.int64
        )
        self.reference_mesh_nodes = None
        if self.geometry_input_mode == "coordinates":
            if not np.array_equal(
                np.sort(self._coordinate_input_indices),
                np.arange(self._coordinate_input_indices.size),
            ):
                raise ValueError(
                    "Mesh-coordinate differentiation currently requires a serial "
                    "mesh with input_global_indices forming a node permutation."
                )
            self.reference_mesh_nodes = np.empty_like(mesh.geometry.x)
            self.reference_mesh_nodes[self._coordinate_input_indices] = mesh.geometry.x

        self.material_inputs = MaterialInputFactory(self)
        self.load_inputs = LoadInputFactory(self)
        self.post_processor = ShellPostProcessor(self)
        self.post = self.post_processor

        if mesh_tags is not None:
            self.set_up_subdomains(mesh_tags)

        if shell_bc_func is None:
            raise ValueError(
                "Please provide the shell bc location function.\n"
                "Example:\n"
                "def ClampedBoundary(x):\n"
                "    return np.less(x[1], 0.0)"
            )
        self.set_up_bcs(shell_bc_func, PENALTY_BC)

        self.shell_pde, self.fea = self._build_analysis_objects()
        self._post_fea = None
        self.pp = self.post_processor

    @property
    def post_fea(self):
        if self._post_fea is None:
            self._post_fea = self._build_post_input_fea()
        return self._post_fea

    def set_up_bcs(self, bc_locs_func, PENALTY_BC):
        if PENALTY_BC:
            mesh = self.mesh
            fdim = mesh.topology.dim - 1
            ds_1 = createCustomMeasure(mesh, fdim, bc_locs_func, measure="ds", tag=100)
            dS_1 = createCustomMeasure(mesh, fdim, bc_locs_func, measure="dS", tag=100)
            self.dss = ds_1(100)
            self.dSS = dS_1(100)
        else:
            self.dss = None
            self.dSS = None

    def set_up_subdomains(self, mesh_tags):
        cd2fe_el = np.argsort(self.mesh.topology.original_cell_index)
        vals = -np.ones(cd2fe_el.shape[0], dtype=np.int32)
        for i, inds in enumerate(mesh_tags.values()):
            vals[inds] = i
        self.association_table = {key: i for i, key in enumerate(mesh_tags.keys())}
        meshtags_fea = dolfinx.mesh.meshtags(self.mesh, 2, cd2fe_el, vals.astype(np.int32))
        self.dxx = ufl.Measure("dx", domain=self.mesh, subdomain_data=meshtags_fea)

    def _ensure_variable(self, value, name):
        if isinstance(value, csdl.Variable):
            return value
        arr = np.asarray(value)
        if arr.shape == ():
            arr = np.array([arr.item()], dtype=float)
        return csdl.Variable(value=arr, name=name)

    def _maybe_variable(self, value, name):
        if value is None:
            return None
        return self._ensure_variable(value, name)

    def _broadcast_variable(self, value, out_shape, name):
        if isinstance(value, csdl.Variable):
            if value.shape == out_shape:
                return value
            if value.shape == (1,):
                return csdl.expand(value, out_shape, action="i->j")
            if value.shape == ():
                return csdl.expand(value, out_shape)
        arr = np.asarray(value)
        if arr.shape == out_shape:
            return csdl.Variable(value=arr, name=name)
        if arr.shape == ():
            return csdl.Variable(value=np.full(out_shape, arr.item()), name=name)
        if arr.shape == (1,):
            return csdl.Variable(value=np.full(out_shape, arr.item()), name=name)
        raise ValueError(f"Cannot broadcast {name} from shape {arr.shape} to {out_shape}.")

    def _assemble_tensor33(self, *components):
        npts = components[0].shape[0]
        tensor = csdl.Variable(value=np.zeros((npts, 3, 3)))
        indices = [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
            (2, 0), (2, 1), (2, 2),
        ]
        for (i, j), comp in zip(indices, components):
            tensor = tensor.set(csdl.slice[:, i, j], comp)
        return tensor

    def _assemble_tensor22(self, *components):
        npts = components[0].shape[0]
        tensor = csdl.Variable(value=np.zeros((npts, 2, 2)))
        indices = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for (i, j), comp in zip(indices, components):
            tensor = tensor.set(csdl.slice[:, i, j], comp)
        return tensor

    def _add_common_solver_settings(self, fea):
        if self.solve_direct:
            fea.custom_solve = custom_solve_direct
        fea.REPORT = False
        fea.record = self.record
        fea.linear_problem = True

    def _add_strong_bcs(self, fea, shell_pde):
        W = shell_pde.W
        bcs = []
        if not self.PENALTY_BC:
            locate_BC1 = dolfinx.fem.locate_dofs_geometrical(
                (W.sub(0), W.sub(0).collapse()[0]), self.shell_bc_func
            )
            locate_BC2 = dolfinx.fem.locate_dofs_geometrical(
                (W.sub(1), W.sub(1).collapse()[0]), self.shell_bc_func
            )
            ubc = Function(W)
            with ubc.vector.localForm() as uloc:
                uloc.set(0.0)
            bcs.append(dolfinx.fem.dirichletbc(ubc, locate_BC1, W.sub(0)))
            bcs.append(dolfinx.fem.dirichletbc(ubc, locate_BC2, W.sub(1)))
        bcs.extend(self._build_gauge_point_bcs(shell_pde))
        fea.bc = bcs

    def _build_gauge_point_bcs(self, shell_pde):
        """Strong point Dirichlet BCs pinning residual rigid-body modes.

        Independent of ``PENALTY_BC``: these target single mesh nodes, which
        the facet-integral penalty mechanism (built for boundary regions,
        where an entity is only selected if *all* its vertices match the
        predicate) cannot reliably pin.
        """
        if not self.gauge_points:
            return []
        W = shell_pde.W
        subspaces = [
            W.sub(0).sub(0), W.sub(0).sub(1), W.sub(0).sub(2),
            W.sub(1).sub(0), W.sub(1).sub(1), W.sub(1).sub(2),
        ]
        zero = Function(W)
        with zero.vector.localForm() as uloc:
            uloc.set(0.0)
        bcs = []
        for point_predicate, components in self.gauge_points:
            for component in components:
                sub = subspaces[component]
                dofs = dolfinx.fem.locate_dofs_geometrical(
                    (sub, sub.collapse()[0]), point_predicate
                )
                if dofs[0].size == 0:
                    raise ValueError(
                        "Gauge-point predicate matched no DOFs for component "
                        f"{component}; check the target coordinate and tolerance."
                    )
                bcs.append(dolfinx.fem.dirichletbc(zero, dofs, sub))
        return bcs

    def _build_analysis_objects(self):
        shell_pde = CompositeShellPDE(
            self.mesh,
            element_wise_material=self.element_wise_material,
            elementwise_pressure=self.elementwise_pressure,
            element=self.element,
        )

        solve_fea = FEA(self.mesh)
        self._add_common_solver_settings(solve_fea)
        solve_fea.recorder_path = os.path.join(self.recorder_path, "abd_solve")
        os.makedirs(solve_fea.recorder_path, exist_ok=True)
        self._add_strong_bcs(solve_fea, shell_pde)

        A = Function(shell_pde.VABD)
        B = Function(shell_pde.VABD)
        D = Function(shell_pde.VABD)
        As = Function(shell_pde.VAs)
        h = Function(shell_pde.VT)
        f = Function(shell_pde.VF)
        m = Function(shell_pde.VF)
        density = Function(shell_pde.VT)
        w = Function(shell_pde.W)
        g = Function(shell_pde.W)
        with g.vector.localForm() as uloc:
            uloc.set(0.0)

        residual_form = shell_pde.pdeRes(
            w=w,
            f=f,
            m=m,
            A=A,
            B=B,
            D=D,
            As=As,
            penalty=self.PENALTY_BC,
            dss=self.dss,
            dSS=self.dSS,
            g=g,
            bc_dof_mask=self.bc_dof_mask,
        )

        solve_fea.add_input("thickness", h, init_val=0.001, record=self.record)
        solve_fea.add_input("F_solid", f, init_val=1.0, record=self.record)
        solve_fea.add_input("M_solid", m, init_val=1.0, record=self.record)
        solve_fea.add_direct_vector_input("load_vector", shape=len(getFuncArray(w)), sign=-1.0, record=False)
        solve_fea.add_input("A", A, init_val=1.0, record=self.record)
        solve_fea.add_input("B", B, init_val=1.0, record=self.record)
        solve_fea.add_input("D", D, init_val=1.0, record=self.record)
        solve_fea.add_input("As", As, init_val=1.0, record=self.record)
        solve_fea.add_input("density", density, init_val=1.0, record=self.record)
        solve_fea.add_mesh_coordinates_input(
            "mesh_nodes",
            record=False,
            input_global_indices=self._coordinate_input_indices,
        )
        solve_fea.add_state(
            name="disp_solid",
            function=w,
            residual_form=residual_form,
            arguments=[
                "thickness", "F_solid", "M_solid", "load_vector",
                "A", "B", "D", "As", "mesh_nodes",
            ],
            record=self.record,
        )

        return shell_pde, solve_fea

    def _build_post_input_fea(self):
        shell_pde = self.shell_pde
        post_fea = FEA(self.mesh)
        post_fea.record = False
        post_fea.recorder_path = os.path.join(self.recorder_path, "abd_post")
        os.makedirs(post_fea.recorder_path, exist_ok=True)
        disp_in = Function(shell_pde.W)
        post_fea.add_input("disp_solid", disp_in, init_val=0.0, record=False)
        post_fea.add_input("thickness", Function(shell_pde.VT), init_val=0.001, record=False)
        post_fea.add_input("F_solid", Function(shell_pde.VF), init_val=1.0, record=False)
        post_fea.add_input("M_solid", Function(shell_pde.VF), init_val=1.0, record=False)
        post_fea.add_input("A", Function(shell_pde.VABD), init_val=1.0, record=False)
        post_fea.add_input("B", Function(shell_pde.VABD), init_val=1.0, record=False)
        post_fea.add_input("D", Function(shell_pde.VABD), init_val=1.0, record=False)
        post_fea.add_input("As", Function(shell_pde.VAs), init_val=1.0, record=False)
        post_fea.add_input("E", Function(shell_pde.VT), init_val=1.0, record=False)
        post_fea.add_input("nu", Function(shell_pde.VT), init_val=0.3, record=False)
        post_fea.add_input("density", Function(shell_pde.VT), init_val=1.0, record=False)
        post_fea.add_mesh_coordinates_input(
            "mesh_nodes",
            record=False,
            input_global_indices=self._coordinate_input_indices,
        )

        shell_pde.set_elastic_model(
            w=disp_in,
            A=post_fea.inputs_dict["A"]["function"],
            B=post_fea.inputs_dict["B"]["function"],
            D=post_fea.inputs_dict["D"]["function"],
            As=post_fea.inputs_dict["As"]["function"],
        )

        return post_fea

    def _material_indices(self, shell_pde):
        if self.element_wise_material:
            return self.mesh.topology.original_cell_index.tolist()
        return self.mesh.geometry.input_global_indices

    def _pressure_indices(self, shell_pde):
        if self.elementwise_pressure:
            return self.mesh.topology.original_cell_index.tolist()
        return self.mesh.geometry.input_global_indices

    def _prepare_common_load_inputs(self, loads, node_disp=None, mesh_nodes=None):
        shell_pde = self.shell_pde
        pressure_mesh_indices = self._pressure_indices(shell_pde)

        shell_inputs = csdl.VariableGroup()

        if getattr(loads, "nodal_forces", None) is not None:
            reshaped_force = csdl.reshape(loads.nodal_forces[pressure_mesh_indices], (-1,))
            pressure_map = shell_pde.construct_force_to_pressure_map()
            pressure_from_forces = csdl.solve_linear(pressure_map.toarray(), reshaped_force)
        else:
            pressure_from_forces = None

        if getattr(loads, "nodal_pressure", None) is not None:
            reshaped_pressure = csdl.reshape(loads.nodal_pressure[pressure_mesh_indices], (-1,))
        else:
            reshaped_pressure = None

        if pressure_from_forces is not None and reshaped_pressure is not None:
            shell_inputs.F_solid = reshaped_pressure + pressure_from_forces
        elif pressure_from_forces is not None:
            shell_inputs.F_solid = pressure_from_forces
        elif reshaped_pressure is not None:
            shell_inputs.F_solid = reshaped_pressure
        else:
            shell_inputs.F_solid = csdl.Variable(value=np.zeros(self.fea.inputs_dict["F_solid"]["shape"]), name="F_solid_zero")
        shell_inputs.F_solid.add_name("F_solid")

        if getattr(loads, "nodal_moments", None) is not None:
            shell_inputs.M_solid = csdl.reshape(loads.nodal_moments[pressure_mesh_indices], (-1,))
        else:
            shell_inputs.M_solid = csdl.Variable(value=np.zeros(self.fea.inputs_dict["M_solid"]["shape"]), name="M_solid_zero")
        shell_inputs.M_solid.add_name("M_solid")

        if getattr(loads, "load_vector", None) is not None:
            shell_inputs.load_vector = loads.load_vector
        else:
            shell_inputs.load_vector = csdl.Variable(value=np.zeros(self.fea.inputs_dict["load_vector"]["shape"]), name="load_vector_zero")
        shell_inputs.load_vector.add_name("load_vector")

        node_disp = self._maybe_variable(node_disp, "node_disp")
        mesh_nodes = self._maybe_variable(mesh_nodes, "mesh_nodes")
        if node_disp is not None and mesh_nodes is not None:
            raise ValueError("Provide either mesh_nodes or node_disp, not both.")
        reference_nodes = csdl.Variable(
            value=self.reference_mesh_nodes,
            name="reference_mesh_nodes",
        )
        if mesh_nodes is None:
            mesh_nodes = reference_nodes if node_disp is None else reference_nodes + node_disp
        shell_inputs.mesh_nodes = mesh_nodes
        shell_inputs.mesh_nodes.add_name("mesh_nodes")
        return shell_inputs

    def _prepare_solver_inputs(self, material, loads, node_disp=None, mesh_nodes=None):
        shell_inputs = self._prepare_common_load_inputs(
            loads,
            node_disp=node_disp,
            mesh_nodes=mesh_nodes,
        )
        material_mesh_indices = self._material_indices(self.shell_pde)

        shell_inputs.thickness = material.thickness[material_mesh_indices]
        shell_inputs.density = material.density[material_mesh_indices]
        shell_inputs.A = material.A[material_mesh_indices]
        shell_inputs.B = material.B[material_mesh_indices]
        shell_inputs.D = material.D[material_mesh_indices]
        shell_inputs.As = material.As[material_mesh_indices]
        if getattr(material, "is_isotropic", False):
            shell_inputs.E = material.E[material_mesh_indices]
            shell_inputs.nu = material.nu[material_mesh_indices]
        else:
            shell_inputs.E = csdl.Variable(value=np.ones(self.fea.inputs_dict["thickness"]["shape"]), name="E_unused")
            shell_inputs.nu = csdl.Variable(value=np.full(self.fea.inputs_dict["thickness"]["shape"], 0.3), name="nu_unused")
        return shell_inputs

    def solve(self, material, loads, node_disp=None, mesh_nodes=None, debug_mode=False):
        node_disp = self._maybe_variable(node_disp, "node_disp")
        mesh_nodes = self._maybe_variable(mesh_nodes, "mesh_nodes")
        shell_inputs = self._prepare_solver_inputs(
            material,
            loads,
            node_disp=node_disp,
            mesh_nodes=mesh_nodes,
        )

        print("=" * 40)
        if getattr(loads, "load_vector", None) is not None:
            load_vector_value = shell_inputs.load_vector.value
            if load_vector_value is None:
                print("Direct generalized shell load-vector norm: deferred (non-inline graph)")
            else:
                print(
                    "Direct generalized shell load-vector norm: {:.6e}".format(
                        np.linalg.norm(load_vector_value)
                    )
                )
        if any(
            getattr(loads, name, None) is not None
            for name in ("nodal_forces", "nodal_pressure", "nodal_moments")
        ):
            field_load_value = shell_inputs.F_solid.value
            if field_load_value is None:
                print("Total field load projected to solid: deferred (non-inline graph)")
            else:
                F_solid_func = Function(self.shell_pde.VF)
                F_solid_func.x.array[:] = field_load_value
                print(
                    "Total field load projected to solid: {}".format(
                        [dolfinx.fem.assemble_scalar(dolfinx.fem.form(F_solid_func[i] * ufl.dx)) for i in range(3)]
                    )
                )
        print("=" * 40)
        print("Solving the RM shell model ...")
        raw_outputs = FEAModel(fea=[self.fea], fea_name="rm_shell").evaluate(shell_inputs, debug_mode=debug_mode)
        print("RM shell solve completed.")
        print("-" * 40)
        return ShellState(
            material=material,
            loads=loads,
            node_disp=node_disp,
            mesh_nodes=mesh_nodes,
            disp_solid=raw_outputs.disp_solid,
            raw_outputs=raw_outputs,
        )

    def _reorder_nodes(self, values):
        fenics_mesh_indices = self.mesh.geometry.input_global_indices
        reverse_fenics_mesh_indices = np.argsort(fenics_mesh_indices).tolist()
        return values[reverse_fenics_mesh_indices]

    def _reorder_elements(self, values):
        element_indices = np.argsort(self.mesh.topology.original_cell_index).tolist()
        return values[element_indices]

    def evaluate(self, *args, **kwargs):
        """Solve the shell problem from convenience inputs and return the default output bundle."""
        if args:
            if len(args) < 5:
                raise ValueError("Legacy RMShellModel.evaluate positional calls require at least 5 arguments.")
            load_arg, thickness, E, nu, density = args[:5]
            node_disp = args[5] if len(args) > 5 else None
            debug_mode = kwargs.pop("debug_mode", False)
            is_pressure = kwargs.pop("is_pressure", True)
            kwargs.update(
                dict(
                    thickness=thickness,
                    E=E,
                    nu=nu,
                    density=density,
                    node_disp=node_disp,
                )
            )
            if is_pressure:
                kwargs["nodal_pressure"] = load_arg
            else:
                kwargs["nodal_forces"] = load_arg

        thickness = kwargs.pop("thickness")
        density = kwargs.pop("density")
        E = kwargs.pop("E", None)
        nu = kwargs.pop("nu", None)
        A = kwargs.pop("A", None)
        B = kwargs.pop("B", None)
        D = kwargs.pop("D", None)
        As = kwargs.pop("As", None)
        nodal_forces = kwargs.pop("nodal_forces", None)
        nodal_pressure = kwargs.pop("nodal_pressure", None)
        nodal_moments = kwargs.pop("nodal_moments", None)
        load_vector = kwargs.pop("load_vector", None)
        node_disp = kwargs.pop("node_disp", None)
        mesh_nodes = kwargs.pop("mesh_nodes", None)
        debug_mode = kwargs.pop("debug_mode", False)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs.keys())}")

        if A is not None or B is not None or D is not None or As is not None:
            material = self.material_inputs.from_abd(
                A=A,
                B=B,
                D=D,
                As=As,
                thickness=thickness,
                density=density,
            )
        else:
            material = self.material_inputs.from_isotropic(
                E=E,
                nu=nu,
                thickness=thickness,
                density=density,
            )

        field_loads = self.load_inputs.from_fields(
            nodal_forces=nodal_forces,
            nodal_pressure=nodal_pressure,
            nodal_moments=nodal_moments,
        )
        vector_loads = None
        if load_vector is not None:
            vector_loads = self.load_inputs.from_vector(load_vector=load_vector)
        loads = self.load_inputs.combine(field_loads, vector_loads)
        state = self.solve(
            material=material,
            loads=loads,
            node_disp=node_disp,
            mesh_nodes=mesh_nodes,
            debug_mode=debug_mode,
        )
        return self.post.evaluate(state=state, debug_mode=debug_mode)

    def assemble_generalized_load_vector(self, nodal_forces=None, nodal_pressure=None, nodal_moments=None, node_disp=None, mesh_nodes=None):
        shell_pde = self.shell_pde
        pressure_mesh_indices = self._pressure_indices(shell_pde)
        deformation_mesh_indices = self.mesh.geometry.input_global_indices

        f_values = np.zeros(self.fea.inputs_dict["F_solid"]["shape"])
        if nodal_forces is not None:
            reshaped_force = np.asarray(nodal_forces)[pressure_mesh_indices].reshape(-1)
            pressure_map = shell_pde.construct_force_to_pressure_map()
            f_values += np.linalg.solve(pressure_map.toarray(), reshaped_force)
        if nodal_pressure is not None:
            f_values += np.asarray(nodal_pressure)[pressure_mesh_indices].reshape(-1)

        m_values = np.zeros(self.fea.inputs_dict["M_solid"]["shape"])
        if nodal_moments is not None:
            m_values += np.asarray(nodal_moments)[pressure_mesh_indices].reshape(-1)

        if node_disp is not None and mesh_nodes is not None:
            raise ValueError("Provide either mesh_nodes or node_disp, not both.")
        current_nodes = self.reference_mesh_nodes.copy()
        if mesh_nodes is not None:
            current_nodes[:] = np.asarray(mesh_nodes).reshape(current_nodes.shape)
        elif node_disp is not None:
            current_nodes += np.asarray(node_disp).reshape(current_nodes.shape)
        self.mesh.geometry.x[:] = current_nodes[deformation_mesh_indices]

        f_func = Function(shell_pde.VF)
        m_func = Function(shell_pde.VF)
        f_func.x.array[:] = f_values
        m_func.x.array[:] = m_values

        return shell_pde.assemble_generalized_load_vector(f=f_func, m=m_func)


class AggregatedStressModel:
    def __init__(self, m: float, rho: int):
        self.m = m
        self.rho = rho

    def evaluate(self, pnorm_stress: csdl.Variable):
        regularized_pnorm = csdl.absolute(pnorm_stress, rho=50.0) + 1.0e-300
        return 1 / self.m * regularized_pnorm ** (1 / self.rho)


class DisplacementExtractionModel:
    def __init__(self, shell_pde):
        self.shell_pde = shell_pde

    def evaluate(self, disp_vec: csdl.Variable):
        disp_extraction_mats = self.shell_pde.construct_nodal_disp_map()
        shape = self.shell_pde.mesh.geometry.x.shape
        nodal_disp_vec = csdl.sparse.matvec(disp_extraction_mats, disp_vec)
        nodal_disp_mat = csdl.transpose(csdl.reshape(nodal_disp_vec, shape=(shape[1], shape[0])))
        fenics_mesh_indices = self.shell_pde.mesh.geometry.input_global_indices
        reverse_fenics_mesh_indices = np.argsort(fenics_mesh_indices).tolist()
        return nodal_disp_mat[reverse_fenics_mesh_indices, :]


class ForceReshapingModel:
    def __init__(self, shell_pde):
        self.shell_pde = shell_pde

    def evaluate(self, nodal_force_mat: csdl.Variable):
        dummy_func = Function(self.shell_pde.VF)
        size = len(dummy_func.x.array)
        fenics_mesh_indices = self.shell_pde.mesh.geometry.input_global_indices
        return csdl.reshape(nodal_force_mat[fenics_mesh_indices, :], shape=(size,))
