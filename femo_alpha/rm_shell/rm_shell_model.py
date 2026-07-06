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

    def __init__(self, material, loads, node_disp, disp_solid, uhat, raw_outputs):
        self.material = material
        self.loads = loads
        self.node_disp = node_disp
        self.disp_solid = disp_solid
        self.uhat = uhat
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


class PostOutputSpec:
    """Specification for one postprocessing output."""

    def __init__(self, name, kind, docstring=None, **metadata):
        self.name = name
        self.kind = kind
        self.docstring = docstring
        self.metadata = metadata


class ShellPostContext:
    """Evaluation context for shell postprocessing outputs."""

    def __init__(self, post_processor, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        self.post = post_processor
        self.shell_model = post_processor.shell_model
        if state is not None:
            material = state.material
            loads = state.loads
            displacement = state.disp_solid
            node_disp = state.node_disp
        else:
            if material is None or loads is None or displacement is None:
                raise ValueError("Provide either a ShellState or material, loads, and displacement.")
        self.state = state
        self.material = material
        self.loads = loads
        self.node_disp = node_disp
        self.displacement = displacement
        self.debug_mode = debug_mode
        self._cache = {}
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
        """Return the deformed area form over the requested region."""
        return self.shell_pde.area_subdomain(
            self.post_fea.inputs_dict["uhat"]["function"],
            self.region_measure(region),
        )

    def compute(self, name):
        """Compute one registered postprocessing output."""
        return self.post.compute(name, context=self)

    def compute_many(self, names):
        """Compute multiple registered postprocessing outputs."""
        return self.post.compute_many(names, context=self)


class ShellPostBuilderFactory:
    """Factory for common region-aware postprocessing output builders."""

    _STRAIN_COMPONENTS = {
        "xx": (0, 0),
        "yy": (1, 1),
        "xy": (0, 1),
        "yx": (1, 0),
    }

    def scalar_form(self, form_fn, arguments, docstring=None):
        """Create a scalar-output builder from a UFL form callback."""

        def builder(context):
            return PostOutputSpec(
                name="",
                kind="scalar_form_builder",
                docstring=docstring,
                form=form_fn(context),
                arguments=arguments,
            )

        return builder

    def field_form(self, form_fn, arguments, element, reorder=None, docstring=None):
        """Create a field-output builder from a UFL form callback."""

        def builder(context):
            return PostOutputSpec(
                name="",
                kind="field_form_builder",
                docstring=docstring,
                form=form_fn(context),
                arguments=arguments,
                element=element,
                reorder=reorder,
            )

        return builder

    def average_strain(self, strain_type="mid", component="xx", region=None):
        """Create a builder for average strain over a tagged region."""
        if component not in self._STRAIN_COMPONENTS:
            raise ValueError(f"Unsupported strain component '{component}'.")

        def builder(context):
            if strain_type == "mid":
                strain_form = context.shell_pde.elastic_model.eps
            elif strain_type == "curvature":
                strain_form = context.shell_pde.elastic_model.kappa
            else:
                raise ValueError(f"Unsupported strain_type '{strain_type}'.")
            i, j = self._STRAIN_COMPONENTS[component]
            measure = context.region_measure(region)
            area_form = context.area_form(region)
            avg_form = strain_form[i, j] * ufl.as_ufl(1.0) * measure
            return PostOutputSpec(
                name="",
                kind="derived_from_forms",
                numerator=PostOutputSpec(
                    name="",
                    kind="scalar_form_builder",
                    form=avg_form,
                    arguments=["disp_solid", "uhat"],
                ),
                denominator=PostOutputSpec(
                    name="",
                    kind="scalar_form_builder",
                    form=area_form,
                    arguments=["uhat"],
                ),
            )

        return builder

    def pnorm_stress(self, rho=100, m=1e-6, region=None, surface="top"):
        """Create a builder for region-restricted p-norm von Mises stress."""

        def builder(context):
            if not getattr(context.material, "is_isotropic", False):
                raise ValueError("pnorm_stress requires material created with from_isotropic(...).")
            post_fea = context.post_fea
            dx_measure = ufl.Measure("dx", domain=context.shell_model.mesh, metadata={"quadrature_degree": 4})
            if region is not None:
                dx_measure = context.region_measure(region)
            pnorm_form = context.shell_pde.pnorm_stress(
                post_fea.inputs_dict["disp_solid"]["function"],
                post_fea.inputs_dict["uhat"]["function"],
                post_fea.inputs_dict["thickness"]["function"],
                post_fea.inputs_dict["E"]["function"],
                post_fea.inputs_dict["nu"]["function"],
                dx_measure,
                m=m,
                rho=rho,
                alpha=None,
                regularization=False,
            )
            return PostOutputSpec(
                name="",
                kind="scalar_form_builder",
                form=pnorm_form,
                arguments=["disp_solid", "thickness", "E", "nu", "uhat"],
            )

        return builder

    def von_mises_stress(self, surface="top"):
        """Create a projected von Mises stress field builder for isotropic materials."""
        surface_map = {"top": "Top", "mid": "Mid", "middle": "Mid", "bottom": "Bot", "bot": "Bot"}
        surface_key = surface.lower()
        if surface_key not in surface_map:
            raise ValueError("surface must be one of 'top', 'mid', or 'bottom'.")
        pde_surface = surface_map[surface_key]

        def builder(context):
            if not getattr(context.material, "is_isotropic", False):
                raise ValueError("von_mises_stress requires material created with from_isotropic(...).")
            post_fea = context.post_fea
            stress_form = context.shell_pde.von_Mises_stress(
                post_fea.inputs_dict["disp_solid"]["function"],
                post_fea.inputs_dict["uhat"]["function"],
                post_fea.inputs_dict["thickness"]["function"],
                post_fea.inputs_dict["E"]["function"],
                post_fea.inputs_dict["nu"]["function"],
                surface=pde_surface,
            )
            return PostOutputSpec(
                name="",
                kind="field_form_builder",
                form=stress_form,
                arguments=["thickness", "disp_solid", "E", "nu", "uhat"],
                element=("DG", 1),
            )

        return builder


class ShellPostProcessor:
    """Registry-driven shell postprocessing interface."""

    DEFAULT_OUTPUTS = (
        "compliance",
        "mass",
        "cg",
        "elastic_energy",
        "disp_extracted",
        "displacements",
        "rotations",
        "aggregated_stress",
    )

    def __init__(self, shell_model):
        self.shell_model = shell_model
        self.builders = ShellPostBuilderFactory()
        self._registry = {}
        self._register_builtin_outputs()

    def _register_builtin_outputs(self):
        self._registry["compliance"] = PostOutputSpec("compliance", "derived_builtin")
        self._registry["mass"] = PostOutputSpec("mass", "scalar_fea_output")
        self._registry["cgx_num"] = PostOutputSpec("cgx_num", "scalar_fea_output")
        self._registry["cgy_num"] = PostOutputSpec("cgy_num", "scalar_fea_output")
        self._registry["cgz_num"] = PostOutputSpec("cgz_num", "scalar_fea_output")
        self._registry["elastic_energy"] = PostOutputSpec("elastic_energy", "scalar_fea_output")
        self._registry["pnorm_stress"] = PostOutputSpec("pnorm_stress", "custom_builder", builder=self.builders.pnorm_stress())
        self._registry["stress"] = PostOutputSpec("stress", "custom_builder", builder=self.builders.von_mises_stress("top"))
        self._registry["stress_top"] = PostOutputSpec("stress_top", "custom_builder", builder=self.builders.von_mises_stress("top"))
        self._registry["stress_mid"] = PostOutputSpec("stress_mid", "custom_builder", builder=self.builders.von_mises_stress("mid"))
        self._registry["stress_bottom"] = PostOutputSpec("stress_bottom", "custom_builder", builder=self.builders.von_mises_stress("bottom"))
        self._registry["rotation"] = PostOutputSpec("rotation", "field_fea_output")
        self._registry["displacement"] = PostOutputSpec("displacement", "field_fea_output")
        self._registry["mid_strain"] = PostOutputSpec("mid_strain", "field_fea_output")
        self._registry["shear_strain"] = PostOutputSpec("shear_strain", "field_fea_output")
        self._registry["curvature"] = PostOutputSpec("curvature", "field_fea_output")
        self._registry["cg"] = PostOutputSpec("cg", "derived_builtin")
        self._registry["aggregated_stress"] = PostOutputSpec("aggregated_stress", "derived_builtin")
        self._registry["disp_extracted"] = PostOutputSpec("disp_extracted", "derived_builtin")
        self._registry["displacements"] = PostOutputSpec("displacements", "derived_builtin")
        self._registry["rotations"] = PostOutputSpec("rotations", "derived_builtin")

    def context(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Create a reusable postprocessing context for a solved or external displacement."""
        return ShellPostContext(
            self,
            state=state,
            material=material,
            loads=loads,
            displacement=displacement,
            node_disp=node_disp,
            debug_mode=debug_mode,
        )

    def add_scalar_output(self, name, builder, docstring=None):
        """Register a custom scalar postprocessing output."""
        self._registry[name] = PostOutputSpec(name, "custom_builder", builder=builder, docstring=docstring)

    def add_field_output(self, name, builder, docstring=None):
        """Register a custom field postprocessing output."""
        self._registry[name] = PostOutputSpec(name, "custom_builder", builder=builder, docstring=docstring)

    def add_derived_output(self, name, compute_fn, docstring=None):
        """Register a custom derived postprocessing output from a CSDL callback."""
        self._registry[name] = PostOutputSpec(
            name,
            "custom_derived",
            compute_fn=compute_fn,
            docstring=docstring,
        )

    def add_output(self, name, builder, docstring=None):
        """Register a custom postprocessing output from a builder callback."""
        self._registry[name] = PostOutputSpec(name, "custom_builder", builder=builder, docstring=docstring)

    def evaluate(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Compatibility wrapper returning the default set of postprocessing outputs."""
        context = self.context(
            state=state,
            material=material,
            loads=loads,
            displacement=displacement,
            node_disp=node_disp,
            debug_mode=debug_mode,
        )
        outputs = self.compute_many(
            self.DEFAULT_OUTPUTS,
            context=context,
        )
        outputs.disp_solid = context.displacement
        outputs.uhat = context.inputs().uhat
        return outputs

    def compute(self, name, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False, context=None):
        """Compute one named postprocessing output."""
        if context is None:
            context = self.context(
                state=state,
                material=material,
                loads=loads,
                displacement=displacement,
                node_disp=node_disp,
                debug_mode=debug_mode,
            )
        if name in context._cache:
            return context._cache[name]
        results = self.compute_many([name], context=context)
        return getattr(results, name)

    def compute_many(self, names, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False, context=None):
        """Compute multiple named postprocessing outputs and return them in a VariableGroup."""
        if context is None:
            context = self.context(
                state=state,
                material=material,
                loads=loads,
                displacement=displacement,
                node_disp=node_disp,
                debug_mode=debug_mode,
            )
        result = csdl.VariableGroup()
        names = list(names)
        pending_form_specs = {}
        pending_field_specs = {}
        for name in names:
            if name in context._cache:
                setattr(result, name, context._cache[name])
                continue
            spec = self._build_spec(name, context)
            if spec.kind == "scalar_fea_output":
                pending_form_specs[name] = self._make_scalar_fea_output_spec(name, context)
            elif spec.kind == "field_fea_output":
                pending_field_specs[name] = self._make_field_fea_output_spec(name, context)
            elif spec.kind == "scalar_form_builder":
                pending_form_specs[name] = spec
            elif spec.kind == "field_form_builder":
                pending_field_specs[name] = spec
            elif spec.kind == "derived_from_forms":
                numerator = self._compute_spec(name + "__num", spec.metadata["numerator"], context)
                denominator = self._compute_spec(name + "__den", spec.metadata["denominator"], context)
                value = numerator / denominator
                context._cache[name] = value
                setattr(result, name, value)
            elif spec.kind == "derived_from_forms_ratio":
                numerator = self._compute_spec(name + "__num", spec.metadata["numerator"], context)
                denominator = self._compute_spec(name + "__den", spec.metadata["denominator"], context)
                value = numerator / denominator
                context._cache[name] = value
                setattr(result, name, value)
            elif spec.kind == "custom_derived":
                value = spec.metadata["compute_fn"](context)
                context._cache[name] = value
                setattr(result, name, value)
            elif spec.kind == "derived_builtin":
                value = self._compute_builtin_derived(name, context)
                context._cache[name] = value
                setattr(result, name, value)
            else:
                raise ValueError(f"Unsupported postprocessing output kind '{spec.kind}' for '{name}'.")

        if pending_form_specs or pending_field_specs:
            fea_outputs = self._evaluate_form_specs(context, pending_form_specs, pending_field_specs)
            for name, value in fea_outputs.items():
                context._cache[name] = value
                setattr(result, name, value)

        for name in names:
            if not hasattr(result, name):
                setattr(result, name, context._cache[name])
        return result

    def mass_properties(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Compute mass and center of gravity outputs together."""
        return self.compute_many(
            ["mass", "cg", "cgx_num", "cgy_num", "cgz_num"],
            state=state,
            material=material,
            loads=loads,
            displacement=displacement,
            node_disp=node_disp,
            debug_mode=debug_mode,
        )

    def kinematics(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Compute displacement and rotation kinematic outputs together."""
        return self.compute_many(
            ["disp_extracted", "displacements", "rotations"],
            state=state,
            material=material,
            loads=loads,
            displacement=displacement,
            node_disp=node_disp,
            debug_mode=debug_mode,
        )

    def strains(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Compute strain-related field outputs together."""
        return self.compute_many(
            ["mid_strain", "shear_strain", "curvature"],
            state=state,
            material=material,
            loads=loads,
            displacement=displacement,
            node_disp=node_disp,
            debug_mode=debug_mode,
        )

    def compliance(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Compute compliance only."""
        return self.compute("compliance", state=state, material=material, loads=loads, displacement=displacement, node_disp=node_disp, debug_mode=debug_mode)

    def mass(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Compute mass only."""
        return self.compute("mass", state=state, material=material, loads=loads, displacement=displacement, node_disp=node_disp, debug_mode=debug_mode)

    def cg(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Compute center of gravity only."""
        return self.compute("cg", state=state, material=material, loads=loads, displacement=displacement, node_disp=node_disp, debug_mode=debug_mode)

    def disp_extracted(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Compute extracted nodal displacements only."""
        return self.compute("disp_extracted", state=state, material=material, loads=loads, displacement=displacement, node_disp=node_disp, debug_mode=debug_mode)

    def mid_strain(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Compute mid-surface strain field only."""
        return self.compute("mid_strain", state=state, material=material, loads=loads, displacement=displacement, node_disp=node_disp, debug_mode=debug_mode)

    def pnorm_stress(self, state=None, material=None, loads=None, displacement=None, node_disp=None, debug_mode=False):
        """Compute the built-in p-norm stress output only."""
        return self.compute("pnorm_stress", state=state, material=material, loads=loads, displacement=displacement, node_disp=node_disp, debug_mode=debug_mode)

    def _build_spec(self, name, context):
        if name not in self._registry:
            raise KeyError(f"Unknown postprocessing output '{name}'.")
        spec = self._registry[name]
        if spec.kind == "custom_builder":
            built = spec.metadata["builder"](context)
            built.name = name
            return built
        return spec

    def _compute_spec(self, name, spec, context):
        temp_group = self.compute_many([], context=context)
        _ = temp_group
        if spec.kind == "scalar_form_builder":
            return self._evaluate_form_specs(context, {name: spec}, {})[name]
        if spec.kind == "field_form_builder":
            return self._evaluate_form_specs(context, {}, {name: spec})[name]
        raise ValueError(f"Unsupported helper spec kind '{spec.kind}'.")

    def _make_scalar_fea_output_spec(self, name, context):
        base = context.post_fea.outputs_dict[name]
        return PostOutputSpec(
            name,
            "scalar_form_builder",
            form=base["form"],
            arguments=base["arguments"],
        )

    def _make_field_fea_output_spec(self, name, context):
        base = context.post_fea.outputs_field_dict[name]
        return PostOutputSpec(
            name,
            "field_form_builder",
            form=base["form"],
            arguments=base["arguments"],
            element=base["function"].function_space.ufl_element(),
            reorder=name,
        )

    def _evaluate_form_specs(self, context, scalar_specs, field_specs):
        fea = copy.copy(context.post_fea)
        fea.outputs_dict = {}
        fea.outputs_field_dict = {}
        for name, spec in scalar_specs.items():
            fea.outputs_dict[name] = dict(
                form=spec.metadata["form"],
                shape=1,
                arguments=spec.metadata["arguments"],
            )
        for name, spec in field_specs.items():
            function_space = dolfinx.fem.FunctionSpace(context.shell_model.mesh, spec.metadata["element"])
            fea.outputs_field_dict[name] = dict(
                form=spec.metadata["form"],
                function=Function(function_space),
                shape=len(getFuncArray(Function(function_space))),
                arguments=spec.metadata["arguments"],
                record=False,
                recorder=None,
            )
        outputs = FEAModel(fea=[fea], fea_name="rm_shell_post_subset").evaluate(
            context.inputs(),
            debug_mode=context.debug_mode,
        )
        result = {}
        for name in scalar_specs:
            result[name] = getattr(outputs, name)
        for name in field_specs:
            value = getattr(outputs, name)
            if name == "mid_strain":
                value = context.shell_model._reorder_elements(value.reshape((-1, 2, 2)))
            elif name == "shear_strain":
                value = context.shell_model._reorder_elements(value.reshape((-1, 2)))
            elif name == "curvature":
                value = context.shell_model._reorder_elements(value.reshape((-1, 2, 2)))
            elif name == "displacement":
                value = context.shell_model._reorder_nodes(value.reshape((-1, 3)))
            elif name == "rotation":
                value = context.shell_model._reorder_nodes(value.reshape((-1, 3)))
            result[name] = value
        return result

    def _compute_builtin_derived(self, name, context):
        if name == "compliance":
            raw_compliance = self._compute_spec(
                "_raw_compliance",
                self._make_scalar_fea_output_spec("compliance", context),
                context,
            )
            compliance = raw_compliance + csdl.vdot(context.inputs().load_vector, context.displacement)
            compliance.add_name("compliance")
            return compliance
        if name == "cg":
            numerators = self.compute_many(["cgx_num", "cgy_num", "cgz_num", "mass"], context=context)
            cg = csdl.concatenate((numerators.cgx_num / numerators.mass, numerators.cgy_num / numerators.mass, numerators.cgz_num / numerators.mass))
            cg.add_name("cg")
            return cg
        if name == "aggregated_stress":
            pnorm = self.compute("pnorm_stress", context=context)
            agg = AggregatedStressModel(m=context.shell_model.m, rho=context.shell_model.rho).evaluate(pnorm)
            agg.add_name("aggregated_stress")
            return agg
        if name == "disp_extracted":
            disp = DisplacementExtractionModel(shell_pde=context.shell_pde).evaluate(context.displacement)
            disp.add_name("disp_extracted")
            return disp
        if name == "displacements":
            displacement = self.compute("displacement", context=context)
            return displacement
        if name == "rotations":
            rotation = self.compute("rotation", context=context)
            return rotation
        raise KeyError(f"Unknown built-in derived output '{name}'.")


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
    ):
        self.mesh = mesh
        self.mesh_tags = mesh_tags
        self.additional_outputs = additional_outputs
        self.shell_bc_func = shell_bc_func
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

        self.nel = mesh.topology.index_map(mesh.topology.dim).size_local
        self.nn = mesh.topology.index_map(0).size_local

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
            self._post_fea = self._build_post_fea()
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
        if self.PENALTY_BC:
            return
        W = shell_pde.W
        locate_BC1 = dolfinx.fem.locate_dofs_geometrical(
            (W.sub(0), W.sub(0).collapse()[0]), self.shell_bc_func
        )
        locate_BC2 = dolfinx.fem.locate_dofs_geometrical(
            (W.sub(1), W.sub(1).collapse()[0]), self.shell_bc_func
        )
        ubc = Function(W)
        with ubc.vector.localForm() as uloc:
            uloc.set(0.0)
        fea.bc = [
            dolfinx.fem.dirichletbc(ubc, locate_BC1, W.sub(0)),
            dolfinx.fem.dirichletbc(ubc, locate_BC2, W.sub(1)),
        ]

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
        uhat = Function(shell_pde.VU)
        w = Function(shell_pde.W)
        g = Function(shell_pde.W)
        with g.vector.localForm() as uloc:
            uloc.set(0.0)

        residual_form = shell_pde.pdeRes(
            w=w,
            uhat=uhat,
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
        solve_fea.add_input("uhat", uhat, init_val=0.0, record=self.record)
        solve_fea.add_state(
            name="disp_solid",
            function=w,
            residual_form=residual_form,
            arguments=["thickness", "F_solid", "M_solid", "load_vector", "A", "B", "D", "As", "uhat"],
            record=self.record,
        )

        return shell_pde, solve_fea

    def _build_post_fea(self):
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
        post_fea.add_input("uhat", Function(shell_pde.VU), init_val=0.0, record=False)

        u_mid, theta = ufl.split(disp_in)
        shell_pde.set_elastic_model(
            w=disp_in,
            uhat=post_fea.inputs_dict["uhat"]["function"],
            A=post_fea.inputs_dict["A"]["function"],
            B=post_fea.inputs_dict["B"]["function"],
            D=post_fea.inputs_dict["D"]["function"],
            As=post_fea.inputs_dict["As"]["function"],
        )
        compliance_form = shell_pde.compliance(
            u_mid,
            theta,
            post_fea.inputs_dict["uhat"]["function"],
            post_fea.inputs_dict["thickness"]["function"],
            post_fea.inputs_dict["F_solid"]["function"],
            post_fea.inputs_dict["M_solid"]["function"],
        )
        cg_x_num_form, cg_y_num_form, cg_z_num_form, mass_form = shell_pde.cg_form(
            post_fea.inputs_dict["uhat"]["function"],
            post_fea.inputs_dict["thickness"]["function"],
            post_fea.inputs_dict["density"]["function"],
        )
        elastic_energy_form = shell_pde.elastic_energy(
            disp_in,
            post_fea.inputs_dict["uhat"]["function"],
            post_fea.inputs_dict["thickness"]["function"],
            None,
        )
        dx_reduced = ufl.Measure("dx", domain=self.mesh, metadata={"quadrature_degree": 4})
        pnorm_stress_form = shell_pde.pnorm_stress(
            disp_in,
            post_fea.inputs_dict["uhat"]["function"],
            post_fea.inputs_dict["thickness"]["function"],
            post_fea.inputs_dict["E"]["function"],
            post_fea.inputs_dict["nu"]["function"],
            dx_reduced,
            m=self.m,
            rho=self.rho,
            alpha=None,
            regularization=False,
        )
        stress_form = shell_pde.von_Mises_stress(
            disp_in,
            post_fea.inputs_dict["uhat"]["function"],
            post_fea.inputs_dict["thickness"]["function"],
            post_fea.inputs_dict["E"]["function"],
            post_fea.inputs_dict["nu"]["function"],
            surface="Top",
        )
        mid_strain = shell_pde.elastic_model.eps
        shear_strain = shell_pde.elastic_model.gamma
        curvature = shell_pde.elastic_model.kappa
        plane_strain_element = ufl.TensorElement("DG", self.mesh.ufl_cell(), degree=0, shape=(2, 2))
        shear_strain_element = ufl.VectorElement("DG", self.mesh.ufl_cell(), degree=0, dim=2)
        rotation_element = ufl.VectorElement("CG", self.mesh.ufl_cell(), degree=1, dim=3)
        displacement_element = ufl.VectorElement("CG", self.mesh.ufl_cell(), degree=1, dim=3)

        post_fea.add_output("compliance", compliance_form, ["disp_solid", "F_solid", "M_solid", "thickness", "uhat"])
        post_fea.add_output("mass", mass_form, ["thickness", "density", "uhat"])
        post_fea.add_output("cgx_num", cg_x_num_form, ["thickness", "density", "uhat"])
        post_fea.add_output("cgy_num", cg_y_num_form, ["thickness", "density", "uhat"])
        post_fea.add_output("cgz_num", cg_z_num_form, ["thickness", "density", "uhat"])
        post_fea.add_output("elastic_energy", elastic_energy_form, ["thickness", "disp_solid", "uhat"])
        post_fea.add_output("pnorm_stress", pnorm_stress_form, ["thickness", "disp_solid", "E", "nu", "uhat"])
        post_fea.add_field_output("stress", stress_form, ["thickness", "disp_solid", "E", "nu", "uhat"], element=("DG", 1), record=False, vtk=False)
        post_fea.add_field_output("mid_strain", mid_strain, ["disp_solid", "uhat"], element=plane_strain_element, vtk=False, record=False)
        post_fea.add_field_output("shear_strain", shear_strain, ["disp_solid", "uhat"], element=shear_strain_element, vtk=False, record=False)
        post_fea.add_field_output("curvature", curvature, ["disp_solid", "uhat"], element=plane_strain_element, vtk=False, record=False)
        post_fea.add_field_output("rotation", theta, ["disp_solid"], element=rotation_element, record=False, vtk=False)
        post_fea.add_field_output("displacement", u_mid, ["disp_solid"], element=displacement_element, record=False, vtk=False)

        return post_fea

    def _material_indices(self, shell_pde):
        if self.element_wise_material:
            return self.mesh.topology.original_cell_index.tolist()
        return self.mesh.geometry.input_global_indices

    def _pressure_indices(self, shell_pde):
        if self.elementwise_pressure:
            return self.mesh.topology.original_cell_index.tolist()
        return self.mesh.geometry.input_global_indices

    def _prepare_common_load_inputs(self, loads, node_disp=None):
        shell_pde = self.shell_pde
        pressure_mesh_indices = self._pressure_indices(shell_pde)
        deformation_mesh_indices = self.mesh.geometry.input_global_indices

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
        if node_disp is None:
            node_disp = csdl.Variable(value=np.zeros((len(deformation_mesh_indices), 3)), name="node_disp")
        shell_inputs.uhat = node_disp[deformation_mesh_indices].reshape((-1,))
        shell_inputs.uhat.add_name("uhat")
        return shell_inputs

    def _prepare_solver_inputs(self, material, loads, node_disp=None):
        shell_inputs = self._prepare_common_load_inputs(loads, node_disp=node_disp)
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

    def solve(self, material, loads, node_disp=None, debug_mode=False):
        node_disp = self._maybe_variable(node_disp, "node_disp")
        shell_inputs = self._prepare_solver_inputs(material, loads, node_disp=node_disp)

        print("=" * 40)
        F_solid_func = Function(self.shell_pde.VF)
        F_solid_func.x.array[:] = shell_inputs.F_solid.value
        print(
            "Total aero force projected to solid: {}".format(
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
            disp_solid=raw_outputs.disp_solid,
            uhat=shell_inputs.uhat,
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
        state = self.solve(material=material, loads=loads, node_disp=node_disp, debug_mode=debug_mode)
        return self.post.evaluate(state=state, debug_mode=debug_mode)

    def assemble_generalized_load_vector(self, nodal_forces=None, nodal_pressure=None, nodal_moments=None, node_disp=None):
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

        uhat_values = np.zeros(self.fea.inputs_dict["uhat"]["shape"])
        if node_disp is not None:
            uhat_values = np.asarray(node_disp)[deformation_mesh_indices].reshape(-1)

        f_func = Function(shell_pde.VF)
        m_func = Function(shell_pde.VF)
        uhat_func = Function(shell_pde.VU)
        f_func.x.array[:] = f_values
        m_func.x.array[:] = m_values
        uhat_func.x.array[:] = uhat_values

        return shell_pde.assemble_generalized_load_vector(uhat=uhat_func, f=f_func, m=m_func)


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
