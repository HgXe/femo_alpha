# RMShell Interface

`RMShellModel` is the main shell analysis interface in FEMO. It uses one `A`, `B`, `D`, and `As` shell formulation for both isotropic and directly supplied stiffness inputs, and supports:

- isotropic material inputs through a high-level helper
- general laminate-style shell inputs through direct `A`, `B`, `D`, and `As` matrices
- field-based loads and direct generalized load vectors
- explicit solve and postprocessing stages
- reusable and extensible postprocessing through `shell.post`

The preferred workflow is:

1. Build a shell model.
2. Construct material inputs.
3. Construct load inputs.
4. Solve for a `ShellState`.
5. Postprocess from either the solved state or an externally supplied displacement.

## Core Workflow

```python
from femo_alpha.rm_shell.rm_shell_model import RMShellModel

shell = RMShellModel(
    mesh,
    shell_bc_func=clamped_boundary,
    element_wise_material=False,
    record=False,
)

material = shell.material_inputs.from_isotropic(
    E=E,
    nu=nu,
    thickness=thickness,
    density=density,
)

loads = shell.load_inputs.from_fields(
    nodal_pressure=nodal_pressure,
    nodal_moments=nodal_moments,
)

state = shell.solve(material=material, loads=loads, node_disp=node_disp)
outputs = shell.post.evaluate(state=state)
```

The old `shell_model.evaluate(...)` convenience entry point is still available for compatibility, but new code should prefer `solve(...)` and `shell.post...`.

## Material Inputs

For isotropic shells:

```python
material = shell.material_inputs.from_isotropic(
    E=E,
    nu=nu,
    thickness=thickness,
    density=density,
)
```

For the general shell interface:

```python
material = shell.material_inputs.from_abd(
    A=A,
    B=B,
    D=D,
    As=As,
    thickness=thickness,
    density=density,
)
```

`from_isotropic(...)` converts `E`, `nu`, and `thickness` into `A`, `B`, `D`, and `As` matrices while preserving `E` and `nu` for isotropic stress recovery. `from_abd(...)` supplies those stiffness matrices directly, but isotropic stress outputs are only available when the material was built with `from_isotropic(...)`.

## Load Inputs

Field-based loads:

```python
loads = shell.load_inputs.from_fields(
    nodal_forces=nodal_forces,
    nodal_pressure=nodal_pressure,
    nodal_moments=nodal_moments,
)
```

Direct generalized load vectors:

```python
loads = shell.load_inputs.from_vector(
    load_vector=load_vector,
)
```

Combined loads:

```python
field_loads = shell.load_inputs.from_fields(
    nodal_pressure=nodal_pressure,
    nodal_moments=nodal_moments,
)
vector_loads = shell.load_inputs.from_vector(
    load_vector=load_vector,
)
loads = shell.load_inputs.combine(field_loads, vector_loads)
```

If an upstream transfer already provides the generalized shell right-hand side, the direct-vector path avoids reconstructing and projecting load fields.

Mesh deformation is not part of the load group. Pass it directly to the solve or postprocessing call:

```python
state = shell.solve(material=material, loads=loads, node_disp=node_disp)
outputs = shell.post.evaluate(state=state)
```

## Solving

`solve(...)` returns a `ShellState` that stores:

- the material and load input groups
- the solved shell displacement variable
- the raw FEA output bundle

```python
state = shell.solve(material=material, loads=loads, node_disp=node_disp)
disp = state.disp_solid
```

## Postprocessing

The postprocessor is exposed as `shell.post`.

Default output bundle:

```python
outputs = shell.post.evaluate(state=state)
```

Selected outputs:

```python
compliance = shell.post.compute("compliance", state=state)
mass_and_cg = shell.post.compute_many(["mass", "cg"], state=state)
```

Grouped helpers:

```python
mass_props = shell.post.mass_properties(state=state)
kinematics = shell.post.kinematics(state=state)
strains = shell.post.strains(state=state)
```

Reusable context:

```python
context = shell.post.context(state=state)
mass = shell.post.compute("mass", context=context)
stress = shell.post.compute("pnorm_stress", context=context)
```

Stress outputs for isotropic materials:

```python
pnorm = shell.post.compute("pnorm_stress", state=state)
top_stress = shell.post.compute("stress", state=state)
mid_stress = shell.post.compute("stress_mid", state=state)
bottom_stress = shell.post.compute("stress_bottom", state=state)
```

`stress` is the top-surface von Mises field. The explicit `stress_top`, `stress_mid`, and `stress_bottom` names are available when a workflow needs to choose the through-thickness recovery surface.

Postprocessing with an external displacement:

```python
external_outputs = shell.post.evaluate(
    material=material,
    loads=loads,
    displacement=state.disp_solid,
)
```

This separation lets you reuse the shell postprocessor even when the displacement comes from outside the FEMO solve path.

## Custom Outputs

Built-in builder helpers can register new outputs quickly:

```python
shell.post.add_scalar_output(
    "avg_eps_x",
    shell.post.builders.average_strain(
        strain_type="mid",
        component="xx",
    ),
)
```

You can also define completely custom form-based outputs by returning a `PostOutputSpec`:

```python
import ufl
from femo_alpha.rm_shell.rm_shell_model import PostOutputSpec

def mean_transverse_displacement_builder(context):
    w = context.post_fea.inputs_dict["disp_solid"]["function"][2]
    measure = context.region_measure()
    return PostOutputSpec(
        name="",
        kind="derived_from_forms",
        numerator=PostOutputSpec(
            name="",
            kind="scalar_form_builder",
            form=w * measure,
            arguments=["disp_solid"],
        ),
        denominator=PostOutputSpec(
            name="",
            kind="scalar_form_builder",
            form=context.area_form(),
            arguments=["uhat"],
        ),
    )

shell.post.add_output("mean_w", mean_transverse_displacement_builder)
mean_w = shell.post.compute("mean_w", state=state)
```

## Related Material

- The tutorial notebook gives a longer end-to-end walkthrough with runnable cells.
- The RMShell direct-load tests show equivalence between field-based and direct generalized load inputs, including derivative checks.
