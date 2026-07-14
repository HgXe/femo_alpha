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

The postprocessor is exposed as `shell.post`. Request the outputs you want, then
call `compute(...)` once.

Default output bundle:

```python
outputs = shell.post.evaluate(state=state)
```

Explicit output bundle:

```python
outputs = (
    shell.post.clear()
    .compliance()
    .elastic_energy()
    .mass_properties()
    .kinematics()
    .compute(state=state)
)
```

Grouped helpers:

```python
outputs = (
    shell.post.clear()
    .mass_properties()
    .kinematics()
    .strains()
    .compute(state=state)
)
```

Reusable context:

```python
context = shell.post.context(state=state)
outputs = (
    shell.post.clear()
    .mass()
    .pnorm_stress()
    .compute(context=context)
)
```

Stress outputs for isotropic materials:

```python
outputs = (
    shell.post.clear()
    .pnorm_stress()
    .stress()
    .stress_mid()
    .stress_bottom()
    .compute(state=state)
)
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

Built-in helpers can register new outputs quickly:

```python
shell.post.clear().average_strain(
    "avg_eps_x",
    strain_type="mid",
    component="xx",
)
outputs = shell.post.compute(state=state)
```

You can also define completely custom form-based outputs directly:

```python
import ufl

def mean_w_numerator(context):
    return context.post_fea.inputs_dict["disp_solid"]["function"][2] * context.region_measure()

shell.post.clear().add_form_ratio(
    "mean_w",
    mean_w_numerator,
    ["disp_solid"],
    lambda context: context.area_form(),
    ["uhat"],
)
outputs = shell.post.compute(state=state)
mean_w = outputs.mean_w
```

## Related Material

- The tutorial notebook gives a longer end-to-end walkthrough with runnable cells.
- The RMShell direct-load tests show equivalence between field-based and direct generalized load inputs, including derivative checks.
