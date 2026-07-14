# Getting started
This page provides instructions for installing `femo_alpha` and running a minimal example.

## Installation

The minimal requirements to use **femo_alpha** for modeling and simulation are `FEniCSx` and `CSDL_alpha`. For modeling aircraft design applications, you may install [CADDEE_alpha](https://github.com/LSDOlab/CADDEE_alpha) to enable coupling with solvers of other disciplines; for optimization, you will also need [ModOpt](https://github.com/LSDOlab/modopt) on top of them for Python bindings of various state-of-the-art optimizers. 

### Installation instructions
It's recommended to use conda for installing the module and its dependencies.

- Create a conda environment for `femo` with a specific Python version (Python 3.9) that is compatible with all of the dependencies
  ```sh
  conda create -n femo python=3.9.10
  ```
  (Python 3.9.7 also works if Python 3.9.10 is unavailable in your conda)
- Activate the conda enviroment 
  ```sh
  conda activate femo
  ```
- Install FEniCSx by `conda-forge`
  ```sh
  conda install -c conda-forge fenics-dolfinx=0.5.1
  ```
- Install `CSDL_alpha` and `ModOpt` by
  ```sh
  pip install git+https://github.com/LSDOlab/CSDL_alpha.git
  pip install git+https://github.com/LSDOlab/modopt.git
  ```
- Install `femo_alpha` as a **user** by 
  ```sh
  pip install git+https://github.com/LSDOlab/femo_alpha.git
  ```
  or install `femo_alpha` as a **developer** by
  ```sh
  git clone https://github.com/LSDOlab/femo_alpha.git
  pip install -e ./femo_alpha
  ```


## Quick verification
To test if your installation was successful, run
`ex_thickness_opt_cantilever_beam.py` from `/femo_alpha/examples/basic_examples/beam_thickness_opt/`.
If everything works correctly, the following terminal output will be displayed.

![beam_thickness_opt](/src/images/beam_thickness_distribution.png "Optimal beam thickness distribution")

```
         ==============
         Scipy summary:
         ==============
         Problem                    : beam_thickness
         Solver                     : scipy_slsqp
         Success                    : True
         Message                    : Optimization terminated successfully
         Objective                  : 23762.153677992977
         Gradient norm              : 100814.27872282796
         Total time                 : 12.408674001693726
         Major iterations           : 111
         Total function evals       : 383
         Total gradient evals       : 111
         ==================================================
Optimization results:
     ['compliance'] [23762.15367799]
     ['volume'] [0.01]
OpenMDAO compliance: 23762.153677443166

```

## RMShell quick start

The recommended shell workflow is now explicit:

1. construct an `RMShellModel`
2. create canonical material and load inputs
3. call `solve(...)`
4. postprocess through `shell.post`

```python
from femo_alpha.rm_shell.rm_shell_model import RMShellModel

shell = RMShellModel(
    mesh,
    shell_bc_func=clamped_boundary,
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
)

state = shell.solve(material=material, loads=loads, node_disp=node_disp)
outputs = (
    shell.post.clear()
    .add_default_outputs()
    .compute(state=state)
)
```

Direct generalized shell load vectors are also supported:

```python
loads = shell.load_inputs.from_vector(
    load_vector=load_vector,
)
state = shell.solve(material=material, loads=loads, node_disp=node_disp)
```

For a fuller walkthrough, including postprocessing and custom outputs, see the RMShell tutorial and interface reference.
