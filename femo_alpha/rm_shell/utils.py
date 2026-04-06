import numpy as np
from dolfinx.fem import VectorFunctionSpace, TensorFunctionSpace, Function
from femo_alpha.rm_shell.linear_shell_fenicsx.kinematics import local_basis_inplane, global_to_local_inplane
from femo_alpha.rm_shell.linear_shell_fenicsx.utils import project
from femo_alpha.fea.utils_dolfinx import getFuncArray

def extract_local_coords(mesh):
    """
    Extract local shell basis coordinates (E0, E1, E2) and transformation matrix (E01) as NumPy arrays.
    
    Parameters:
    -----------
    mesh : dolfinx.mesh.Mesh
        The shell mesh
    
    Returns:
    --------
    dict with keys:
        'E0' : ndarray, shape (n_elements, 3)
            Local in-plane tangent vector (parametric direction 0)
        'E1' : ndarray, shape (n_elements, 3)
            Local in-plane tangent vector (parametric direction 1)
        'E2' : ndarray, shape (n_elements, 3)
            Local normal vector
        'E01' : ndarray, shape (n_elements, 2, 3)
            2x3 transformation matrix; E01[i,j,k] is k-th component of i-th basis vector
        'centroids' : ndarray, shape (n_elements, 3)
            Element centroids in global coordinates
    """
    E0, E1, E2 = local_basis_inplane(mesh)
    E01 = global_to_local_inplane(E0, E1)

    # Element reordering maps
    # fenics_to_original_el[fenics_el] = original_el
    fenics_to_original_el = np.asarray(mesh.topology.original_cell_index, dtype=np.int64)
    # original_to_fenics_el[original_el] = fenics_el
    original_to_fenics_el = np.argsort(fenics_to_original_el)

    # Optional node maps (often useful too)
    # fenics_to_original_node[fenics_node] = original_node
    fenics_to_original_node = np.asarray(mesh.geometry.input_global_indices, dtype=np.int64)
    # original_to_fenics_node[original_node] = fenics_node
    original_to_fenics_node = np.argsort(fenics_to_original_node)

    # Project to DG0 function spaces
    Vvec = VectorFunctionSpace(mesh, ("DG", 0))
    e0_fun = Function(Vvec)
    e1_fun = Function(Vvec)
    e2_fun = Function(Vvec)

    project(E0, e0_fun, lump_mass=False)
    project(E1, e1_fun, lump_mass=False)
    project(E2, e2_fun, lump_mass=False)

    Vten = TensorFunctionSpace(mesh, ("DG", 0), shape=(2, 3))
    e01_fun = Function(Vten)
    project(E01, e01_fun, lump_mass=False)

    # Extract as numpy arrays
    n_elements = mesh.topology.index_map(mesh.topology.dim).size_local

    E0_array = getFuncArray(e0_fun).reshape((n_elements, 3))
    E1_array = getFuncArray(e1_fun).reshape((n_elements, 3))
    E2_array = getFuncArray(e2_fun).reshape((n_elements, 3))
    E01_array = getFuncArray(e01_fun).reshape((n_elements, 2, 3))

    # Compute element centroids
    centroids = np.zeros((n_elements, 3))
    for i in range(n_elements):
        cell = mesh.topology.connectivity(mesh.topology.dim, 0).links(i)
        centroids[i] = mesh.geometry.x[cell].mean(axis=0)

    return {
        'E0': E0_array.copy(),
        'E1': E1_array.copy(),
        'E2': E2_array.copy(),
        'E01': E01_array.copy(),
        'centroids': centroids.copy(),
        "fenics_to_original_el": fenics_to_original_el,
        "original_to_fenics_el": original_to_fenics_el,
        "fenics_to_original_node": fenics_to_original_node,
        "original_to_fenics_node": original_to_fenics_node,
    }