import dolfinx
from dolfinx.fem import Function
import csdl_alpha as csdl
import ufl
import numpy as np
from mpi4py import MPI

from femo_alpha.fea.fea_dolfinx import FEA
from femo_alpha.fea.utils_dolfinx import (createCustomMeasure, convertToDense,
                                            assemble)
from femo_alpha.rm_shell.rm_shell_pde_composite import RMShellPDE
from femo_alpha.csdl_alpha_opt.fea_model import FEAModel
from femo_alpha.rm_shell.linear_shell_fenicsx.linear_shell_model import custom_solve_direct
from femo_alpha.rm_shell.linear_shell_fenicsx.kinematics import strain2D_local_to_global
from femo_alpha.rm_shell.linear_shell_fenicsx.kinematics import voigt2D


class RMShellModelComposite:
    '''
    Class for the RM shell model for aircraft optimization
    ------------------------------------------------------
    Args:
    
    mesh: dolfinx.mesh object for the shell mesh
    shell_bc_func: callable for shell Dirichlet BC locations - returns True if 
                    it is the boundary location, otherwise returns False
    record: boolean to record the FEA model variables in xdmf format
    element_wise_material: boolean to indicate if the material properties are
                            element-wise or nodal-wise
    PENALTY_BC: boolean to indicate if the Dirichlet BC is enforced using penalty
                method
    elementwise_pressure: boolean to indicate if the pressure is element-wise or
                            nodal-wise
    '''
    def __init__(
        self, 
        mesh: dolfinx.mesh,
        shell_bc_func: callable=None, 
        element_wise_material=False,
        rho=100,
        PENALTY_BC=True,
        additional_outputs=None,
        mesh_tags=None,
        record=True,
        elementwise_pressure=False,
        solve_direct=False,
        recorder_path='records',
        strain_heights = None,
        element = 'CG2CG1',
    ):
        '''
        Parameters:
        -----------
        mesh: dolfinx.mesh object for the shell mesh
        shell_bc_func: callable for shell Dirichlet BC locations - returns True if
                       it is the boundary location, otherwise returns False
        element_wise_material: boolean to indicate if the material properties are
                               defined element-wise or node-wise
        PENALTY_BC: boolean to indicate if the penalty method is used for the
                    Dirichlet BC
        additional_outputs: dictionary of callable functions to compute additional
                            scalar outputs for the shell model {name:(function, tags)}
        mesh_tags: dictionary {tag: [inds]} where inds are the indicies of elements under
                    the tag. No duplucate indices.
        record: boolean to record the FEA model variables in xdmf format
        rho: float, density of the shell material
        elementwise_pressure: boolean to indicate if the pressure is defined element-wise or node-wise
        solve_direct: boolean to indicate if the linear system is solved directly (True) or iteratively (False)
        recorder_path: string, path to save the recorded FEA variables if record is True
        strain_heights: list of floats, the heights at which to compute the strain, 
                        as a fraction of the total thickness offset from the center (ie, -0.5 to 0.5).
        '''
        self.mesh = mesh
        self.mesh_tags = mesh_tags
        self.additional_outputs = additional_outputs
        self.shell_bc_func = shell_bc_func # shell bc information
        self.element_wise_material = element_wise_material
        self.record = record
        self.recorder_path = recorder_path
        self.m, self.rho = 1e-6, rho
        self.PENALTY_BC = PENALTY_BC
        self.solve_direct = solve_direct
        self.strain_heights = strain_heights

        self.nel = mesh.topology.index_map(mesh.topology.dim).size_local
        self.nn = mesh.topology.index_map(0).size_local
        self.elementwise_pressure = elementwise_pressure
        self.element = element

        if mesh_tags is not None:
            self.set_up_subdomains(mesh_tags)

        if shell_bc_func is not None:
            self.set_up_bcs(shell_bc_func, PENALTY_BC)
        else:
            raise ValueError('Please provide the shell bc location function.\n \
                             Example:\n \
                             def ClampedBoundary(x):\n \
                                return np.less(x[1], 0.0)')
            
        self.set_up_fea()

    def set_up_bcs(self, bc_locs_func, PENALTY_BC): 
        '''
        Set up the boundary conditions for the shell model and the tip displacement
        ** helper function for aircraft optimization with clamped root bc **
        '''
        if PENALTY_BC:
            mesh = self.mesh
            fdim = mesh.topology.dim - 1
            ds_1 = createCustomMeasure(mesh, fdim, bc_locs_func, measure='ds', tag=100)
            dS_1 = createCustomMeasure(mesh, fdim, bc_locs_func, measure='dS', tag=100)

            self.dss = ds_1(100) # custom ds measure for the Dirichlet BC
            self.dSS = dS_1(100) # custom ds measure for the Dirichlet BC
        else:
            self.dss = None
            self.dSS = None


    def set_up_subdomains(self, mesh_tags):
        '''
        Convert mesh tags to dolfinx mesh tags and set up the measure.
        Creates self.dxx, which is a dolfinx mesh tag measure for the shell model.
        Also creates self.association_table, which is a dictionary that maps the
        meshtag indices to the input tags.

        Parameters:
        -----------
        mesh_tags: dictionary {tag: [inds]} where inds are the indicies of elements under
                    the tag. No duplucate indices.
        '''

        # maps from the external fea element inds to the internal fea element inds 
        cd2fe_el = np.argsort(self.mesh.topology.original_cell_index)

        # need to make vals, which is an array of length num_elements, where vals[i] is the
        # index of the tag of the i-th element (or -1 if no tag)
        vals = -np.ones(cd2fe_el.shape[0], dtype=np.int32)
        for i, inds in enumerate(mesh_tags.values()):
            vals[inds] = i

        # print(vals)
        # exit()

        # association table allows us to map the input tags to the meshtag indices
        self.association_table = {key:i for i, key in enumerate(mesh_tags.keys())}

        # create the mesh tags
        meshtags_fea = dolfinx.mesh.meshtags(self.mesh, 2, cd2fe_el, vals.astype(np.int32))

        # create the measure
        self.dxx = measure = ufl.Measure('dx', domain=self.mesh, subdomain_data=meshtags_fea)

    def set_up_fea(self):
        '''
        Set up the FEMO FEA model for RM shell analysis
        '''
        print('-'*40)
        print('Setting up the FEA model for RM shell analysis ...')
        mesh = self.mesh
        shell_pde = self.shell_pde = RMShellPDE(mesh, 
                                                element_wise_material=self.element_wise_material,
                                                elementwise_pressure=self.elementwise_pressure,
                                                element=self.element)
        dss = self.dss
        dSS = self.dSS

        PENALTY_BC = self.PENALTY_BC

        fea = FEA(mesh)
        if self.solve_direct:
            fea.custom_solve = custom_solve_direct
        # fea.PDE_SOLVER = 'Newton'
        fea.REPORT = False
        fea.record = self.record
        fea.recorder_path = self.recorder_path
        fea.linear_problem = True
        # Add input to the PDE problem:
        A = Function(shell_pde.VABD)
        B = Function(shell_pde.VABD)
        D = Function(shell_pde.VABD)
        As = Function(shell_pde.VAs)
        h = Function(shell_pde.VT)
        f = Function(shell_pde.VF)
        density = Function(shell_pde.VT)
        uhat = Function(shell_pde.VU)

        # Add state to the PDE problem:
        w_space = shell_pde.W
        w = Function(w_space)

        # Set up strong boundary condition
        if not PENALTY_BC:
            W = shell_pde.W
            locate_BC1 = dolfinx.fem.locate_dofs_geometrical((W.sub(0), W.sub(0).collapse()[0]),
                                                self.shell_bc_func)
            locate_BC2 = dolfinx.fem.locate_dofs_geometrical((W.sub(1), W.sub(1).collapse()[0]),
                                                self.shell_bc_func)
            ubc =  Function(W)
            with ubc.vector.localForm() as uloc:
                uloc.set(0.)

            bcs = [dolfinx.fem.dirichletbc(ubc, locate_BC1, W.sub(0)),
                    dolfinx.fem.dirichletbc(ubc, locate_BC2, W.sub(1)),]
            fea.bc = bcs

        # Composite Material PDE residual form
        g = Function(shell_pde.W)
        with g.vector.localForm() as uloc:
            uloc.set(0.)
        residual_form = shell_pde.pdeRes(w=w, # displacement
                                         uhat=uhat, # mesh displacement
                                         f=f, # force
                                         A = A, B=B, D=D, As=As, # composite material stiffness matrices
                                         penalty=PENALTY_BC, 
                                         dss=dss, dSS=dSS, g=g)

        # Add output to the PDE problem:
        u_mid, theta = ufl.split(w)
        compliance_form = shell_pde.compliance(u_mid,uhat,h,f)
        # mass_form = shell_pde.mass(uhat, h, density)
        cg_x_num_form, cg_y_num_form, cg_z_num_form, mass_form = shell_pde.cg_form(
            uhat, h, density
        )
        elastic_energy_form = shell_pde.elastic_energy(w,uhat,h,None)
        dx_reduced = ufl.Measure('dx', domain=mesh, 
                                 metadata={'quadrature_degree':4})
        
        mid_strain = shell_pde.elastic_model.eps
        shear_strain = shell_pde.elastic_model.gamma
        curvature = shell_pde.elastic_model.kappa

        strain_form_voigt = voigt2D(mid_strain)
        eps_x = strain_form_voigt[0]
        eps_y = strain_form_voigt[1]
        gamma_xy = strain_form_voigt[2]
        eps_1 = (eps_x + eps_y)/2 + ufl.sqrt(((eps_x - eps_y)/2)**2 + (0.5 * gamma_xy)**2)


        fea.add_input('thickness', h, init_val=0.001, record=self.record)
        fea.add_input('F_solid', f, init_val=1., record=self.record)
        fea.add_input('A', A, init_val=1., record=self.record)
        fea.add_input('B', B, init_val=1., record=self.record)
        fea.add_input('D', D, init_val=1., record=self.record)
        fea.add_input('As', As, init_val=1., record=self.record)
        fea.add_input('density', density, init_val=1., record=self.record)
        fea.add_input('uhat', uhat, init_val=0., record=self.record)

        fea.add_state(name='disp_solid',
                        function=w,
                        residual_form=residual_form,
                        arguments=['thickness','F_solid',
                                    'A','B','D','As','uhat'])
        fea.add_output(name='compliance',
                        form=compliance_form,
                        arguments=['disp_solid','F_solid','thickness','uhat'])
        fea.add_output(name='mass',
                        form=mass_form,
                        arguments=['thickness','density','uhat'])
        fea.add_output(name='cgx_num',
                        form=cg_x_num_form,
                        arguments=['thickness','density','uhat'])
        fea.add_output(name='cgy_num',
                        form=cg_y_num_form,
                        arguments=['thickness','density','uhat'])
        fea.add_output(name='cgz_num',
                        form=cg_z_num_form,
                        arguments=['thickness','density','uhat'])
        fea.add_output(name='elastic_energy',
                        form=elastic_energy_form,
                        arguments=['thickness','disp_solid','uhat'])
        
        fea.add_field_output(name='eps_1',
                             form=eps_1,
                             arguments=['thickness','disp_solid','uhat'],
                             element=('DG',0),
                             vtk=False,
                             record=self.record)
        
        plane_strain_element = ufl.TensorElement('DG', mesh.ufl_cell(), degree=0, shape=(2,2))
        fea.add_field_output(name='mid_strain',
                            form=mid_strain,
                            arguments=['disp_solid','uhat'],
                            element=plane_strain_element,
                            vtk=False,
                            record=self.record)
        shear_strain_element = ufl.VectorElement('DG', mesh.ufl_cell(), degree=0, dim=2)
        fea.add_field_output(name='shear_strain',
                            form=shear_strain,
                            arguments=['disp_solid','uhat'],
                            element=shear_strain_element,
                            vtk=False,
                            record=self.record)
        curvature_element = ufl.TensorElement('DG', mesh.ufl_cell(), degree=0, shape=(2,2))
        fea.add_field_output(name='curvature',
                            form=curvature,
                            arguments=['disp_solid','uhat'],
                            element=curvature_element,
                            vtk=False,
                            record=self.record)
        
        
        if self.strain_heights is not None:
            shear_strain_form = shell_pde.elastic_model.local_shear_strains()
            shear_strain_element = ufl.VectorElement('DG', mesh.ufl_cell(), degree=0, dim=2)
            membrane_strain_element = ufl.TensorElement('DG', mesh.ufl_cell(), degree=0, shape=(2,2))
            fea.add_field_output(
                name='shear_strain',
                form=shear_strain_form,
                arguments=['thickness','disp_solid','uhat'],
                element=shear_strain_element,
                record=self.record,
            )
            for i, height in enumerate(self.strain_heights):
                membrane_strain_form = shell_pde.elastic_model.local_membrane_strains(h*height)
                fea.add_field_output(
                    name=f'membrane_strain_{i}',
                    form=membrane_strain_form,
                    arguments=['thickness','disp_solid','uhat'],
                    element=membrane_strain_element,
                    record=self.record,
                )
                
        # Add rotation field output
        rotation_form = theta
        rotation_element = ufl.VectorElement('CG', mesh.ufl_cell(), degree=1, dim=3)
        fea.add_field_output(
            name='rotation',
            form=rotation_form,
            arguments=['disp_solid'],
            element=rotation_element,
            record=self.record,
            vtk=False
        )

        # Add displacement field output
        displacement_form = u_mid
        displacement_element = ufl.VectorElement('CG', mesh.ufl_cell(), degree=1, dim=3)
        fea.add_field_output(
            name='displacement',
            form=displacement_form,
            arguments=['disp_solid'],
            element=displacement_element,
            record=self.record,
            vtk=False
        )

        self.fea = fea

    def evaluate_modal_fea(self, shell_pde, A_val, B_val, D_val, As_val, h_val, density_val):
        from femo_alpha.rm_shell.linear_shell_fenicsx.linear_shell_model import (MaterialModelComposite2, 
                                                                          ElasticModelModal)
        w = Function(shell_pde.W)
        h = Function(shell_pde.VT)
        A = Function(shell_pde.VABD)
        B = Function(shell_pde.VABD)
        D = Function(shell_pde.VABD)
        As = Function(shell_pde.VAs)
        density = Function(shell_pde.VT)
        h.x.array[:] = h_val
        A.x.array[:] = A_val
        B.x.array[:] = B_val
        D.x.array[:] = D_val
        As.x.array[:] = As_val
        density.x.array[:] = density_val
        material_model = MaterialModelComposite2(A=A, B=B, D=D, As=As)
        elastic_model_modal = ElasticModelModal(self.mesh,
                                                w, material_model.CLT)
        elastic_energy = elastic_model_modal.elasticEnergy()
        f_0 = dolfinx.fem.Constant(shell_pde.mesh, (0.0,0.0,0.0))
        elastic_res = elastic_model_modal.weakFormResidual(elastic_energy, f_0)

        K = ufl.derivative(elastic_res, w)
        K_compiled = dolfinx.fem.form(K)
        
        inertia_res = elastic_model_modal.inertialResidual(density, h)
        M = ufl.derivative(inertia_res, w)
        M_compiled = dolfinx.fem.form(M)

        hh = ufl.TrialFunction(shell_pde.VT)
        dKdh = ufl.derivative(K, h, hh)
        dMdh = ufl.derivative(M, h, hh)
        dKdh = ufl.replace(dKdh, {hh: h})
        dMdh = ufl.replace(dMdh, {hh: h})

        dKdh_compiled = dolfinx.fem.form(dKdh)
        dMdh_compiled = dolfinx.fem.form(dMdh)

        # print(dolfinx.fem.assemble_scalar(dolfinx.fem.form(elastic_energy)))
        # print(dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(elastic_res)).getArray())
        
        # K_mat = dolfinx.fem.petsc.assemble_matrix(K_compiled, self.fea.bc)
        # # K_mat = dolfinx.fem.petsc.assemble_matrix(K_compiled)
        # K_mat.assemble()
        # K_dense = K_mat.convert("dense")
        # print(K_mat.getSize())
        # print(K_dense.getDenseArray())
        
        # M_mat = dolfinx.fem.petsc.assemble_matrix(M_compiled)
        # M_mat.assemble()
        # M_dense = M_mat.convert("dense")
        # print(M_mat.getSize())
        # print(M_dense.getDenseArray())
        # exit()
        dKdh_list = []
        dMdh_list = []

        # [RX] this process is extremely memory intensive. 
        #  It takes ~7GB of memory for a 10x50 mesh
        for i in range(len(h.x.array)):
            h.x.array[:] = 0.0
            h.x.array[i] = 0.2
            print("-------------------------")
            print("     Iteration: ", i)
            print("-------------------------")
            h.x.scatter_forward()
            # print("h: ", h.x.array)

            dKdh_mat_i = dolfinx.fem.petsc.assemble_matrix(dKdh_compiled)
            # dKdh_mat_i = dolfinx.fem.petsc.assemble_matrix(dKdh_compiled, self.fea.bc)
            dKdh_mat_i.assemble()
            dMdh_mat_i = dolfinx.fem.petsc.assemble_matrix(dMdh_compiled)
            dMdh_mat_i.assemble()
            dKdh_list.append(dKdh_mat_i)
            dMdh_list.append(dMdh_mat_i)
            print(dKdh_mat_i.convert("dense").getDenseArray())
            print(dMdh_mat_i.convert("dense").getDenseArray())
            # dKdh_list.append(dKdh_mat_i.convert("dense").getDenseArray())
            # dMdh_list.append(dMdh_mat_i.convert("dense").getDenseArray())
        # print("dKdh_list: ", dKdh_list)
        # print("dMdh_list: ", dMdh_list)
        # exit()
        
    def evaluate(self, 
                thickness: csdl.Variable,
                A: csdl.Variable,
                B: csdl.Variable,
                D: csdl.Variable,
                As: csdl.Variable,
                density: csdl.Variable,
                nodal_forces: csdl.Variable = None,
                nodal_pressure: csdl.Variable = None,
                node_disp: csdl.Variable = None,
                debug_mode=False) -> csdl.VariableGroup:
        '''
        Parameters:
        -----------
        Vector csdl.Variable:
            > force_vector: the force vector applied on the shell mesh nodes
            > thickness: the thickness on the shell mesh nodes
            > A: the A matrix on the shell mesh nodes
            > B: the B matrix on the shell mesh nodes
            > D: the D matrix on the shell mesh nodes
            > As: the As matrix on the shell mesh nodes
            > density: the density on the shell mesh nodes

        Returns:
        --------
        Vector csdl.Variable:
            > disp_solid: the displacements (3 translational dofs, 3 rotation dofs)
                            on the shell mesh nodes
            > stress: the von Mises stress on the shell mesh elements
        Scalar csdl.Variable:
            > aggregated_stress: the aggregated stress of the shell model
            > compliance: the compliance of the shell model
            > tip_disp: the tip displacement of the shell model
            > mass: the mass of the shell model
            > cg: the center of gravity location of the shell model [x,y,z]
        '''
        shell_inputs = csdl.VariableGroup()

        #:::::::::::::::::::::: Prepare the inputs :::::::::::::::::::::::::::::
        # sort the material properties based on FEniCS indices
        if self.element_wise_material:
            material_mesh_indices = self.shell_pde.mesh.topology.original_cell_index.tolist()
        else:
            material_mesh_indices = self.shell_pde.mesh.geometry.input_global_indices
        shell_inputs.thickness = thickness[material_mesh_indices]
        shell_inputs.A = A[material_mesh_indices]
        shell_inputs.B = B[material_mesh_indices]
        shell_inputs.D = D[material_mesh_indices]
        shell_inputs.As = As[material_mesh_indices]
        shell_inputs.density = density[material_mesh_indices]

        # reshape the force matrix to vector and sort indices
        if self.elementwise_pressure:
            pressure_mesh_indices = self.shell_pde.mesh.topology.original_cell_index.tolist()
        else:
            pressure_mesh_indices = self.shell_pde.mesh.geometry.input_global_indices
        
        if nodal_forces is not None:
            reshaped_force = csdl.reshape(nodal_forces[pressure_mesh_indices], (-1,))
            # Compute nodal pressures based on forces
            print('Converting forces to pressures ...')
            A = self.shell_pde.construct_force_to_pressure_map()
            pressure_from_forces = csdl.solve_linear(A.toarray(), reshaped_force)
        if nodal_pressure is not None:
            reshaped_pressure = csdl.reshape(nodal_pressure[pressure_mesh_indices], (-1,))

        if nodal_forces is not None and nodal_pressure is not None:
            shell_inputs.F_solid = reshaped_pressure + pressure_from_forces
        elif nodal_forces is not None:
            shell_inputs.F_solid = pressure_from_forces
        elif nodal_pressure is not None:
            shell_inputs.F_solid = reshaped_pressure
        else:
            raise ValueError('Please provide either nodal forces or pressure vector.')

        shell_inputs.F_solid.add_name('F_solid')

        print("="*40)
        F_solid_func = Function(self.shell_pde.VF)
        F_solid_func.x.array[:] = shell_inputs.F_solid.value
        print("Total aero force projected to solid: {}".format(
            [dolfinx.fem.assemble_scalar(dolfinx.fem.form(F_solid_func[i]*ufl.dx)) for i in range(3)]))
        print("="*40)

        # sort the nodal mesh deformation based on FEniCS indices
        deformation_mesh_indices = self.shell_pde.mesh.geometry.input_global_indices
        if node_disp is None:
            node_disp = csdl.Variable(value=0.0, shape=(len(deformation_mesh_indices), 3), 
                                      name='node_disp')
        reshaped_node_disp = node_disp[deformation_mesh_indices].reshape((-1,))
        reshaped_node_disp.add_name('uhat')
        shell_inputs.uhat = reshaped_node_disp

        #:::::::::::::::::::::: Evaluate the model :::::::::::::::::::::::::::::
        # Evaluate the shell model
        print('Evaluating the RM shell model ...')
        solid_model = FEAModel(fea=[self.fea], fea_name='rm_shell')
        shell_outputs = solid_model.evaluate(shell_inputs, debug_mode=debug_mode)

        #:::::::::::::::::::::: Postprocess the outputs ::::::::::::::::::::::::
        # disp_extraction_model = DisplacementExtractionModel(shell_pde=self.shell_pde)
        # disp_extracted = disp_extraction_model.evaluate(shell_outputs.disp_solid)
        # disp_extracted.add_name('disp_extracted')
        # shell_outputs.disp_extracted = disp_extracted
        
        # rot_extraction_model = RotationExtractionModel(shell_pde=self.shell_pde)
        # theta_extracted = rot_extraction_model.evaluate(shell_outputs.disp_solid)
        # theta_extracted.add_name('theta_extracted')
        # shell_outputs.theta_extracted = theta_extracted
        
        # compute cg location
        cg_x = shell_outputs.cgx_num / shell_outputs.mass
        cg_y = shell_outputs.cgy_num / shell_outputs.mass
        cg_z = shell_outputs.cgz_num / shell_outputs.mass
        shell_outputs.cg = csdl.concatenate([cg_x, cg_y, cg_z])
        shell_outputs.cg.add_name('cg')

        print('RM shell model evaluation completed.')
        print('-'*40)

        element_indices = np.argsort(self.shell_pde.mesh.topology.original_cell_index).tolist()

        # Re-order the cell-wise field outputs to match the input mesh element ordering (FEniCS --> CADDEE)
        if self.strain_heights is not None:
            shell_outputs.shear_strain = shell_outputs.shear_strain.reshape((-1, 2))[element_indices, :]
            for i, height in enumerate(self.strain_heights):
                shell_outputs.__setattr__(
                    f'membrane_strain_{i}',
                    shell_outputs.__getattribute__(f'membrane_strain_{i}').reshape((-1, 2, 2))[element_indices, :, :],
                )

        # Re-order the roatation output to match the input mesh node ordering (FEniCS --> CADDEE)
        fenics_mesh_indices = self.shell_pde.mesh.geometry.input_global_indices
        reverse_fenics_mesh_indices = np.argsort(fenics_mesh_indices).tolist()
        shell_outputs.rotations = shell_outputs.rotation.reshape((-1, 3))[reverse_fenics_mesh_indices, :]

        # Re-order the displacement output to match the input mesh node ordering (FEniCS --> CADDEE)
        shell_outputs.displacements = shell_outputs.displacement.reshape((-1, 3))[reverse_fenics_mesh_indices, :]

        # Re-order the strain outputs to match the input mesh element ordering (FEniCS --> CADDEE)
        shell_outputs.mid_strain = shell_outputs.mid_strain.reshape((-1, 2, 2))[element_indices, :, :]
        shell_outputs.shear_strain = shell_outputs.shear_strain.reshape((-1, 2))[element_indices, :]
        shell_outputs.curvature = shell_outputs.curvature.reshape((-1, 2, 2))[element_indices, :, :]

        # self.evaluate_modal_fea(self.shell_pde, 
        #                       shell_inputs.E.value, 
        #                       shell_inputs.nu.value, 
        #                       shell_inputs.thickness.value, 
        #                       shell_inputs.density.value)

        return shell_outputs


class AggregatedStressModel:
    '''
    Compute the aggregated stress
    '''
    def __init__(self, m: float, rho: int):
        self.m = m
        self.rho = rho

    def evaluate(self, pnorm_stress: csdl.Variable):
        aggregated_stress = 1/self.m*pnorm_stress**(1/self.rho)
        return aggregated_stress

class DisplacementExtractionModel:
    '''
    Extract and reshape displacement vector into matrix
    '''
    def __init__(self, shell_pde: RMShellPDE):
        self.shell_pde = shell_pde

    def evaluate(self, disp_vec: csdl.Variable):
        shell_pde = self.shell_pde

        disp_extraction_mats = shell_pde.construct_nodal_disp_map()
        # Both vector or tensors need to be numpy arrays
        shape = shell_pde.mesh.geometry.x.shape
        # contains nodal displacements only (CG1)
        nodal_disp_vec = csdl.sparse.matvec(disp_extraction_mats, disp_vec)
        nodal_disp_mat = csdl.transpose(csdl.reshape(nodal_disp_vec, shape=(shape[1],shape[0])))

        # reorder the matrix to match the importing mesh node indices
        # FEniCS --> CADDEE
        fenics_mesh_indices = self.shell_pde.mesh.geometry.input_global_indices
        reverse_fenics_mesh_indices = np.argsort(fenics_mesh_indices).tolist()
        reordered_nodal_disp_mat = nodal_disp_mat[reverse_fenics_mesh_indices,:]
        return reordered_nodal_disp_mat

class RotationExtractionModel:
    '''
    Extract and reshape rotation vector into matrix (num_nodes, 2)
    Mirrors DisplacementExtractionModel but for CG1 rotations (no interpolation needed)
    '''
    def __init__(self, shell_pde: RMShellPDE):
        self.shell_pde = shell_pde

    def evaluate(self, disp_vec: csdl.Variable):
        shell_pde = self.shell_pde

        rot_extraction_mats = shell_pde.construct_nodal_rot_map()
        # Extract rotation vector from mixed state
        nodal_rot_vec = csdl.sparse.matvec(rot_extraction_mats, disp_vec)
        num_nodes = shell_pde.mesh.topology.index_map(0).size_local
        # Reshape: (2*num_nodes,) -> (2, num_nodes) -> transpose -> (num_nodes, 2)
        nodal_rot_mat = csdl.transpose(csdl.reshape(nodal_rot_vec, shape=(2, num_nodes)))

        # Reorder to match CADDEE mesh node indices (FEniCS --> CADDEE)
        fenics_mesh_indices = shell_pde.mesh.geometry.input_global_indices
        reverse_fenics_mesh_indices = np.argsort(fenics_mesh_indices).tolist()
        reordered_nodal_rot_mat = nodal_rot_mat[reverse_fenics_mesh_indices, :]
        return reordered_nodal_rot_mat

class ForceReshapingModel:
    '''
    Reshape force matrix to vector
    '''
    def __init__(self, shell_pde: RMShellPDE):
        self.shell_pde = shell_pde

    def evaluate(self, nodal_force_mat: csdl.Variable):
        shell_pde = self.shell_pde
        dummy_func = Function(shell_pde.VF)
        size = len(dummy_func.x.array)
        # reorder the matrix to match the FEniCS mesh node indices
        # CADDEE --> FEniCS
        fenics_mesh_indices = self.shell_pde.mesh.geometry.input_global_indices    
        output = csdl.reshape(nodal_force_mat[fenics_mesh_indices,:], shape=(size,))
        return output
