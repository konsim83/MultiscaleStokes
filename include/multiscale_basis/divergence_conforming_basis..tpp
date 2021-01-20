#pragma once

// My headers
#include <multiscale_basis/divergence_conforming_basis.h>

MSSTOKES_OPEN_NAMESPACE

DivergenceStabilizedBasis::DivergenceStabilizedBasis(
  const CoreModelData::Parameters &                parameters,
  typename Triangulation<3>::active_cell_iterator &global_cell,
  bool                                             is_first_cell,
  unsigned int                                     local_subdomain,
  MPI_Comm                                         mpi_communicator)
  : mpi_communicator(mpi_communicator)
  , parameters(parameters_ms)
  , parameter_filename(parameter_filename_)
  , triangulation()
  , fe(FE_Q<dim>(parameters.velocity_degree),
       dim,
       (parameters.use_locally_conservative_discretization ?
          static_cast<const FiniteElement<dim> &>(
            FE_DGP<dim>(parameters.stokes_velocity_degree - 1)) :
          static_cast<const FiniteElement<dim> &>(
            FE_Q<dim>(parameters.stokes_velocity_degree - 1))),
       1)
  , dof_handler(triangulation)
  , modified_basis_constraints(fe.base_element(0).n_dofs_per_cell)
  , sparsity_pattern()
  , modified_basis(fe.base_element(0).n_dofs_per_cell)
  , system_rhs(fe.base_element(0).n_dofs_per_cell)
  , global_element_matrix(fe.dofs_per_cell, fe.dofs_per_cell)
  , global_element_rhs(fe.dofs_per_cell)
  , global_weights(fe.dofs_per_cell, 0)
  , global_cell_id(global_cell->id())
  , is_first_cell(is_first_cell)
  , global_cell_it(global_cell)
  , local_subdomain(local_subdomain)
  , corner_points(GeometryInfo<dim>::vertices_per_cell, Point<dim>())
  , length_system_basis(modified_basis_constraints.size())
  , is_built_global_element_matrix(false)
  , is_set_global_weights(false)
  , is_set_cell_data(false)
{
  for (unsigned int vertex_n = 0;
       vertex_n < GeometryInfo<dim>::vertices_per_cell;
       ++vertex_n)
    {
      corner_points[vertex_n] = global_cell_it->vertex(vertex_n);
    }

  is_set_cell_data = true;
}

DivergenceStabilizedBasis::DivergenceStabilizedBasis(
  const DivergenceStabilizedBasis &other)
  : mpi_communicator(other.mpi_communicator)
  , parameters(other.parameters)
  , parameter_filename(other.parameter_filename)
  , triangulation() // must be constructed deliberately, but is empty on
                    // copying anyway
  , fe(FE_RaviartThomas<3>(parameters.degree),
       1,
       FE_DGQ<3>(parameters.degree),
       1)
  , dof_handler(triangulation)
  , modified_basis_constraints(other.modified_basis_constraints)
  , sparsity_pattern()
  , modified_basis(other.modified_basis)
  , system_rhs(other.system_rhs)
  , global_rhs(other.global_rhs)
  , global_element_matrix(other.global_element_matrix)
  , global_element_rhs(other.global_element_rhs)
  , global_weights(other.global_weights)
  , global_solution(other.global_solution)
  , inner_schur_preconditioner(other.inner_schur_preconditioner)
  , global_cell_id(other.global_cell_id)
  , is_first_cell(other.is_first_cell)
  , global_cell_it(other.global_cell_it)
  , local_subdomain(other.local_subdomain)
  , corner_points(other.corner_points)
  , length_system_basis(other.length_system_basis)
  , is_built_global_element_matrix(other.is_built_global_element_matrix)
  , is_set_global_weights(other.is_set_global_weights)
  , is_set_cell_data(other.is_set_cell_data)
{
  global_cell_id = global_cell_it->id();

  for (unsigned int vertex_n = 0;
       vertex_n < GeometryInfo<dim>::vertices_per_cell;
       ++vertex_n)
    {
      corner_points[vertex_n] = global_cell_it->vertex(vertex_n);
    }

  is_set_cell_data = true;
}

DivergenceStabilizedBasis::~DivergenceStabilizedBasis()
{
  system_matrix.clear();

  for (unsigned int n_basis = 0; n_basis < GeometryInfo<dim>::faces_per_cell;
       ++n_basis)
    {
      modified_basis_constraints[n_basis].clear();
    }

  dof_handler.clear();
}

void
DivergenceStabilizedBasis::setup_grid()
{
  Assert(is_set_cell_data, ExcMessage("Cell data must be set first."));

  GridGenerator::general_cell(triangulation,
                              corner_points,
                              /* colorize faces */ false);

  triangulation.refine_global(parameters.n_refine_local);
}

void
DivergenceStabilizedBasis::setup_system_matrix()
{
  dof_handler.distribute_dofs(fe);

  DoFRenumbering::Cuthill_McKee(dof_handler);

  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise(dof_handler, block_component);

  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
  const unsigned int n_u = dofs_per_block[0];
  const unsigned int n_p = dofs_per_block[1];

  {
    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: " << triangulation.n_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (" << n_u << '+' << n_p << ')' << std::endl;
  }

  {
    BlockDynamicSparsityPattern dsp(2, 2);
    dsp.block(0, 0).reinit(n_u, n_u);
    dsp.block(1, 0).reinit(n_p, n_u);
    dsp.block(0, 1).reinit(n_u, n_p);
    dsp.block(1, 1).reinit(n_p, n_p);
    dsp.collect_sizes();

    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        if (!((c == dim) && (d == dim)))
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;
    DoFTools::make_sparsity_pattern(
      dof_handler, coupling, dsp, constraints, false);
    sparsity_pattern.copy_from(dsp);
  }

  {
    BlockDynamicSparsityPattern preconditioner_dsp(2, 2);
    preconditioner_dsp.block(0, 0).reinit(n_u, n_u);
    preconditioner_dsp.block(1, 0).reinit(n_p, n_u);
    preconditioner_dsp.block(0, 1).reinit(n_u, n_p);
    preconditioner_dsp.block(1, 1).reinit(n_p, n_p);
    preconditioner_dsp.collect_sizes();

    Table<2, DoFTools::Coupling> preconditioner_coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        if (((c == dim) && (d == dim)))
          preconditioner_coupling[c][d] = DoFTools::always;
        else
          preconditioner_coupling[c][d] = DoFTools::none;
    DoFTools::make_sparsity_pattern(dof_handler,
                                    preconditioner_coupling,
                                    preconditioner_dsp,
                                    constraints,
                                    false);
    preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
  }

  assembled_matrix.reinit(sparsity_pattern);
  assembled_preconditioner.reinit(preconditioner_sparsity_pattern);

  system_matrix.reinit(sparsity_pattern);
  preconditioner_matrix.reinit(preconditioner_sparsity_pattern);

  global_solution.reinit(dofs_per_block);
  global_rhs.reinit(dofs_per_block);
}

void
DivergenceStabilizedBasis::setup_basis_dofs()
{
  Assert(is_set_cell_data, ExcMessage("Cell data must be set first."));

  Timer timer;

  if (parameters.verbose)
    {
      std::cout << "	Setting up dofs for H(div) part.....";

      timer.restart();
    }

  ShapeFun::ShapeFunctionVector<dim> std_shape_function_stokes(
    fe.base_element(0), global_cell_it);

  FEValuesExtractors::Vector velocities(0);
  for (unsigned int n_basis = 0; n_basis < modified_basis.size(); ++n_basis)
    {
      std_shape_function_stokes.set_index(n_basis);

      // set constraints (first hanging nodes, then flux)
      modified_basis_constraints[n_basis].clear();

      DoFTools::make_hanging_node_constraints(
        dof_handler, modified_basis_constraints[n_basis]);

      VectorTools::interpolate_boundary_values(
        dof_handler,
        /* boundary_id*/ 0,
        std_shape_function_stokes,
        modified_basis_constraints[n_basis],
        fe.component_mask(velocities));

      modified_basis_constraints[n_basis].close();
    }

  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise(dof_handler, block_component);

  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

  for (unsigned int n_basis = 0; n_basis < modified_basis.size(); ++n_basis)
    {
      modified_basis[n_basis].reinit(dofs_per_block);
      system_rhs[n_basis].reinit(dofs_per_block);
    }

  if (parameters.verbose)
    {
      timer.stop();
      std::cout << "done in   " << timer.cpu_time() << "   seconds."
                << std::endl;
    }
}

void
DivergenceStabilizedBasis::assemble_system()
{
  Timer timer;
  if (parameters.verbose)
    {
      std::cout << "	Assembling local linear system in cell   "
                << global_cell_id.to_string() << ".....";

      timer.restart();
    }
  // Choose appropriate quadrature rules
  QGauss<3> quadrature_formula(parameters.degree + 2);

  // Get relevant quantities to be updated from finite element
  FEValues<3> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);

  // Define some abbreviations
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  // Declare local contributions and reserve memory
  FullMatrix<double>          local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>              local_rhs(dofs_per_cell);
  std::vector<Vector<double>> local_rhs_v(GeometryInfo<dim>::faces_per_cell,
                                          Vector<double>(dofs_per_cell));

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // equation data
  const EquationData::RightHandSideParsed right_hand_side(parameter_filename,
                                                          /* n_components */ 1);
  const EquationData::DiffusionInverse_A  a_inverse(parameter_filename);
  const EquationData::ReactionRate        reaction_rate;

  // allocate
  std::vector<double>       rhs_values(n_q_points);
  std::vector<double>       reaction_rate_values(n_q_points);
  std::vector<Tensor<2, 3>> a_inverse_values(n_q_points);

  // define extractors
  const FEValuesExtractors::Vector flux(0);
  const FEValuesExtractors::Scalar concentration(3);

  // ------------------------------------------------------------------
  // loop over cells
  typename DoFHandler<3>::active_cell_iterator cell =
                                                 dof_handler.begin_active(),
                                               endc = dof_handler.end();
  for (; cell != endc; ++cell)
    {
      fe_values.reinit(cell);

      local_matrix = 0;
      local_rhs    = 0;

      for (unsigned int n_basis = 0;
           n_basis < GeometryInfo<dim>::faces_per_cell;
           ++n_basis)
        {
          local_rhs_v[n_basis] = 0;
        }

      right_hand_side.value_list(fe_values.get_quadrature_points(), rhs_values);
      reaction_rate.value_list(fe_values.get_quadrature_points(),
                               reaction_rate_values);
      a_inverse.value_list(fe_values.get_quadrature_points(), a_inverse_values);

      // loop over quad points
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              // Test functions
              const Tensor<1, 3> phi_i_sigma = fe_values[flux].value(i, q);
              const double div_phi_i_sigma   = fe_values[flux].divergence(i, q);
              const double phi_i_u = fe_values[concentration].value(i, q);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  // Trial functions
                  const Tensor<1, 3> phi_j_sigma = fe_values[flux].value(j, q);
                  const double       div_phi_j_sigma =
                    fe_values[flux].divergence(j, q);
                  const double phi_j_u = fe_values[concentration].value(j, q);

                  /*
                   * Discretize
                   * K^{-1}sigma + grad(u) = 0
                   * div(sigma) + alpha*u = f , where
                   * alpha<0 (this is important) This is
                   * the simplest form of a
                   * diffusion-reaction equation where an
                   * anisotropic diffusion and reaction
                   * are in balance in a heterogeneous
                   * medium. A multiscale reaction rate is
                   * also possible and can easily be
                   * added.
                   */
                  local_matrix(i, j) +=
                    (phi_i_sigma * a_inverse_values[q] *
                       phi_j_sigma               /* Block (0, 0)*/
                     - div_phi_i_sigma * phi_j_u /* Block (0, 1)*/
                     + phi_i_u * div_phi_j_sigma /* Block (1, 0)*/
                     + reaction_rate_values[q] * phi_i_u *
                         phi_j_u /* Block (1, 1)*/
                     ) *
                    fe_values.JxW(q);
                } // end for ++j

              // Only for use in global assembly
              local_rhs(i) += phi_i_u * rhs_values[q] * fe_values.JxW(q);

              // Only for use in local solving. Critical for
              // Darcy type problem. (Think of LBB between
              // RT0-DGQ0)
              for (unsigned int n_basis = 0;
                   n_basis < GeometryInfo<dim>::faces_per_cell;
                   ++n_basis)
                {
                  // Note the sign here.
                  if (parameters.is_laplace)
                    {
                      const double scale = 1 / volume_measure;
                      if (n_basis == 0)
                        local_rhs_v[n_basis](i) +=
                          -phi_i_u * scale * fe_values.JxW(q);
                      if (n_basis == 1)
                        local_rhs_v[n_basis](i) +=
                          phi_i_u * scale * fe_values.JxW(q);
                      if (n_basis == 2)
                        local_rhs_v[n_basis](i) +=
                          -phi_i_u * scale * fe_values.JxW(q);
                      if (n_basis == 3)
                        local_rhs_v[n_basis](i) +=
                          phi_i_u * scale * fe_values.JxW(q);
                      if (n_basis == 4)
                        local_rhs_v[n_basis](i) +=
                          -phi_i_u * scale * fe_values.JxW(q);
                      if (n_basis == 5)
                        local_rhs_v[n_basis](i) +=
                          phi_i_u * scale * fe_values.JxW(q);
                    }
                  else
                    local_rhs_v[n_basis](i) += 0;
                }
            } // end for ++i
        }     // end for ++q

      // Only for use in global assembly.
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          global_rhs(local_dof_indices[i]) += local_rhs(i);
        }

      // Add to global matrix. Take care of constraints later.
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              assembled_matrix.add(local_dof_indices[i],
                                   local_dof_indices[j],
                                   local_matrix(i, j));
            }

          for (unsigned int n_basis = 0;
               n_basis < GeometryInfo<dim>::faces_per_cell;
               ++n_basis)
            {
              system_rhs[n_basis](local_dof_indices[i]) +=
                local_rhs_v[n_basis](i);
            }
        }
      // ------------------------------------------
    } // end for ++cell

  if (parameters.verbose)
    {
      timer.stop();
      std::cout << "done in   " << timer.cpu_time() << "   seconds."
                << std::endl;
    }
} // end assemble()

void
DivergenceStabilizedBasis::solve_direct(unsigned int n_basis)
{
  Timer timer;
  if (parameters.verbose)
    {
      std::cout << "	Solving linear system (SparseDirectUMFPACK) in cell   "
                << global_cell_id.to_string() << "   for basis   " << n_basis
                << ".....";

      timer.restart();
    }

  // for convenience define an alias
  const BlockVector<double> &system_rhs = system_rhs[n_basis];
  BlockVector<double> &      solution   = modified_basis[n_basis];

  // use direct solver
  SparseDirectUMFPACK A_inv;
  A_inv.initialize(system_matrix);

  A_inv.vmult(solution, system_rhs);

  modified_basis_constraints[n_basis].distribute(solution);

  if (parameters.verbose)
    {
      timer.stop();
      std::cout << "done in   " << timer.cpu_time() << "   seconds."
                << std::endl;
    }
}

void
DivergenceStabilizedBasis::solve_iterative(unsigned int n_basis)
{
  Timer timer;
  Timer inner_timer;

  // ------------------------------------------
  // Make a preconditioner for each system matrix
  if (parameters.verbose)
    {
      std::cout << "	Computing preconditioner in cell   "
                << global_cell_id.to_string() << "   for basis   " << n_basis
                << "   .....";

      timer.restart();
    }

  // for convenience define an alias
  const BlockVector<double> &system_rhs = system_rhs[n_basis];
  BlockVector<double> &      solution   = modified_basis[n_basis];

  inner_schur_preconditioner = std::make_shared<
    typename LinearSolvers::LocalInnerPreconditioner<3>::type>();

  typename LinearSolvers::LocalInnerPreconditioner<3>::type::AdditionalData
    data;
  inner_schur_preconditioner->initialize(system_matrix.block(0, 0), data);

  if (parameters.verbose)
    {
      timer.stop();
      std::cout << "done in   " << timer.cpu_time() << "   seconds."
                << std::endl;
    }
  // ------------------------------------------

  // Now solve.
  if (parameters.verbose)
    {
      std::cout << "	Solving linear system (iteratively, with "
                   "preconditioner) in cell   "
                << global_cell_id.to_string() << "   for basis   " << n_basis
                << "   .....";

      timer.restart();
    }

  // Construct inverse of upper left block
  const LinearSolvers::InverseMatrix<
    SparseMatrix<double>,
    typename LinearSolvers::LocalInnerPreconditioner<3>::type>
    A_inverse(system_matrix.block(0, 0), *inner_schur_preconditioner);

  Vector<double> tmp(system_rhs.block(0).size());
  {
    // Set up Schur complement
    LinearSolvers::SchurComplement<
      typename LinearSolvers::LocalInnerPreconditioner<3>::type>
      schur_complement(system_matrix, A_inverse, dof_handler);

    // Compute schur_rhs = -g + C*A^{-1}*f
    Vector<double> schur_rhs(system_rhs.block(1).size());

    A_inverse.vmult(tmp, system_rhs.block(0));
    system_matrix.block(1, 0).vmult(schur_rhs, tmp);
    schur_rhs -= system_rhs.block(1);

    if (parameters.verbose)
      {
        inner_timer.restart();
      }

    LinearAlgebra::SchurComplement<typename InnerPreconditioner<dim>::type>
                             schur_complement(system_matrix, A_inverse);
    SolverControl            solver_control(solution.block(1).size(),
                                 1e-6 * schur_rhs.l2_norm(),
                                 /* log_history */ false,
                                 /* log_result */ false);
    SolverCG<Vector<double>> cg_solver(solver_control);

    SparseILU<double> preconditioner;
    preconditioner.initialize(preconditioner_matrix.block(1, 1),
                              SparseILU<double>::AdditionalData());

    InverseMatrix<SparseMatrix<double>, SparseILU<double>> Mp_inverse(
      preconditioner_matrix.block(1, 1), preconditioner);

    cg_solver.solve(schur_complement, solution.block(1), schur_rhs, Mp_inverse);

    modified_basis_constraints[n_basis].distribute(solution);

    if (parameters.verbose)
      {
        inner_timer.stop();

        std::cout << std::endl
                  << "		- Iterative Schur complement solver "
                     "converged in   "
                  << solver_control.last_step()
                  << "   iterations.	Time:	" << inner_timer.cpu_time()
                  << "   seconds." << std::endl;
      }
  }

  {
    if (parameters.verbose)
      {
        inner_timer.restart();
      }

    // use computed pressure to solve for velocity
    system_matrix.block(0, 1).vmult(tmp, solution.block(1));
    tmp *= -1;
    tmp += system_rhs.block(0);

    // Solve for velocity
    A_inverse.vmult(solution.block(0), tmp);

    modified_basis_constraints[n_basis].distribute(solution);

    if (parameters.verbose)
      {
        inner_timer.stop();

        std::cout << "		- Outer solver completed.   Time:   "
                  << inner_timer.cpu_time() << "   seconds." << std::endl;
      }
  }

  if (parameters.verbose)
    {
      timer.stop();
      std::cout << "		- done in   " << timer.cpu_time() << "   seconds."
                << std::endl;
    }
}



void
DivergenceStabilizedBasis::assemble_global_element_matrix()
{
  // First, reset.
  global_element_matrix = 0;

  // Get lengths of tmp vectors for assembly
  std::vector<types::global_dof_index> dofs_per_component =
    DoFTools::count_dofs_per_fe_component(dof_handler);
  const unsigned int n_sigma = dofs_per_component[0],
                     n_u     = dofs_per_component[3];

  Vector<double> tmp_u(n_u), tmp_sigma(n_sigma);

  // This assembles the local contribution to the global global matrix
  // with an algebraic trick. It uses the local system matrix stored in
  // the respective basis object.
  unsigned int block_row, block_col;

  BlockVector<double> *test_vec_ptr, *trial_vec_ptr;

  for (unsigned int i_test = 0; i_test < length_system_basis; ++i_test)
    {
      test_vec_ptr =
        &(modified_basis.at(i_test % GeometryInfo<dim>::faces_per_cell));

      if (i_test < GeometryInfo<dim>::faces_per_cell)
        block_row = 0;
      else
        block_row = 1;

      for (unsigned int i_trial = 0; i_trial < length_system_basis; ++i_trial)
        {
          trial_vec_ptr =
            &(modified_basis.at(i_trial % GeometryInfo<dim>::faces_per_cell));

          if (i_trial < GeometryInfo<dim>::faces_per_cell)
            block_col = 0;
          else
            block_col = 1;

          if (block_row == 0) /* This means we are testing with sigma. */
            {
              if (block_col == 0) /* This means trial function is sigma. */
                {
                  assembled_matrix.block(block_row, block_col)
                    .vmult(tmp_sigma, trial_vec_ptr->block(block_col));
                  global_element_matrix(i_test, i_trial) +=
                    (test_vec_ptr->block(block_row) * tmp_sigma);
                  tmp_sigma = 0;
                }
              if (block_col == 1) /* This means trial function is u. */
                {
                  assembled_matrix.block(block_row, block_col)
                    .vmult(tmp_sigma, trial_vec_ptr->block(block_col));
                  global_element_matrix(i_test, i_trial) +=
                    (test_vec_ptr->block(block_row) * tmp_sigma);
                  tmp_sigma = 0;
                }
            }  // end if
          else /* This means we are testing with u. */
            {
              if (block_col == 0) /* This means trial function is sigma. */
                {
                  assembled_matrix.block(block_row, block_col)
                    .vmult(tmp_u, trial_vec_ptr->block(block_col));
                  global_element_matrix(i_test, i_trial) +=
                    (test_vec_ptr->block(block_row) * tmp_u);
                  tmp_u = 0;
                }
              if (block_col == 1) /* This means trial function is u. */
                {
                  assembled_matrix.block(block_row, block_col)
                    .vmult(tmp_u, trial_vec_ptr->block(block_col));
                  global_element_matrix(i_test, i_trial) +=
                    test_vec_ptr->block(block_row) * tmp_u;
                  tmp_u = 0;
                }
            } // end else
        }     // end for i_trial

      if (i_test >= GeometryInfo<dim>::faces_per_cell)
        {
          block_row = 1;
          // If we are testing with u we possibly have a
          // right-hand side.
          global_element_rhs(i_test) +=
            test_vec_ptr->block(block_row) * global_rhs.block(block_row);
        }
    } // end for i_test

  is_built_global_element_matrix = true;
}

void
DivergenceStabilizedBasis::output_basis()
{
  Timer timer;
  if (parameters.verbose)
    {
      std::cout << "	Writing local basis in cell   "
                << global_cell_id.to_string() << ".....";

      timer.restart();
    }

  for (unsigned int n_basis = 0; n_basis < GeometryInfo<dim>::faces_per_cell;
       ++n_basis)
    {
      BlockVector<double> &basis = modified_basis[n_basis];

      std::vector<std::string> solution_names(3, "sigma");
      solution_names.push_back("u");

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(
          3, DataComponentInterpretation::component_is_part_of_vector);
      interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);

      DataOut<3> data_out;
      data_out.attach_dof_handler(dof_handler);

      data_out.add_data_vector(basis,
                               solution_names,
                               DataOut<3>::type_dof_data,
                               interpretation);

      data_out.build_patches();

      // filename
      std::string filename = "basis_stokes-dq";
      filename += ".div";
      filename += "." + Utilities::int_to_string(local_subdomain, 5);
      filename += ".cell-" + global_cell_id.to_string();
      filename += ".index-";
      filename += Utilities::int_to_string(n_basis, 2);
      filename += ".vtu";

      std::ofstream output(parameters.dirname_output + "/" + filename);
      data_out.write_vtu(output);
    }

  if (parameters.verbose)
    {
      timer.stop();
      std::cout << "done in   " << timer.cpu_time() << "   seconds."
                << std::endl;
    }
}

void
DivergenceStabilizedBasis::output_global_solution_in_cell()
{
  DataOut<3> data_out;
  data_out.attach_dof_handler(dof_handler);

  std::vector<std::string> solution_names(3, "sigma");
  solution_names.emplace_back("u");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      3, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);

  data_out.add_data_vector(global_solution,
                           solution_names,
                           DataOut<3>::type_dof_data,
                           data_component_interpretation);

  // Postprocess
  std::unique_ptr<RTDQ_PostProcessor> postprocessor(
    new RTDQ_PostProcessor(parameter_filename));
  data_out.add_data_vector(global_solution, *postprocessor);

  data_out.build_patches();

  std::ofstream output(parameters.dirname_output + "/" +
                       parameters.filename_global);
  data_out.write_vtu(output);
}

void
DivergenceStabilizedBasis::set_global_weights(
  const std::vector<double> &weights)
{
  // Copy assignment of global weights
  global_weights = weights;

  // reinitialize the global solution on this cell
  global_solution = 0;

  const unsigned int dofs_per_cell_sigma = fe.base_element(0).n_dofs_per_cell();
  const unsigned int dofs_per_cell_u     = fe.base_element(1).n_dofs_per_cell();

  // First set block 0
  for (unsigned int i = 0; i < dofs_per_cell_sigma; ++i)
    global_solution.block(0).sadd(1,
                                  global_weights[i],
                                  modified_basis[i].block(0));

  // Then set block 1
  for (unsigned int i = 0; i < dofs_per_cell_u; ++i)
    global_solution.block(1).sadd(1,
                                  global_weights[i + dofs_per_cell_sigma],
                                  modified_basis[i].block(1));

  is_set_global_weights = true;
}

void
DivergenceStabilizedBasis::set_u_to_std()
{
  for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
    modified_basis[i].block(1) = 1;
}

void
DivergenceStabilizedBasis::project_velocity_divergence_on_pressure_space(
  unsigned int n_basis)
{
  // // Quadrature used for projection
  // QGauss<3> quad_rule(3);

  // // Set up vector shape function from finite element on current cell
  // ShapeFun::BasisRaviartThomas<3> std_shape_function_div(global_cell_it,
  //                                                        /* degree */ 0);

  // DoFHandler<3> dof_handler_fake(triangulation);
  // dof_handler_fake.distribute_dofs(fe.base_element(0));

  // if (parameters.renumber_dofs)
  //   {
  //     throw std::runtime_error("Renumbering DoFs not allowed when sanity "
  //                              "checking basis for sigma.");
  //   }

  // AffineConstraints<double> constraints;
  // constraints.clear();
  // DoFTools::make_hanging_node_constraints(dof_handler_fake, constraints);
  // constraints.close();

  // for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
  //   {
  //     modified_basis.at(i).block(0).reinit(dof_handler_fake.n_dofs());

  //     std_shape_function_div.set_index(i);

  //     VectorTools::project(dof_handler_fake,
  //                          constraints,
  //                          quad_rule,
  //                          std_shape_function_div,
  //                          modified_basis[i].block(0));
  //   }

  // dof_handler_fake.clear();
}

void
DivergenceStabilizedBasis::set_filename_global()
{
  parameters.filename_global +=
    ("." + Utilities::int_to_string(local_subdomain, 5) + ".cell-" +
     global_cell_id.to_string() + ".vtu");
}

const FullMatrix<double> &
DivergenceStabilizedBasis::get_global_element_matrix() const
{
  return global_element_matrix;
}

const Vector<double> &
DivergenceStabilizedBasis::get_global_element_rhs() const
{
  return global_element_rhs;
}

const std::string &
DivergenceStabilizedBasis::get_filename_global() const
{
  return parameters.filename_global;
}

void
DivergenceStabilizedBasis::run()
{
  Timer timer;

  if (true)
    {
      char processor_name[MPI_MAX_PROCESSOR_NAME];
      int  name_len;
      MPI_Get_processor_name(processor_name, &name_len);
      std::string proc_name(processor_name, name_len);

      std::cout << "	Solving for basis in cell   "
                << global_cell_id.to_string() << "   [machine: " << proc_name
                << " | rank: "
                << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                << "]   .....";
      timer.restart();
    }

  // Create grid
  setup_grid();

  // Reserve space for system matrices
  setup_system_matrix();

  // Set up boundary conditions and other constraints
  setup_basis_dofs_div();

  // Assemble
  assemble_system();

  for (unsigned int n_basis = 0; n_basis < GeometryInfo<dim>::faces_per_cell;
       ++n_basis)
    {
      project_velocity_divergence_on_pressure_space(unsigned int n_basis);

      // This is for curl.
      system_matrix.reinit(sparsity_pattern);
      preconditioner_matrix.reinit(preconditioner_sparsity_pattern);

      system_matrix.copy_from(assembled_matrix);
      preconditioer_matrix.copy_from(assembled_preconditioner);

      // Now take care of constraints
      modified_basis_constraints[n_basis].condense(system_matrix,
                                                   system_rhs[n_basis]);
      modified_basis_constraints[n_basis].condense(preconditioer_matrix,
                                                   system_rhs[n_basis]);

      // Now solve
      if (parameters.use_direct_solver)
        solve_direct(n_basis);
      else
        {
          solve_iterative(n_basis);
        }
    } // end for

  {
    // Free memory as much as possible
    system_matrix.clear();
    preconditioner_matrix.clear();
    sparsity_pattern.reinit(0, 0);
    preconditioner_sparsity_pattern.reinit(0, 0);
    for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; ++i)
      {
        modified_basis_constraints[i].clear();
      }
  }

  assemble_global_element_matrix();

  {
    // Free more memory as much as possible
    assembled_matrix.clear();
    preconditioer_matrix.clear();
  }

  // We need to set a filename for the global solution on the current cell
  set_filename_global();

  if (is_first_cell)
    {
      try
        {
          Tools::create_data_directory(parameters.dirname_output);
        }
      catch (std::runtime_error &e)
        {
          // No exception handling here.
        }
      output_basis();
    }

  if (true)
    {
      timer.stop();

      std::cout << "done in   " << timer.cpu_time() << "   seconds."
                << std::endl;
    }
}

MSSTOKES_CLOSE_NAMESPACE