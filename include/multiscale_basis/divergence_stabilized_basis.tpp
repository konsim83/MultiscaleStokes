#pragma once

// My headers
#include <multiscale_basis/divergence_stabilized_basis.h>

MSSTOKES_OPEN_NAMESPACE

template <int dim>
DivergenceStabilizedBasis<dim>::DivergenceStabilizedBasis(
  const CoreModelData::Parameters &                  parameters,
  typename Triangulation<dim>::active_cell_iterator &global_cell,
  bool                                               is_first_cell,
  unsigned int                                       local_subdomain,
  CoreModelData::TemperatureForcing<dim> &           temperature_forcing,
  MPI_Comm                                           mpi_communicator)
  : mpi_communicator(mpi_communicator)
  , parameters(parameters)
  , triangulation()
  , fe(FE_Q<dim>(parameters.stokes_velocity_degree),
       dim,
       (parameters.use_locally_conservative_discretization ?
          static_cast<const FiniteElement<dim> &>(
            FE_DGP<dim>(parameters.stokes_velocity_degree - 1)) :
          static_cast<const FiniteElement<dim> &>(
            FE_Q<dim>(parameters.stokes_velocity_degree - 1))),
       1)
  , velocity_fe(FE_Q<dim>(parameters.stokes_velocity_degree), dim)
  , dof_handler(triangulation)
  , velocity_basis_constraints(velocity_fe.dofs_per_cell)
  , pressure_basis_constraints(fe.base_element(1).dofs_per_cell)
  , sparsity_pattern()
  , preconditioner_sparsity_pattern()
  , velocity_basis(velocity_fe.dofs_per_cell)
  , pressure_basis(fe.base_element(1).dofs_per_cell)
  , system_rhs(fe.dofs_per_cell)
  , global_element_matrix(fe.dofs_per_cell, fe.dofs_per_cell)
  , global_element_rhs(fe.dofs_per_cell)
  , global_weights(fe.dofs_per_cell, 0)
  , global_cell_id(global_cell->id())
  , is_first_cell(is_first_cell)
  , global_cell(global_cell)
  , local_subdomain(local_subdomain)
  , corner_points(GeometryInfo<dim>::vertices_per_cell, Point<dim>())
  , temperature_forcing_ptr(&temperature_forcing)
  , is_built_global_element_matrix(false)
  , is_set_global_weights(false)
  , is_set_cell_data(false)
{
  for (unsigned int vertex_n = 0;
       vertex_n < GeometryInfo<dim>::vertices_per_cell;
       ++vertex_n)
    {
      corner_points[vertex_n] = global_cell->vertex(vertex_n);
    }

  is_set_cell_data = true;
}

template <int dim>
DivergenceStabilizedBasis<dim>::DivergenceStabilizedBasis(
  const DivergenceStabilizedBasis &other)
  : mpi_communicator(other.mpi_communicator)
  , parameters(other.parameters)
  , triangulation() // must be constructed deliberately, but is empty on
                    // copying anyway
  , fe(FE_Q<dim>(parameters.stokes_velocity_degree),
       dim,
       (parameters.use_locally_conservative_discretization ?
          static_cast<const FiniteElement<dim> &>(
            FE_DGP<dim>(parameters.stokes_velocity_degree - 1)) :
          static_cast<const FiniteElement<dim> &>(
            FE_Q<dim>(parameters.stokes_velocity_degree - 1))),
       1)
  , velocity_fe(FE_Q<dim>(parameters.stokes_velocity_degree), dim)
  , dof_handler(triangulation)
  , velocity_basis_constraints(other.velocity_basis_constraints)
  , pressure_basis_constraints(other.pressure_basis_constraints)
  , sparsity_pattern()
  , preconditioner_sparsity_pattern()
  , velocity_basis(other.velocity_basis)
  , pressure_basis(other.pressure_basis)
  , system_rhs(other.system_rhs)
  , global_rhs(other.global_rhs)
  , global_element_matrix(other.global_element_matrix)
  , global_element_rhs(other.global_element_rhs)
  , global_weights(other.global_weights)
  , global_solution(other.global_solution)
  , inner_schur_preconditioner(other.inner_schur_preconditioner)
  , global_cell_id(other.global_cell_id)
  , is_first_cell(other.is_first_cell)
  , global_cell(other.global_cell)
  , local_subdomain(other.local_subdomain)
  , corner_points(other.corner_points)
  , temperature_forcing_ptr(other.temperature_forcing_ptr)
  , is_built_global_element_matrix(other.is_built_global_element_matrix)
  , is_set_global_weights(other.is_set_global_weights)
  , is_set_cell_data(other.is_set_cell_data)
{
  global_cell_id = global_cell->id();

  for (unsigned int vertex_n = 0;
       vertex_n < GeometryInfo<dim>::vertices_per_cell;
       ++vertex_n)
    {
      corner_points[vertex_n] = global_cell->vertex(vertex_n);
    }

  is_set_cell_data = true;
}

template <int dim>
DivergenceStabilizedBasis<dim>::~DivergenceStabilizedBasis()
{
  system_matrix.clear();

  for (unsigned int index_velocity_dof = 0;
       index_velocity_dof < velocity_basis.size();
       ++index_velocity_dof)
    {
      velocity_basis_constraints[index_velocity_dof].clear();
    }

  dof_handler.clear();
}

template <int dim>
void
DivergenceStabilizedBasis<dim>::setup_grid()
{
  Assert(is_set_cell_data, ExcMessage("Cell data must be set first."));

  GridGenerator::general_cell(triangulation,
                              corner_points,
                              /* colorize faces */ false);

  triangulation.refine_global(parameters.refinements_basis);
}

template <int dim>
void
DivergenceStabilizedBasis<dim>::setup_system_matrix()
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
    std::cout << "      Number of active cells: "
              << triangulation.n_active_cells() << std::endl
              << "      Total number of cells: " << triangulation.n_cells()
              << std::endl
              << "      Number of degrees of freedom: " << dof_handler.n_dofs()
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

    DoFTools::make_sparsity_pattern(dof_handler, coupling, dsp);

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
                                    preconditioner_dsp);

    preconditioner_sparsity_pattern.copy_from(preconditioner_dsp);
  }

  assembled_matrix.reinit(sparsity_pattern);
  assembled_preconditioner.reinit(preconditioner_sparsity_pattern);

  global_solution.reinit(dofs_per_block);
  global_rhs.reinit(dofs_per_block);
}

template <int dim>
void
DivergenceStabilizedBasis<dim>::setup_basis_dofs()
{
  Assert(is_set_cell_data, ExcMessage("Cell data must be set first."));

  Timer timer;

  if (parameters.verbose_basis)
    {
      std::cout << "      Setting up dofs for velocity basis...";

      timer.restart();
    }

  { // Velocity constraints
    ShapeFun::ShapeFunctionVector<dim> std_shape_function_stokes(velocity_fe,
                                                                 global_cell);
    Functions::ZeroFunction<dim>       zero_fun_scalar(1);

    // zero_fun_and_std_basis_pressure
    ShapeFun::FunctionConcatinator<dim> std_basis_velocity_and_zero_fun(
      std_shape_function_stokes, zero_fun_scalar);

    FEValuesExtractors::Vector velocity(0);
    for (unsigned int index_velocity_dof = 0;
         index_velocity_dof < velocity_basis.size();
         ++index_velocity_dof)
      {
        std_shape_function_stokes.set_shape_fun_index(index_velocity_dof);

        // set constraints (first hanging nodes, then velocity)
        velocity_basis_constraints[index_velocity_dof].clear();

        DoFTools::make_hanging_node_constraints(
          dof_handler, velocity_basis_constraints[index_velocity_dof]);

        VectorTools::interpolate_boundary_values(
          dof_handler,
          /* boundary_id*/ 0,
          std_basis_velocity_and_zero_fun,
          velocity_basis_constraints[index_velocity_dof],
          fe.component_mask(velocity));

        velocity_basis_constraints[index_velocity_dof].close();
      }
  }

  { // Pressure constraints
    Functions::ZeroFunction<dim>       zero_fun_vector(dim);
    ShapeFun::ShapeFunctionScalar<dim> std_basis_pressure(fe.base_element(1),
                                                          global_cell);

    // zero_fun_and_std_basis_pressure
    ShapeFun::FunctionConcatinator<dim> zero_fun_and_std_basis_pressure(
      zero_fun_vector, std_basis_pressure);

    FEValuesExtractors::Scalar pressure(dim);
    for (unsigned int index_pressure_dof = 0;
         index_pressure_dof < pressure_basis.size();
         ++index_pressure_dof)
      {
        std_basis_pressure.set_shape_fun_index(index_pressure_dof);

        // set constraints (first hanging nodes, then velocity)
        pressure_basis_constraints[index_pressure_dof].clear();

        DoFTools::make_hanging_node_constraints(
          dof_handler, pressure_basis_constraints[index_pressure_dof]);

        VectorTools::interpolate_boundary_values(
          dof_handler,
          /* boundary_id*/ 0,
          zero_fun_and_std_basis_pressure,
          pressure_basis_constraints[index_pressure_dof],
          fe.component_mask(pressure));

        pressure_basis_constraints[index_pressure_dof].close();
      }
  }

  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1;
  DoFRenumbering::component_wise(dof_handler, block_component);

  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);

  for (unsigned int index_velocity_dof = 0;
       index_velocity_dof < velocity_basis.size();
       ++index_velocity_dof)
    {
      velocity_basis[index_velocity_dof].reinit(dofs_per_block);
      system_rhs[index_velocity_dof].reinit(dofs_per_block);
    }
  for (unsigned int index_pressure_dof = 0;
       index_pressure_dof < pressure_basis.size();
       ++index_pressure_dof)
    {
      pressure_basis[index_pressure_dof].reinit(dofs_per_block);
      system_rhs[velocity_basis.size() + index_pressure_dof].reinit(
        dofs_per_block);
    }

  if (parameters.verbose_basis)
    {
      timer.stop();
      std::cout << " done in   " << timer.cpu_time() << "   seconds."
                << std::endl;
    }
}

template <int dim>
void
DivergenceStabilizedBasis<dim>::assemble_system()
{
  Timer timer;
  timer.restart();

  assembled_matrix         = 0;
  assembled_preconditioner = 0;
  global_rhs               = 0;
  for (unsigned int n_basis = 0; n_basis < fe.dofs_per_cell; ++n_basis)
    {
      system_rhs[n_basis] = 0;
    }

  // Choose appropriate quadrature rules
  QGauss<dim> quadrature_formula(parameters.stokes_velocity_degree + 2);

  // Get relevant quantities to be updated from finite element
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  // Define some abbreviations
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  // Declare local contributions and reserve memory
  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> local_preconditioner_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     local_global_rhs(dofs_per_cell);
  std::vector<Vector<double>> local_system_rhs(fe.dofs_per_cell,
                                               Vector<double>(dofs_per_cell));

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // equation data
  ShapeFun::ShapeFunctionVectorDivergence div_std_velocity_shape(velocity_fe,
                                                                 global_cell);

  // allocate
  std::vector<double>              temperature_forcing_values(n_q_points);
  std::vector<std::vector<double>> div_std_velocity_shape_values(
    velocity_basis.size(), std::vector<double>(n_q_points));

  // define extractors
  const FEValuesExtractors::Vector velocity(0);
  const FEValuesExtractors::Scalar pressure(dim);

  std::vector<Tensor<1, dim>>          phi_u(dofs_per_cell);
  std::vector<SymmetricTensor<2, dim>> symgrad_phi_u(dofs_per_cell);
  std::vector<double>                  div_phi_u(dofs_per_cell);
  std::vector<double>                  phi_p(dofs_per_cell);

  // ------------------------------------------------------------------
  // loop over cells
  typename DoFHandler<dim>::active_cell_iterator cell =
                                                   dof_handler.begin_active(),
                                                 endc = dof_handler.end();
  for (; cell != endc; ++cell)
    {
      fe_values.reinit(cell);

      // Reset local matrices
      local_matrix     = 0;
      local_global_rhs = 0;
      for (unsigned int n_basis = 0; n_basis < fe.dofs_per_cell; ++n_basis)
        {
          local_system_rhs[n_basis] = 0;
        }

      // Get function values
      for (unsigned int index_velocity_dof = 0;
           index_velocity_dof < velocity_basis.size();
           ++index_velocity_dof)
        {
          div_std_velocity_shape.set_shape_fun_index(index_velocity_dof);

          div_std_velocity_shape.value_list(
            fe_values.get_quadrature_points(),
            div_std_velocity_shape_values[index_velocity_dof]);
        }

      temperature_forcing_ptr->value_list(fe_values.get_quadrature_points(),
                                          temperature_forcing_values);

      // loop over quad points
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          // Get the shape function values
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              phi_u[k]         = fe_values[velocity].value(k, q);
              symgrad_phi_u[k] = fe_values[velocity].symmetric_gradient(k, q);
              div_phi_u[k]     = fe_values[velocity].divergence(k, q);
              phi_p[k]         = fe_values[pressure].value(k, q);
            }

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  local_matrix(i, j) +=
                    (2 * parameters.physical_constants.kinematic_viscosity *
                       (symgrad_phi_u[i] * symgrad_phi_u[j]) -
                     div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j]) *
                    fe_values.JxW(q);

                  local_preconditioner_matrix(i, j) +=
                    (phi_p[i] * phi_p[j]) * fe_values.JxW(q);
                } // end for ++j

              const Tensor<1, dim> gravity =
                CoreModelData::vertical_gravity_vector(
                  fe_values.quadrature_point(q),
                  parameters.physical_constants.gravity_constant);

              // Only for use in global assembly
              local_global_rhs(i) += temperature_forcing_values[q] *
                                     (phi_u[i] * gravity) * fe_values.JxW(q);

              // The right-hand side should be zero apart from the pressure
              // component. This needs to be processed later to contain the
              // orthogonal projection of the divergence of the velocity shape
              // function in the pressure componenent
              for (unsigned int index_velocity_dof = 0;
                   index_velocity_dof < velocity_basis.size();
                   ++index_velocity_dof)
                {
                  local_system_rhs[index_velocity_dof](i) +=
                    phi_p[i] *
                    div_std_velocity_shape_values[index_velocity_dof][q] *
                    fe_values.JxW(q);
                }

              // Nothing to do for pressure so far
              for (unsigned int index_pressure_dof = 0;
                   index_pressure_dof < pressure_basis.size();
                   ++index_pressure_dof)
                {
                  local_system_rhs[index_pressure_dof + velocity_basis.size()](
                    i) += 0;
                }
            } // end for ++i
        }     // end for ++q

      // Only for use in global assembly.
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          global_rhs(local_dof_indices[i]) += local_global_rhs(i);
        }

      // Add to global matrix. Take care of constraints later.
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              assembled_matrix.add(local_dof_indices[i],
                                   local_dof_indices[j],
                                   local_matrix(i, j));

              assembled_preconditioner.add(local_dof_indices[i],
                                           local_dof_indices[j],
                                           local_preconditioner_matrix(i, j));
            }

          for (unsigned int index_velocity_dof = 0;
               index_velocity_dof < velocity_basis.size();
               ++index_velocity_dof)
            {
              system_rhs[index_velocity_dof](local_dof_indices[i]) +=
                local_system_rhs[index_velocity_dof](i);
            }

          for (unsigned int index_pressure_dof = 0;
               index_pressure_dof < pressure_basis.size();
               ++index_pressure_dof)
            {
              system_rhs[index_pressure_dof + velocity_basis.size()](
                local_dof_indices[i]) +=
                local_system_rhs[index_pressure_dof + velocity_basis.size()](i);
            }
        }
      // ------------------------------------------
    } // end for ++cell

  timer.stop();
  if (parameters.verbose_basis)
    {
      std::cout << "      Assembling local linear system ... done in   "
                << timer.cpu_time() << "   seconds." << std::endl;
    }
} // end assemble()

template <int dim>
void
DivergenceStabilizedBasis<dim>::solve_direct(unsigned int n_basis)
{
  Timer timer;

  if (parameters.verbose_basis)
    {
      std::cout << "      Invoke direct solver for basis   " << n_basis
                << "   ...";

      timer.restart();
    }

  // for convenience define an alias
  const BlockVector<double> &this_system_rhs = system_rhs[n_basis];
  BlockVector<double> &      solution        = velocity_basis[n_basis];

  // use direct solver
  SparseDirectUMFPACK A_inv;
  A_inv.initialize(system_matrix);

  A_inv.vmult(solution, this_system_rhs);

  if (n_basis < velocity_basis.size())
    velocity_basis_constraints[n_basis].distribute(solution);
  else
    pressure_basis_constraints[n_basis - velocity_basis.size()].distribute(
      solution);

  if (parameters.verbose_basis)
    {
      timer.stop();
      std::cout
        << "      Solving linear system (SparseDirectUMFPACK) for basis   "
        << n_basis << "... done in   " << timer.cpu_time() << "   seconds."
        << std::endl;
    }
}

template <int dim>
void
DivergenceStabilizedBasis<dim>::solve_iterative(unsigned int n_basis)
{
  Timer timer;
  Timer inner_timer;

  // ------------------------------------------
  // Make a preconditioner for each system matrix
  if (parameters.verbose_basis)
    {
      std::cout << "      Computing preconditioner for basis   " << n_basis
                << "   ...";

      timer.restart();
    }

  // for convenience define an alias
  const BlockVector<double> &this_system_rhs = system_rhs[n_basis];
  BlockVector<double> &      solution        = velocity_basis[n_basis];

  inner_schur_preconditioner = std::make_shared<
    typename LinearAlgebra::LocalInnerPreconditioner<dim>::type>();

  typename LinearAlgebra::LocalInnerPreconditioner<dim>::type::AdditionalData
    data;
  inner_schur_preconditioner->initialize(system_matrix.block(0, 0), data);

  if (parameters.verbose_basis)
    {
      timer.stop();
      std::cout << "done in   " << timer.cpu_time() << "   seconds."
                << std::endl;
    }
  // ------------------------------------------

  // Now solve.
  if (parameters.verbose_basis)
    {
      std::cout << "      Solving linear system (iteratively, with "
                   "preconditioner) for basis   "
                << n_basis << "   ...";

      timer.restart();
    }

  // Construct inverse of upper left block
  const LinearAlgebra::InverseMatrix<
    SparseMatrix<double>,
    typename LinearAlgebra::LocalInnerPreconditioner<dim>::type>
    A_inverse(system_matrix.block(0, 0), *inner_schur_preconditioner);

  Vector<double> tmp(this_system_rhs.block(0).size());
  {
    // Compute schur_rhs = -g + C*A^{-1}*f
    Vector<double> schur_rhs(this_system_rhs.block(1).size());

    A_inverse.vmult(tmp, this_system_rhs.block(0));
    system_matrix.block(1, 0).vmult(schur_rhs, tmp);
    schur_rhs -= this_system_rhs.block(1);

    if (parameters.verbose_basis)
      {
        inner_timer.restart();
      }

    // Set up Schur complement
    LinearAlgebra::SchurComplement<
      typename LinearAlgebra::LocalInnerPreconditioner<dim>::type>
      schur_complement(system_matrix, A_inverse);

    SolverControl solver_control(solution.block(1).size(),
                                 1e-6 * schur_rhs.l2_norm(),
                                 /* log_history */ false,
                                 /* log_result */ false);

    SolverCG<Vector<double>> cg_solver(solver_control);

    // This is the preconditioner of the outer preconditioner which is just the
    // inverse mass on the pressure space
    SparseILU<double> preconditioner;
    preconditioner.initialize(preconditioner_matrix.block(1, 1),
                              SparseILU<double>::AdditionalData());

    const LinearAlgebra::InverseMatrix<SparseMatrix<double>, SparseILU<double>>
      Mp_inverse(preconditioner_matrix.block(1, 1), preconditioner);

    cg_solver.solve(schur_complement, solution.block(1), schur_rhs, Mp_inverse);

    if (n_basis < velocity_basis.size())
      velocity_basis_constraints[n_basis].distribute(solution);
    else
      pressure_basis_constraints[n_basis - velocity_basis.size()].distribute(
        solution);

    if (parameters.verbose_basis)
      {
        inner_timer.stop();

        std::cout << std::endl
                  << "            - Iterative Schur complement solver "
                     "converged in   "
                  << solver_control.last_step()
                  << "   iterations.      Time:      " << inner_timer.cpu_time()
                  << "   seconds." << std::endl;
      }
  }

  {
    if (parameters.verbose_basis)
      {
        inner_timer.restart();
      }

    // use computed pressure to solve for velocity
    system_matrix.block(0, 1).vmult(tmp, solution.block(1));
    tmp *= -1;
    tmp += this_system_rhs.block(0);

    // Solve for velocity
    A_inverse.vmult(solution.block(0), tmp);

    if (n_basis < velocity_basis.size())
      velocity_basis_constraints[n_basis].distribute(solution);
    else
      pressure_basis_constraints[n_basis - velocity_basis.size()].distribute(
        solution);

    if (parameters.verbose_basis)
      {
        inner_timer.stop();

        std::cout << "            - Outer solver completed.   Time:   "
                  << inner_timer.cpu_time() << "   seconds." << std::endl;
      }
  }

  if (parameters.verbose_basis)
    {
      timer.stop();
      std::cout << "            - done in   " << timer.cpu_time()
                << "   seconds." << std::endl;
    }
}


template <int dim>
void
DivergenceStabilizedBasis<dim>::project_standard_basis_on_velocity_space()
{
  Timer timer;
  timer.restart();

  // Quadrature used for projection
  const QGauss<dim>     quad_rule(parameters.stokes_velocity_degree + 1);
  const QGauss<dim - 1> quad_rule_boundary(parameters.stokes_velocity_degree);

  ShapeFun::ShapeFunctionVector<dim> std_shape_function_stokes(velocity_fe,
                                                               global_cell);
  Functions::ZeroFunction<dim>       zero_fun_scalar(1);

  // zero_fun_and_std_basis_pressure
  ShapeFun::FunctionConcatinator<dim> std_basis_velocity_and_zero_fun(
    std_shape_function_stokes, zero_fun_scalar);

  FEValuesExtractors::Vector velocity(0);
  for (unsigned int index_velocity_dof = 0;
       index_velocity_dof < velocity_basis.size();
       ++index_velocity_dof)
    {
      std_shape_function_stokes.set_shape_fun_index(index_velocity_dof);

      VectorTools::project(dof_handler,
                           velocity_basis_constraints[index_velocity_dof],
                           quad_rule,
                           std_basis_velocity_and_zero_fun,
                           velocity_basis[index_velocity_dof],
                           /* enforce_zero_boundary */ false,
                           quad_rule_boundary,
                           /* project_to_boundary_first */ true);
    }

  timer.stop();
  if (parameters.verbose_basis)
    {
      std::cout
        << "      Projecting standard basis on velocity space ... done in   "
        << timer.cpu_time() << "   seconds." << std::endl;
    }
}



template <int dim>
void
DivergenceStabilizedBasis<dim>::project_standard_basis_on_pressure_space()
{
  Timer timer;
  timer.restart();

  // Quadrature used for projection
  const QGauss<dim>     quad_rule(parameters.stokes_velocity_degree + 1);
  const QGauss<dim - 1> quad_rule_boundary(parameters.stokes_velocity_degree);

  Functions::ZeroFunction<dim>       zero_fun_vector(dim);
  ShapeFun::ShapeFunctionScalar<dim> std_basis_pressure(fe.base_element(1),
                                                        global_cell);

  // zero_fun_and_std_basis_pressure
  ShapeFun::FunctionConcatinator<dim> zero_fun_and_std_basis_pressure(
    zero_fun_vector, std_basis_pressure);

  for (unsigned int index_pressure_dof = 0;
       index_pressure_dof < pressure_basis.size();
       ++index_pressure_dof)
    {
      std_basis_pressure.set_shape_fun_index(index_pressure_dof);

      VectorTools::project(dof_handler,
                           pressure_basis_constraints[index_pressure_dof],
                           quad_rule,
                           zero_fun_and_std_basis_pressure,
                           pressure_basis[index_pressure_dof],
                           /* enforce_zero_boundary */ false,
                           quad_rule_boundary,
                           /* project_to_boundary_first */ true);
    }

  timer.stop();
  if (parameters.verbose_basis)
    {
      std::cout
        << "      Projecting standard basis on pressure space ... done in   "
        << timer.cpu_time() << "   seconds." << std::endl;
    }
}

template <int dim>
void
DivergenceStabilizedBasis<dim>::assemble_global_element_matrix()
{
  // First, reset.
  global_element_matrix = 0;
  global_element_rhs    = 0;

  // Get lengths of tmp vectors for assembly
  std::vector<unsigned int> block_component(dim + 1, 0);
  block_component[dim] = 1;
  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler, block_component);
  const unsigned int n_u = dofs_per_block[0];
  const unsigned int n_p = dofs_per_block[1];

  Vector<double> tmp_u(n_u), tmp_p(n_p);

  // This assembles the local contribution to the global global matrix
  // with an algebraic trick. It uses the local system matrix stored in
  // the respective basis object.
  unsigned int block_row, block_col;

  Vector<double> *test_vec_ptr, *trial_vec_ptr;
  // std::shared_ptr<SparseMatrix<double>> relevant_block;

  const unsigned int dofs_per_cell   = fe.dofs_per_cell;
  const unsigned int dofs_per_cell_u = velocity_basis.size();
  // const unsigned int dofs_per_cell_p = pressure_basis.size();

  for (unsigned int i_test = 0; i_test < dofs_per_cell; ++i_test)
    {
      if (i_test < dofs_per_cell_u)
        {
          test_vec_ptr = &(velocity_basis[i_test].block(0));
          block_row    = 0;
        }
      else
        {
          test_vec_ptr = &(pressure_basis[i_test - dofs_per_cell_u].block(1));
          block_row    = 1;
        }

      for (unsigned int i_trial = 0; i_trial < dofs_per_cell; ++i_trial)
        {
          if (i_trial < dofs_per_cell_u)
            {
              block_col     = 0;
              trial_vec_ptr = &(velocity_basis[i_trial].block(0));
            }
          else
            {
              block_col = 1;
              trial_vec_ptr =
                &(pressure_basis[i_trial - dofs_per_cell_u].block(1));
            }

          // relevant_block = &(assembled_matrix.block(block_row, block_col));

          if (block_row == 0) /* This means we are testing with velocity. */
            {
              assembled_matrix.block(block_row, block_col)
                .vmult(tmp_u, *trial_vec_ptr);
              global_element_matrix(i_test, i_trial) += (*test_vec_ptr) * tmp_u;
              tmp_u = 0;
            }  // end if
          else /* This means we are testing with pressure. */
            {
              assembled_matrix.block(block_row, block_col)
                .vmult(tmp_p, *trial_vec_ptr);
              global_element_matrix(i_test, i_trial) += (*test_vec_ptr) * tmp_p;
              tmp_p = 0;
            } // end if
        }     // end for i_trial

      if (i_test <= dofs_per_cell_u)
        {
          // block_row = 0 in this case

          // If we are testing with u we possibly have a
          // right-hand side.
          global_element_rhs(i_test) +=
            (*test_vec_ptr) * global_rhs.block(block_row);
        }
    } // end for i_test

  is_built_global_element_matrix = true;
}

template <int dim>
void
DivergenceStabilizedBasis<dim>::output_basis()
{
  Timer timer;
  timer.restart();

  for (unsigned int n_basis = 0; n_basis < fe.dofs_per_cell; ++n_basis)
    {
      std::vector<std::string> solution_names(dim, "velocity");
      solution_names.push_back("pressure");

      std::vector<DataComponentInterpretation::DataComponentInterpretation>
        interpretation(
          dim, DataComponentInterpretation::component_is_part_of_vector);
      interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);

      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);

      data_out.add_data_vector(
        (n_basis < velocity_basis.size() ?
           velocity_basis[n_basis] :
           pressure_basis[n_basis - velocity_basis.size()]),
        solution_names,
        DataOut<dim>::type_dof_data,
        interpretation);

      data_out.build_patches();

      // filename
      std::string filename = "basis_stokes";
      filename += ".div";
      filename += "." + Utilities::int_to_string(local_subdomain, 5);
      filename += ".cell-" + global_cell_id.to_string();
      filename += ".index-";
      filename += Utilities::int_to_string(n_basis, 2);
      filename += ".vtu";

      std::ofstream output(parameters.dirname_output + "/" + filename);
      data_out.write_vtu(output);
    }

  timer.stop();
  if (parameters.verbose_basis)
    {
      std::cout << "      Writing local basis ... done in   "
                << timer.cpu_time() << "   seconds." << std::endl;
    }
}

template <int dim>
void
DivergenceStabilizedBasis<dim>::output_global_solution_in_cell()
{
  Assert(
    is_set_global_weights,
    ExcMessage(
      "Global weights must be set before local output of global solution."));

  // First the forcing projection with a different DoFHandler
  DoFHandler<dim> forcing_dof_handler(triangulation);
  FE_Q<dim>       forcing_fe(parameters.stokes_velocity_degree);
  forcing_dof_handler.distribute_dofs(forcing_fe);

  AffineConstraints<double> no_constraints;
  no_constraints.clear();
  DoFTools::make_hanging_node_constraints(forcing_dof_handler, no_constraints);
  no_constraints.close();

  Vector<double> bouyancy_forcing(forcing_dof_handler.n_dofs());
  VectorTools::project(forcing_dof_handler,
                       no_constraints,
                       QGauss<dim>(parameters.stokes_velocity_degree + 1),
                       *temperature_forcing_ptr,
                       bouyancy_forcing);

  // Now join the Stokes and the forcing dofs
  const FESystem<dim> joint_fe(fe, 1, forcing_fe, 1);

  DoFHandler<dim> joint_dof_handler(triangulation);
  joint_dof_handler.distribute_dofs(joint_fe);

  Assert(joint_dof_handler.n_dofs() ==
           dof_handler.n_dofs() + forcing_dof_handler.n_dofs(),
         ExcInternalError());

  Vector<double> joint_solution;

  joint_solution.reinit(joint_dof_handler.n_dofs());

  {
    std::vector<types::global_dof_index> local_joint_dof_indices(
      joint_fe.dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
    std::vector<types::global_dof_index> local_forcing_dof_indices(
      forcing_fe.dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
      joint_cell = joint_dof_handler.begin_active(),
      joint_endc = joint_dof_handler.end(), cell = dof_handler.begin_active(),
      forcing_cell = forcing_dof_handler.begin_active();
    for (; joint_cell != joint_endc; ++joint_cell, ++cell, ++forcing_cell)
      {
        if (joint_cell->is_locally_owned())
          {
            joint_cell->get_dof_indices(local_joint_dof_indices);
            cell->get_dof_indices(local_dof_indices);
            forcing_cell->get_dof_indices(local_forcing_dof_indices);

            for (unsigned int i = 0; i < joint_fe.dofs_per_cell; ++i)
              if (joint_fe.system_to_base_index(i).first.first == 0)
                {
                  Assert(joint_fe.system_to_base_index(i).second <
                           local_dof_indices.size(),
                         ExcInternalError());

                  joint_solution(local_joint_dof_indices[i]) = global_solution(
                    local_dof_indices[joint_fe.system_to_base_index(i).second]);
                }
              else
                {
                  Assert(joint_fe.system_to_base_index(i).first.first == 1,
                         ExcInternalError());
                  Assert(joint_fe.system_to_base_index(i).second <
                           local_forcing_dof_indices.size(),
                         ExcInternalError());

                  joint_solution(local_joint_dof_indices[i]) = bouyancy_forcing(
                    local_forcing_dof_indices[joint_fe.system_to_base_index(i)
                                                .second]);
                }
          } // end if is_locally_owned()
      }     // end for ++joint_cell
  }

  Postprocessor<dim> postprocessor(
    Utilities::MPI::this_mpi_process(mpi_communicator));

  DataOut<dim> data_out;
  data_out.attach_dof_handler(joint_dof_handler);

  data_out.add_data_vector(joint_solution, postprocessor);

  data_out.build_patches(parameters.stokes_velocity_degree);

  const std::string filename =
    (parameters.filename_output + "." +
     Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
     ".vtu");
  std::ofstream output(parameters.dirname_output + "/" + filename);
  data_out.write_vtu(output);

  forcing_dof_handler.clear();
}

template <int dim>
void
DivergenceStabilizedBasis<dim>::set_global_weights(
  const std::vector<double> &weights)
{
  // Copy assignment of global weights
  global_weights = weights;

  // reinitialize the global solution on this cell
  global_solution = 0;

  const unsigned int dofs_per_cell_u = velocity_basis.size();
  const unsigned int dofs_per_cell_p = pressure_basis.size();

  // First set block 0
  for (unsigned int i = 0; i < dofs_per_cell_u; ++i)
    global_solution.block(0).sadd(1,
                                  global_weights[i],
                                  velocity_basis[i].block(0));

  // Then set block 1
  for (unsigned int i = 0; i < dofs_per_cell_p; ++i)
    global_solution.block(1).sadd(1,
                                  global_weights[i + dofs_per_cell_u],
                                  pressure_basis[i].block(1));

  is_set_global_weights = true;
}

template <int dim>
void
DivergenceStabilizedBasis<dim>::project_velocity_divergence_on_pressure_space()
{
  PreconditionJacobi<SparseMatrix<double>> mass_preconditioner;
  mass_preconditioner.initialize(assembled_preconditioner.block(1, 1));

  const LinearAlgebra::InverseMatrix<SparseMatrix<double>,
                                     PreconditionJacobi<SparseMatrix<double>>>
    pressure_mass_inverse(assembled_preconditioner.block(1, 1),
                          mass_preconditioner);

  for (unsigned int index_velocity_dof = 0;
       index_velocity_dof < velocity_basis.size();
       ++index_velocity_dof)
    {
      // Copy the rhs from the assembled system since this is not exactly what
      // we need (we need the orthogonal projection)
      const Vector<double> rhs_tmp(velocity_basis[index_velocity_dof].block(1));
      velocity_basis[index_velocity_dof].block(1) = 0;

      pressure_mass_inverse.vmult(velocity_basis[index_velocity_dof].block(1),
                                  rhs_tmp);
    }
}

template <int dim>
void
DivergenceStabilizedBasis<dim>::set_filename_global()
{
  parameters.filename_output +=
    ("." + Utilities::int_to_string(local_subdomain, 5) + ".cell-" +
     global_cell_id.to_string() + ".vtu");
}

template <int dim>
const FullMatrix<double> &
DivergenceStabilizedBasis<dim>::get_global_element_matrix() const
{
  Assert(
    is_built_global_element_matrix,
    ExcMessage(
      "Global element matrix and global right-hand side must first be built on local level."));

  return global_element_matrix;
}

template <int dim>
const Vector<double> &
DivergenceStabilizedBasis<dim>::get_global_element_rhs() const
{
  Assert(
    is_built_global_element_matrix,
    ExcMessage(
      "Global element matrix and global right-hand side must first be built on local level."));

  return global_element_rhs;
}

template <int dim>
const std::string &
DivergenceStabilizedBasis<dim>::get_filename_global() const
{
  return parameters.filename_output;
}

template <int dim>
void
DivergenceStabilizedBasis<dim>::run()
{
  Timer timer;
  timer.restart();

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int  name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  std::string proc_name(processor_name, name_len);

  if (parameters.verbose_basis)
    {
      std::cout << "      [cell: " << global_cell_id.to_string()
                << " | machine: " << proc_name << " | rank: "
                << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << "]"
                << std::endl
                << "         Solving for basis ..." << std::endl;
    }

  // Create grid
  setup_grid();

  // Reserve space for system matrices
  setup_system_matrix();

  // Set up boundary conditions and other constraints
  setup_basis_dofs();

  // Since we do not solve for a modified pressure just project all standard
  // basis functions
  project_standard_basis_on_pressure_space();

  // Assemble
  assemble_system();

  // This is to make sure that the global divergence maps the modified
  // velocity basis into the right space
  project_velocity_divergence_on_pressure_space();

  if (true)
    {
      project_standard_basis_on_velocity_space();
    }
  else
    {
      // for (unsigned int n_basis = 0; n_basis < velocity_basis.size();
      // ++n_basis)
      //   {
      //     // The assembled matrices do not contain boundary conditions so
      //     copy
      //     // them and apply the constraints
      //     system_matrix.reinit(sparsity_pattern);
      //     preconditioner_matrix.reinit(preconditioner_sparsity_pattern);

      //     system_matrix.copy_from(assembled_matrix);
      //     preconditioner_matrix.copy_from(assembled_preconditioner);

      //     // Now take care of constraints
      //     velocity_basis_constraints[n_basis].condense(system_matrix,
      //                                                  system_rhs[n_basis]);
      //     velocity_basis_constraints[n_basis].condense(preconditioner_matrix);

      //     // Now solve
      //     if (true)
      //       solve_direct(n_basis);
      //     else
      //       {
      //         solve_iterative(n_basis);
      //       }
      //   } // end for
    } // end if

  assemble_global_element_matrix();

  {
    // Free as much memory as possible
    system_matrix.clear();
    preconditioner_matrix.clear();
    sparsity_pattern.reinit(0, 0);
    preconditioner_sparsity_pattern.reinit(0, 0);
    for (unsigned int index_velocity_dof = 0;
         index_velocity_dof < velocity_basis.size();
         ++index_velocity_dof)
      {
        velocity_basis_constraints[index_velocity_dof].clear();
      }
    for (unsigned int index_pressure_dof = 0;
         index_pressure_dof < pressure_basis.size();
         ++index_pressure_dof)
      {
        pressure_basis_constraints[index_pressure_dof].clear();
      }
  }

  {
    // Free more memory as much as possible
    assembled_matrix.clear();
    preconditioner_matrix.clear();
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

  { // Always time the run function
    timer.stop();

    std::cout << "      [cell: " << global_cell_id.to_string()
              << " | machine: " << proc_name
              << " | rank: " << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
              << "]" << std::endl
              << "         Solving for basis ... done in   " << timer.cpu_time()
              << "   seconds." << std::endl
              << std::endl;
  }
}

MSSTOKES_CLOSE_NAMESPACE