#include <core/domain_geometry.h>
#include <core/stokes_model.h>

MSSTOKES_OPEN_NAMESPACE

template <int dim>
MultiscaleStokesModel<dim>::MultiscaleStokesModel(
  CoreModelData::Parameters &parameters_)
  : DomainGeometry<dim>()
  , parameters(parameters_)
  , mapping(1)
  , stokes_fe(FE_Q<dim>(parameters.stokes_velocity_degree),
              dim,
              (parameters.use_locally_conservative_discretization ?
                 static_cast<const FiniteElement<dim> &>(
                   FE_DGP<dim>(parameters.stokes_velocity_degree - 1)) :
                 static_cast<const FiniteElement<dim> &>(
                   FE_Q<dim>(parameters.stokes_velocity_degree - 1))),
              1)
  , stokes_dof_handler(this->triangulation)
  , temperature_forcing(this->domain_center,
                        parameters.physical_constants.reference_temperature,
                        parameters.physical_constants.expansion_coefficient,
                        /* variance */ parameters.variance)
{}



template <int dim>
MultiscaleStokesModel<dim>::~MultiscaleStokesModel()
{}



/////////////////////////////////////////////////////////////
// System and dof setup
/////////////////////////////////////////////////////////////

template <int dim>
void
MultiscaleStokesModel<dim>::setup_stokes_matrices(
  const std::vector<IndexSet> &stokes_partitioning,
  const std::vector<IndexSet> &stokes_relevant_partitioning)
{
  stokes_matrix.clear();
  LA::BlockSparsityPattern     sp(stokes_partitioning,
                              stokes_partitioning,
                              stokes_relevant_partitioning,
                              this->mpi_communicator);
  Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
  for (unsigned int c = 0; c < dim + 1; ++c)
    for (unsigned int d = 0; d < dim + 1; ++d)
      if (!((c == dim) && (d == dim)))
        coupling[c][d] = DoFTools::always;
      else
        coupling[c][d] = DoFTools::none;

  DoFTools::make_sparsity_pattern(stokes_dof_handler,
                                  coupling,
                                  sp,
                                  stokes_constraints,
                                  false,
                                  Utilities::MPI::this_mpi_process(
                                    this->mpi_communicator));
  sp.compress();

  stokes_matrix.reinit(sp);
}



template <int dim>
void
MultiscaleStokesModel<dim>::setup_stokes_preconditioner(
  const std::vector<IndexSet> &stokes_partitioning,
  const std::vector<IndexSet> &stokes_relevant_partitioning)
{
  Amg_preconditioner.reset();
  Mp_preconditioner.reset();

  stokes_preconditioner_matrix.clear();
  LA::BlockSparsityPattern sp(stokes_partitioning,
                              stokes_partitioning,
                              stokes_relevant_partitioning,
                              this->mpi_communicator);

  Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
  for (unsigned int c = 0; c < dim + 1; ++c)
    for (unsigned int d = 0; d < dim + 1; ++d)
      if (c == d)
        coupling[c][d] = DoFTools::always;
      else
        coupling[c][d] = DoFTools::none;

  DoFTools::make_sparsity_pattern(stokes_dof_handler,
                                  coupling,
                                  sp,
                                  stokes_constraints,
                                  false,
                                  Utilities::MPI::this_mpi_process(
                                    this->mpi_communicator));
  sp.compress();

  stokes_preconditioner_matrix.reinit(sp);
}



template <int dim>
void
MultiscaleStokesModel<dim>::setup_dofs()
{
  TimerOutput::Scope timing_section(
    this->computing_timer, "MultiscaleStokesModel - setup dofs of systems");

  /*
   * Setup dof handlers for stokes
   */
  std::vector<unsigned int> stokes_sub_blocks(dim + 1, 0);
  stokes_sub_blocks[dim] = 1;

  stokes_dof_handler.distribute_dofs(stokes_fe);

  DoFRenumbering::component_wise(stokes_dof_handler, stokes_sub_blocks);

  /*
   * Count dofs
   */
  const std::vector<types::global_dof_index> stokes_dofs_per_block =
    DoFTools::count_dofs_per_fe_block(stokes_dof_handler, stokes_sub_blocks);
  const unsigned int n_u = stokes_dofs_per_block[0],
                     n_p = stokes_dofs_per_block[1];

  /*
   * Comma separated large numbers
   */
  std::locale s = this->pcout.get_stream().getloc();
  this->pcout.get_stream().imbue(std::locale(""));

  /*
   * Print some mesh and dof info
   */
  this->pcout << "   Number of active cells: "
              << this->triangulation.n_global_active_cells() << " (on "
              << this->triangulation.n_levels() << " levels)" << std::endl
              << "   Number of degrees of freedom: " << n_u + n_p << " (" << n_u
              << " + " << n_p << ")" << std::endl
              << std::endl;
  this->pcout.get_stream().imbue(s);

  /*
   * Setup partitioners to store what dofs and matrix entries are stored on
   * the local processor
   */
  {
    stokes_index_set = stokes_dof_handler.locally_owned_dofs();
    stokes_partitioning.push_back(stokes_index_set.get_view(0, n_u));
    stokes_partitioning.push_back(stokes_index_set.get_view(n_u, n_u + n_p));
    DoFTools::extract_locally_relevant_dofs(stokes_dof_handler,
                                            stokes_relevant_set);
    stokes_relevant_partitioning.push_back(
      stokes_relevant_set.get_view(0, n_u));
    stokes_relevant_partitioning.push_back(
      stokes_relevant_set.get_view(n_u, n_u + n_p));
  }


  /*
   * Setup constraints and boundary values for Stokes. Make sure this is
   * consistent with the initial data.
   */
  {
    stokes_constraints.clear();
    stokes_constraints.reinit(stokes_relevant_set);
    DoFTools::make_hanging_node_constraints(stokes_dof_handler,
                                            stokes_constraints);

    {
      std::vector<
        GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>>
        periodicity_vector;

      /*
       * All dimensions up to the last are periodic (z-direction is always
       * bounded from below and form above)
       */
      for (unsigned int d = 0; d < dim - 1; ++d)
        {
          GridTools::collect_periodic_faces(stokes_dof_handler,
                                            /*b_id1*/ 2 * (d + 1) - 2,
                                            /*b_id2*/ 2 * (d + 1) - 1,
                                            /*direction*/ d,
                                            periodicity_vector);
        }

      DoFTools::make_periodicity_constraints<DoFHandler<dim>>(
        periodicity_vector, stokes_constraints);

      FEValuesExtractors::Vector velocity_components(0);

      // No-slip on boundary id 2/4 (lower in 2d/3d)
      VectorTools::interpolate_boundary_values(
        stokes_dof_handler,
        (dim == 2 ? 2 : 4),
        Functions::ZeroFunction<dim>(dim + 1),
        stokes_constraints,
        stokes_fe.component_mask(velocity_components));

      // No-flux on boundary id 3/5 (upper in 2d/3d)
      std::set<types::boundary_id> no_normal_flux_boundaries;
      no_normal_flux_boundaries.insert((dim == 2 ? 3 : 5));

      VectorTools::compute_no_normal_flux_constraints(stokes_dof_handler,
                                                      0,
                                                      no_normal_flux_boundaries,
                                                      stokes_constraints,
                                                      mapping);
    }

    stokes_constraints.close();
  }

  /*
   * Setup the matrix and vector objects.
   */
  setup_stokes_matrices(stokes_partitioning, stokes_relevant_partitioning);
  setup_stokes_preconditioner(stokes_partitioning,
                              stokes_relevant_partitioning);

  stokes_rhs.reinit(stokes_partitioning,
                    stokes_relevant_partitioning,
                    this->mpi_communicator,
                    /* is_writable */ true);
  stokes_solution.reinit(stokes_relevant_partitioning, this->mpi_communicator);
}


/////////////////////////////////////////////////////////////
// Initialize multiscale basis
/////////////////////////////////////////////////////////////

template <int dim>
void
MultiscaleStokesModel<dim>::initialize_and_compute_basis()
{
  TimerOutput::Scope t(this->computing_timer,
                       "Modified basis initialization and computation");

  typename Triangulation<dim>::active_cell_iterator cell = stokes_dof_handler
                                                             .begin_active(),
                                                    endc =
                                                      stokes_dof_handler.end();
  /*
   * This is only to identify first cell (for setting output flag for basis).
   */
  const CellId first_cell = cell->id();

  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          const bool is_first_cell =
            ((first_cell == cell->id()) &&
             (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0));

          DivergenceStabilizedBasis<dim> current_cell_problem(
            parameters,
            cell,
            is_first_cell,
            this->triangulation.locally_owned_subdomain(),
            temperature_forcing,
            this->mpi_communicator);

          CellId current_cell_id(cell->id());
          cell_basis_map.emplace(current_cell_id, current_cell_problem);
        }
    } // end ++cell

  /*
   * Now each node possesses a set of basis objects.
   * We need to compute them on each node and could even do so in
   * a locally threaded way. Here we use a serial version. Threading happens
   * within the basis.
   */
  typename BasisMap::iterator it_basis    = cell_basis_map.begin(),
                              it_endbasis = cell_basis_map.end();
  for (; it_basis != it_endbasis; ++it_basis)
    {
      (it_basis->second).run();
    }
}

/////////////////////////////////////////////////////////////
// Assembly Stokes preconditioner
/////////////////////////////////////////////////////////////


template <int dim>
void
MultiscaleStokesModel<dim>::local_assemble_stokes_preconditioner(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  Assembly::Scratch::StokesPreconditioner<dim> &        scratch,
  Assembly::CopyData::StokesPreconditioner<dim> &       data)
{
  const unsigned int dofs_per_cell = stokes_fe.dofs_per_cell;
  const unsigned int n_q_points = scratch.stokes_fe_values.n_quadrature_points;

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);



  scratch.stokes_fe_values.reinit(cell);

  cell->get_dof_indices(data.local_dof_indices);
  data.local_matrix = 0;

  for (unsigned int q = 0; q < n_q_points; ++q)
    {
      for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          scratch.grad_phi_u[k] =
            scratch.stokes_fe_values[velocities].gradient(k, q);
          scratch.phi_p[k] = scratch.stokes_fe_values[pressure].value(k, q);
        }

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              data.local_matrix(i, j) +=
                (parameters.physical_constants.kinematic_viscosity *
                   scalar_product(scratch.grad_phi_u[i],
                                  scratch.grad_phi_u[j]) +
                 (1. / parameters.physical_constants.kinematic_viscosity) *
                   (scratch.phi_p[i] * scratch.phi_p[j])) *
                scratch.stokes_fe_values.JxW(q);
            }
        }
    }
}


template <int dim>
void
MultiscaleStokesModel<dim>::copy_local_to_global_stokes_preconditioner(
  const Assembly::CopyData::StokesPreconditioner<dim> &data)
{
  stokes_constraints.distribute_local_to_global(data.local_matrix,
                                                data.local_dof_indices,
                                                stokes_preconditioner_matrix);
}


template <int dim>
void
MultiscaleStokesModel<dim>::assemble_stokes_preconditioner()
{
  TimerOutput::Scope timer_section(this->computing_timer,
                                   "   Assembly Stokes preconditioner");

  stokes_preconditioner_matrix = 0;
  const QGauss<dim> quadrature_formula(parameters.stokes_velocity_degree + 1);
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  WorkStream::run(
    CellFilter(IteratorFilters::LocallyOwnedCell(),
               stokes_dof_handler.begin_active()),
    CellFilter(IteratorFilters::LocallyOwnedCell(), stokes_dof_handler.end()),
    std::bind(&MultiscaleStokesModel<dim>::local_assemble_stokes_preconditioner,
              this,
              std::placeholders::_1,
              std::placeholders::_2,
              std::placeholders::_3),
    std::bind(
      &MultiscaleStokesModel<dim>::copy_local_to_global_stokes_preconditioner,
      this,
      std::placeholders::_1),
    Assembly::Scratch::StokesPreconditioner<dim>(stokes_fe,
                                                 quadrature_formula,
                                                 mapping,
                                                 update_JxW_values |
                                                   update_values |
                                                   update_gradients),
    Assembly::CopyData::StokesPreconditioner<dim>(stokes_fe));

  stokes_preconditioner_matrix.compress(VectorOperation::add);
}



template <int dim>
void
MultiscaleStokesModel<dim>::build_stokes_preconditioner()
{
  TimerOutput::Scope timer_section(this->computing_timer,
                                   "   Build Stokes preconditioner");

  this->pcout << "   Assembling and building Stokes block preconditioner..."
              << std::flush;

  assemble_stokes_preconditioner();

  Mp_preconditioner = std::make_shared<Block_11_PreconType>();
  typename Block_11_PreconType::AdditionalData Mp_preconditioner_data;
  Mp_preconditioner->initialize(stokes_preconditioner_matrix.block(1, 1),
                                Mp_preconditioner_data);


  std::vector<std::vector<bool>> constant_modes_velocity;
  FEValuesExtractors::Vector     velocity_components(0);
  DoFTools::extract_constant_modes(stokes_dof_handler,
                                   stokes_fe.component_mask(
                                     velocity_components),
                                   constant_modes_velocity);
  Amg_preconditioner = std::make_shared<Block_00_PreconType>();
  typename Block_00_PreconType::AdditionalData Amg_data;
  /*
   * This is relevant to AMG preconditioners
   */
  Amg_data.constant_modes        = constant_modes_velocity;
  Amg_data.elliptic              = true;
  Amg_data.higher_order_elements = true;
  Amg_data.smoother_sweeps       = 1;
  Amg_data.aggregation_threshold = 0.02;
  Amg_preconditioner->initialize(stokes_preconditioner_matrix.block(0, 0),
                                 Amg_data);

  this->pcout << std::endl;
}



/////////////////////////////////////////////////////////////
// Assembly Stokes system
/////////////////////////////////////////////////////////////


template <int dim>
void
MultiscaleStokesModel<dim>::assemble_stokes_system()
{
  TimerOutput::Scope timer_section(this->computing_timer,
                                   "   Assemble Stokes system");

  this->pcout << "   Assembling Stokes system..." << std::flush;

  stokes_matrix = 0;
  stokes_rhs    = 0;

  const unsigned int dofs_per_cell = stokes_fe.dofs_per_cell;

  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     local_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // loop over cells
  typename DoFHandler<dim>::active_cell_iterator cell = stokes_dof_handler
                                                          .begin_active(),
                                                 endc =
                                                   stokes_dof_handler.end();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          typename BasisMap::iterator it_basis =
            cell_basis_map.find(cell->id());

          local_matrix = 0;
          local_rhs    = 0;

          local_matrix = (it_basis->second).get_global_element_matrix();
          local_rhs    = (it_basis->second).get_global_element_rhs();

          // Add to global matrix, include constraints
          cell->get_dof_indices(local_dof_indices);
          stokes_constraints.distribute_local_to_global(local_matrix,
                                                        local_rhs,
                                                        local_dof_indices,
                                                        stokes_matrix,
                                                        stokes_rhs);
        }
    } // end for ++cell

  stokes_matrix.compress(VectorOperation::add);
  stokes_rhs.compress(VectorOperation::add);

  this->pcout << std::endl;
}



template <int dim>
double
MultiscaleStokesModel<dim>::get_maximal_velocity() const
{
  const QIterated<dim> quadrature_formula(QTrapez<1>(),
                                          parameters.stokes_velocity_degree);
  const unsigned int   n_q_points = quadrature_formula.size();

  FEValues<dim>               fe_values(mapping,
                          stokes_fe,
                          quadrature_formula,
                          update_values);
  std::vector<Tensor<1, dim>> velocity_values(n_q_points);

  const FEValuesExtractors::Vector velocities(0);
  double                           max_local_velocity = 0;

  for (const auto &cell : stokes_dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        fe_values[velocities].get_function_values(stokes_solution,
                                                  velocity_values);
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            max_local_velocity =
              std::max(max_local_velocity, velocity_values[q].norm());
          }
      }

  double max_global_velocity =
    Utilities::MPI::max(max_local_velocity, this->mpi_communicator);

  this->pcout << "   Max velocity (with dimensions): " << max_global_velocity
              << " m/s" << std::endl;

  return max_global_velocity;
}


/////////////////////////////////////////////////////////////
// solve
/////////////////////////////////////////////////////////////

template <int dim>
void
MultiscaleStokesModel<dim>::solve()
{
  TimerOutput::Scope timer_section(this->computing_timer,
                                   "   Solve Stokes system");
  this->pcout
    << "   Solving Stokes system with (block preconditioned solver)... "
    << std::flush;

  LA::MPI::BlockVector distributed_stokes_solution(stokes_rhs);
  distributed_stokes_solution = stokes_solution;

  const unsigned int
    start = (distributed_stokes_solution.block(0).size() +
             distributed_stokes_solution.block(1).local_range().first),
    end   = (distributed_stokes_solution.block(0).size() +
           distributed_stokes_solution.block(1).local_range().second);

  for (unsigned int i = start; i < end; ++i)
    if (stokes_constraints.is_constrained(i))
      distributed_stokes_solution(i) = 0;

  PrimitiveVectorMemory<LA::MPI::BlockVector> mem;
  unsigned int                                n_iterations = 0;
  const double  solver_tolerance = 1e-8 * stokes_rhs.l2_norm();
  SolverControl solver_control(30,
                               solver_tolerance,
                               /* log_history */ true,
                               /* log_result */ true);

  try
    {
      const LinearAlgebra::BlockSchurPreconditioner<Block_00_PreconType,
                                                    Block_11_PreconType>
        preconditioner(stokes_matrix,
                       stokes_preconditioner_matrix,
                       *Mp_preconditioner,
                       *Amg_preconditioner,
                       false);

      SolverFGMRES<LA::MPI::BlockVector> solver(
        solver_control,
        mem,
        SolverFGMRES<LA::MPI::BlockVector>::AdditionalData(30));

      solver.solve(stokes_matrix,
                   distributed_stokes_solution,
                   stokes_rhs,
                   preconditioner);

      n_iterations = solver_control.last_step();
    }
  catch (SolverControl::NoConvergence &)
    {
      const LinearAlgebra::BlockSchurPreconditioner<Block_00_PreconType,
                                                    Block_11_PreconType>
        preconditioner(stokes_matrix,
                       stokes_preconditioner_matrix,
                       *Mp_preconditioner,
                       *Amg_preconditioner,
                       /* full_preconditioned_solve*/ true);

      SolverControl solver_control_refined(stokes_matrix.m(), solver_tolerance);

      SolverFGMRES<LA::MPI::BlockVector> solver(
        solver_control_refined,
        mem,
        SolverFGMRES<LA::MPI::BlockVector>::AdditionalData(50));

      solver.solve(stokes_matrix,
                   distributed_stokes_solution,
                   stokes_rhs,
                   preconditioner);

      n_iterations =
        (solver_control.last_step() + solver_control_refined.last_step());
    }
  stokes_constraints.distribute(distributed_stokes_solution);

  stokes_solution = distributed_stokes_solution;

  this->pcout << n_iterations << " iterations." << std::endl;
}


/////////////////////////////////////////////////////////////
// Output results and related stuff
/////////////////////////////////////////////////////////////


template <int dim>
void
MultiscaleStokesModel<dim>::send_global_weights_to_cell()
{
  // For each cell we get dofs_per_cell values
  const unsigned int                   dofs_per_cell = stokes_fe.dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // active cell iterator
  typename DoFHandler<dim>::active_cell_iterator cell = stokes_dof_handler
                                                          .begin_active(),
                                                 endc =
                                                   stokes_dof_handler.end();
  for (; cell != endc; ++cell)
    {
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(local_dof_indices);
          std::vector<double> extracted_weights(dofs_per_cell, 0);
          stokes_solution.extract_subvector_to(local_dof_indices,
                                               extracted_weights);

          typename BasisMap::iterator it_basis =
            cell_basis_map.find(cell->id());
          (it_basis->second).set_global_weights(extracted_weights);
        }
    } // end ++cell
}

template <int dim>
std::vector<std::string>
MultiscaleStokesModel<dim>::collect_filenames_on_mpi_process() const
{
  std::vector<std::string> filename_list;

  typename BasisMap::const_iterator it_basis    = cell_basis_map.begin(),
                                    it_endbasis = cell_basis_map.end();
  for (; it_basis != it_endbasis; ++it_basis)
    {
      filename_list.push_back((it_basis->second).get_filename_global());
    }

  return filename_list;
}


template <int dim>
void
MultiscaleStokesModel<dim>::output_results()
{
  TimerOutput::Scope timer_section(this->computing_timer,
                                   "Postprocessing and output");

  this->pcout << "   Writing multiscale stabilized Stokes solution... "
              << std::flush;

  // ---------------------------------------------------
  // write local fine solution
  typename BasisMap::iterator it_basis    = cell_basis_map.begin(),
                              it_endbasis = cell_basis_map.end();

  for (; it_basis != it_endbasis; ++it_basis)
    {
      (it_basis->second).output_global_solution_in_cell();
    }

  // Gather local filenames
  std::vector<std::vector<std::string>> filename_list_list =
    Utilities::MPI::gather(this->mpi_communicator,
                           collect_filenames_on_mpi_process(),
                           /* root_process = */ 0);

  std::vector<std::string> filenames_on_cell;
  for (unsigned int i = 0; i < filename_list_list.size(); ++i)
    for (unsigned int j = 0; j < filename_list_list[i].size(); ++j)
      filenames_on_cell.push_back(filename_list_list[i][j]);
  // ---------------------------------------------------

  DataOut<dim> data_out;
  data_out.attach_dof_handler(stokes_dof_handler);

  // pvtu-record for all local fine outputs
  if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
    {
      std::string filename_master = parameters.filename_output;
      filename_master += "_fine";
      filename_master += ".pvtu";

      std::ofstream master_output(parameters.dirname_output + "/" +
                                  filename_master);
      data_out.write_pvtu_record(master_output, filenames_on_cell);
    }

  this->pcout << std::endl;
}


/////////////////////////////////////////////////////////////
// Run function
/////////////////////////////////////////////////////////////


template <int dim>
void
MultiscaleStokesModel<dim>::run()
{
  TimerOutput::Scope timer_section(
    this->computing_timer, "MultiscaleStokesModel - global run function");

  // call refinement routine in base class
  this->refine_global(parameters.initial_global_refinement);

  setup_dofs();

  initialize_and_compute_basis();

  try
    {
      Tools::create_data_directory(parameters.dirname_output);
    }
  catch (std::runtime_error &e)
    {
      // No exception handling here.
    }

  assemble_stokes_system();
  build_stokes_preconditioner();

  solve();

  send_global_weights_to_cell();

  output_results();

  this->computing_timer.print_summary();

  this->pcout << "----------------------------------------" << std::endl;
}

MSSTOKES_CLOSE_NAMESPACE
