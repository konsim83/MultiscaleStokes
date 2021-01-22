#include <core/domain_geometry.h>
#include <core/stokes_model.h>

MSSTOKES_OPEN_NAMESPACE

template <int dim>
StokesModel<dim>::StokesModel(CoreModelData::Parameters &parameters_)
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
{}



template <int dim>
StokesModel<dim>::~StokesModel()
{}



/////////////////////////////////////////////////////////////
// System and dof setup
/////////////////////////////////////////////////////////////

template <int dim>
void
StokesModel<dim>::setup_stokes_matrices(
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
StokesModel<dim>::setup_stokes_preconditioner(
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
StokesModel<dim>::setup_dofs()
{
  TimerOutput::Scope timing_section(this->computing_timer,
                                    "StokesModel - setup dofs of systems");

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
  this->pcout << "Number of active cells: "
              << this->triangulation.n_global_active_cells() << " (on "
              << this->triangulation.n_levels() << " levels)" << std::endl
              << "Number of degrees of freedom: " << n_u + n_p << " (" << n_u
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
// Assembly Stokes preconditioner
/////////////////////////////////////////////////////////////


template <int dim>
void
StokesModel<dim>::local_assemble_stokes_preconditioner(
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
StokesModel<dim>::copy_local_to_global_stokes_preconditioner(
  const Assembly::CopyData::StokesPreconditioner<dim> &data)
{
  stokes_constraints.distribute_local_to_global(data.local_matrix,
                                                data.local_dof_indices,
                                                stokes_preconditioner_matrix);
}


template <int dim>
void
StokesModel<dim>::assemble_stokes_preconditioner()
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
    std::bind(&StokesModel<dim>::local_assemble_stokes_preconditioner,
              this,
              std::placeholders::_1,
              std::placeholders::_2,
              std::placeholders::_3),
    std::bind(&StokesModel<dim>::copy_local_to_global_stokes_preconditioner,
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
StokesModel<dim>::build_stokes_preconditioner()
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
StokesModel<dim>::local_assemble_stokes_system(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  Assembly::Scratch::StokesSystem<dim> &                scratch,
  Assembly::CopyData::StokesSystem<dim> &               data)
{
  const unsigned int dofs_per_cell =
    scratch.stokes_fe_values.get_fe().dofs_per_cell;
  const unsigned int n_q_points = scratch.stokes_fe_values.n_quadrature_points;

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  scratch.stokes_fe_values.reinit(cell);

  data.local_matrix = 0;
  data.local_rhs    = 0;

  /*
   * Function values of temperature forcing
   */
  CoreModelData::TemperatureForcing<dim> temperature_forcing(
    this->domain_center,
    parameters.physical_constants.reference_temperature,
    parameters.physical_constants.expansion_coefficient,
    /* variance */ parameters.variance);
  std::vector<double> temperature_forcing_values(n_q_points);
  temperature_forcing.value_list(
    scratch.stokes_fe_values.get_quadrature_points(),
    temperature_forcing_values);

  for (unsigned int q = 0; q < n_q_points; ++q)
    {
      for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          scratch.phi_u[k] = scratch.stokes_fe_values[velocities].value(k, q);
          scratch.grads_phi_u[k] =
            scratch.stokes_fe_values[velocities].symmetric_gradient(k, q);
          scratch.div_phi_u[k] =
            scratch.stokes_fe_values[velocities].divergence(k, q);
          scratch.phi_p[k] = scratch.stokes_fe_values[pressure].value(k, q);
        }

      /*
       * Move everything to the LHS here.
       */
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              data.local_matrix(i, j) +=
                ((parameters.physical_constants.kinematic_viscosity * 2 *
                  scratch.grads_phi_u[i] *
                  scratch.grads_phi_u[j]) // eps(v):sigma(eps(u))
                 -
                 (scratch.div_phi_u[i] *
                  scratch.phi_p[j]) // div(v)*p (solve for scaled pressure dt*p)
                 - (scratch.phi_p[i] * scratch.div_phi_u[j]) // q*div(u)
                 ) *
                scratch.stokes_fe_values.JxW(q);
            }
        }

      const Tensor<1, dim> gravity = CoreModelData::vertical_gravity_vector(
        scratch.stokes_fe_values.quadrature_point(q),
        parameters.physical_constants.gravity_constant);

      /*
       * This is only the RHS
       */
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          data.local_rhs(i) += temperature_forcing_values[q] *
                               (scratch.phi_u[i] * gravity) *
                               scratch.stokes_fe_values.JxW(q);
        }
    }

  cell->get_dof_indices(data.local_dof_indices);
}



template <int dim>
void
StokesModel<dim>::copy_local_to_global_stokes_system(
  const Assembly::CopyData::StokesSystem<dim> &data)
{
  stokes_constraints.distribute_local_to_global(data.local_matrix,
                                                data.local_rhs,
                                                data.local_dof_indices,
                                                stokes_matrix,
                                                stokes_rhs);
}



template <int dim>
void
StokesModel<dim>::assemble_stokes_system()
{
  TimerOutput::Scope timer_section(this->computing_timer,
                                   "   Assemble Stokes system");

  this->pcout << "   Assembling Stokes system..." << std::flush;

  stokes_matrix = 0;
  stokes_rhs    = 0;

  const QGauss<dim> quadrature_formula(parameters.stokes_velocity_degree + 1);
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  WorkStream::run(
    CellFilter(IteratorFilters::LocallyOwnedCell(),
               stokes_dof_handler.begin_active()),
    CellFilter(IteratorFilters::LocallyOwnedCell(), stokes_dof_handler.end()),
    std::bind(&StokesModel<dim>::local_assemble_stokes_system,
              this,
              std::placeholders::_1,
              std::placeholders::_2,
              std::placeholders::_3),
    std::bind(&StokesModel<dim>::copy_local_to_global_stokes_system,
              this,
              std::placeholders::_1),
    Assembly::Scratch::StokesSystem<dim>(
      stokes_fe,
      mapping,
      quadrature_formula,
      (update_values | update_quadrature_points | update_JxW_values |
       update_gradients)),
    Assembly::CopyData::StokesSystem<dim>(stokes_fe));

  stokes_matrix.compress(VectorOperation::add);
  stokes_rhs.compress(VectorOperation::add);

  this->pcout << std::endl;
}



template <int dim>
double
StokesModel<dim>::get_maximal_velocity() const
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
StokesModel<dim>::solve()
{
  TimerOutput::Scope timer_section(this->computing_timer,
                                   "   Solve Stokes system");
  this->pcout
    << "   Solving Stokes system for one time step with (block preconditioned solver)... "
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
// Output results
/////////////////////////////////////////////////////////////


template <int dim>
void
StokesModel<dim>::output_results()
{
  TimerOutput::Scope timer_section(this->computing_timer,
                                   "Postprocessing and output");

  this->pcout << "   Writing Stokes solution... " << std::flush;

  // First the forcing projection with a different DoFHandler
  DoFHandler<dim> forcing_dof_handler(this->triangulation);
  FE_Q<dim>       forcing_fe(parameters.stokes_velocity_degree);
  forcing_dof_handler.distribute_dofs(forcing_fe);

  AffineConstraints<double> no_constraints;
  no_constraints.clear();
  DoFTools::make_hanging_node_constraints(forcing_dof_handler, no_constraints);
  no_constraints.close();

  IndexSet locally_owned_forcing_dofs =
    forcing_dof_handler.locally_owned_dofs();
  IndexSet locally_relevant_forcing_dofs;
  DoFTools::extract_locally_relevant_dofs(forcing_dof_handler,
                                          locally_relevant_forcing_dofs);
  LA::MPI::Vector distributed_bouyancy_forcing(locally_owned_forcing_dofs);

  VectorTools::project(forcing_dof_handler,
                       no_constraints,
                       QGauss<dim>(parameters.stokes_velocity_degree + 1),
                       CoreModelData::TemperatureForcing<dim>(
                         this->domain_center,
                         parameters.physical_constants.reference_temperature,
                         parameters.physical_constants.expansion_coefficient,
                         /* variance */ parameters.variance),
                       distributed_bouyancy_forcing);

  LA::MPI::Vector bouyancy_forcing(locally_relevant_forcing_dofs,
                                   this->mpi_communicator);
  bouyancy_forcing = distributed_bouyancy_forcing;

  // Now join the Stokes and the forcing dofs
  const FESystem<dim> joint_fe(stokes_fe, 1, forcing_fe, 1);

  DoFHandler<dim> joint_dof_handler(this->triangulation);
  joint_dof_handler.distribute_dofs(joint_fe);

  Assert(joint_dof_handler.n_dofs() ==
           stokes_dof_handler.n_dofs() + forcing_dof_handler.n_dofs(),
         ExcInternalError());

  LA::MPI::Vector joint_solution;

  joint_solution.reinit(joint_dof_handler.locally_owned_dofs(),
                        this->mpi_communicator);

  {
    std::vector<types::global_dof_index> local_joint_dof_indices(
      joint_fe.dofs_per_cell);
    std::vector<types::global_dof_index> local_stokes_dof_indices(
      stokes_fe.dofs_per_cell);
    std::vector<types::global_dof_index> local_forcing_dof_indices(
      forcing_fe.dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
      joint_cell   = joint_dof_handler.begin_active(),
      joint_endc   = joint_dof_handler.end(),
      stokes_cell  = stokes_dof_handler.begin_active(),
      forcing_cell = forcing_dof_handler.begin_active();
    for (; joint_cell != joint_endc;
         ++joint_cell, ++stokes_cell, ++forcing_cell)
      {
        if (joint_cell->is_locally_owned())
          {
            joint_cell->get_dof_indices(local_joint_dof_indices);
            stokes_cell->get_dof_indices(local_stokes_dof_indices);
            forcing_cell->get_dof_indices(local_forcing_dof_indices);

            for (unsigned int i = 0; i < joint_fe.dofs_per_cell; ++i)
              if (joint_fe.system_to_base_index(i).first.first == 0)
                {
                  Assert(joint_fe.system_to_base_index(i).second <
                           local_stokes_dof_indices.size(),
                         ExcInternalError());

                  joint_solution(local_joint_dof_indices[i]) = stokes_solution(
                    local_stokes_dof_indices[joint_fe.system_to_base_index(i)
                                               .second]);
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

  joint_solution.compress(VectorOperation::insert);

  IndexSet locally_relevant_joint_dofs(joint_dof_handler.n_dofs());
  DoFTools::extract_locally_relevant_dofs(joint_dof_handler,
                                          locally_relevant_joint_dofs);

  LA::MPI::Vector locally_relevant_joint_solution;
  locally_relevant_joint_solution.reinit(locally_relevant_joint_dofs,
                                         this->mpi_communicator);
  locally_relevant_joint_solution = joint_solution;


  Postprocessor<dim> postprocessor(
    Utilities::MPI::this_mpi_process(this->mpi_communicator));

  DataOut<dim> data_out;
  data_out.attach_dof_handler(joint_dof_handler);

  data_out.add_data_vector(locally_relevant_joint_solution, postprocessor);


  // data_out.add_data_vector(stokes_dof_handler, stokes_solution,
  // postprocessor);


  // std::vector<std::string> forcing_names(1, "vertical buoyancy force");

  // std::vector<DataComponentInterpretation::DataComponentInterpretation>
  //   interpretation(1, DataComponentInterpretation::component_is_scalar);


  data_out.build_patches(parameters.stokes_velocity_degree);

  const std::string filename =
    (parameters.filename_output + "." +
     Utilities::int_to_string(this->triangulation.locally_owned_subdomain(),
                              4) +
     ".vtu");
  std::ofstream output(parameters.dirname_output + "/" + filename);
  data_out.write_vtu(output);

  /*
   * Write pvtu record
   */
  if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
           ++i)
        filenames.push_back(std::string(parameters.filename_output) + "." +
                            Utilities::int_to_string(i, 4) + ".vtu");

      const std::string pvtu_master_filename =
        (parameters.filename_output + ".pvtu");
      std::ofstream pvtu_master(parameters.dirname_output + "/" +
                                pvtu_master_filename);
      data_out.write_pvtu_record(pvtu_master, filenames);
    }

  forcing_dof_handler.clear();

  this->pcout << std::endl;
}


/////////////////////////////////////////////////////////////
// Run function
/////////////////////////////////////////////////////////////


template <int dim>
void
StokesModel<dim>::run()
{
  TimerOutput::Scope timer_section(this->computing_timer,
                                   "StokesModel - global run function");

  // call refinement routine in base class
  this->refine_global(parameters.initial_global_refinement);

  setup_dofs();

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

  output_results();

  this->computing_timer.print_summary();

  this->pcout << "----------------------------------------" << std::endl;
}

MSSTOKES_CLOSE_NAMESPACE
