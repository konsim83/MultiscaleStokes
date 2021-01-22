#pragma once

// Deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>


// STL
#include <fstream>
#include <iostream>
#include <limits>
#include <locale>
#include <memory>
#include <string>
#include <vector>


// MsStokes
#include <base/config.h>
#include <base/utilities.h>
#include <core/domain_geometry.h>
#include <core/stokes_model_assembly.h>
#include <linear_algebra/block_schur_preconditioner.hpp>
#include <linear_algebra/preconditioner_block_identity.hpp>
#include <model_data/core_model_data.h>
#include <model_data/stokes_model_data.h>
#include <model_data/stokes_model_parameters.h>
#include <multiscale_basis/divergence_stabilized_basis.h>
#include <postprocessor/postprocessor.h>


MSSTOKES_OPEN_NAMESPACE

/*!
 * @class MultiscaleStokesModel
 *
 * @brief Class to implement a nD Stokes problem on a cube with hyper shell topology.
 */
template <int dim>
class MultiscaleStokesModel : protected DomainGeometry<dim>
{
public:
  MultiscaleStokesModel(CoreModelData::Parameters &parameters);
  ~MultiscaleStokesModel();

  void
  run();

private:
  void
  setup_dofs();

  void
  initialize_and_compute_basis();

  void
  setup_stokes_matrices(
    const std::vector<IndexSet> &stokes_partitioning,
    const std::vector<IndexSet> &stokes_relevant_partitioning);

  void
  setup_stokes_preconditioner(
    const std::vector<IndexSet> &stokes_partitioning,
    const std::vector<IndexSet> &stokes_relevant_partitioning);

  void
  assemble_stokes_preconditioner();

  void
  build_stokes_preconditioner();

  void
  assemble_stokes_system();

  double
  get_maximal_velocity() const;

  void
  solve();

  /*!
   * After computing the global multiscale
   * solution we need to send the global weights to each
   * basis object. This then defines the global solution.
   */
  void
  send_global_weights_to_cell();

  /*!
   * Collect all file names of
   * basis objects on each processor.
   */
  std::vector<std::string>
  collect_filenames_on_mpi_process() const;

  void
  output_results();

  /*!
   * Parameter class.
   */
  CoreModelData::Parameters &parameters;

  const MappingQ<dim> mapping;

  const FESystem<dim>       stokes_fe;
  DoFHandler<dim>           stokes_dof_handler;
  AffineConstraints<double> stokes_constraints;

  IndexSet              stokes_index_set, stokes_relevant_set;
  std::vector<IndexSet> stokes_partitioning, stokes_relevant_partitioning;

  LA::BlockSparseMatrix stokes_matrix;
  LA::BlockSparseMatrix stokes_preconditioner_matrix;

  LA::MPI::BlockVector stokes_rhs;
  LA::MPI::BlockVector stokes_solution;

  using Block_00_PreconType = LA::PreconditionAMG;
  using Block_11_PreconType = LA::PreconditionJacobi;
  std::shared_ptr<Block_00_PreconType> Amg_preconditioner;
  std::shared_ptr<Block_11_PreconType> Mp_preconditioner;

  void
  local_assemble_stokes_preconditioner(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::StokesPreconditioner<dim> &        scratch,
    Assembly::CopyData::StokesPreconditioner<dim> &       data);

  void
  copy_local_to_global_stokes_preconditioner(
    const Assembly::CopyData::StokesPreconditioner<dim> &data);

  void
  local_assemble_stokes_system(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::StokesSystem<dim> &                scratch,
    Assembly::CopyData::StokesSystem<dim> &               data);

  void
  copy_local_to_global_stokes_system(
    const Assembly::CopyData::StokesSystem<dim> &data);

  /**
   * Convenience typedef for STL map holding basis functions for each
   * coarse cell.
   */
  using BasisMap = std::map<CellId, DivergenceStabilizedBasis<dim>>;

  /**
   * Basis vector maps cell_id to local basis.
   */
  BasisMap cell_basis_map;

  CoreModelData::TemperatureForcing<dim> temperature_forcing;
};

// Extern template instantiations
extern template class MultiscaleStokesModel<2>;
extern template class MultiscaleStokesModel<3>;

MSSTOKES_CLOSE_NAMESPACE
