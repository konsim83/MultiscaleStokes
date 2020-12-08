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
#include <deal.II/fe/fe_dgq.h>
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


// AquaPlanet
#include <base/config.h>
#include <base/utilities.h>
#include <core/domain_geometry.h>
#include <core/stokes_model_assembly.h>
#include <linear_algebra/approximate_inverse.hpp>
#include <linear_algebra/approximate_schur_complement.hpp>
#include <linear_algebra/block_schur_preconditioner.hpp>
#include <linear_algebra/inverse_matrix.hpp>
#include <linear_algebra/schur_complement.hpp>
#include <model_data/core_model_data.h>
#include <model_data/stokes_model_data.h>
#include <model_data/stokes_model_parameters.h>


MSSTOKES_OPEN_NAMESPACE

/*!
 * @class StokesModel
 *
 * @brief Class to implement a nD Stokes problem on a cube with hyper shell topology.
 */
template <int dim>
class StokesModel : protected DomainGeometry<dim>
{
public:
  StokesModel(CoreModelData::Parameters &parameters);
  ~StokesModel();

  void
  run();

private:
  class Postprocessor : public DataPostprocessor<dim>
  {
  public:
    Postprocessor(const unsigned int partition);

    virtual void
    evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &inputs,
      std::vector<Vector<double>> &computed_quantities) const override;

    virtual std::vector<std::string>
    get_names() const override;

    virtual std::vector<
      DataComponentInterpretation::DataComponentInterpretation>
    get_data_component_interpretation() const override;

    virtual UpdateFlags
    get_needed_update_flags() const override;

  private:
    const unsigned int partition;
  };

  void
  setup_dofs();

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
  solve_block_preconditioned();

  void
  solve_Schur_complement();

  void
  output_results();

  void
  print_paramter_info() const;

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

  class Postprocessor;

  using InnerPreconditionerType = typename LA::PreconditionILU;
  std::shared_ptr<InnerPreconditionerType>         inner_schur_preconditioner;
  typename InnerPreconditionerType::AdditionalData data;
  bool is_initialized_inner_schur_preconditioner = false;
};

// Extern template instantiations
extern template class StokesModel<2>;
extern template class StokesModel<3>;

MSSTOKES_CLOSE_NAMESPACE
