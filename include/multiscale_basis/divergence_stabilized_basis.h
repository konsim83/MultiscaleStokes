#pragma once

// Deal.ii MPI
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

// STL
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

// my headers
#include <base/config.h>
#include <base/utilities.h>
#include <functions/function_concatinator.hpp>
#include <functions/shape_fun_vector.hpp>
#include <functions/shape_fun_vector_divergence.hpp>
#include <functions/shape_function_scalar.hpp>
#include <linear_algebra/inverse_matrix.hpp>
#include <linear_algebra/local_inner_preconditioner.hpp>
#include <linear_algebra/schur_complement.hpp>
#include <model_data/stokes_model_data.h>
#include <model_data/stokes_model_parameters.h>
#include <postprocessor/postprocessor.h>

MSSTOKES_OPEN_NAMESPACE

template <int dim>
class DivergenceStabilizedBasis
{
public:
  /*!
   * Constructor deleted.
   */
  DivergenceStabilizedBasis() = delete;

  DivergenceStabilizedBasis(
    const CoreModelData::Parameters &                  parameters,
    typename Triangulation<dim>::active_cell_iterator &global_cell,
    bool                                               is_first_cell,
    unsigned int                                       local_subdomain,
    CoreModelData::TemperatureForcing<dim> &           temperature_forcing,
    MPI_Comm                                           mpi_communicator);

  DivergenceStabilizedBasis(const DivergenceStabilizedBasis<dim> &other);

  ~DivergenceStabilizedBasis();

  void
  run();

  void
  output_global_solution_in_cell();

  const FullMatrix<double> &
  get_global_element_matrix() const;

  const Vector<double> &
  get_global_element_rhs() const;

  const std::string &
  get_filename_global() const;

  void
  set_global_weights(const std::vector<double> &global_weights);

private:
  void
  setup_grid();
  void
  setup_system_matrix();

  void
  setup_basis_dofs();

  void
  project_velocity_divergence_on_pressure_space();

  void
  assemble_system();

  void
  project_standard_basis_on_velocity_space();

  void
  project_standard_basis_on_pressure_space();

  void
  assemble_global_element_matrix();

  void
  set_filename_global();

  void
  solve_direct(unsigned int n_basis);

  void
  solve_iterative(unsigned int n_basis);

  void
  write_exact_solution_in_cell();

  void
  output_basis();

  MPI_Comm mpi_communicator;

  CoreModelData::Parameters parameters;

  Triangulation<dim> triangulation;
  FESystem<dim>      fe;
  FESystem<dim>      velocity_fe;
  DoFHandler<dim>    dof_handler;

  std::vector<AffineConstraints<double>> velocity_basis_constraints;
  std::vector<AffineConstraints<double>> pressure_basis_constraints;

  BlockSparsityPattern sparsity_pattern;
  BlockSparsityPattern preconditioner_sparsity_pattern;

  // System matrix without constraints, assembled once
  BlockSparseMatrix<double> assembled_matrix;
  BlockSparseMatrix<double> assembled_preconditioner;

  // System and preconditioner matrix with constraints
  BlockSparseMatrix<double> system_matrix;
  BlockSparseMatrix<double> preconditioner_matrix;

  std::vector<BlockVector<double>> velocity_basis;
  std::vector<BlockVector<double>> pressure_basis;
  std::vector<BlockVector<double>> system_rhs;

  // This is only for global assembly
  BlockVector<double> global_rhs;

  FullMatrix<double>  global_element_matrix;
  Vector<double>      global_element_rhs;
  std::vector<double> global_weights;

  // This the global solution after the global weights have been set. It is a
  // linear combination of the modified basis.
  BlockVector<double> global_solution;

  // Shared pointer to preconditioner type for each system matrix
  std::shared_ptr<typename LinearAlgebra::LocalInnerPreconditioner<dim>::type>
    inner_schur_preconditioner;

  CellId global_cell_id;

  bool is_first_cell;

  typename Triangulation<dim>::active_cell_iterator global_cell;

  const unsigned int local_subdomain;

  std::vector<Point<dim>> corner_points;

  SmartPointer<const CoreModelData::TemperatureForcing<dim>>
    temperature_forcing_ptr;

  bool is_built_global_element_matrix;
  bool is_set_global_weights;
  bool is_set_cell_data;
};

// Extern template instantiations
extern template class DivergenceStabilizedBasis<2>;
extern template class DivergenceStabilizedBasis<3>;

MSSTOKES_CLOSE_NAMESPACE