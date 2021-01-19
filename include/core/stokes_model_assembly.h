#pragma once

// Deal.ii
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
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

// MsStokes
#include <base/config.h>

MSSTOKES_OPEN_NAMESPACE

namespace Assembly
{
  namespace Scratch
  {
    ////////////////////////////////////
    /// Stokes preconditioner
    ////////////////////////////////////

    template <int dim>
    struct StokesPreconditioner
    {
      StokesPreconditioner(const FiniteElement<dim> &stokes_fe,
                           const Quadrature<dim> &   stokes_quadrature,
                           const Mapping<dim> &      mapping,
                           const UpdateFlags         update_flags);

      StokesPreconditioner(const StokesPreconditioner<dim> &data);

      FEValues<dim> stokes_fe_values;

      std::vector<Tensor<2, dim>> grad_phi_u;
      std::vector<double>         phi_p;
    };


    ////////////////////////////////////
    /// Stokes system
    ////////////////////////////////////

    template <int dim>
    struct StokesSystem : public StokesPreconditioner<dim>
    {
      StokesSystem(const FiniteElement<dim> &stokes_fe,
                   const Mapping<dim> &      mapping,
                   const Quadrature<dim> &   stokes_quadrature,
                   const UpdateFlags         stokes_update_flags);

      StokesSystem(const StokesSystem<dim> &data);

      FEValues<dim> stokes_fe_values;

      std::vector<Tensor<1, dim>>          phi_u;
      std::vector<SymmetricTensor<2, dim>> grads_phi_u;
      std::vector<double>                  div_phi_u;
    };
  } // namespace Scratch



  namespace CopyData
  {
    ////////////////////////////////////
    /// Stokes preconditioner copy
    ////////////////////////////////////

    template <int dim>
    struct StokesPreconditioner
    {
      StokesPreconditioner(const FiniteElement<dim> &stokes_fe);
      StokesPreconditioner(const StokesPreconditioner<dim> &data);

      StokesPreconditioner<dim> &
      operator=(const StokesPreconditioner<dim> &) = default;

      FullMatrix<double>                   local_matrix;
      std::vector<types::global_dof_index> local_dof_indices;
    };


    ////////////////////////////////////
    /// Stokes system copy
    ////////////////////////////////////

    template <int dim>
    struct StokesSystem : public StokesPreconditioner<dim>
    {
      StokesSystem(const FiniteElement<dim> &stokes_fe);

      StokesSystem(const StokesSystem<dim> &data);

      Vector<double> local_rhs;
    };
  } // namespace CopyData
} // namespace Assembly

// Extern template instantiations
extern template class Assembly::Scratch::StokesPreconditioner<2>;
extern template class Assembly::Scratch::StokesSystem<2>;

extern template class Assembly::CopyData::StokesPreconditioner<2>;
extern template class Assembly::CopyData::StokesSystem<2>;

extern template class Assembly::Scratch::StokesPreconditioner<3>;
extern template class Assembly::Scratch::StokesSystem<3>;

extern template class Assembly::CopyData::StokesPreconditioner<3>;
extern template class Assembly::CopyData::StokesSystem<3>;

MSSTOKES_CLOSE_NAMESPACE
