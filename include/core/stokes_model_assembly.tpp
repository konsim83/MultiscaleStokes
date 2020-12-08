#include <core/stokes_model_assembly.h>

MSSTOKES_OPEN_NAMESPACE

namespace Assembly
{
  namespace Scratch
  {
    ////////////////////////////////////
    /// Stokes preconditioner
    ////////////////////////////////////

    template <int dim>
    StokesPreconditioner<dim>::StokesPreconditioner(
      const FiniteElement<dim> &stokes_fe,
      const Quadrature<dim> &   stokes_quadrature,
      const Mapping<dim> &      mapping,
      const UpdateFlags         update_flags)
      : stokes_fe_values(mapping, stokes_fe, stokes_quadrature, update_flags)
      , grad_phi_u(stokes_fe.dofs_per_cell)
      , phi_p(stokes_fe.dofs_per_cell)
    {}


    template <int dim>
    StokesPreconditioner<dim>::StokesPreconditioner(
      const StokesPreconditioner<dim> &scratch)
      : stokes_fe_values(scratch.stokes_fe_values.get_mapping(),
                         scratch.stokes_fe_values.get_fe(),
                         scratch.stokes_fe_values.get_quadrature(),
                         scratch.stokes_fe_values.get_update_flags())
      , grad_phi_u(scratch.grad_phi_u)
      , phi_p(scratch.phi_p)
    {}



    ////////////////////////////////////
    /// Stokes system
    ////////////////////////////////////

    template <int dim>
    StokesSystem<dim>::StokesSystem(const FiniteElement<dim> &stokes_fe,
                                    const Mapping<dim> &      mapping,
                                    const Quadrature<dim> &   stokes_quadrature,
                                    const UpdateFlags stokes_update_flags)
      : StokesPreconditioner<dim>(stokes_fe,
                                  stokes_quadrature,
                                  mapping,
                                  stokes_update_flags)
      , stokes_fe_values(mapping,
                         stokes_fe,
                         stokes_quadrature,
                         stokes_update_flags)
      , grads_phi_u(stokes_fe.dofs_per_cell)
      , div_phi_u(stokes_fe.dofs_per_cell)
    {}


    template <int dim>
    StokesSystem<dim>::StokesSystem(const StokesSystem<dim> &scratch)
      : StokesPreconditioner<dim>(scratch)
      , stokes_fe_values(scratch.stokes_fe_values.get_mapping(),
                         scratch.stokes_fe_values.get_fe(),
                         scratch.stokes_fe_values.get_quadrature(),
                         scratch.stokes_fe_values.get_update_flags())
      , grads_phi_u(scratch.grads_phi_u)
      , div_phi_u(scratch.div_phi_u)
    {}
  } // namespace Scratch



  namespace CopyData
  {
    ////////////////////////////////////
    /// Stokes preconditioner copy
    ////////////////////////////////////

    template <int dim>
    StokesPreconditioner<dim>::StokesPreconditioner(
      const FiniteElement<dim> &stokes_fe)
      : local_matrix(stokes_fe.dofs_per_cell, stokes_fe.dofs_per_cell)
      , local_dof_indices(stokes_fe.dofs_per_cell)
    {}


    template <int dim>
    StokesPreconditioner<dim>::StokesPreconditioner(
      const StokesPreconditioner<dim> &data)
      : local_matrix(data.local_matrix)
      , local_dof_indices(data.local_dof_indices)
    {}


    ////////////////////////////////////
    /// Stokes system copy
    ////////////////////////////////////

    template <int dim>
    StokesSystem<dim>::StokesSystem(const FiniteElement<dim> &stokes_fe)
      : StokesPreconditioner<dim>(stokes_fe)
      , local_rhs(stokes_fe.dofs_per_cell)
    {}


    template <int dim>
    StokesSystem<dim>::StokesSystem(const StokesSystem<dim> &data)
      : StokesPreconditioner<dim>(data)
      , local_rhs(data.local_rhs)
    {}
  } // namespace CopyData
} // namespace Assembly

MSSTOKES_CLOSE_NAMESPACE
