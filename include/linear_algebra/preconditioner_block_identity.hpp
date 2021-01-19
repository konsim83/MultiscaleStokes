#pragma once

// Deal.ii
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/numerics/vector_tools.h>


// MsStokes
#include <base/config.h>

MSSTOKES_OPEN_NAMESPACE

namespace LinearAlgebra
{
  template <typename DoFHandlerType>
  class PreconditionerBlockIdentity
  {
  public:
    PreconditionerBlockIdentity(DoFHandlerType &dof_handler)
      : dof_handler(dof_handler)
    {}

    void
    vmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const
    {
      dst = src;
    }

  private:
    DoFHandlerType &dof_handler;
  };

} // namespace LinearAlgebra


MSSTOKES_CLOSE_NAMESPACE
