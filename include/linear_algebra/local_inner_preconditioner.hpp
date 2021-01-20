#pragma once

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>


MSSTOKES_OPEN_NAMESPACE

namespace LinearAlgebra
{
  using namespace dealii;

  template <int dim>
  class LocalInnerPreconditioner;

  /*!
   * @class LocalInnerPreconditioner<2>
   *
   * @brief Encapsulation of preconditioner type for local problems in 2D
   */
  template <>
  class LocalInnerPreconditioner<2>
  {
  public:
    using type = SparseDirectUMFPACK;
  };

  /*!
   * @class LocalInnerPreconditioner<3>
   *
   * @brief Encapsulation of preconditioner type for local problems in 3D
   */
  template <>
  class LocalInnerPreconditioner<3>
  {
  public:
    using type = SparseILU<double>;
  };

} // end namespace LinearAlgebra

MSSTOKES_CLOSE_NAMESPACE
