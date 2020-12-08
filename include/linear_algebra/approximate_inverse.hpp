#pragma once

// Deal.ii
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

// STL
#include <memory>

// AquaPlanet
#include <base/config.h>

MSSTOKES_OPEN_NAMESPACE

namespace LinearAlgebra
{
  /*!
   * @class ApproximateInverseMatrix
   *
   * @brief Approximate inverse matrix
   *
   * Approximate inverse matrix through use of preconditioner and a limited
   * number of CG iterations.
   */
  template <typename MatrixType, typename PreconditionerType>
  class ApproximateInverseMatrix : public Subscriptor
  {
  public:
    /*!
     * Constructor.
     *
     * @param m
     * @param preconditioner
     * @param n_iter
     */
    ApproximateInverseMatrix(const MatrixType &        m,
                             const PreconditionerType &preconditioner,
                             const unsigned int        n_iter,
                             bool                      use_simple_cg = true);

    /*!
     * Matrix vector multiplication. VectorType template can be serial or
     * distributed.
     *
     * @param dst
     * @param src
     */
    template <typename VectorType>
    void
    vmult(VectorType &dst, const VectorType &src) const;

  private:
    /*!
     * Smart pointer to matrix.
     */
    const SmartPointer<const MatrixType> matrix;

    /*!
     * Pointer to type of preconsitioner.
     */
    const PreconditionerType &preconditioner;

    /*!
     * Maximum number of CG iterations.
     */
    const unsigned int max_iter;

    const bool use_simple_cg;
  };


  ///////////////////////////////////////
  /// Implementation
  ///////////////////////////////////////

  template <typename MatrixType, typename PreconditionerType>
  ApproximateInverseMatrix<MatrixType, PreconditionerType>::
    ApproximateInverseMatrix(const MatrixType &        m,
                             const PreconditionerType &preconditioner,
                             const unsigned int        n_iter,
                             bool                      use_simple_cg)
    : matrix(&m)
    , preconditioner(preconditioner)
    , max_iter(n_iter)
    , use_simple_cg(use_simple_cg)
  {}

  template <typename MatrixType, typename PreconditionerType>
  template <typename VectorType>
  void
  ApproximateInverseMatrix<MatrixType, PreconditionerType>::vmult(
    VectorType &      dst,
    const VectorType &src) const
  {
    SolverControl solver_control(/* max_iter */ max_iter, 1e-6 * src.l2_norm());

    dst = 0;

    try
      {
        if (use_simple_cg)
          {
            SolverCG<VectorType> local_solver(solver_control);
            local_solver.solve(*matrix, dst, src, preconditioner);
          }
        else
          {
            SolverGMRES<VectorType> local_solver(solver_control);
            local_solver.solve(*matrix, dst, src, preconditioner);
          }
      }
    catch (std::exception &e)
      {
        Assert(false, ExcMessage(e.what()));
      }
  }

} // namespace LinearAlgebra

MSSTOKES_CLOSE_NAMESPACE
