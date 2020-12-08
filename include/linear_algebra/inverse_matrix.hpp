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
   * @class InverseMatrix
   *
   * @brief Implements an iterative inverse
   *
   * Implement the inverse matrix of a given matrix through
   * its action by a preconditioned CG solver. This class also
   * works with MPI.
   *
   * @note The inverse is not constructed explicitly.
   */
  template <typename MatrixType, typename PreconditionerType>
  class InverseMatrix : public Subscriptor
  {
  public:
    /*!
     * Constructor.
     *
     * @param m
     * @param preconditioner
     */
    InverseMatrix(const MatrixType &        m,
                  const PreconditionerType &preconditioner,
                  bool                      use_simple_cg = true);

    /*!
     * Matrix-vector multiplication.
     *
     * @param[out] dst
     * @param[in] src
     */
    template <typename VectorType>
    void
    vmult(VectorType &dst, const VectorType &src) const;

  private:
    /*!
     * Samrt pointer to system matrix.
     */
    const SmartPointer<const MatrixType> matrix;

    /*!
     * Preconditioner.
     */
    const PreconditionerType &preconditioner;

    const bool use_simple_cg;
  };


  ///////////////////////////////////////
  /// Implementation
  ///////////////////////////////////////


  template <typename MatrixType, typename PreconditionerType>
  InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
    const MatrixType &        m,
    const PreconditionerType &preconditioner,
    bool                      use_simple_cg)
    : matrix(&m)
    , preconditioner(preconditioner)
    , use_simple_cg(use_simple_cg)
  {}

  template <typename MatrixType, typename PreconditionerType>
  template <typename VectorType>
  void
  InverseMatrix<MatrixType, PreconditionerType>::vmult(
    VectorType &      dst,
    const VectorType &src) const
  {
    SolverControl solver_control(std::max(static_cast<std::size_t>(src.size()),
                                          static_cast<std::size_t>(1000)),
                                 1e-6 * src.l2_norm());

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

} // end namespace LinearAlgebra

MSSTOKES_CLOSE_NAMESPACE
