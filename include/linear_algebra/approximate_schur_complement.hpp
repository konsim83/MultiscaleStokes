#pragma once

// Deal.ii
#include <deal.II/base/subscriptor.h>

// STL
#include <memory>
#include <vector>

// AquaPlanet
#include <base/config.h>

MSSTOKES_OPEN_NAMESPACE

namespace LinearAlgebra
{
  /*!
   * @class ApproximateSchurComplement
   *
   * @brief Implements a MPI parallel approximate Schur complement
   *
   * Implements a parallel approximate Schur complement through the use of a
   * preconditioner for the inner inverse matrix, i.e., if we want to solve
   * \f{eqnarray}{
   *	\left(
   *	\begin{array}{cc}
   *		A & B^T \\
   *		B & 0
   *	\end{array}
   *	\right)
   *	\left(
   *	\begin{array}{c}
   *		\sigma \\
   *		u
   *	\end{array}
   *	\right)
   *	=
   *	\left(
   *	\begin{array}{c}
   *		f \\
   *		0
   *	\end{array}
   *	\right)
   * \f}
   * and know that \f$A\f$ is invertible then we choose a preconditioner
   *\f$P_A\f$ and define the approximate Schur complement as \f{eqnarray}{
   *\tilde S = BP_A^{-1}B^T \f} to solve for \f$u\f$.
   */
  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  class ApproximateSchurComplement : public Subscriptor
  {
  private:
    /*!
     * Typedef for convenience.
     */
    using BlockType = typename BlockMatrixType::BlockType;

  public:
    /*!
     * Constructor.
     *
     * @param system_matrix
     * @param owned_partitioning
     * @param mpi_communicator
     */
    ApproximateSchurComplement(const BlockMatrixType &      system_matrix,
                               const std::vector<IndexSet> &owned_partitioning,
                               MPI_Comm                     mpi_communicator);

    /*!
     * Matrix-vector product.
     *
     * @param dst
     * @param src
     */
    void
    vmult(VectorType &dst, const VectorType &src) const;

  private:
    /*!
     * Samrt pointer to system matrix.
     */
    const SmartPointer<const BlockMatrixType> system_matrix;

    /*!
     * Preconditioner.
     */
    PreconditionerType preconditioner;

    /*!
     * Index set to initialize tmp vectors using only locally owned partition.
     */
    const std::vector<IndexSet> &owned_partitioning;

    /*!
     * Relevant MPI communicator.
     */
    MPI_Comm mpi_communicator;

    /*!
     * Muatable distributed types for temporary vectors.
     */
    mutable VectorType tmp1, tmp2;
  };


  ///////////////////////////////////////
  /// Implementation
  ///////////////////////////////////////


  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  ApproximateSchurComplement<BlockMatrixType, VectorType, PreconditionerType>::
    ApproximateSchurComplement(const BlockMatrixType &      system_matrix,
                               const std::vector<IndexSet> &owned_partitioning,
                               MPI_Comm                     mpi_communicator)
    : system_matrix(&system_matrix)
    , preconditioner()
    , owned_partitioning(owned_partitioning)
    , mpi_communicator(mpi_communicator)
    , tmp1(owned_partitioning[0], mpi_communicator)
    , tmp2(owned_partitioning[0], mpi_communicator)
  {
    typename PreconditionerType::AdditionalData data;
    preconditioner.initialize(system_matrix.block(0, 0), data);
  }

  template <typename BlockMatrixType,
            typename VectorType,
            typename PreconditionerType>
  void
  ApproximateSchurComplement<BlockMatrixType, VectorType, PreconditionerType>::
    vmult(VectorType &dst, const VectorType &src) const
  {
    system_matrix->block(0, 1).vmult(tmp1, src);
    preconditioner.vmult(tmp2, tmp1);
    system_matrix->block(1, 0).vmult(dst, tmp2);
  }

} // namespace LinearAlgebra

MSSTOKES_CLOSE_NAMESPACE
