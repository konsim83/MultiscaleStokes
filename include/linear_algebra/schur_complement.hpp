#pragma once

// Deal.ii
#include <deal.II/base/subscriptor.h>

// STL
#include <memory>
#include <vector>

// My headers
#include <base/config.h>

MSSTOKES_OPEN_NAMESPACE

namespace LinearAlgebra
{
  /*!
   * @class SchurComplement
   *
   * @brief Implements a MPI parallel Schur complement
   *
   * Implements a parallel Schur complement through the use of an inner inverse
   * matrix, i.e., if we want to solve
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
   * and know that \f$A\f$ is invertible then we first define the inverse and
   *define the Schur complement as \f{eqnarray}{ \tilde S = BP_A^{-1}B^T \f}
   *to solve for \f$u\f$. The inverse must be separately given to the class as
   *an input argument.
   */
  template <typename BlockMatrixType,
            typename VectorType,
            typename InverseMatrixType>
  class SchurComplement : public Subscriptor
  {
  private:
    using BlockType = typename BlockMatrixType::BlockType;

  public:
    /*!
     * Constructor. The user must take care to pass the correct inverse of the
     * upper left block of the system matrix.
     *
     * @param system_matrix
     * 	Block Matrix
     * @param relevant_inverse_matrix
     * 	Inverse of upper left block of the system matrix.
     * @param owned_partitioning
     * @param mpi_communicator
     */
    SchurComplement(const BlockMatrixType &      system_matrix,
                    const InverseMatrixType &    relevant_inverse_matrix,
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
     * Smart pointer to system matrix block 01.
     */
    const SmartPointer<const BlockType> block_01;

    /*!
     * Smart pointer to system matrix block 10.
     */
    const SmartPointer<const BlockType> block_10;

    /*!
     * Smart pointer to inverse upper left block of the system matrix.
     */
    const SmartPointer<const InverseMatrixType> relevant_inverse_matrix;

    /*!
     * Index set to initialize tmp vectors using only locally owned partition.
     */
    const std::vector<IndexSet> &owned_partitioning;

    /*!
     * Current MPI communicator.
     */
    MPI_Comm mpi_communicator;

    /*!
     * Muatable types for temporary vectors.
     */
    mutable VectorType tmp1, tmp2;
  };


  ///////////////////////////////////////
  /// Implementation
  ///////////////////////////////////////


  template <typename BlockMatrixType,
            typename VectorType,
            typename InverseMatrixType>
  SchurComplement<BlockMatrixType, VectorType, InverseMatrixType>::
    SchurComplement(const BlockMatrixType &      system_matrix,
                    const InverseMatrixType &    relevant_inverse_matrix,
                    const std::vector<IndexSet> &owned_partitioning,
                    MPI_Comm                     mpi_communicator)
    : block_01(&(system_matrix.block(0, 1)))
    , block_10(&(system_matrix.block(1, 0)))
    , relevant_inverse_matrix(&relevant_inverse_matrix)
    , owned_partitioning(owned_partitioning)
    , mpi_communicator(mpi_communicator)
    , tmp1(owned_partitioning[0], mpi_communicator)
    , tmp2(owned_partitioning[0], mpi_communicator)
  {}

  template <typename BlockMatrixType,
            typename VectorType,
            typename InverseMatrixType>
  void
  SchurComplement<BlockMatrixType, VectorType, InverseMatrixType>::vmult(
    VectorType &      dst,
    const VectorType &src) const
  {
    block_01->vmult(tmp1, src);
    relevant_inverse_matrix->vmult(tmp2, tmp1);
    block_10->vmult(dst, tmp2);
  }
} // end namespace LinearAlgebra

MSSTOKES_CLOSE_NAMESPACE
