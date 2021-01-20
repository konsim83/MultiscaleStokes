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
  template <class PreconditionerType>
  class SchurComplement : public Subscriptor
  {
  public:
    SchurComplement(
      const BlockSparseMatrix<double> &system_matrix,
      const InverseMatrix<SparseMatrix<double>, PreconditionerType> &A_inverse);
    void
    vmult(Vector<double> &dst, const Vector<double> &src) const;

  private:
    const SmartPointer<const BlockSparseMatrix<double>> system_matrix;
    const SmartPointer<
      const InverseMatrix<SparseMatrix<double>, PreconditionerType>>
                           A_inverse;
    mutable Vector<double> tmp1, tmp2;
  };
  template <class PreconditionerType>
  SchurComplement<PreconditionerType>::SchurComplement(
    const BlockSparseMatrix<double> &system_matrix,
    const InverseMatrix<SparseMatrix<double>, PreconditionerType> &A_inverse)
    : system_matrix(&system_matrix)
    , A_inverse(&A_inverse)
    , tmp1(system_matrix.block(0, 0).m())
    , tmp2(system_matrix.block(0, 0).m())
  {}
  template <class PreconditionerType>
  void
  SchurComplement<PreconditionerType>::vmult(Vector<double> &      dst,
                                             const Vector<double> &src) const
  {
    system_matrix->block(0, 1).vmult(tmp1, src);
    A_inverse->vmult(tmp2, tmp1);
    system_matrix->block(1, 0).vmult(dst, tmp2);
  }

} // end namespace LinearAlgebra

MSSTOKES_CLOSE_NAMESPACE
