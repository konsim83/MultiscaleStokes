#pragma once

// Deal.ii
#include <deal.II/base/function.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/full_matrix.h>

// STL
#include <cmath>
#include <fstream>

// My Headers
#include <base/config.h>

MSSTOKES_OPEN_NAMESPACE

namespace ShapeFun
{
  template <int dim>
  class BasisQ1 : public Function<dim>
  {
  public:
    BasisQ1() = delete;

    BasisQ1(const typename Triangulation<dim>::active_cell_iterator &cell);

    BasisQ1(const BasisQ1<dim> &);

    void
    set_index(unsigned int index);

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;

    virtual void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<double> &          values,
               const unsigned int             component = 0) const override;

  private:
    unsigned int index_basis;

    FullMatrix<double> coeff_matrix;
  };

  /*
   * Declare specializations in 2D
   */
  template <>
  BasisQ1<2>::BasisQ1(
    const typename Triangulation<2>::active_cell_iterator &cell);

  template <>
  double
  BasisQ1<2>::value(const Point<2> &, const unsigned int) const;

  template <>
  void
  BasisQ1<2>::value_list(const std::vector<Point<2>> &,
                         std::vector<double> &,
                         const unsigned int) const;

  /*
   * Declare specializations in 3D
   */
  template <>
  double
  BasisQ1<3>::value(const Point<3> &, const unsigned int) const;

  template <>
  BasisQ1<3>::BasisQ1(
    const typename Triangulation<3>::active_cell_iterator &cell);

  template <>
  void
  BasisQ1<3>::value_list(const std::vector<Point<3>> &,
                         std::vector<double> &,
                         const unsigned int) const;

  // exernal template instantiations
  extern template class BasisQ1<2>;
  extern template class BasisQ1<3>;

} // namespace ShapeFun

MSSTOKES_CLOSE_NAMESPACE
