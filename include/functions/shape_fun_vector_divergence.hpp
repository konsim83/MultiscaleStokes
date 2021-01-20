#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <vector>

MSSTOKES_OPEN_NAMESPACE

namespace ShapeFun
{
  using namespace dealii;

  /*!
   * @class ShapeFunctionVectorDivergence
   *
   * @brief Class for evaluations of divergence of vector valued shape functions.
   *
   * @note This is slow. Only use for quick and dirty prototyping and sanity checks.
   */
  template <int dim>
  class ShapeFunctionVectorDivergence : public Function<dim>
  {
  public:
    ShapeFunctionVectorDivergence(
      const FiniteElement<dim> &                         fe,
      typename Triangulation<dim>::active_cell_iterator &cell);

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;

    virtual void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<double> &          values,
               const unsigned int             component = 0) const override;

    void
    set_current_cell(
      const typename Triangulation<dim>::active_cell_iterator &cell);

    void
    set_shape_fun_index(unsigned int index);

  private:
    SmartPointer<const FiniteElement<dim>> fe_ptr;
    const unsigned int                     dofs_per_cell;
    unsigned int                           shape_fun_index;

    const MappingQ<dim> mapping;

    typename Triangulation<dim>::active_cell_iterator *current_cell_ptr;

    const FEValuesExtractors::Vector flux;
  };


  template <int dim>
  ShapeFunctionVectorDivergence<dim>::ShapeFunctionVectorDivergence(
    const FiniteElement<dim> &                         fe,
    typename Triangulation<dim>::active_cell_iterator &cell)
    : Function<dim>(1)
    , fe_ptr(&fe)
    , dofs_per_cell(fe_ptr->dofs_per_cell)
    , shape_fun_index(0)
    , mapping(1)
    , current_cell_ptr(&cell)
    , flux(0)
  {}

  template <int dim>
  void
  ShapeFunctionVectorDivergence<dim>::set_current_cell(
    const typename Triangulation<dim>::active_cell_iterator &cell)
  {
    current_cell_ptr = &cell;
  }

  template <int dim>
  void
  ShapeFunctionVectorDivergence<dim>::set_shape_fun_index(unsigned int index)
  {
    shape_fun_index = index;
  }

  template <int dim>
  double
  ShapeFunctionVectorDivergence<dim>::value(
    const Point<dim> &p,
    const unsigned int /* component = 0 */) const
  {
    // Map physical points to reference cell
    Point<dim> point_on_ref_cell(
      mapping.transform_real_to_unit_cell(*current_cell_ptr, p));

    // Copy-assign a fake quadrature rule form mapped point
    Quadrature<dim> fake_quadrature(point_on_ref_cell);

    // Update he fe_values object
    FEValues<dim> fe_values(*fe_ptr,
                            fake_quadrature,
                            update_values | update_gradients |
                              update_quadrature_points);

    fe_values.reinit(*current_cell_ptr);

    return fe_values[flux].divergence(shape_fun_index, /* q_index */ 0);
  }

  template <int dim>
  void
  ShapeFunctionVectorDivergence<dim>::value_list(
    const std::vector<Point<dim>> &points,
    std::vector<double> &          values,
    const unsigned int /* component = 0 */) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    const unsigned int n_q_points = points.size();

    // Map physical points to reference cell
    std::vector<Point<dim>> points_on_ref_cell(n_q_points);
    for (unsigned int i = 0; i < n_q_points; ++i)
      {
        points_on_ref_cell.at(i) =
          mapping.transform_real_to_unit_cell(*current_cell_ptr, points.at(i));
      }

    // Copy-assign a fake quadrature rule form mapped point
    Quadrature<dim> fake_quadrature(points_on_ref_cell);

    // Update he fe_values object
    FEValues<dim> fe_values(*fe_ptr,
                            fake_quadrature,
                            update_values | update_gradients |
                              update_quadrature_points);

    fe_values.reinit(*current_cell_ptr);

    for (unsigned int i = 0; i < n_q_points; ++i)
      {
        values.at(i) = fe_values[flux].divergence(shape_fun_index,
                                                  /* q_index */ i);
      }
  }

} // namespace ShapeFun

MSSTOKES_CLOSE_NAMESPACE