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

// My Headers
#include <base/config.h>

MSSTOKES_OPEN_NAMESPACE

namespace ShapeFun
{
  using namespace dealii;

  template <int dim>
  class ShapeFunctionVector : public Function<dim>
  {
  public:
    ShapeFunctionVector(const FiniteElement<dim> &                  fe,
                        typename Triangulation<dim>::cell_iterator &cell);

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &value) const override;

    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  values) const override;

    void
    tensor_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Tensor<1, dim>> &  values) const;

    void
    set_current_cell(const typename Triangulation<dim>::cell_iterator &cell);

    void
    set_shape_fun_index(unsigned int index);

  private:
    SmartPointer<const FiniteElement<dim>> fe_ptr;
    const unsigned int                     dofs_per_cell;
    unsigned int                           shape_fun_index;

    const MappingQGeneric<dim> mapping;

    typename Triangulation<dim>::cell_iterator *current_cell_ptr;
  };

  template <int dim>
  ShapeFunctionVector<dim>::ShapeFunctionVector(
    const FiniteElement<dim> &                  fe,
    typename Triangulation<dim>::cell_iterator &cell)
    : Function<dim>(dim + 1)
    , fe_ptr(&fe)
    , dofs_per_cell(fe_ptr->dofs_per_cell)
    , shape_fun_index(0)
    , mapping(1)
    , current_cell_ptr(&cell)
  {
    // Make sure FE has dim components
    Assert(fe_ptr->n_components() == dim,
           ExcDimensionMismatch(dim, fe_ptr->n_components()));
  }

  template <int dim>
  void
  ShapeFunctionVector<dim>::set_current_cell(
    const typename Triangulation<dim>::cell_iterator &cell)
  {
    current_cell_ptr = &cell;
  }

  template <int dim>
  void
  ShapeFunctionVector<dim>::set_shape_fun_index(unsigned int index)
  {
    shape_fun_index = index;
  }

  template <int dim>
  void
  ShapeFunctionVector<dim>::vector_value(const Point<dim> &p,
                                         Vector<double> &  value) const
  {
    // Map physical points to reference cell
    Point<dim> point_on_ref_cell(
      mapping.transform_real_to_unit_cell(*current_cell_ptr, p));

    // Copy-assign a fake quadrature rule form mapped point
    Quadrature<dim> fake_quadrature(point_on_ref_cell);

    // Update he fe_values object
    FEValues<dim> fe_values(mapping,
                            *fe_ptr,
                            fake_quadrature,
                            update_values | update_quadrature_points);

    fe_values.reinit(*current_cell_ptr);

    for (unsigned int i = 0; i < dim; ++i)
      {
        value[i] = fe_values.shape_value_component(shape_fun_index,
                                                   /* q_index */ 0,
                                                   i);
      }

    value[dim] = 0;
  }

  template <int dim>
  void
  ShapeFunctionVector<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    const unsigned int n_q_points = points.size();

    // Map physical points to reference cell
    std::vector<Point<dim>> points_on_ref_cell(n_q_points);
    for (unsigned int i = 0; i < n_q_points; ++i)
      {
        points_on_ref_cell[i] =
          mapping.transform_real_to_unit_cell(*current_cell_ptr, points[i]);
      }

    // Copy-assign a fake quadrature rule form mapped point
    Quadrature<dim> fake_quadrature(points_on_ref_cell);

    // Update he fe_values object
    FEValues<dim> fe_values(mapping,
                            *fe_ptr,
                            fake_quadrature,
                            update_values | update_quadrature_points);

    fe_values.reinit(*current_cell_ptr);

    for (unsigned int i = 0; i < n_q_points; ++i)
      {
        for (unsigned int component = 0; component < dim; ++component)
          {
            values[i][component] =
              fe_values.shape_value_component(shape_fun_index,
                                              /* q_index */ i,
                                              component);
          }
        values[i][dim] = 0;
      }
  }

  template <int dim>
  void
  ShapeFunctionVector<dim>::tensor_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Tensor<1, dim>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    const unsigned int n_q_points = points.size();

    // Map physical points to reference cell
    std::vector<Point<dim>> points_on_ref_cell(n_q_points);
    for (unsigned int i = 0; i < n_q_points; ++i)
      {
        points_on_ref_cell[i] =
          mapping.transform_real_to_unit_cell(*current_cell_ptr, points[i]);
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
        for (unsigned int component = 0; component < dim; ++component)
          {
            values[i][component] =
              fe_values.shape_value_component(shape_fun_index,
                                              /* q_index */ i,
                                              component);
          }
        values[i][dim] = 0;
      }
  }

} // namespace ShapeFun

MSSTOKES_CLOSE_NAMESPACE