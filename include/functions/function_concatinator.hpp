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

  /*!
   * @class FunctionConcatinator
   *
   * @brief Concatination of Function<dim> objects
   *
   * This class represents a vector function that is made of two
   * concatinated <code> Function<dim> </code> objects.
   *
   */
  template <int dim>
  class FunctionConcatinator : public Function<dim>
  {
  public:
    /*!
     * Constructor takes two function objects and concatinates
     * them to a vecor function.
     *
     * @param function1
     * @param function2
     */
    FunctionConcatinator(const Function<dim> &function1,
                         const Function<dim> &function2);

    /*!
     * Value of concatinated function of a given component.
     *
     * @param p
     * @param component
     * @return
     */
    virtual double
    value(const Point<dim> &p, const unsigned int component) const override;

    /*!
     * Vector value of concatinated function.
     *
     * @param p
     * @param value
     */
    virtual void
    vector_value(const Point<dim> &p, Vector<double> &value) const override;

    /*!
     * Vector value list of concatinated function.
     *
     * @param points
     * @param values
     */
    virtual void
    vector_value_list(const std::vector<Point<dim>> &points,
                      std::vector<Vector<double>> &  values) const override;

  private:
    /*!
     * Smart pointer to first componenet of input.
     */
    SmartPointer<const Function<dim>> function_ptr1;

    /*!
     * Smart pointer to second componenet of input.
     */
    SmartPointer<const Function<dim>> function_ptr2;
  };


  template <int dim>
  FunctionConcatinator<dim>::FunctionConcatinator(
    const Function<dim> &function1,
    const Function<dim> &function2)
    : Function<dim>(function1.n_components + function2.n_components)
    , function_ptr1(&function1)
    , function_ptr2(&function2)
  {}

  template <int dim>
  double
  FunctionConcatinator<dim>::value(const Point<dim> & p,
                                   const unsigned int component) const
  {
    if (component < function_ptr1->n_components)
      {
        Vector<double> value1(function_ptr1->n_components);
        function_ptr1->vector_value(p, value1);
        return value1(component);
      }
    else
      {
        Vector<double> value2(function_ptr2->n_components);
        function_ptr2->vector_value(p, value2);
        return value2(component - function_ptr1->n_components);
      }
  }

  template <int dim>
  void
  FunctionConcatinator<dim>::vector_value(const Point<dim> &p,
                                          Vector<double> &  value) const
  {
    Vector<double> value1(function_ptr1->n_components);
    function_ptr1->vector_value(p, value1);

    Vector<double> value2(function_ptr2->n_components);
    function_ptr2->vector_value(p, value2);

    for (unsigned int j = 0; j < function_ptr1->n_components; ++j)
      value(j) = value1(j);
    for (unsigned int j = 0; j < function_ptr2->n_components; ++j)
      value(function_ptr1->n_components + j) = value2(j);
  }

  template <int dim>
  void
  FunctionConcatinator<dim>::vector_value_list(
    const std::vector<Point<dim>> &points,
    std::vector<Vector<double>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    Vector<double> value1(function_ptr1->n_components);
    Vector<double> value2(function_ptr2->n_components);

    for (unsigned int i = 0; i < points.size(); ++i)
      {
        Assert(values[i].size() ==
                 (function_ptr1->n_components + function_ptr2->n_components),
               ExcDimensionMismatch(values[i].size(),
                                    (function_ptr1->n_components +
                                     function_ptr2->n_components)));

        value1 = 0;
        value2 = 0;
        function_ptr1->vector_value(points[i], value1);
        function_ptr2->vector_value(points[i], value2);

        for (unsigned int j = 0; j < function_ptr1->n_components; ++j)
          {
            values[i](j) = value1(j);
          }
        for (unsigned int j = 0; j < function_ptr2->n_components; ++j)
          {
            values[i](function_ptr1->n_components + j) = value2(j);
          }
      }
  }

} // namespace ShapeFun

MSSTOKES_CLOSE_NAMESPACE
