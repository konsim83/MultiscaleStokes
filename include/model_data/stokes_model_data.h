#pragma once

// C++ STL
#include <cmath>
#include <string>

// Deal.ii
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/vector.h>

// AquaPlanet
#include <base/config.h>
#include <model_data/core_model_data.h>


MSSTOKES_OPEN_NAMESPACE

namespace CoreModelData
{
  /*!
   * Temerature initial values for rising warm bubble test.
   */
  template <int dim>
  class TemperatureForcing : public Function<dim>
  {
  public:
    /*!
     * Constructor.
     */
    TemperatureForcing(const Point<dim> &center,
                       const double      reference_temperature,
                       const double      expansion_coefficient,
                       const double      variance);


    /*!
     * Return temperature value at a single point.
     *
     * @param p
     * @param component
     * @return
     */
    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;

    /*!
     * Return temperature value as a vector at a single point.
     *
     * @param points
     * @param values
     */
    virtual void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<double> &          values,
               const unsigned int             component = 0) const override;

  private:
    Point<dim>     center;
    Tensor<2, dim> covariance_matrix;
    const double   reference_temperature;
    const double   expansion_coefficient;
    const double   variance;
  };
} // namespace CoreModelData

// Extern template instantiations
extern template class CoreModelData::TemperatureForcing<2>;
extern template class CoreModelData::TemperatureForcing<3>;

MSSTOKES_CLOSE_NAMESPACE
