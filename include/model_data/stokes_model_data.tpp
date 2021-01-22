#include <model_data/stokes_model_data.h>

MSSTOKES_OPEN_NAMESPACE

template <int dim>
CoreModelData::TemperatureForcing<dim>::TemperatureForcing(
  const Point<dim> &center,
  const double      reference_temperature,
  const double      expansion_coefficient,
  const double      variance)
  : Function<dim>(1)
  , center(center)
  , reference_temperature(reference_temperature)
  , expansion_coefficient(expansion_coefficient)
  , variance(variance)
{
  covariance_matrix = 0;

  for (unsigned int d = 0; d < dim; ++d)
    {
      covariance_matrix[d][d] = variance;
    }
}


template <int dim>
double
CoreModelData::TemperatureForcing<dim>::value(const Point<dim> &p,
                                              const unsigned int) const
{
  /*
   * This is a normalized Gaussian
   */
  double temperature =
    sqrt(determinant(covariance_matrix)) *
    exp(-0.5 * scalar_product(p - center, covariance_matrix * (p - center))) /
    sqrt(std::pow(2 * numbers::PI, dim));

  return (1 - expansion_coefficient * (temperature - reference_temperature));
  // return temperature;
}



template <int dim>
void
CoreModelData::TemperatureForcing<dim>::value_list(
  const std::vector<Point<dim>> &points,
  std::vector<double> &          values,
  const unsigned int) const
{
  Assert(points.size() == values.size(),
         ExcDimensionMismatch(points.size(), values.size()));

  for (unsigned int p = 0; p < points.size(); ++p)
    {
      const double temperature =
        sqrt(determinant(covariance_matrix)) *
        exp(-0.5 * scalar_product(points[p] - center,
                                  covariance_matrix * (points[p] - center))) /
        sqrt(std::pow(2 * numbers::PI, dim));
      // values[p] = temperature;
      values[p] =
        (1 - expansion_coefficient * (temperature - reference_temperature));
    }
}

MSSTOKES_CLOSE_NAMESPACE
