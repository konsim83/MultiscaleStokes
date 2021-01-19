#pragma once

// Deal.ii
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

// AquaPlanet
#include <base/config.h>
#include <model_data/physical_constants.h>


MSSTOKES_OPEN_NAMESPACE

/*!
 * @namespace CoreModelData
 *
 * Namespace containing namespaces for different models such as Boussinesq
 * appromimation or the full primitive equations
 */
namespace CoreModelData
{
  /*!
   * Compute vertical gravity vector at a given point.
   *
   * @return vertical gravity vector
   */
  template <int dim>
  Tensor<1, dim>
  vertical_gravity_vector(const Point<dim> &p, const double gravity_constant);

} // namespace CoreModelData

/*
 * Extern template instantiations
 */
extern template Tensor<1, 2>
CoreModelData::vertical_gravity_vector<2>(const Point<2> &p,
                                          const double    gravity_constant);
extern template Tensor<1, 3>
CoreModelData::vertical_gravity_vector<3>(const Point<3> &p,
                                          const double    gravity_constant);
MSSTOKES_CLOSE_NAMESPACE
