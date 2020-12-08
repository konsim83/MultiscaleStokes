#include <model_data/core_model_data.h>

MSSTOKES_OPEN_NAMESPACE

namespace CoreModelData
{
  template <int dim>
  Tensor<1, dim>
  vertical_gravity_vector(const Point<dim> & /* p */,
                          const double gravity_constant)
  {
    Tensor<1, dim> e_z;
    e_z[dim - 1] = 1;

    return -gravity_constant * e_z;
  }

} // namespace CoreModelData

MSSTOKES_CLOSE_NAMESPACE
