#include <model_data/core_model_data.h>

#include <model_data/core_model_data.tpp>

MSSTOKES_OPEN_NAMESPACE

template Tensor<1, 2>
CoreModelData::vertical_gravity_vector<2>(const Point<2> &p,
                                          const double    gravity_constant);
template Tensor<1, 3>
CoreModelData::vertical_gravity_vector<3>(const Point<3> &p,
                                          const double    gravity_constant);

MSSTOKES_CLOSE_NAMESPACE
