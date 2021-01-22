#pragma once

// MsStokes
#include <postprocessor/postprocessor.h>

MSSTOKES_OPEN_NAMESPACE

template <int dim>
Postprocessor<dim>::Postprocessor(const unsigned int partition)
  : partition(partition)
{}



template <int dim>
std::vector<std::string>
Postprocessor<dim>::get_names() const
{
  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.emplace_back("divergence");
  solution_names.emplace_back("pressure");
  solution_names.emplace_back("vertical_buoyancy_force");
  solution_names.emplace_back("partition");

  return solution_names;
}



template <int dim>
std::vector<DataComponentInterpretation::DataComponentInterpretation>
Postprocessor<dim>::get_data_component_interpretation() const
{
  // Velocity
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation(dim,
                   DataComponentInterpretation::component_is_part_of_vector);

  // Divergence
  interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  // Pressure
  interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  // Vertical temperature forcing
  interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  // Partition
  interpretation.push_back(DataComponentInterpretation::component_is_scalar);

  return interpretation;
}



template <int dim>
UpdateFlags
Postprocessor<dim>::get_needed_update_flags() const
{
  return update_values | update_gradients | update_quadrature_points;
}



template <int dim>
void
Postprocessor<dim>::evaluate_vector_field(
  const DataPostprocessorInputs::Vector<dim> &inputs,
  std::vector<Vector<double>> &               computed_quantities) const
{
  const unsigned int n_quadrature_points = inputs.solution_values.size();

  Assert(inputs.solution_gradients.size() == n_quadrature_points,
         ExcInternalError());
  Assert(computed_quantities.size() == n_quadrature_points, ExcInternalError());
  Assert(inputs.solution_values[0].size() == dim + 2, ExcInternalError());

  /*
   * TODO: Rescale to physical quantities here.
   */
  for (unsigned int q = 0; q < n_quadrature_points; ++q)
    {
      // Velocity
      for (unsigned int d = 0; d < dim; ++d)
        computed_quantities[q](d) = inputs.solution_values[q](d);

      // Divergence
      computed_quantities[q](dim) = 0;
      for (unsigned int d = dim; d < 2 * dim; ++d)
        {
          computed_quantities[q](dim) +=
            inputs.solution_gradients[q][d - dim][d - dim];
        }

      const double pressure           = (inputs.solution_values[q](dim));
      computed_quantities[q](dim + 1) = pressure;

      // should be negative since gravity points downward
      computed_quantities[q](dim + 2) = -inputs.solution_values[q](dim + 1);

      computed_quantities[q](dim + 3) = partition;
    }
}

MSSTOKES_CLOSE_NAMESPACE