#pragma once

// Deal.II
#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/numerics/data_postprocessor.h>

// MsStokes
#include <base/config.h>

MSSTOKES_OPEN_NAMESPACE

template <int dim>
class Postprocessor : public DataPostprocessor<dim>
{
public:
  Postprocessor(const unsigned int partition);

  virtual void
  evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &computed_quantities) const override;

  virtual std::vector<std::string>
  get_names() const override;

  virtual std::vector<DataComponentInterpretation::DataComponentInterpretation>
  get_data_component_interpretation() const override;

  virtual UpdateFlags
  get_needed_update_flags() const override;

private:
  const unsigned int partition;
};

// Extern template instantiations
extern template class Postprocessor<2>;
extern template class Postprocessor<3>;

MSSTOKES_CLOSE_NAMESPACE