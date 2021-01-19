#include <model_data/physical_constants.h>

MSSTOKES_OPEN_NAMESPACE


CoreModelData::PhysicalConstants::PhysicalConstants(
  const std::string &parameter_filename)
  : reference_temperature(293.15)
  , density(1.29)
  , expansion_coefficient(1 / 273.15)
  , dynamic_viscosity(1.82e-5)
  , kinematic_viscosity(dynamic_viscosity / density)
  , gravity_constant(9.81)
{
  ParameterHandler prm;
  PhysicalConstants::declare_parameters(prm);

  std::ifstream parameter_file(parameter_filename);
  if (!parameter_file)
    {
      parameter_file.close();
      std::ofstream parameter_out(parameter_filename);
      prm.print_parameters(parameter_out, ParameterHandler::Text);
      AssertThrow(false,
                  ExcMessage(
                    "Input parameter file <" + parameter_filename +
                    "> not found. Creating a template file of the same name."));
    }
  prm.parse_input(parameter_file,
                  /* filename = */ "generated_parameter.in",
                  /* last_line = */ "",
                  /* skip_undefined = */ true);
  parse_parameters(prm);
}



void
CoreModelData::PhysicalConstants::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Physical Constants");
  {
    prm.declare_entry("reference temperature",
                      "293.15",
                      Patterns::Double(0),
                      "Reference temperature.");

    prm.declare_entry("density",
                      "1.29",
                      Patterns::Double(0),
                      "Reference density.");

    prm.declare_entry("expansion coefficient",
                      "0.003661",
                      Patterns::Double(0),
                      "Thermal expansion coefficient std: ideal gas.");

    prm.declare_entry("dynamic viscosity",
                      "1.82e-5",
                      Patterns::Double(0),
                      "Dynamic viscosity.");

    prm.declare_entry("gravity constant",
                      "9.81",
                      Patterns::Double(0),
                      "Gravity constant");
  }
  prm.leave_subsection();
}



void
CoreModelData::PhysicalConstants::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Physical Constants");
  {
    reference_temperature = prm.get_double("reference temperature");

    density               = prm.get_double("density"); /* kg / m^3 */
    expansion_coefficient = prm.get_double("expansion coefficient");

    dynamic_viscosity   = prm.get_double("dynamic viscosity");
    kinematic_viscosity = dynamic_viscosity / density;

    gravity_constant = prm.get_double("gravity constant");
  }
  prm.leave_subsection();
}

MSSTOKES_CLOSE_NAMESPACE
