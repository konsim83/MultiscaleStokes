#include <model_data/physical_constants.h>

MSSTOKES_OPEN_NAMESPACE


CoreModelData::PhysicalConstants::PhysicalConstants(
  const std::string &parameter_filename)
  : pressure(1.01325e+5)
  , omega(7.272205e-5)
  , density(1.29)
  , universal_gas_constant(8.31446261815324)
  , specific_gas_constant_dry(2)
  , expansion_coefficient(1 / 273.15)
  , dynamic_viscosity(1.82e-5)
  , kinematic_viscosity(dynamic_viscosity)
  , specific_heat_p(1.005)
  , specific_heat_v(0.718)
  , thermal_conductivity(2.62e-2)
  , thermal_diffusivity(thermal_conductivity / (specific_heat_p * 1.01325e+5))
  , radiogenic_heating(7.4e-12)
  , gravity_constant(9.81)
  , speed_of_sound(331.5)
  , atm_height(4)
  , R0(6.371000e+6)
  , R1(R0 + 1.0e+5)
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
    prm.declare_entry("omega",
                      "7.272205e-5",
                      Patterns::Double(0),
                      "Reference angular velocity [rad/s].");

    prm.declare_entry("average atm pressure",
                      "1.01325e+5",
                      Patterns::Double(0),
                      "Reference pressure.");

    prm.declare_entry("density",
                      "1.29",
                      Patterns::Double(0),
                      "Reference density.");

    prm.declare_entry("universal gas constant",
                      "8.31446261815324",
                      Patterns::Double(0),
                      "Universal gas constant.");

    prm.declare_entry("specific gas constant dry",
                      "287.0",
                      Patterns::Double(0),
                      "Specific gas constant of dry air.");

    prm.declare_entry("expansion coefficient",
                      "0.003661",
                      Patterns::Double(0),
                      "Thermal expansion coefficient std: ideal gas.");

    prm.declare_entry("dynamic viscosity",
                      "1.82e-5",
                      Patterns::Double(0),
                      "Dynamic viscosity.");

    prm.declare_entry("specific heat p",
                      "1.005",
                      Patterns::Double(0),
                      "Specific heat constant (isobaric)");

    prm.declare_entry("specific heat v",
                      "0.718",
                      Patterns::Double(0),
                      "Specific heat constant (isochoric).");

    prm.declare_entry("thermal conductivity",
                      "2.62e-2",
                      Patterns::Double(0),
                      "Thermal conductivity.");

    prm.declare_entry("radiogenic heating",
                      "7.4e-12",
                      Patterns::Double(0),
                      "Radiogenic heating.");

    prm.declare_entry("gravity constant",
                      "9.81",
                      Patterns::Double(0),
                      "Gravity constant");

    prm.declare_entry("speed of sound",
                      "331.5",
                      Patterns::Double(0),
                      "Speed of sound.");

    prm.declare_entry("atm height",
                      "1.0e+5",
                      Patterns::Double(0),
                      "Height of atmosphere.");

    prm.declare_entry("R0",
                      "6.371000e+6",
                      Patterns::Double(0),
                      "Inner radius of atmosphere.");
  }
  prm.leave_subsection();
}



void
CoreModelData::PhysicalConstants::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Physical Constants");
  {
    pressure = prm.get_double("average atm pressure"); /* Pa */
    omega    = prm.get_double("omega");                /* 1/s */
    density  = prm.get_double("density");              /* kg / m^3 */

    universal_gas_constant    = prm.get_double("universal gas constant");
    specific_gas_constant_dry = prm.get_double("specific gas constant dry");
    expansion_coefficient     = prm.get_double("expansion coefficient");

    dynamic_viscosity = prm.get_double("dynamic viscosity");

    kinematic_viscosity = dynamic_viscosity / density;

    specific_heat_p      = prm.get_double("specific heat p");
    specific_heat_v      = prm.get_double("specific heat v");
    thermal_conductivity = prm.get_double("thermal conductivity");

    thermal_diffusivity = thermal_conductivity / (specific_heat_p * pressure);

    radiogenic_heating = prm.get_double("radiogenic heating");
    gravity_constant   = prm.get_double("gravity constant");
    speed_of_sound     = prm.get_double("speed of sound");
    atm_height         = prm.get_double("atm height");
    R0                 = prm.get_double("R0");

    R1 = R0 + atm_height;
  }
  prm.leave_subsection();
}

MSSTOKES_CLOSE_NAMESPACE
