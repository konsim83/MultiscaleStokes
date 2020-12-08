#include <model_data/physical_constants.h>

MSSTOKES_OPEN_NAMESPACE


CoreModelData::ReferenceQuantities::ReferenceQuantities(
  const std::string &parameter_filename)
  : time(0)
  , velocity(10)
  , length(1e+4)
  , temperature_ref(273.15)
  , temperature_change(5)
{
  ParameterHandler prm;
  ReferenceQuantities::declare_parameters(prm);

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
CoreModelData::ReferenceQuantities::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Boussinesq Model");
  {
    prm.enter_subsection("Reference quantities");
    {
      prm.declare_entry("velocity",
                        "10",
                        Patterns::Double(0),
                        "Reference velocity");

      prm.declare_entry("length",
                        "1e+4",
                        Patterns::Double(0),
                        "Reference length.");

      prm.declare_entry("temperature",
                        "273.15",
                        Patterns::Double(0),
                        "Reference temperature at bottom.");

      prm.declare_entry("temperature change",
                        "5",
                        Patterns::Double(0),
                        "Reference temperature change.");
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();
}



void
CoreModelData::ReferenceQuantities::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Boussinesq Model");
  {
    prm.enter_subsection("Reference quantities");
    {
      velocity           = prm.get_double("velocity");           /* m/s */
      length             = prm.get_double("length");             /* m */
      temperature_ref    = prm.get_double("temperature");        /* K */
      temperature_change = prm.get_double("temperature change"); /* K */
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();

  time = length / velocity; /* s */
}

MSSTOKES_CLOSE_NAMESPACE
