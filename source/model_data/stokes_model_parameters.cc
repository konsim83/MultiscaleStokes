#include <model_data/stokes_model_parameters.h>

MSSTOKES_OPEN_NAMESPACE


CoreModelData::Parameters::Parameters(const std::string &parameter_filename)
  : space_dimension(2)
  , physical_constants(parameter_filename)
  , initial_global_refinement(2)
  , stokes_velocity_degree(2)
  , use_locally_conservative_discretization(true)
  , solver_diagnostics_print_level(1)
  , hello_from_cluster(false)
{
  ParameterHandler prm;
  CoreModelData::Parameters::declare_parameters(prm);

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
CoreModelData::Parameters::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Stokes Model");
  {
    prm.enter_subsection("Mesh parameters");
    {
      prm.declare_entry("initial global refinement",
                        "3",
                        Patterns::Integer(0),
                        "The number of global refinement steps performed on "
                        "the initial coarse mesh, before the problem is first "
                        "solved there.");
    }
    prm.leave_subsection();

    prm.declare_entry("space dimension",
                      "2",
                      Patterns::Integer(2, 3),
                      "Spatial dimension of the problem.");

    prm.declare_entry("stokes velocity degree",
                      "2",
                      Patterns::Integer(1),
                      "The polynomial degree to use for the velocity variables "
                      "in the NSE system.");

    prm.declare_entry(
      "use locally conservative discretization",
      "true",
      Patterns::Bool(),
      "Whether to use a Navier-Stokes discretization that is locally "
      "conservative at the expense of a larger number of degrees "
      "of freedom, or to go with a cheaper discretization "
      "that does not locally conserve mass (although it is "
      "globally conservative.");

    prm.declare_entry("solver diagnostics level",
                      "1",
                      Patterns::Integer(0),
                      "Output level for solver for debug purposes.");

    prm.declare_entry("filename output",
                      "dycore",
                      Patterns::FileName(),
                      "Base filename for output.");

    prm.declare_entry("dirname output",
                      "data-output",
                      Patterns::FileName(),
                      "Name of output directory.");

    prm.declare_entry(
      "hello from cluster",
      "false",
      Patterns::Bool(),
      "Output some (node) information of each MPI process (rank, node name, number of threads).");
  }
  prm.leave_subsection();
}



void
CoreModelData::Parameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Stokes Model");
  {
    prm.enter_subsection("Mesh parameters");
    {
      initial_global_refinement = prm.get_integer("initial global refinement");
    }
    prm.leave_subsection();

    space_dimension = prm.get_integer("space dimension");

    stokes_velocity_degree = prm.get_integer("stokes velocity degree");

    use_locally_conservative_discretization =
      prm.get_bool("use locally conservative discretization");

    solver_diagnostics_print_level =
      prm.get_integer("solver diagnostics level");

    filename_output = prm.get("filename output");
    dirname_output  = prm.get("dirname output");

    hello_from_cluster = prm.get_bool("hello from cluster");
  }
  prm.leave_subsection();
}


MSSTOKES_CLOSE_NAMESPACE
