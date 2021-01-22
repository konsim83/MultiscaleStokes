#pragma once

// Deal.ii
#include <deal.II/base/parameter_handler.h>

// AquaPlanet
#include <base/config.h>
#include <model_data/physical_constants.h>

MSSTOKES_OPEN_NAMESPACE

namespace CoreModelData
{
  /*!
   * @struct Paramters
   *
   * Struct holding parameters for a bouyancy Boussinesq model.
   */
  struct Parameters
  {
    Parameters(const std::string &parameter_filename);

    void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);

    unsigned int space_dimension;

    CoreModelData::PhysicalConstants physical_constants;

    unsigned int initial_global_refinement;

    unsigned int stokes_velocity_degree;

    bool use_locally_conservative_discretization;

    unsigned int solver_diagnostics_print_level;

    std::string filename_output;
    std::string dirname_output;

    bool hello_from_cluster;


    bool         verbose_basis = true;
    unsigned int refinements_basis;
  };

} // namespace CoreModelData

MSSTOKES_CLOSE_NAMESPACE
