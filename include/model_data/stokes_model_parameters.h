#pragma once

// Deal.ii
#include <deal.II/base/parameter_handler.h>

// AquaPlanet
#include <base/config.h>
#include <model_data/physical_constants.h>
#include <model_data/reference_quantities.h>

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

    CoreModelData::ReferenceQuantities reference_quantities;
    CoreModelData::PhysicalConstants   physical_constants;

    unsigned int initial_global_refinement;

    unsigned int stokes_velocity_degree;

    bool use_locally_conservative_discretization;

    unsigned int solver_diagnostics_print_level;

    bool use_schur_complement_solver;
    bool use_direct_solver;

    std::string filename_output;
    std::string dirname_output;

    bool hello_from_cluster;
  };

} // namespace CoreModelData

MSSTOKES_CLOSE_NAMESPACE