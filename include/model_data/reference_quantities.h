#pragma once

// Deal.ii
#include <deal.II/base/parameter_handler.h>

// AquaPlanet
#include <base/config.h>

MSSTOKES_OPEN_NAMESPACE

namespace CoreModelData
{
  /*!
   * @struct ReferenceQuantities
   *
   * Struct contains reference quantities for non-dimensionalization
   */
  struct ReferenceQuantities
  {
    ReferenceQuantities(const std::string &parameter_filename);

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);

    /*!
     * Reference time is one hour.
     */
    double time; /* s */

    /*!
     * Reference velocity.
     */
    double velocity; /* m/s */

    /*!
     * Reference length.
     */
    double length; /* m */

    /*!
     * Reference temperature 273.15 K (0 degree Celsius).
     */
    double temperature_ref; /* K */

    /*!
     * Reference temperature change.
     */
    double temperature_change; /* K */

  }; // struct ReferenceQuantities

} // namespace CoreModelData

MSSTOKES_CLOSE_NAMESPACE
