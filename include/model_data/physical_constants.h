#pragma once

// Deal.ii
#include <deal.II/base/parameter_handler.h>

// AquaPlanet
#include <base/config.h>

MSSTOKES_OPEN_NAMESPACE

namespace CoreModelData
{
  /*!
   * @struct PhysicalConstants
   *
   * Struct containing physical constants.
   */
  struct PhysicalConstants
  {
    PhysicalConstants(const std::string &parameter_filename);

    static void
    declare_parameters(ParameterHandler &prm);

    void
    parse_parameters(ParameterHandler &prm);

    /*!
     * Reference temperature. This is not really a physical constant.
     */
    double reference_temperature;

    /*!
     * Reference density of air at bottom reference
     * temperature.
     */
    double density; /* kg / m^3 */

    /*!
     * Thermal expansion coefficient (beta) of air at bottom reference
     * temperature.
     */
    double expansion_coefficient;

    /*!
     * Dynamic viscosity (eta or mu) of air at bottom reference
     * temperature.
     */
    double dynamic_viscosity; /* kg/(m*s) */

    /*!
     * Dynamics viscosity (nu) of air at bottom reference
     * temperature.
     */
    double kinematic_viscosity;

    /*!
     * Gravity constant.
     */
    double gravity_constant; /* m/s^2 */

    /*!
     * A year in seconds.
     */
    static constexpr double year_in_seconds = 60 * 60 * 24 * 365.2425; /* s */

    /*!
     * A day in seconds.
     */
    static constexpr double day_in_seconds = 60 * 60 * 24; /* s */

    /*!
     * An hour in seconds.
     */
    static constexpr double hour_in_seconds = 60 * 60; /* s */

  }; // struct PhysicalConstants

} // namespace CoreModelData

MSSTOKES_CLOSE_NAMESPACE
