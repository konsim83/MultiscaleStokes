#pragma once

// Deal.ii
#include <deal.II/base/parameter_handler.h>

// AquaPlanet
#include <base/config.h>
#include <model_data/reference_quantities.h>

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
     * Earth reference pressure.
     */
    double pressure; /* Pa */

    /*!
     * Earth angular velocity.
     */
    double omega; /* 1/s */

    /*!
     * Reference density of air at bottom reference
     * temperature.
     */
    double density; /* kg / m^3 */

    /*!
     * Universal gas constant.
     */
    double universal_gas_constant; /* J/(mol*K) */

    /*!
     * Specific gas constant of dry air.
     */
    double specific_gas_constant_dry; /* J/(kg*K) */

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
     * Specific heat capacity of air under constant pressure.
     */
    double specific_heat_p; /* J / (K*kg) */

    /*!
     * Specific heat capacity of air under isochoric changes of state.
     */
    double specific_heat_v; /* J / (K*kg) */

    /*!
     * Thermal conductivity (kappa, k or lambda) of air at bottom reference
     * temperature.
     */
    double thermal_conductivity; /* W/(m*K) */

    /*!
     * Thermal diffusivity (alpha or a) of air at bottom reference
     * temperature.
     */
    double thermal_diffusivity;

    /*!
     * A good part of the earth's heat loss through the surface is due
     * to the decay of radioactive elements (uranium, thorium,
     * potassium).
     */
    double radiogenic_heating; /* W / kg */

    /*!
     * Gravity constant.
     */
    double gravity_constant; /* m/s^2 */

    /*!
     * Speed of sound.
     */
    double speed_of_sound; /* m/s */

    /*!
     * Height of atmosphere (up to mesosphere)
     */
    double atm_height; /* m */

    /*!
     * Inner earth radius
     */
    double R0; /* m */

    /*!
     * Earth radius plus height of mesosphere.
     */
    double R1; /* m */

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
