#pragma once

#include <base/config.h>
#include <sys/stat.h>

#include <stdexcept>
#include <string>

MSSTOKES_OPEN_NAMESPACE

/*!
 * @namespace Tools
 *
 * @brief Namespace containing all tools that do not fit other more specific namespaces.
 */
namespace Tools
{
  /*!
   * @brief Creates a directory with a name from a given string
   *
   * @param dir_name
   */
  void
  create_data_directory(std::string dir_name);

} // namespace Tools

MSSTOKES_CLOSE_NAMESPACE
