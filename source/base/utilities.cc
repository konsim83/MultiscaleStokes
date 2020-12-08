#include <base/utilities.h>

MSSTOKES_OPEN_NAMESPACE

namespace Tools
{
  void
  create_data_directory(std::string dir_name)
  {
    const int dir_err =
      mkdir(dir_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (-1 == dir_err)
      {
        throw std::runtime_error(
          "Error creating directory! It might already exist or you do not have write permissions in this folder.");
      }
  }

} // namespace Tools

MSSTOKES_CLOSE_NAMESPACE
