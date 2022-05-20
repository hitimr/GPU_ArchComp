#pragma once
#ifdef __linux__
#include <libgen.h>       // dirname
#include <linux/limits.h> // PATH_MAX
#include <string>
#include <unistd.h> // readlink
#else
#error Platform not supported
#endif // linux

#include "common.hpp"

namespace misc
{
static std::string get_proj_root_dir()
{
#ifdef __linux__
  // Get path to exe
  char result[PATH_MAX];
  ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
  std::string path;
  if (count != -1)
  {
    path = dirname(result);
  }

#else
#error Platform not supported
#endif // linux

  std::string root_folder_name(PROJ_ROOT_FOLDER_NAME);
  size_t pos = path.find(root_folder_name);
  if (pos == std::string::npos)
  {
    throw std::runtime_error("Could not locate project root dir");
  }

  path = path.erase(pos + root_folder_name.size()) + "/";

  return path;
}
} // namespace misc