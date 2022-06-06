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

std::string get_input_file()
{
  // Load input file
  // I havent found a good way for adding defaults tring to boost::options so im doing it by hand
  std::string input_file = misc::get_proj_root_dir().append(
      g_options.count("inputfile") ? g_options["inputfile"].as<std::vector<std::string>>()[0]
                                   : DEFAULT_INPUT_FILE);

  return input_file;
}

std::string get_gt_file()
{
  std::string file_name = get_input_file();
  size_t len = file_name.size();
  std::string gt_file = file_name.insert(len - 4, "_mst_gt");

  return gt_file;
}

std::string get_output_file()
{
  std::string file_name = misc::get_proj_root_dir().append("/out/MST_calculated.csv");
  return file_name;
}

std::string get_reference_output_file()
{
  std::string file_name = misc::get_proj_root_dir().append("/out/MST_reference.csv");
  return file_name;
}

} // namespace misc