#include "common.hpp"
#include "kruskal.hpp"
#include "edgelist.hpp"
#include "misc.hpp"
#include <iostream>
#include <string>

// debugging
#include <chrono>
#include <thread>


OptionsT g_options;
Benchmarker g_benchmarker;

OptionsT parse_options(int ac, char **av)
{
  po::options_description desc("Allowed options");

  // clang-format off
    desc.add_options()
        ("help,h", 
        "produce help message")

        ("mst-kernel,m",
        po::value<int>()->default_value(MST_KERNEL_REGULAR_KRUSKAL),
        "Kernel used calculating MST. 0 = regular kruskal, 1 = filter-kruskal")

        ("sort-kernel,s",
        po::value<int>()->default_value(SORT_KERNEL_GPU_BUBBLE_MULT),
        "Kernel used for sorting [int]")

        ("inputfile,i", 
        po::value<std::vector<std::string>>(), 
        "input file containing graph data. Filepath must be relative to Ex2/");
  // clang-format on

  // Boiler-plate Boost options stuff
  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

  // Produce help message
  if (vm.count("help"))
  {
    std::cout << desc << "\n";
    exit(EXIT_SUCCESS);
  }

  return vm;
}

int main(int ac, char **av)
{
  g_options = parse_options(ac, av);
  g_benchmarker = Benchmarker();

  // Load input file
  // I havent found a good way for adding defaults tring to boost::options so im doing it by hand
  std::string input_file = misc::get_proj_root_dir().append(
      g_options.count("inputfile") ? g_options["inputfile"].as<std::vector<std::string>>()[0]
                                 : DEFAULT_INPUT_FILE);
  std::cout << "Loading " << input_file << std::endl;

  EdgeList edgelist(input_file.c_str());


  calculate_mst(edgelist);

  g_benchmarker.print_timings();


  return 0;
}