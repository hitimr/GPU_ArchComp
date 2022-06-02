#include "common.hpp"
#include "edgelist.hpp"
#include "kruskal.hpp"
#include "misc.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
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
        po::value<int>()->default_value(MST_DEFAULT_KERNEL),
        "Kernel used calculating MST. 0 = regular kruskal, 1 = filter-kruskal")

        ("sort-kernel,s",
        po::value<int>()->default_value(SORT_DEFAULT_KERNEL),
        "Kernel used for sort() [int]")

        ("partition-kernel,p",
        po::value<int>()->default_value(PARTITION_DEFAULT_KERNEL),
        "Kernel used for partition() [int]")

        ("filter-kernel,f",
        po::value<int>()->default_value(DEFAULT_FILTER_KERNEL),
        "Kernel used for filter() [int]")

        ("compress-kernel,c",
        po::value<int>()->default_value(DEFAULT_COMPRESS_KERNEL),
        "Kernel used for compress() [int]")


        ("inputfile,i", 
        po::value<std::vector<std::string>>(), 
        "input file containing graph data. Filepath must be relative to Ex2/")

        ("ouputfile_timings,t", 
        po::value<std::vector<std::string>>(), 
        "Output file for timings. Filepath must be relative to Ex2/");
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
  srand(0);

  EdgeList edgelist(misc::get_input_file());
  EdgeList MST_reference(misc::get_gt_file());
  std::cout << edgelist.size() << " Edges loaded" << std::endl;

  // Perform MST Calculation
  EdgeList MST = calculate_mst(edgelist);

  // Print timings to console
  std::cout << std::endl << "Benchmark results:" << std::endl;
  g_benchmarker.print_timings();

  // Export timings if specified
  if (g_options.count("ouputfile_timings"))
  {
    g_benchmarker.export_csv(g_options["ouputfile_timings"].as<std::vector<std::string>>()[0]);
  }

  // Check Solution
  // TODO: perform full check. i.e. verify all data not just the sum of weigths
  if (MST.weigth() != MST_reference.weigth())
  {
    std::cout << "Error! Weight of MST does not match reference" << std::endl;
    std::cout << "Calculated: " << MST.weigth() << std::endl;
    std::cout << "Reference:" <<  MST_reference.weigth() << std::endl;
    throw std::runtime_error("Weight of MST does not match reference");
  }

  return 0;
}
