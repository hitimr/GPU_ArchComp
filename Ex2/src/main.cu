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

int main(int ac, char **av)
{
  g_options = misc::parse_options(ac, av);
  g_benchmarker = Benchmarker();
  srand(0);

  // Check Device Support
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if (!deviceProp.canMapHostMemory)
  {
    throw std::runtime_error("Error! Device does not support pinned memory!");
  }

  EdgeList edgelist(misc::get_input_file());
  EdgeList MST_reference(misc::get_gt_file());
  std::cout << edgelist.size() << " Edges loaded" << std::endl;

  // Perform MST Calculation
  EdgeList MST;
  for (int i = 0; i < g_options["repetitions"].as<int>(); i++)
  {
    MST = calculate_mst(edgelist);
  }

  // Print timings to console
  std::cout << std::endl << "Benchmark results:" << std::endl;
  g_benchmarker.print_timings();

  // save solution to csv
  MST.write_to_file(misc::get_output_file());
  MST_reference.write_to_file(misc::get_reference_output_file());

  // Export timings if specified
  if (g_options.count("ouputfile_timings"))
  {
    g_benchmarker.export_csv(g_options["ouputfile_timings"].as<std::vector<std::string>>()[0]);
  }

  // Check Solution
  if (MST.weigth() != MST_reference.weigth())
  {
    std::cout << "Error! Weight of MST does not match reference" << std::endl;
    std::cout << "Calculated: " << MST.weigth() << std::endl;
    std::cout << "Reference: " << MST_reference.weigth() << std::endl;
    throw std::runtime_error("Weight of MST does not match reference");
  }

  cudaDeviceReset();

  return 0;
}
