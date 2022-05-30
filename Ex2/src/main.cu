#include "common.hpp"
#include "edgelist.hpp"
#include "kruskal.hpp"
#include "union_find.hpp"
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

/*
int main(int ac, char **av)
{

  
  g_options = parse_options(ac, av);
  g_benchmarker = Benchmarker();
  srand(0);

  // Load input file
  // I havent found a good way for adding defaults tring to boost::options so im doing it by hand
  std::string input_file = misc::get_proj_root_dir().append(
      g_options.count("inputfile") ? g_options["inputfile"].as<std::vector<std::string>>()[0]
                                   : DEFAULT_INPUT_FILE);
  std::cout << "Loading " << input_file << std::endl;
  EdgeList edgelist(input_file.c_str());

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
  
  return 0;
}
*/


// this alternative main tests the UnionFindPC class
int main(){

  int size = 9;
  //UnionFind P(size);
  UnionFindPC P(size);

  for(int i = 0; i < 4; ++i){
    P.link(i, i+1);
  }

  for(int i = 5; i < size - 1; ++i){
    P.link(i, i+1);
  }

  for(int i = 0; i < size; ++i){
    std::cout << "i: " << i << ", parent: " << P.get_parent(i) << std::endl;
  }

  std::cout << "now call find ..." << std::endl;
  std::cout << "find(0): " << P.find(0) << std::endl;
  std::cout << "find(5): " << P.find(5) << std::endl;

  for(int i = 0; i < size; ++i){
    std::cout << "i: " << i << ", parent: " << P.get_parent(i) << std::endl;
  }

  return 0;
}
