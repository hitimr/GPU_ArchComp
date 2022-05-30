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

__global__ void test_kernel(EdgeList *E)
{
  // You have access to all member variables of the class
  printf("Number of Edges (on GPU) %d \n", (int)E->num_edges);

  // you can even use class methods :D
  printf("Number of Edges (on GPU; from class method): %d \n", (int)E->size());


  // the index-operator does not work (yet) but you can use at() instead
  Edge first_edge = E->at(0);
  printf("Weight of first edge (on GPU): %i\n", (int)first_edge.weight);

  // unfortunately CUDA is not able to handle std::vectors
  // so we need to access special device-pointers for arrays or vectors
  // Lets make some changes to the List
  for (int i = 0; i < 3; i++)
    E->d_val[i] = 42;

  printf("First 3 edge weights after modification on GPU:\n");
  printf("[%i,%i,%i]\n", E->d_val[0], E->d_val[1], E->d_val[2]);
}

void gpu_interface_demo(EdgeList &E)
{
  // Currently the EdgeList resides on CPU only
  std::cout << "Number of Edges (on CPU): " << E.num_edges << std::endl;

  // To send or update a copy of E to GPU use sync_hostToDevice();
  E.sync_hostToDevice();

  // In order to pass the GPU version of the EdgeList to a kernel use E.gpu (instead of just E)
  test_kernel<<<1, 1>>>(E.gpu);
  cudaDeviceSynchronize();

  // Kernel has finished. Now we need to sync changes back to the host
  printf("First 3 edge weights before sync (from CPU):\n");
  printf("[%i,%i,%i]\n", E.val[0], E.val[1], E.val[2]);

  E.sync_deviceToHost();

  printf("First 3 edge weights after sync (from CPU):\n");
  printf("[%i,%i,%i]\n", E.val[0], E.val[1], E.val[2]);
}

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

  // gpu_interface_demo(edgelist);
  
  // Perform MST Calculation
  EdgeList MST = calculate_mst(edgelist);

  // Print timings to console
  std::cout << std::endl << "Benchmark results:" << std::endl;
  g_benchmarker.print_timings();

  int sum = MST.weigth();

  // Export timings if specified
  if (g_options.count("ouputfile_timings"))
  {
    g_benchmarker.export_csv(g_options["ouputfile_timings"].as<std::vector<std::string>>()[0]);
  }

  

  return 0;
}


