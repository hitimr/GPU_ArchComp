#include "common.hpp"
#include "edgelist.hpp"
#include "kruskal.hpp"
#include "misc.hpp"
#include "partition.hpp"

#include <cassert>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>

Benchmarker g_benchmarker;
OptionsT g_options;

int main(int ac, char **av)
{

  //   po::store(po::parse_command_line(ac, av), options);
  g_options = misc::parse_options(ac, av);

  std::vector<int> kernels = {FILTER_KERNEL_CPU_NAIVE, FILTER_KERNEL_GPU, FILTER_KERNEL_THRUST};

  for (int kernel : kernels)
  {
    // Init
    EdgeList E(misc::get_input_file());
    EdgeList E_test(misc::get_input_file());
    UnionFind P(E.num_edges);
    UnionFind P_test(E.num_edges);
    EdgeList T(E.num_nodes - 1); // EdgeList is empty but required memory is already allocated
    EdgeList T_test(E.num_nodes - 1);

    EdgeList E_leq;      // less or equal than threshold
    EdgeList E_big;      // bigger than threshold
    EdgeList E_leq_test; // less or equal than threshold
    EdgeList E_big_test; // bigger than threshold

    int p = pivot(E);
    partition(E, E_leq, E_big, p, g_options["partition-kernel"].as<int>());
    partition(E_test, E_leq_test, E_big_test, p, g_options["partition-kernel"].as<int>());

    if (E_leq.size() != 0)
    {
      kruskal(E_leq, P, T);
      kruskal(E_leq_test, P_test, T_test);
    }

    E_big.sync_hostToDevice();
    E_big_test.sync_hostToDevice();

    filter(E_big, P, FILTER_KERNEL_CPU_NAIVE);
    filter(E_big_test, P_test, kernel);

    E_big.sync_deviceToHost();
    E_big_test.sync_deviceToHost();

    assert(E_big.size() == E_big_test.size());

    for (size_t i = 0; i < E_big.size(); i++)
    {
      assert(E_big[i].weight == E_big_test[i].weight);
      assert(E_big[i].source == E_big_test[i].source);
      assert(E_big[i].target == E_big_test[i].target);
    }
  }

  return 0;
}
