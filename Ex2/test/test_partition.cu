#include "common.hpp"
#include "edgelist.hpp"
#include "misc.hpp"
#include "partition.hpp"

#include <cassert>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>

#define TEST_SIZE (int)10

Benchmarker g_benchmarker;
OptionsT g_options;

void swap(EdgeList &E, int i, int j)
{
  Edge edge_i = E[i];
  Edge edge_j = E[j];

  E.val[i] = edge_j.weight;
  E.coo1[i] = edge_j.source;
  E.coo2[i] = edge_j.target;

  E.val[j] = edge_i.weight;
  E.coo1[j] = edge_i.source;
  E.coo2[j] = edge_i.target;
}

void test_partition_kernel(int kernel)
{
  EdgeList E, E_leq, E_big;

  E.resize_and_set_num_edges(TEST_SIZE);

  // Fill EdgeList with 1,2,3...
  std::iota(&E.coo1[0], &E.coo1[E.size()], 0);
  std::iota(&E.coo2[0], &E.coo2[E.size()], -10);
  std::iota(&E.val[0], &E.val[E.size()], 0);

  // Shuffle
  /*
  srand(0);
  for (int n = 0; n < TEST_SIZE; n++)
  {
    int i = rand() % E.size();
    int j = rand() % E.size();

    swap(E, i, j);
  }
  */

  int thresh = E.size() / 2;
  partition(E, E_leq, E_big, thresh, kernel);
  E.sync_deviceToHost();

  assert(E_big.size() + E_leq.size() == E.size());

  // verify
  for (size_t i = 0; i < E_leq.size(); i++)
  {
    assert(E_leq.val[i] <= thresh);
  }

  for (size_t i = 0; i < E_big.size(); i++)
  {
    assert(E_big.val[i] > thresh);
  }

  return;
}

int main(int ac, char **av)
{

  // po::store(po::parse_command_line(ac, av), options);
  g_options = misc::parse_options(ac, av);

  std::vector<int> kernels = {PARTITION_KERNEL_GPU, PARTITION_KERNEL_CPU_NAIVE,
                              PARTITION_KERNEL_STREAMS, PARTITION_KERNEL_THRUST};

  for (auto kernel : kernels)
  {
    test_partition_kernel(kernel);
    std::cout << "Kernel " << kernel << "/" << kernels.size() << " passed" << std::endl;
  }

  return 0;
}