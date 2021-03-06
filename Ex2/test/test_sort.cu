#include "common.hpp"
#include "edgelist.hpp"
#include "sort.hpp"
#include "misc.hpp"

#include <cassert>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>

#define TEST_SIZE (int)1e6
#define TEST_SIZE_BUBBLE_SORT (int)1e3

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

void test_sort_kernel(int kernel)
{
  EdgeList E;

  if (kernel == SORT_KERNEL_GPU_BUBBLE_MULT)
  {
    E.resize_and_set_num_edges(TEST_SIZE_BUBBLE_SORT);
  }
  else
  {
    E.resize_and_set_num_edges(TEST_SIZE);
  }

  // Fill EdgeList with 1,2,3...
  std::iota(&E.coo1[0], &E.coo1[E.size()], 0);
  std::iota(&E.coo2[0], &E.coo2[E.size()], 0);
  std::iota(&E.val[0], &E.val[E.size()], 0);

  // Shuffle
  srand(0);
  for (int n = 0; n < TEST_SIZE; n++)
  {
    int i = rand() % E.size();
    int j = rand() % E.size();

    swap(E, i, j);
  }

  sort_edgelist(E, kernel);
  E.sync_deviceToHost();

  // verify
  for (size_t i = 0; i < E.size(); i++)
  {
    assert(E.coo1[i] == i);
    assert(E.coo2[i] == i);
    assert(E.val[i] == i);
  }

  return;
}

int main(int ac, char **av)
{

  // po::store(po::parse_command_line(ac, av), options);
  g_options = misc::parse_options(ac, av);

  std::vector<int> kernels = {SORT_KERNEL_GPU_BUBBLE_MULT, SORT_KERNEL_MERGE_SORT,
                              SORT_KERNEL_THRUST, SORT_KERNEL_RADIX};

  for (auto kernel : kernels)
  {
    test_sort_kernel(kernel);
    std::cout << "Kernel " << kernel << "/" << kernels.size() << " passed" << std::endl;
  }

  return 0;
}