#include "common.hpp"
#include "kruskal.hpp"
#include "partition.hpp"
#include "sort.hpp"
#include "union_find.hpp"
#include <cassert>
#include <stdio.h>
#include <vector>

EdgeList calculate_mst(EdgeList &edgelist)
{
  g_benchmarker.start("total");
  g_benchmarker.start("Initialize");
  UnionFind P(edgelist.num_edges);
  EdgeList T(edgelist.num_nodes - 1); // EdgeList is empty but required memory is already allocated
  g_benchmarker.stop("Initialize");

  switch (g_options["mst-kernel"].as<int>())
  {
  case MST_KERNEL_REGULAR_KRUSKAL:
    kruskal(edgelist, P, T);
    break;

  case MST_KERNEL_FILTER_KRUSKAL:
    filter_kruskal(edgelist, P, T);
    break;

  default:
    throw std::invalid_argument("Unknown MST kernel");
  }

  g_benchmarker.stop("total");
  return T;
}

bool kruskal_threshold(EdgeList &E)
{
  // TODO: remove magic number
  // arbitrary for now
  if (E.num_edges < 4096)
  {
    return true;
  }
  else
  {
    return false;
  }
}

int pivot(const EdgeList &E)
{
  // rand() is sufficient for pivot elements
  int pos = rand() % E.size();
  return E.val[pos];
}

void filter_kruskal(EdgeList &E, UnionFind &P, EdgeList &T)
{
  if (kruskal_threshold(E))
  {
    kruskal(E, P, T);
    P.compress(g_options["compress-kernel"].as<int>());
  }
  else
  {
    EdgeList E_leq; // less or equal than threshold
    EdgeList E_big; // bigger than threshold

    int p = pivot(E);
    partition(E, E_leq, E_big, p, g_options["partition-kernel"].as<int>());

    if (E_leq.size() != 0)
    {
      filter_kruskal(E_leq, P, T);
    }
    
    filter(E_big, P, g_options["filter-kernel"].as<int>());

    if (E_big.size() != 0)
    {
      filter_kruskal(E_big, P, T);
    }
  }
}

// std::vector<int> kruskal(std::vector<int> &coo1, std::vector<int> &coo2, std::vector<int> &val,
// const size_t num_nodes, bool debug = false){
void kruskal(EdgeList &E, UnionFind &P, EdgeList &T)
{
  g_benchmarker.start("Kruskal()");

  // this will sort all three arrays according to the values in the first one
  sort_edgelist(E, g_options["sort-kernel"].as<int>());

  // grow MST
  g_benchmarker.start("grow MST");
  for (size_t i = 0; i < E.num_edges; i++)
  {
    if (P.find(E.coo1[i]) != P.find(E.coo2[i]))
    {
      T.append_edge(E[i]);
      P.my_union(E.coo1[i], E.coo2[i]);
    }
  }
  g_benchmarker.stop("grow MST");
  g_benchmarker.stop("Kruskal()");
}