#include "common.hpp"
#include "kruskal.hpp"
#include "sort.hpp"
#include "union_find.hpp"
#include <cassert>
#include <vector>

void calculate_mst(EdgeList &edgelist)
{
  g_benchmarker.start("calculate_mst()");

  g_benchmarker.start("Initialize Data structures");
  UnionFind P(edgelist.num_edges);
  EdgeList T(edgelist.num_nodes-1); // EdgeList is empty but required memory is already allocated
  g_benchmarker.stop("Initialize Data structures");


  switch (g_options["mst-kernel"].as<int>())
  {
  case MST_KERNEL_REGULAR_KRUSKAL:
    kruskal(edgelist, P, T);
    break;

  default:
    throw std::invalid_argument("Unknown MST kernel");
  }

  g_benchmarker.stop("calculate_mst()");
}

// std::vector<int> kruskal(std::vector<int> &coo1, std::vector<int> &coo2, std::vector<int> &val,
// const size_t num_nodes, bool debug = false){
void kruskal(EdgeList &E, UnionFind &P, EdgeList &T)
{
  g_benchmarker.start("Kruskal()");

  // this will sort all three arrays according to the values in the first one
  sort_edgelist(E, g_options["sort-kernel"].as<int>());

  // grow MST
  for (size_t i = 0; i < E.num_edges; i++)
  {
    if (P.find(E.coo1[i]) != P.find(E.coo2[i]))
    {
      T.append_edge(E[i]);
      P.my_union(E.coo1[i], E.coo2[i]);
    }
  }
  
  g_benchmarker.stop("Kruskal()");
}