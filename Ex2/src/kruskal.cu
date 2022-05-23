#include "common.hpp"
#include "kruskal.hpp"
#include "sort.hpp"
#include "union_find.hpp"

#include <cassert>
#include <vector>

void calculate_mst(Graph &graph)
{
  g_benchmarker.start("calculate MST");

  g_benchmarker.start("Initialize Union-Find DS");
  UnionFind P(graph.numEdges());
  g_benchmarker.stop("Initialize Union-Find DS");
  
  std::vector<int> T(graph.numVertices() - 1, 1);

  switch (g_options["mst-kernel"].as<int>())
  {
  case MST_KERNEL_REGULAR_KRUSKAL:
    kruskal(graph, P, T);
    break;

  default:
    throw std::invalid_argument("Unknown MST kernel");
  }

  g_benchmarker.stop("calculate MST");
}

// std::vector<int> kruskal(std::vector<int> &coo1, std::vector<int> &coo2, std::vector<int> &val,
// const size_t num_nodes, bool debug = false){
std::vector<int> kruskal(Graph &graph, UnionFind &P, std::vector<int> &T)
{
  std::vector<int> coo1 = graph.getCoo1();
  std::vector<int> coo2 = graph.getCoo2();
  std::vector<int> val = graph.getWeights();

  assert((coo1.size() == coo2.size()) && (coo1.size() == val.size()));

  int num_nodes = T.size() + 1;
  // std::vector<int> T(num_nodes - 1, -1);
  // UnionFind P(num_nodes);

  // this will sort all three arrays according to the values in the first one
  sort_edgelist(g_options["sort-kernel"].as<int>(), val, coo1, coo2);

#ifdef DEBUG
  std::vector<int> find;
  find.resize(num_nodes);
#endif

  // grow MST
  int tree_pos = 0;
  for (size_t i = 0; i < val.size(); ++i)
  {
    if (P.find(coo1[i]) != P.find(coo2[i]))
    {
      T[tree_pos] = i;
      tree_pos++;
      P.my_union(coo1[i], coo2[i]);
    }
  }
  return T;
}