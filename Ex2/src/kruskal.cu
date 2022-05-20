
#include "union_find.hpp"
#include "sort.hpp"
#include <cassert>
#include <vector>

// std::vector<int> kruskal(std::vector<int> &coo1, std::vector<int> &coo2, std::vector<int> &val,
// const size_t num_nodes, bool debug = false){
std::vector<int> kruskal(std::vector<int> &coo1, std::vector<int> &coo2, std::vector<int> &val,
                         UnionFind &P, std::vector<int> &T)
{

  assert((coo1.size() == coo2.size()) && (coo1.size() == val.size()));

  int num_nodes = T.size() + 1;
  // std::vector<int> T(num_nodes - 1, -1);
  // UnionFind P(num_nodes);

  // this will sort all three arrays according to the values in the first one
  gpu_bubble_sort_mult(val, coo1, coo2);
  

#ifdef DEBUG
      std::vector<int> find;
  find.resize(num_nodes);
#endif

  // grow MST
  int tree_pos = 0;
  for (size_t i = 0; i < val.size(); ++i)
  {
#ifdef DEBUG
    for (int i = 0; i < find.size(); ++i)
      find[i] = P.find(i);
    int find1 = P.find(coo1[i]);
    int find2 = P.find(coo2[i]);
#endif

    if (P.find(coo1[i]) != P.find(coo2[i]))
    {
      T[tree_pos] = i;
      tree_pos++;
      P.my_union(coo1[i], coo2[i]);
    }
  }
  return T;
}