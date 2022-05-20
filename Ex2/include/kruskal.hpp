
#include "union_find.hpp"
#include <vector>

// std::vector<int> kruskal(std::vector<int> &coo1, std::vector<int> &coo2, std::vector<int> &val,
// const size_t num_nodes, bool debug = false){
std::vector<int> kruskal(std::vector<int> &coo1, std::vector<int> &coo2, std::vector<int> &val,
                         UnionFind &P, std::vector<int> &T);