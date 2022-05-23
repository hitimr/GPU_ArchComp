#pragma once
#include "graph.hpp"
#include "union_find.hpp"
#include <vector>

void calculate_mst(Graph &graph);
std::vector<int> kruskal(Graph &graph, UnionFind &P, std::vector<int> &T);