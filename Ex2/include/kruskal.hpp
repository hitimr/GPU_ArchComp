#pragma once
#include "edgelist.hpp"
#include "union_find.hpp"
#include <vector>

void calculate_mst(EdgeList &edgelist);
void kruskal(EdgeList &edgelist, UnionFind &P, EdgeList &T);