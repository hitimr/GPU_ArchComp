#pragma once
#include "edgelist.hpp"
#include "union_find.hpp"
#include <vector>

// general wrapper for MST algorithms
EdgeList calculate_mst(EdgeList &edgelist);

// regular kruskal algorithm
void kruskal(EdgeList &edgelist, UnionFind &P, EdgeList &T);

// filter kruskal algorithm
void filter_kruskal(EdgeList &edgelist, UnionFind &P, EdgeList &T);

// subroutine required for filter_kruskal
bool kruskal_threshold(EdgeList &E);

int pivot(const EdgeList &E);