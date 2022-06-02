#pragma once
#include "edgelist.hpp"
#include "union_find.hpp"
#include <vector>

void partition(EdgeList &E, EdgeList &E_leq, EdgeList &E_ge, int threshold, int kernel);
void partition_cpu_naive(const EdgeList &E, EdgeList &E_leq, EdgeList &E_ge, int threshold);
void exclusive_scan(int const *input, int *output, int N);
void partition_inclusive_scan(EdgeList &E, EdgeList &E_leq, EdgeList &E_ge, int threshold);


void filter(EdgeList &E, UnionFind &P, int kernel);
void filter_cpu_naive(EdgeList &E, UnionFind &P);
void filter_gpu_naive(EdgeList &E, UnionFind &P);