#pragma once
#include "edgelist.hpp"
#include <vector>

void partition(EdgeList &E, EdgeList &E_leq, EdgeList &E_ge, int threshold, int kernel);

void partition_cpu_naive(const EdgeList &E, EdgeList &E_leq, EdgeList &E_ge, int threshold);

void exclusive_scan(int const *input, int *output, int N);
void partition_inclusive_scan(EdgeList &E, EdgeList &E_leq, EdgeList &E_ge, int threshold);