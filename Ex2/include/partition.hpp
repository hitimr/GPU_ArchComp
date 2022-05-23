#pragma once
#include "edgelist.hpp"
#include <vector>

void partition(const EdgeList &E, EdgeList &E_leq, EdgeList &E_ge, int threshold, int kernel);

void partition_cpu_naive(const EdgeList &E, EdgeList &E_leq, EdgeList &E_ge, int threshold);

void exclusive_scan(int const *input, int *output, int N);
std::vector<std::vector<int>> partition_inclusive_scan(std::vector<int> &vec, int threshold);