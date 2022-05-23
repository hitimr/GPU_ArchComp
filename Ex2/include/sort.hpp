#pragma once
#include "edgelist.hpp"
#include <vector>

void sort_edgelist(EdgeList &E, int kernel);

void gpu_bubble_sort_mult(std::vector<int> &vec, std::vector<int> &v2, std::vector<int> &v3);