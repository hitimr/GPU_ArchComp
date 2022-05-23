#pragma once
#include <vector>


template <typename ARGS> void sort(SortKernel kernel, ARGS &&args);
void gpu_bubble_sort_mult(std::vector<int> &vec, std::vector<int> &v2, std::vector<int> &v3);