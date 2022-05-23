#pragma once
#include <vector>

void sort_edgelist(int kernel, std::vector<int> &val, std::vector<int> &coo1,
                   std::vector<int> &coo2);
                   
void gpu_bubble_sort_mult(std::vector<int> &vec, std::vector<int> &v2, std::vector<int> &v3);