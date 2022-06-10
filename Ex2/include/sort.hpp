#pragma once
#include "edgelist.hpp"
#include <vector>

void sort_edgelist(EdgeList &E, int kernel);

void gpu_bubble_sort_mult(EdgeList &E);
void gpu_thrust_sort_three(std::vector<int> &vec, std::vector<int> &v2, std::vector<int> &v3);
void improved_mergesort_three(EdgeList &E);
void radix_sort(std::vector<int> &vec, std::vector<int> &vec1, std::vector<int> &vec2);