#pragma once
#include "edgelist.hpp"
#include <vector>

void sort_edgelist(EdgeList &E, int kernel);

void gpu_bubble_sort_mult(EdgeList &E);
void gpu_thrust_sort_three(EdgeList &E);
void improved_mergesort_three(EdgeList &E);
void radix_sort(EdgeList &E);
void assemble_coo(EdgeList &E, int * indices);