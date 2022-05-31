#pragma once
#include<edgelist.hpp>
#include<union_find.hpp>


void filter(EdgeList &E, UnionFind &P, int kernel);

void filter_cpu_naive(EdgeList &E, UnionFind &P);

void filter_gpu_naive(EdgeList &E, UnionFind &P);

void remove_if(EdgeList &E, EdgeList &E_new, UnionFind *P);