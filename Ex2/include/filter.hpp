#pragma once
#include<edgelist.hpp>
#include<union_find.hpp>


void filter(EdgeList &E, UnionFind &P, int kernel);

void filter_cpu_naive(EdgeList &E, UnionFind &P);