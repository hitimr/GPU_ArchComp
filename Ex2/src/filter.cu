#include "benchmarker.hpp"
#include "common.hpp"
#include "filter.hpp"

void filter(EdgeList &E, UnionFind &P, int kernel)
{
  g_benchmarker.start("filter()");

  switch (kernel)
  {
  case FILTER_KERNEL_CPU_NAIVE:
    filter_cpu_naive(E, P);
    break;

  default:
    throw std::invalid_argument("Unknown filter kernel");
  }

  g_benchmarker.stop("filter()");
}

void filter_cpu_naive(EdgeList &E, UnionFind &P)
{
  EdgeList E_filt;
  E_filt.reserve(E.size());

  for (size_t i = 0; i < E.size(); i++)
  {
    Edge e = E[i];
    if(P.find(e.source) != P.find(e.target))
    {
        E_filt.append_edge(e);
    }    
  }

  E = E_filt;
}