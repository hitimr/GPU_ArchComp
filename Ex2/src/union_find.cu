#include "common.hpp"
#include "union_find.hpp"


void UnionFind::compress(int kernel)
{
    switch(kernel)
    {
    case COMPRESS_KERNEL_CPU_NAIVE:
        //compress_cpu_nbaive();
        break;

    default:
        throw std::invalid_argument("Unknown compress kernel");
    }        
}

