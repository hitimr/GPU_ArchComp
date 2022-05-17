//#pragma once
#include <iostream>

#include "graph.hpp"
#include "kruskal.hpp"
#include "gpu_sorting.cu"
#include "assert.h"


int main()
{
    std::cout << "Hello, World" << std::endl;
    Graph input("10_0.2_1_100.csv");
    input.getCOOReepresentation();
    input.getCSRRepresentation();
    input.getELLRepresentation();

    std::vector<int> coo1, coo2, val;
    std::tie(coo1, coo2, val) = input.getCOOReepresentation();

    std::cout << std::endl;
    std::cout << "printing input graph..." << std::endl;
    print_COO(coo1, coo2, val);

    std::cout << std::endl;
    std::cout << "---------------------------------------------"<< std::endl;
    std::cout << "run sorting..." << std::endl;
    //my_sorting(coo1, coo2, val);
    gpu_bubble_sort_mult(val,coo1,coo2);
    std::cout << "printing sorted graph..." << std::endl;
    print_COO(coo1, coo2, val);

    std::cout << std::endl;
    std::cout << "---------------------------------------------"<< std::endl;
    std::cout << "intput.size = " << input.size << std::endl;
    std::cout << "run old kruskal..." << std::endl;
    int num_nodes = input.size;
    std::vector<size_t> mst = kruskal_old(coo1, coo2, val, num_nodes);

    std::cout << std::endl;
    std::cout << "printing mst as edges..." << std::endl;
    print_MST(coo1, coo2, val, mst);
    std::cout << "total weight: " <<  total_weight(val,mst) << std::endl;
    std::cout << "count nodes of the MST: " <<  count_nodes_MST(coo1, coo2, mst, num_nodes) << std::endl;

/*
    std::cout << std::endl;
    std::cout << "printing mst as edge indices..." << std::endl;
    print_vector(mst);
*/

    std::tie(coo1, coo2, val) = input.getCOOReepresentation();
    std::cout << std::endl;
    std::cout << "---------------------------------------------"<< std::endl;
    std::cout << "intput.size = " << input.size << std::endl;
    std::cout << "run new kruskal..." << std::endl;

    UnionFind P(num_nodes);
    std::vector<int> T(num_nodes - 1, -1);
    std::vector<int> mst2 = kruskal(coo1, coo2, val, P, T);

    std::cout << std::endl;
    std::cout << "printing mst as edges..." << std::endl;
    print_MST(coo1, coo2, val, mst2);
    std::cout << "total weight: " <<  total_weight(val,mst2) << std::endl;

    return 0;
}