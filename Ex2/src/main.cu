//#pragma once
#include <iostream>

#include "graph.hpp"
#include "simonstuff.hpp"
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

    std::cout << "printing input graph..." << std::endl;
    print_COO(coo1, coo2, val);
    std::cout << "run sorting..." << std::endl;
    my_sorting(coo1, coo2, val);
    std::cout << "printing sorted graph..." << std::endl;
    print_COO(coo1, coo2, val);

    std::cout << "intput.size = " << input.size << std::endl;
    std::cout << "run kruskal..." << std::endl;
    std::vector<size_t> mst = kruskal(coo1, coo2, val, input.size);
    std::cout << "printing mst as edge indices..." << std::endl;
    print_vector(mst);
    std::cout << "printing mst as edges..." << std::endl;
    print_MST(coo1, coo2, val, mst);
    std::cout << "total weight: " <<  total_weight(val,mst) << std::endl;

    return 0;
}