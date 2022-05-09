//#pragma once
#include <iostream>

#include "graph.hpp"
#include "simonstuff.hpp"



int main()
{
    std::cout << "Hello, World" << std::endl;
    Graph input("10_0.2_1_100.csv");
    input.getCOOReepresentation();
    input.getCSRRepresentation();
    input.getELLRepresentation();

    //print_tuple(input.getCOOReepresentation());
    print_COO(input);
    my_sorting(input);
    //print_tuple(20);

    return 0;
}