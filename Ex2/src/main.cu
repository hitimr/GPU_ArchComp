#include <iostream>

#include "graph.hpp"

int main()
{
    std::cout << "Hello, World" << std::endl;
    Graph input("10_0.2_1_100.csv");
    input.getCOOReepresentation();
    input.getCSRRepresentation();
    input.getELLRepresentation();
    return 0;
}