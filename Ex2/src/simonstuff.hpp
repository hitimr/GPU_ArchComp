//#pragma once
#include <vector>
#include <iostream>
#include <tuple>

// for the sorting
#include <algorithm>
#include <numeric>

#include "graph.hpp"


template <typename T>
void print_vector(std::vector<T> print_this){
    for(size_t i = 0; i < print_this.size(); ++i){
        std::cout << print_this[i] << std::endl;
    }
}

// sorting several vectors by one of them... taken from
//  https://stackoverflow.com/questions/17074324/how-can-i-sort-two-vectors-in-the-same-way-with-criteria-that-uses-only-one-of

template <typename T, typename Compare>
std::vector<std::size_t> sort_permutation(
    const std::vector<T>& vec,
    Compare compare)
{
    std::vector<std::size_t> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
        [&](std::size_t i, std::size_t j){ return compare(vec[i], vec[j]); });
    return p;
}

template <typename T>
std::vector<T> apply_permutation(
    const std::vector<T>& vec,
    const std::vector<std::size_t>& p)
{
    std::vector<T> sorted_vec(vec.size());
    std::transform(p.begin(), p.end(), sorted_vec.begin(),
        [&](std::size_t i){ return vec[i]; });
    return sorted_vec;
}

/* 
Example use
vector<MyObject> vectorA;
vector<int> vectorB;

auto p = sort_permutation(vectorA,
    [](T const& a, T const& b){ *some comparison* });

vectorA = apply_permutation(vectorA, p);
vectorB = apply_permutation(vectorB, p);
*/

void my_sorting(Graph some_graph){

    std::vector<int> c1, c2, val; 
    std::tie(c1, c2, val) = some_graph.getCOOReepresentation();

    auto p = sort_permutation(val,
        [](int const& a, int const& b){ return (a < b); });

    val = apply_permutation(val, p);
    c1 = apply_permutation(c1, p);
    c2 = apply_permutation(c2, p);


    std::cout << "printing sorted..." << std::endl;
    std::cout << "COO_1 | " << "COO_2 | " << "VAL" << std::endl;
    std::cout << "-------------------------------" << std::endl;

    //std::tie(coo1, coo2, val) = print_this.getCOOReepresentation();

    for(size_t i = 0; i < c1.size(); ++i){
        std::cout << c1[i] << " | " << c2[i] << " | " << val[i] << std::endl;
    }
}




//void print_tuple(std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> print_this){
void print_COO(Graph print_this){
//void print_tuple(int someint){
//    std::cout << "this is some int: " << someint << std::endl;
    std::cout << "printing tuple..." << std::endl;

    std::vector<int> coo1, coo2, val; 
    //coo1 = std::get<0>(print_this.getCOOReepresentation());

    std::cout << "COO_1 | " << "COO_2 | " << "VAL" << std::endl;
    std::cout << "-------------------------------" << std::endl;

    std::tie(coo1, coo2, val) = print_this.getCOOReepresentation();

    for(size_t i = 0; i < coo1.size(); ++i){
        std::cout << coo1[i] << " | " << coo2[i] << " | " << val[i] << std::endl;
    }


/*
    temp = std::get<0>(print_this.getCOOReepresentation());
    std::cout << "first vector: " << std::endl;
    std::cout << "num elements: " << temp.size() << std::endl;
    print_vector(temp);

    temp = std::get<1>(print_this.getCOOReepresentation());
    std::cout << "second vector: " << std::endl;
    std::cout << "num elements: " << temp.size() << std::endl;
    print_vector(temp);

    temp = std::get<2>(print_this.getCOOReepresentation());
    std::cout << "third vector: " << std::endl;
    std::cout << "num elements: " << temp.size() << std::endl;
    print_vector(temp);

*/

}