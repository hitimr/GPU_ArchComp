//#pragma once
#include "assert.h"
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

//void print_tuple(std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> print_this){
//void print_COO(Graph print_this){
void print_COO(std::vector<int> coo1, std::vector<int> coo2, std::vector<int> val){
//    std::cout << "printing COO..." << std::endl;

    std::cout << "COO_1 | " << "COO_2 | " << "VAL" << std::endl;
    std::cout << "-------------------------------" << std::endl;

//    std::tie(coo1, coo2, val) = print_this.getCOOReepresentation();

    for(size_t i = 0; i < coo1.size(); ++i){
        std::cout << coo1[i] << " | " << coo2[i] << " | " << val[i] << std::endl;
    }

}


void print_MST(std::vector<int> coo1, std::vector<int> coo2, std::vector<int> val, std::vector<size_t> mst){
    std::cout << "COO_1 | " << "COO_2 | " << "VAL" << std::endl;
    std::cout << "-------------------------------" << std::endl;

    for(size_t i = 0; i < mst.size(); ++i){
        std::cout << coo1[mst[i]] << " | " << coo2[mst[i]] << " | " << val[mst[i]] << std::endl;
    }

}




// ----------------------------------------
//                SORTING
// ----------------------------------------

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

void my_sorting(std::vector<int> &coo1, std::vector<int> &coo2, std::vector<int> &val){
    assert((coo1.size() == coo2.size()) && (coo1.size() == val.size()));
// this sorts the three vectors inplace - i.e. vectors will be changed
    auto p = sort_permutation(val,
        [](int const& a, int const& b){ return (a < b); });

    val = apply_permutation(val, p);
    coo1 = apply_permutation(coo1, p);
    coo2 = apply_permutation(coo2, p);
}



// ----------------------------------------
//                KRUSKAL
// ----------------------------------------

void add_node_to_group(const size_t b, const size_t a, std::vector<int> &groups){
    // this will change the group of all nodes that belong to the group of node b to the group of node a
    // if b does not belong to a group, it will only set the group of b to group of a

    if (groups[b] == -1) {
        groups[b] = groups[a];
        return;
    }

    for(size_t i = 0; i < groups.size(); ++i){
        if (groups[i] == groups[b])
            groups[i] = groups[a];
    }
}


std::vector<size_t> kruskal(std::vector<int> &coo1, std::vector<int> &coo2, std::vector<int> &val, const size_t num_nodes, bool debug = false){
    assert((coo1.size() == coo2.size()) && (coo1.size() == val.size()));
    std::vector<int> groups(num_nodes,-1);

    // initialize minimal spanning tree as list of edge indices.
    std::vector<size_t> mst(num_nodes - 1, 0);

    int number_edges_found = 0;
    int group_counter = 0;

    if (debug) std::cout << "debugging..." << std::endl;

    int a, b;
    for(size_t i = 0; i < coo1.size(); ++i){
        if (debug) std::cout << "debugging... i = " << i << std::endl;

        if (number_edges_found >= num_nodes - 1) break;

        // edge between a and b
        a = coo1[i];
        b = coo2[i];

        // if they are in the same group: skip edge
        if ((groups[a] == groups[b]) && (groups[a] != -1)) {
            if (debug) std::cout << "skip edge" << std::endl; 
            continue;
        }

        // if they are both not grouped: create new group
        if ((groups[a] == groups[b]) && (groups[a] == -1)){
            mst[number_edges_found] = i;
            groups[a] = group_counter;
            groups[b] = group_counter;
            group_counter++;
            number_edges_found++;
            if (debug) std::cout << "create new group with index " << group_counter - 1 << std::endl; 
            continue;
        }

        // if one of them belongs to a group: add the other to the group (possibly swallowing their whole group)
        if (groups[a] != -1){
            mst[number_edges_found] = i;
            number_edges_found++;
            add_node_to_group(b, a, groups);
            if (debug) std::cout << "add second node to first node group" << std::endl; 
            continue;
        }

        if (groups[b] != -1){
            mst[number_edges_found] = i;
            number_edges_found++;
            add_node_to_group(a, b, groups);
            if (debug) std::cout << "add first node to second node group" << std::endl; 
            continue;
        }
    }

    return mst;
}

int total_weight(const std::vector<int> &val, const std::vector<size_t> &mst){
    int sum = 0;
    for(size_t i = 0; i < mst.size(); ++i)
        sum += val[mst[i]];
    return sum;
}