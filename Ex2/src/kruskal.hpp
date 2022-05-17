//#pragma once
#include "assert.h"
#include <vector>
#include <iostream>
#include <tuple>

// for the sorting
#include <algorithm>
#include <numeric>

#include "graph.hpp"
#include "gpu_sorting.cu"

//#define DEBUG


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

template<typename T>
void print_MST(std::vector<int> coo1, std::vector<int> coo2, std::vector<int> val, std::vector<T> mst){
    std::cout << "COO_1 | " << "COO_2 | " << "VAL" << std::endl;
    std::cout << "-------------------------------" << std::endl;

    for(size_t i = 0; i < mst.size(); ++i){
        std::cout << coo1[mst[i]] << " | " << coo2[mst[i]] << " | " << val[mst[i]] << std::endl;
    }

}




// ----------------------------------------
//              CPU SORTING
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
//                NEW KRUSKAL
// ----------------------------------------

class UnionFind
{
    private: 
    std::vector<int> parent;

    public:
    UnionFind(size_t size){
        parent.resize(size);
        for(size_t i = 0; i < parent.size(); ++i)
            parent[i] = i;
    }

    int find(int i){
        if (parent[i] == i) 
            return i;
        else 
            return find(parent[i]);
    }

    void link(int i, int j){
        assert(find(i) != find(j));
        parent[i] = j;
    }

    void my_union(int i, int j){
        if (find(i) != find(j)) 
            link(find(i),find(j));
    }
};


//std::vector<int> kruskal(std::vector<int> &coo1, std::vector<int> &coo2, std::vector<int> &val, const size_t num_nodes, bool debug = false){
std::vector<int> kruskal(std::vector<int> &coo1, std::vector<int> &coo2, std::vector<int> &val, UnionFind &P, std::vector<int> &T){

    assert((coo1.size() == coo2.size()) && (coo1.size() == val.size()));

    int num_nodes = T.size() + 1;
    //std::vector<int> T(num_nodes - 1, -1);
    //UnionFind P(num_nodes);
    gpu_bubble_sort_mult(val,coo1,coo2); // this will sort all three arrays according to the values in the first one

    #ifdef DEBUG
        std::vector<int> find;
        find.resize(num_nodes);
    #endif

    // grow MST
    int tree_pos = 0;
    for(size_t i = 0; i < val.size(); ++i){
        #ifdef DEBUG
            for(int i = 0; i < find.size(); ++i)
                find[i] = P.find(i);
            int find1 = P.find(coo1[i]);
            int find2 = P.find(coo2[i]);
        #endif

        if (P.find(coo1[i]) != P.find(coo2[i])){
            T[tree_pos] = i;
            tree_pos++;
            P.my_union(coo1[i], coo2[i]);
        }
    }
    return T;

/*
    // Sort all edges by weight
    sorted = E.sort_values(by="weight", ascending=True)

    # grow MST
    for index, edge in sorted.iterrows():
        if P.find(edge["u"]) != P.find(edge["v"]):
            T = T.append(edge)
            P.union(edge["u"], edge["v"])
    T = T.sort_values(by=["u", "v"]).astype(int)
    return E, T.reindex(), P
*/
}


// ----------------------------------------
//                OLD KRUSKAL
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


std::vector<size_t> kruskal_old(std::vector<int> &coo1, std::vector<int> &coo2, std::vector<int> &val, const int num_nodes, bool debug = false){
    assert((coo1.size() == coo2.size()) && (coo1.size() == val.size()));

    gpu_bubble_sort_mult(val,coo1,coo2);

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


template <typename T>
int total_weight(const std::vector<int> &val, const std::vector<T> &mst){
    int sum = 0;
    for(size_t i = 0; i < mst.size(); ++i)
        sum += val[mst[i]];
    return sum;
}

// T is somtimes int sometimes size_t
template <typename T>
int count_nodes_MST(const std::vector<int> &coo1, const std::vector<int> &coo2, const std::vector<T> &mst, const int num_nodes){
    // this counts the nodes of a spanning tree
    std::vector<bool> check(num_nodes,false);

    for(size_t i = 0; i < mst.size(); ++i){
        check[coo1[mst[i]]] = true;
        check[coo2[mst[i]]] = true;
    }
    int sum = 0;
    for(size_t i = 0; i < check.size(); ++i)
        sum += check[i];    
    return sum;
}