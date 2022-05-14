# pragma once
#include "assert.h"
#include <vector>
#include <iostream>
#include <tuple>


template<typename T>
bool check_sorting(std::vector<T> &vec){
    for(size_t i = 0; i < vec.size() - 1; ++i)
        if(vec[i] > vec[i+1]) return false;
    return true;
}

template<typename T>
void swap(std::vector<T> &vec, size_t i, size_t j){
    T temp = vec[i];
    vec[i] = vec[j];
    vec[j] = temp;
}

template<typename T>
void even_pass(std::vector<T> &vec){
    for(size_t i = 0; i < size_t(vec.size()/2); ++i)
        if (vec[2*i] > vec[2*i+1])
            swap(vec, 2*i, 2*i+1);
}

template<typename T>
void odd_pass(std::vector<T> &vec){
    for(size_t i = 0; i < size_t(vec.size()/2 - 1); ++i)
        if (vec[2*i+1] > vec[2*i+2])
            swap(vec, 2*i+1, 2*i+2);
}

template<typename T>
void bubble_sort(std::vector<T> &vec){
    for(size_t i = 0; i < size_t(vec.size()/2); ++i){
        even_pass(vec);
        odd_pass(vec);
    }
}