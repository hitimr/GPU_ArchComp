# pragma once
#include <iostream>


__global__ void check_array(int* vec, int* smaller, int* greater, size_t size, int threshold){

    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(size_t i = thread_id; i < size; i += num_threads){
        smaller[i] = vec[i] <= threshold;
        greater[i] = vec[i] > threshold;
    }
}