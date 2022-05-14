# pragma once
#include "assert.h"
#include <vector>
#include <iostream>
#include <tuple>

#define BLOCK_SIZE 256 
#define GRID_SIZE 256 


//--------------------------------------
//       sort one single array
//--------------------------------------
__device__ void gpu_swap(int* vec, size_t i, size_t j){
    int temp = vec[i];
    vec[i] = vec[j];
    vec[j] = temp;
}

__global__ void gpu_even_pass(int* vec, size_t size){

    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(size_t i = thread_id; i < size_t(size/2); i += num_threads){
        if (vec[2*i] > vec[2*i+1])
            gpu_swap(vec, 2*i, 2*i+1);
    }
}

__global__ void gpu_odd_pass(int* vec, size_t size){

    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(size_t i = thread_id; i < size_t(size/2 - 1); i += num_threads){
        if (vec[2*i+1] > vec[2*i+2])
            gpu_swap(vec, 2*i+1, 2*i+2);
    }
}

void gpu_bubble_sort(std::vector<int> &vec){

    size_t size = vec.size(); 
    int num_bytes = vec.size() * sizeof(int);

    // allocate
    int* d_vec;
    cudaMalloc((void**)&d_vec, num_bytes);

    // copy
    cudaMemcpy(d_vec, vec.data(), num_bytes, cudaMemcpyHostToDevice);

    // sort
    for(size_t i = 0; i < size_t(size/2); ++i){
        gpu_even_pass<<<BLOCK_SIZE, GRID_SIZE>>>(d_vec, size);
        cudaDeviceSynchronize();            
        gpu_odd_pass<<<BLOCK_SIZE, GRID_SIZE>>>(d_vec, size);
        cudaDeviceSynchronize();            
    }

    // copy back
    cudaMemcpy(vec.data(), d_vec, sizeof(int) * vec.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_vec);
}


//--------------------------------------
//       sort three arrays
//--------------------------------------
__global__ void gpu_even_pass_mult(int* vec, size_t size, int* v2, int* v3){

    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(size_t i = thread_id; i < size_t(size/2); i += num_threads){
        if (vec[2*i] > vec[2*i+1]){
            gpu_swap(vec, 2*i, 2*i+1);
            gpu_swap(v2, 2*i, 2*i+1);
            gpu_swap(v3, 2*i, 2*i+1);
        }
    }
}


__global__ void gpu_odd_pass_mult(int* vec, size_t size, int* v2, int* v3){

    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(size_t i = thread_id; i < size_t(size/2 - 1); i += num_threads){
        if (vec[2*i+1] > vec[2*i+2]){
            gpu_swap(vec, 2*i+1, 2*i+2);
            gpu_swap(v2, 2*i+1, 2*i+2);
            gpu_swap(v3, 2*i+1, 2*i+2);
        }
    }
}


void gpu_bubble_sort_mult(std::vector<int> &vec, std::vector<int> &v2, std::vector<int> &v3){

    size_t size = vec.size(); 
    int num_bytes = vec.size() * sizeof(int);

    // allocate
    int *d_vec, *d_v2, *d_v3;
    cudaMalloc((void**)&d_vec, num_bytes);
    cudaMalloc((void**)&d_v2, num_bytes);
    cudaMalloc((void**)&d_v3, num_bytes);

    // copy
    cudaMemcpy(d_vec, vec.data(), num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v2, v2.data(), num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v3, v3.data(), num_bytes, cudaMemcpyHostToDevice);

    // sort
    for(size_t i = 0; i < size_t(size/2); ++i){
        gpu_even_pass_mult<<<BLOCK_SIZE, GRID_SIZE>>>(d_vec, size, d_v2, d_v3);
        cudaDeviceSynchronize();            
        gpu_odd_pass_mult<<<BLOCK_SIZE, GRID_SIZE>>>(d_vec, size, d_v2, d_v3);
        cudaDeviceSynchronize();            
    }

    // copy back
    cudaMemcpy(vec.data(), d_vec, sizeof(int) * vec.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(v2.data(), d_v2, sizeof(int) * vec.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(v3.data(), d_v3, sizeof(int) * vec.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_vec);
    cudaFree(d_v2);
    cudaFree(d_v3);
}



// ------------------------------------------------
//                CPU - SORTING
// ------------------------------------------------

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
