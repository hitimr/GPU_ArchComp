# pragma once
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>

#define BLOCKSIZE 256
#define GRIDSIZE 256


__global__ void check_array(int* vec, int* smaller, int* greater, size_t size, int threshold){

    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(size_t i = thread_id; i < size; i += num_threads){
        smaller[i] = vec[i] <= threshold;
        greater[i] = vec[i] > threshold;
    }
}

__global__ void create_partitioned_array(int* values, int* truth_values, int* scanned_values, int* new_array, size_t size)
{
    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(size_t i = thread_id; i < size; i += num_threads){
        if(truth_values[i])
          new_array[scanned_values[i]-1] = values[i];
    } 
}

// first kernel for exclusive scan
__global__ void scan_kernel_1(int const *X,
                              int *Y,
                              int N,
                              int *carries)
{
  __shared__ double shared_buffer[BLOCKSIZE];
  int my_value;
 
  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);
  unsigned int block_offset = 0;
 
  // run scan on each section
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
  {
    // load data:
    my_value = (i < N) ? X[i] : 0;
 
    // inclusive scan in shared buffer:
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
      __syncthreads();
      shared_buffer[threadIdx.x] = my_value;
      __syncthreads();
      if (threadIdx.x >= stride)
        my_value += shared_buffer[threadIdx.x - stride];
    }
    __syncthreads();
    shared_buffer[threadIdx.x] = my_value;
    __syncthreads();
 
    // exclusive scan requires us to write a zero value at the beginning of each block
    my_value = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;
 
    // write to output array
    if (i < N)
      Y[i] = block_offset + my_value;
 
    block_offset += shared_buffer[blockDim.x-1];
  }
 
  // write carry:
  if (threadIdx.x == 0)
    carries[blockIdx.x] = block_offset;
 
}
 
// second kernel for exclusive scan - exclusive-scan of carries
__global__ void scan_kernel_2(int *carries)
{
  __shared__ int shared_buffer[BLOCKSIZE];
 
  // load data:
  double my_carry = carries[threadIdx.x];
 
  // exclusive scan in shared buffer:
 
  for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
  {
    __syncthreads();
    shared_buffer[threadIdx.x] = my_carry;
    __syncthreads();
    if (threadIdx.x >= stride)
      my_carry += shared_buffer[threadIdx.x - stride];
  }
  __syncthreads();
  shared_buffer[threadIdx.x] = my_carry;
  __syncthreads();
 
  // write to output array
  carries[threadIdx.x] = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;
}
 
__global__ void scan_kernel_3(int *Y, int N,
                              int const *carries)
{
  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);
 
  __shared__ int shared_offset;
 
  if (threadIdx.x == 0)
    shared_offset = carries[blockIdx.x];
 
  __syncthreads();
 
  // add offset to each element in the block:
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
    if (i < N)
      Y[i] += shared_offset;
}

 __global__ void makeInclusive(int *Y, int N, const int *X)
 {
     for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < N-1; i += gridDim.x * blockDim.x) {
        Y[i] = Y[i+1];
    }
    if (blockDim.x * blockIdx.x + threadIdx.x == 0)
        Y[N-1] += X[N-1];
 }

 
// all scan kernels from many cores lecture from Dr. Rupp so far
void exclusive_scan(int const * input,
                    int       * output, int N)
{
 
  int *carries;
  cudaMalloc(&carries, sizeof(int) * GRIDSIZE);
 
  // First step: Scan within each thread group and write carries
  scan_kernel_1<<<GRIDSIZE, BLOCKSIZE>>>(input, output, N, carries);
 
  // Second step: Compute offset for each thread group (exclusive scan for each thread group)
  scan_kernel_2<<<1, GRIDSIZE>>>(carries);
 
  // Third step: Offset each thread group accordingly
  scan_kernel_3<<<GRIDSIZE, BLOCKSIZE>>>(output, N, carries);

  // Make inclusive
  makeInclusive<<<GRIDSIZE, BLOCKSIZE>>>(output, N, input);
 
  cudaFree(carries);
}


std::vector<std::vector<int>> partition_on_condition(std::vector<int> &vec, int threshold)
{
    size_t size = vec.size();
    int num_bytes = vec.size() * sizeof(int);

    // allocate
    int *d_vec, *d_v2, *d_v3, *d_v4, *d_v5, *d_v6, *d_v7;
    cudaMalloc((void**)&d_vec, num_bytes);
    cudaMalloc((void**)&d_v2, num_bytes);
    cudaMalloc((void**)&d_v3, num_bytes);
    cudaMalloc((void**)&d_v4, num_bytes);
    cudaMalloc((void**)&d_v5, num_bytes);

    cudaMemcpy(d_vec, vec.data(), num_bytes, cudaMemcpyHostToDevice);

    check_array<<<GRIDSIZE, BLOCKSIZE>>>(d_vec, d_v2, d_v3, size, threshold);

    exclusive_scan(d_v2, d_v4, size);
    exclusive_scan(d_v3, d_v5, size);

    int sum_smaller[1];
    int sum_greater[1];
    cudaMemcpy(sum_smaller, d_v4 + size-1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(sum_greater, d_v5 + size-1, sizeof(int), cudaMemcpyDeviceToHost);

    cudaMalloc((void**)&d_v6, sizeof(int)*sum_smaller[0]);
    cudaMalloc((void**)&d_v7, sizeof(int)*sum_greater[0]);

    std::vector<int> partitioned_array_smaller(sum_smaller[0]);
    std::vector<int> partitioned_array_greater(sum_greater[0]);

    create_partitioned_array<<<GRIDSIZE, BLOCKSIZE>>>(d_vec, d_v2, d_v4, d_v6, size);
    create_partitioned_array<<<GRIDSIZE, BLOCKSIZE>>>(d_vec, d_v3, d_v5, d_v7, size);

    cudaMemcpy(partitioned_array_smaller.data(), d_v6, sizeof(int)*sum_smaller[0], cudaMemcpyDeviceToHost);
    cudaMemcpy(partitioned_array_greater.data(), d_v7, sizeof(int)*sum_greater[0], cudaMemcpyDeviceToHost);

    std::vector<std::vector<int>> result(2);
    result[0] = partitioned_array_smaller;
    result[1] = partitioned_array_greater;

    cudaFree(d_vec);
    cudaFree(d_v2);
    cudaFree(d_v3);
    cudaFree(d_v4);
    cudaFree(d_v5);
    cudaFree(d_v6);
    cudaFree(d_v7);

    return result;
}