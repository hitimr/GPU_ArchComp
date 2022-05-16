# pragma once
#include <iostream>
#include <algorithm>


__global__ void check_array(int* vec, int* smaller, int* greater, size_t size, int threshold){

    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(size_t i = thread_id; i < size; i += num_threads){
        smaller[i] = vec[i] <= threshold;
        greater[i] = vec[i] > threshold;
    }
}

// first kernel for exclusive scan
__global__ void scan_kernel_1(int const *X,
                              int *Y,
                              int N,
                              int *carries)
{
  __shared__ double shared_buffer[256];
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
  __shared__ int shared_buffer[256];
 
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

 
void exclusive_scan(int const * input,
                    int       * output, int N)
{
  int num_blocks = 256;
  int threads_per_block = 256;
 
  int *carries;
  cudaMalloc(&carries, sizeof(int) * num_blocks);
 
  // First step: Scan within each thread group and write carries
  scan_kernel_1<<<num_blocks, threads_per_block>>>(input, output, N, carries);
 
  // Second step: Compute offset for each thread group (exclusive scan for each thread group)
  scan_kernel_2<<<1, num_blocks>>>(carries);
 
  // Third step: Offset each thread group accordingly
  scan_kernel_3<<<num_blocks, threads_per_block>>>(output, N, carries);
 
  cudaFree(carries);
}