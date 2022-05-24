#include "common.hpp"
#include "edgelist.hpp"
#include "partition.hpp"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>


void partition(EdgeList &E, EdgeList &E_leq, EdgeList &E_ge, int threshold, int kernel)
{
  g_benchmarker.start("partition()");

  switch (kernel)
  {
  case PARTITION_KERNEL_CPU_NAIVE:
    partition_cpu_naive(E, E_leq, E_ge, threshold);
    break;

  case PARTITION_KERNEL_GPU:
    //partition_inclusive_scan();
    break;

  default:
    throw std::invalid_argument("Unknown partition kernel");
  }

  g_benchmarker.stop("partition()");
}

void partition_cpu_naive(const EdgeList &E, EdgeList &E_leq, EdgeList &E_ge, int threshold)
{
  // allocate both to max size so vectors dont grow
  size_t max_size = E.size();
  E_leq.reserve(max_size);
  E_ge.reserve(max_size);

  for (size_t i = 0; i < E.size(); i++)
  {
    Edge e = E[i];
    if (e.weight <= threshold)
    {
      E_leq.append_edge(E[i]);
    }
    else
    {
      E_ge.append_edge(E[i]);
    }
  }
}

__global__ void check_array(int *vec, int *smaller, int *greater, size_t size, int threshold)
{

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;

  for (size_t i = thread_id; i < size; i += num_threads)
  {
    smaller[i] = vec[i] <= threshold;
    greater[i] = vec[i] > threshold;
  }
}

__global__ void create_partitioned_array(int *values, int *truth_values, int *scanned_values,
                                         int *new_array, size_t size)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;

  for (size_t i = thread_id; i < size; i += num_threads)
  {
    if (truth_values[i])
      new_array[scanned_values[i] - 1] = values[i];
  }
}

// first kernel for exclusive scan
// the whole array X to be scanned gets contigously distributed among the blocks.
// every block performs an exclusive scan on its array and writes the results to Y.
// Y then is partly exclusively scanned. what is still missing, is adding the end value of block
// i-1 to all elements in block i. This will later happen in a second kernel.
// all the blockoffsets get stored in carries.
__global__ void scan_kernel_1(int const *X, int *Y, int N, int *carries)
{
  __shared__ double shared_buffer[BLOCKSIZE];
  int my_value;

  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x * blockIdx.x;
  unsigned int block_stop = work_per_thread * blockDim.x * (blockIdx.x + 1);
  unsigned int block_offset = 0;

  // run scan on each section, this for loop is necessary if there are more elements in the array
  // than there are threads in total.
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
  {
    // load data:
    my_value = (i < N) ? X[i] : 0;

    // inclusive scan in shared buffer:
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
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

    // write to output array
    if (i < N)
      Y[i] = block_offset + my_value;

    block_offset += shared_buffer[blockDim.x - 1];
  }

  // write carry:
  if (threadIdx.x == 0)
    carries[blockIdx.x] = block_offset;
}

// Y is partly exclusively scanned. Here the offsets of each block get added.
__global__ void scan_kernel_2(int *Y, int N, int const *carries)
{
  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x * blockIdx.x;
  unsigned int block_stop = work_per_thread * blockDim.x * (blockIdx.x + 1);

  __shared__ int shared_offset;
  __shared__ int shared_buffer[GRIDSIZE];

  // load data:
  int my_carry = carries[threadIdx.x];

  // exclusive scan in the carries array
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
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

  if (threadIdx.x == 0)
    shared_offset = (blockIdx.x > 0) ? shared_buffer[blockIdx.x - 1] : 0;
  ;

  __syncthreads();

  // add offset to each element in the block:
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
    if (i < N)
      Y[i] += shared_offset;
}

// The general structure of the exclusive scan and most parts in the respective kernel codes are
// taken from Dr. Rupp's lecture "Computational Sciende on Many Core Architectures". We simplified
// it a bit to be restricted to work when GRIDSIZE == BLOCKSIZE. That saves us launching one
// intermediate kernel to perform an exclusive scan on the carries array.
void exclusive_scan(int const *input, int *output, int N)
{

  int *carries;
  cudaMalloc(&carries, sizeof(int) * GRIDSIZE);

  // First step: Scan within each thread group and write carries
  scan_kernel_1<<<GRIDSIZE, BLOCKSIZE>>>(input, output, N, carries);

  cudaDeviceSynchronize();

  // Second step: Offset each thread group accordingly
  scan_kernel_2<<<GRIDSIZE, BLOCKSIZE>>>(output, N, carries);

  cudaFree(carries);
}

// void partition_inclusive_scan(E, E_leq, E_big, threshold)
std::vector<std::vector<int>> partition_inclusive_scan(std::vector<int> &vec, int threshold)
{
  //E.sync_hostToDervice();
  size_t size = vec.size();
  int num_bytes = vec.size() * sizeof(int);

  // allocate
  int *d_vec, *d_v2, *d_v3, *d_v4, *d_v5, *d_v6, *d_v7;
  cudaMalloc((void **)&d_vec, num_bytes);
  cudaMalloc((void **)&d_v2, num_bytes);
  cudaMalloc((void **)&d_v3, num_bytes);
  cudaMalloc((void **)&d_v4, num_bytes);
  cudaMalloc((void **)&d_v5, num_bytes);

  cudaMemcpy(d_vec, vec.data(), num_bytes, cudaMemcpyHostToDevice);

  check_array<<<GRIDSIZE, BLOCKSIZE>>>(d_vec, d_v2, d_v3, size, threshold);
  //checkarray(E.gpu)

  exclusive_scan(d_v2, d_v4, size);
  exclusive_scan(d_v3, d_v5, size);

  int sum_smaller[1];
  int sum_greater[1];

  cudaMemcpy(sum_smaller, d_v4 + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(sum_greater, d_v5 + size - 1, sizeof(int), cudaMemcpyDeviceToHost);

  cudaMalloc((void **)&d_v6, sizeof(int) * sum_smaller[0]);
  cudaMalloc((void **)&d_v7, sizeof(int) * sum_greater[0]);

  std::vector<int> partitioned_array_smaller(sum_smaller[0]);
  std::vector<int> partitioned_array_greater(sum_greater[0]);

  // E_leq.sync_deviceToHost();
  // E_leq.reserve(sum_smaller);
  // E_big.sync_deviceToHost();
  // E_big.reserve(sum_greater);




  create_partitioned_array<<<GRIDSIZE, BLOCKSIZE>>>(d_vec, d_v2, d_v4, d_v6, size);
  create_partitioned_array<<<GRIDSIZE, BLOCKSIZE>>>(d_vec, d_v3, d_v5, d_v7, size);

  cudaMemcpy(partitioned_array_smaller.data(), d_v6, sizeof(int) * sum_smaller[0],
             cudaMemcpyDeviceToHost);
  cudaMemcpy(partitioned_array_greater.data(), d_v7, sizeof(int) * sum_greater[0],
             cudaMemcpyDeviceToHost);

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