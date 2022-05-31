#include "common.hpp"
#include "edgelist.hpp"
#include "partition.hpp"
#include "inclusive_scan_kernel.hpp"
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
    partition_inclusive_scan(E, E_leq, E_ge, threshold);
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

__global__ void create_partitioned_array(int *values, int *start, int *target, 
                                         int *truth_values, int *scanned_values,
                                         int *new_array_values, int *new_array_start, int *new_array_target,
                                         size_t size)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;

  for (size_t i = thread_id; i < size; i += num_threads)
  {
    if (truth_values[i]){
      new_array_values[scanned_values[i] - 1] = values[i];
      new_array_start[scanned_values[i] - 1] = start[i];
      new_array_target[scanned_values[i] - 1] = target[i];
    }
  }
}

// void partition_inclusive_scan(E, E_leq, E_big, threshold)
void partition_inclusive_scan(EdgeList &E, EdgeList &E_leq, EdgeList &E_ge, int threshold)
{
  size_t size = E.val.size();
  int num_bytes = E.val.size() * sizeof(int);

  // allocate
  int *d_E_val, *d_E_coo1, *d_E_coo2, *d_truth_small, *d_truth_big, *d_scanned_truth_small, *d_scanned_truth_big;
  cudaMalloc((void **)&d_E_val, num_bytes);
  cudaMalloc((void **)&d_E_coo1, num_bytes);
  cudaMalloc((void **)&d_E_coo2, num_bytes);
  cudaMalloc((void **)&d_truth_small, num_bytes);
  cudaMalloc((void **)&d_truth_big, num_bytes);
  cudaMalloc((void **)&d_scanned_truth_small, num_bytes);
  cudaMalloc((void **)&d_scanned_truth_big, num_bytes);

  cudaMemcpy(d_E_val, E.val.data(), num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_E_coo1, E.coo1.data(), num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_E_coo2, E.coo2.data(), num_bytes, cudaMemcpyHostToDevice);

  check_array<<<GRIDSIZE, BLOCKSIZE>>>(d_E_val, d_truth_small, d_truth_big, size, threshold);

  inclusive_scan(d_truth_small, d_scanned_truth_small, size);
  inclusive_scan(d_truth_big, d_scanned_truth_big, size);

  int sum_smaller[1];
  int sum_greater[1];

  cudaMemcpy(sum_smaller, d_scanned_truth_small + size - 1, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(sum_greater, d_scanned_truth_big + size - 1, sizeof(int), cudaMemcpyDeviceToHost);

  int *d_E_leq_val, *d_E_leq_coo1, *d_E_leq_coo2, *d_E_ge_val, *d_E_ge_coo1, *d_E_ge_coo2;
  cudaMalloc((void **)&d_E_leq_val, sizeof(int) * sum_smaller[0]);
  cudaMalloc((void **)&d_E_leq_coo1, sizeof(int) * sum_smaller[0]);
  cudaMalloc((void **)&d_E_leq_coo2, sizeof(int) * sum_smaller[0]);
  cudaMalloc((void **)&d_E_ge_val, sizeof(int) * sum_greater[0]);
  cudaMalloc((void **)&d_E_ge_coo1, sizeof(int) * sum_greater[0]);
  cudaMalloc((void **)&d_E_ge_coo2, sizeof(int) * sum_greater[0]);
  
  // reserve some space here for leq and ge vectors
  E_leq.resize_and_set_num_edges(sum_smaller[0]);
  E_ge.resize_and_set_num_edges(sum_greater[0]);

  create_partitioned_array<<<GRIDSIZE, BLOCKSIZE>>>(d_E_val, d_E_coo1, d_E_coo2, d_truth_small, d_scanned_truth_small, d_E_leq_val, d_E_leq_coo1, d_E_leq_coo2, size);
  create_partitioned_array<<<GRIDSIZE, BLOCKSIZE>>>(d_E_val, d_E_coo1, d_E_coo2, d_truth_big, d_scanned_truth_big, d_E_ge_val, d_E_ge_coo1, d_E_ge_coo2, size);

  cudaMemcpy(E_leq.val.data(), d_E_leq_val, sizeof(int) * sum_smaller[0], cudaMemcpyDeviceToHost);
  cudaMemcpy(E_leq.coo1.data(), d_E_leq_coo1, sizeof(int) * sum_smaller[0], cudaMemcpyDeviceToHost);
  cudaMemcpy(E_leq.coo2.data(), d_E_leq_coo2, sizeof(int) * sum_smaller[0], cudaMemcpyDeviceToHost);
             
  cudaMemcpy(E_ge.val.data(), d_E_ge_val, sizeof(int) * sum_greater[0], cudaMemcpyDeviceToHost);
  cudaMemcpy(E_ge.coo1.data(), d_E_ge_coo1, sizeof(int) * sum_greater[0], cudaMemcpyDeviceToHost);
  cudaMemcpy(E_ge.coo2.data(), d_E_ge_coo2, sizeof(int) * sum_greater[0], cudaMemcpyDeviceToHost);

  cudaFree(d_E_val);
  cudaFree(d_E_coo1);
  cudaFree(d_E_coo2);
  cudaFree(d_E_leq_val);
  cudaFree(d_E_leq_coo1);
  cudaFree(d_E_leq_coo2);
  cudaFree(d_E_ge_val);
  cudaFree(d_E_ge_coo1);
  cudaFree(d_E_ge_coo2);
  cudaFree(d_truth_small);
  cudaFree(d_truth_big);
  cudaFree(d_scanned_truth_small);
  cudaFree(d_scanned_truth_big);

}