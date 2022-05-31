#include "benchmarker.hpp"
#include "common.hpp"
#include "filter.hpp"
#include "inclusive_scan_kernel.hpp"
#include "union_find.hpp"

void filter(EdgeList &E, UnionFind &P, int kernel)
{
  g_benchmarker.start("filter()");

  switch (kernel)
  {
  case FILTER_KERNEL_CPU_NAIVE:
    filter_cpu_naive(E, P);
    break;

  case FILTER_KERNEL_GPU:
    filter_gpu_naive(E, P);
    break;


  default:
    throw std::invalid_argument("Unknown filter kernel");
  }

  g_benchmarker.stop("filter()");
}

void filter_cpu_naive(EdgeList &E, UnionFind &P)
{
  EdgeList E_filt;
  E_filt.reserve(E.size());

  for (size_t i = 0; i < E.size(); i++)
  {
    Edge e = E[i];
    if(P.find(e.source) != P.find(e.target))
    {
        E_filt.append_edge(e);
    }    
  }

  E = E_filt;
}

__device__ int find_gpu(int i, int *group_array)
{
  if (group_array[i] == i)
    return i;
  else
    return find_gpu(group_array[i], group_array);
}


__global__ void check_array_filter(int *group_array, int *coo1, int *coo2, int *truth_array, int size)
{

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;

  for (size_t i = thread_id; i < size; i += num_threads)
  {
    truth_array[i] = (find_gpu(coo1[i], group_array) != find_gpu(coo2[i], group_array));
  }
}


__global__ void create_partitioned_array_filter(int *values, int *start, int *target, 
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


void remove_if(EdgeList &E, EdgeList &E_new, UnionFind &P)
{
  size_t size = E.val.size();
  int num_bytes = E.val.size() * sizeof(int);

  int num_bytes_parents = P.parent.size() * sizeof(int);


  // allocate
  int *d_E_val, *d_E_coo1, *d_E_coo2, *d_truth, *d_scanned_truth, *d_parents;
  cudaMalloc((void **)&d_E_val, num_bytes);
  cudaMalloc((void **)&d_E_coo1, num_bytes);
  cudaMalloc((void **)&d_E_coo2, num_bytes);
  cudaMalloc((void **)&d_truth, num_bytes);
  cudaMalloc((void **)&d_scanned_truth, num_bytes);
  cudaMalloc((void **)&d_parents, num_bytes_parents);


  cudaMemcpy(d_E_val, E.val.data(), num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_E_coo1, E.coo1.data(), num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_E_coo2, E.coo2.data(), num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_parents, P.parent.data(), num_bytes_parents, cudaMemcpyHostToDevice);


  check_array_filter<<<GRIDSIZE, BLOCKSIZE>>>(d_parents, d_E_coo1, d_E_coo2, d_truth, size);

  inclusive_scan(d_truth, d_scanned_truth, size);

  int sum[1];

  cudaMemcpy(sum, d_scanned_truth + size - 1, sizeof(int), cudaMemcpyDeviceToHost);

  int *d_E_new_val, *d_E_new_coo1, *d_E_new_coo2;
  cudaMalloc((void **)&d_E_new_val, sizeof(int) * sum[0]);
  cudaMalloc((void **)&d_E_new_coo1, sizeof(int) * sum[0]);
  cudaMalloc((void **)&d_E_new_coo2, sizeof(int) * sum[0]);

  // reserve some space here for leq and ge vectors
  E_new.resize_and_set_num_edges(sum[0]);

  create_partitioned_array_filter<<<GRIDSIZE, BLOCKSIZE>>>(d_E_val, d_E_coo1, d_E_coo2, d_truth, d_scanned_truth, d_E_new_val, d_E_new_coo1, d_E_new_coo2, size);

  cudaMemcpy(E_new.val.data(), d_E_new_val, sizeof(int) * sum[0], cudaMemcpyDeviceToHost);
  cudaMemcpy(E_new.coo1.data(), d_E_new_coo1, sizeof(int) * sum[0], cudaMemcpyDeviceToHost);
  cudaMemcpy(E_new.coo2.data(), d_E_new_coo2, sizeof(int) * sum[0], cudaMemcpyDeviceToHost);

  cudaFree(d_E_val);
  cudaFree(d_E_coo1);
  cudaFree(d_E_coo2);
  cudaFree(d_E_new_val);
  cudaFree(d_E_new_coo1);
  cudaFree(d_E_new_coo2);
  cudaFree(d_truth);
  cudaFree(d_scanned_truth);
}


void filter_gpu_naive(EdgeList &E, UnionFind &P)
{
  EdgeList E_filt;
  remove_if(E, E_filt, P);

  E = E_filt;
}