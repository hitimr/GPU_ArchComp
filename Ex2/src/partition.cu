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
  E.sync_deviceToHost();  // TODO: remove before release

  g_benchmarker.start("partition()");

  switch (kernel)
  {
  case PARTITION_KERNEL_CPU_NAIVE:
    partition_cpu_naive(E, E_leq, E_ge, threshold);
    break;

  case PARTITION_KERNEL_GPU:
    partition_inclusive_scan(E, E_leq, E_ge, threshold);
    break;

  case PARTITION_KERNEL_THRUST:
    partition_thrust(E, E_leq, E_ge, threshold);
    break;

  default:
    throw std::invalid_argument("Unknown partition kernel");
  }

  g_benchmarker.stop("partition()");
}

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

  case FILTER_KERNEL_THRUST:
    filter_thrust(E, P);
    break;

  default:
    throw std::invalid_argument("Unknown filter kernel");
  }

  g_benchmarker.stop("filter()");
}

void partition_cpu_naive(EdgeList &E, EdgeList &E_leq, EdgeList &E_ge, int threshold)
{
  E.sync_deviceToHost();
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

// first kernel for inclusive scan
// the whole array X to be scanned gets contigously distributed among the blocks.
// every block performs an inclusive scan on its array and writes the results to Y.
// Y then is partly inclusively scanned. what is still missing, is adding the end value of block
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

// Y is partly inclusively scanned. Here the offsets of each block get added.
__global__ void scan_kernel_2(int *Y, int N, int const *carries)
{
  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x * blockIdx.x;
  unsigned int block_stop = work_per_thread * blockDim.x * (blockIdx.x + 1);

  __shared__ int shared_offset;
  __shared__ int shared_buffer[GRIDSIZE];

  // load data:
  int my_carry = carries[threadIdx.x];

  // inclusive scan in the carries array
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

// The general structure of the inclusive scan and most parts in the respective kernel codes are
// taken from Dr. Rupp's lecture "Computational Sciende on Many Core Architectures". We simplified
// it a bit to be restricted to work when GRIDSIZE == BLOCKSIZE. That saves us launching one
// intermediate kernel to perform an inclusive scan on the carries array.
void inclusive_scan(int const *input, int *output, int N)
{

  int *carries;
  cudaMalloc(&carries, sizeof(int) * GRIDSIZE);

  // First step: Scan within each thread group and write carries
  scan_kernel_1<<<GRIDSIZE, BLOCKSIZE>>>(input, output, N, carries);

  cudaDeviceSynchronize();

  // Second step: Offset each thread group accordingly
  scan_kernel_2<<<GRIDSIZE, BLOCKSIZE>>>(output, N, carries);

  cudaDeviceSynchronize();

  cudaFree(carries);
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

__global__ void create_partitioned_array(int *values, int *start, int *target, int *truth_values,
                                         int *scanned_values, int *new_array_values,
                                         int *new_array_start, int *new_array_target, size_t size)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;

  for (size_t i = thread_id; i < size; i += num_threads)
  {
    if (truth_values[i])
    {
      new_array_values[scanned_values[i] - 1] = values[i];
      new_array_start[scanned_values[i] - 1] = start[i];
      new_array_target[scanned_values[i] - 1] = target[i];
    }
  }
}

// void partition_inclusive_scan(E, E_leq, E_big, threshold)
void partition_inclusive_scan(EdgeList &E, EdgeList &E_leq, EdgeList &E_ge, int threshold)
{
  E.sync_hostToDevice();

  size_t size = E.val.size();
  int num_bytes = E.val.size() * sizeof(int);

  // allocate
  int *d_truth_small, *d_truth_big, *d_scanned_truth_small, *d_scanned_truth_big;
  // cudaMalloc((void **)&d_E_val, num_bytes);
  cudaMalloc((void **)&d_truth_small, num_bytes);
  cudaMalloc((void **)&d_truth_big, num_bytes);
  cudaMalloc((void **)&d_scanned_truth_small, num_bytes);
  cudaMalloc((void **)&d_scanned_truth_big, num_bytes);

  check_array<<<GRIDSIZE, BLOCKSIZE>>>(E.d_val, d_truth_small, d_truth_big, size, threshold);

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

  create_partitioned_array<<<GRIDSIZE, BLOCKSIZE>>>(E.d_val, E.d_coo1, E.d_coo2, d_truth_small,
                                                    d_scanned_truth_small, d_E_leq_val,
                                                    d_E_leq_coo1, d_E_leq_coo2, size);
  create_partitioned_array<<<GRIDSIZE, BLOCKSIZE>>>(E.d_val, E.d_coo1, E.d_coo2, d_truth_big,
                                                    d_scanned_truth_big, d_E_ge_val, d_E_ge_coo1,
                                                    d_E_ge_coo2, size);

  cudaMemcpy(E_leq.val.data(), d_E_leq_val, sizeof(int) * sum_smaller[0], cudaMemcpyDeviceToHost);
  cudaMemcpy(E_leq.coo1.data(), d_E_leq_coo1, sizeof(int) * sum_smaller[0], cudaMemcpyDeviceToHost);
  cudaMemcpy(E_leq.coo2.data(), d_E_leq_coo2, sizeof(int) * sum_smaller[0], cudaMemcpyDeviceToHost);

  cudaMemcpy(E_ge.val.data(), d_E_ge_val, sizeof(int) * sum_greater[0], cudaMemcpyDeviceToHost);
  cudaMemcpy(E_ge.coo1.data(), d_E_ge_coo1, sizeof(int) * sum_greater[0], cudaMemcpyDeviceToHost);
  cudaMemcpy(E_ge.coo2.data(), d_E_ge_coo2, sizeof(int) * sum_greater[0], cudaMemcpyDeviceToHost);

  // cudaFree(d_E_val);
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

void filter_cpu_naive(EdgeList &E, UnionFind &P)
{
  E.sync_deviceToHost();

  EdgeList E_filt;
  E_filt.reserve(E.size());

  for (size_t i = 0; i < E.size(); i++)
  {
    Edge e = E[i];
    if (P.find(e.source) != P.find(e.target))
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

__global__ void check_array_filter(int *group_array, int *coo1, int *coo2, int *truth_array,
                                   int size)
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
                                                int *new_array_values, int *new_array_start,
                                                int *new_array_target, size_t size)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;

  for (size_t i = thread_id; i < size; i += num_threads)
  {
    if (truth_values[i])
    {
      new_array_values[scanned_values[i] - 1] = values[i];
      new_array_start[scanned_values[i] - 1] = start[i];
      new_array_target[scanned_values[i] - 1] = target[i];
    }
  }
}

void filter_gpu_naive(EdgeList &E, UnionFind &P)
{
  E.sync_hostToDevice();
  EdgeList E_new;

  size_t size = E.val.size();
  int num_bytes = E.val.size() * sizeof(int);

  int num_bytes_parents = P.parent.size() * sizeof(int);

  // allocate
  int *d_truth, *d_scanned_truth, *d_parents;
  cudaMalloc((void **)&d_truth, num_bytes);
  cudaMalloc((void **)&d_scanned_truth, num_bytes);
  cudaMalloc((void **)&d_parents, num_bytes_parents);

  cudaMemcpy(d_parents, P.parent.data(), num_bytes_parents, cudaMemcpyHostToDevice);

  check_array_filter<<<GRIDSIZE, BLOCKSIZE>>>(d_parents, E.d_coo1, E.d_coo2, d_truth, size);

  inclusive_scan(d_truth, d_scanned_truth, size);

  int sum[1];

  cudaMemcpy(sum, d_scanned_truth + size - 1, sizeof(int), cudaMemcpyDeviceToHost);

  // reserve some space here for leq and ge vectors
  E_new.resize_and_set_num_edges(sum[0]);
  E_new.set_owner(DEVICE);

  create_partitioned_array_filter<<<GRIDSIZE, BLOCKSIZE>>>(E.d_val, E.d_coo1, E.d_coo2, d_truth,
                                                           d_scanned_truth, E_new.d_val,
                                                           E_new.d_coo1, E_new.d_coo2, size);

  // cudaMemcpy(E_new.val.data(), d_E_new_val, sizeof(int) * sum[0], cudaMemcpyDeviceToHost);
  // cudaMemcpy(E_new.coo1.data(), d_E_new_coo1, sizeof(int) * sum[0], cudaMemcpyDeviceToHost);
  // cudaMemcpy(E_new.coo2.data(), d_E_new_coo2, sizeof(int) * sum[0], cudaMemcpyDeviceToHost);
  //E_new.sync_deviceToHost();

  cudaFree(d_truth);
  cudaFree(d_scanned_truth);

  E = E_new;
}




// condition for partitioning with thrust
struct is_less_equal
{
  int threshold;
  is_less_equal(int t): threshold(t) {}

  __host__ __device__
  bool operator()(const int &x)
  {
    return x <= threshold;
  }
};


struct is_greater
{
  int threshold;
  is_greater(int t): threshold(t) {}

  __host__ __device__
  bool operator()(const int &x)
  {
    return x > threshold;
  }
};

void partition_thrust(EdgeList &E, EdgeList &E_leq, EdgeList &E_ge, int threshold){

  E.sync_hostToDevice();
  size_t size_E = E.val.size();

  thrust::device_ptr<int> ptr_E_d_val(&E.d_val[0]);
  thrust::device_ptr<int> ptr_E_d_coo1(&E.d_coo1[0]);
  thrust::device_ptr<int> ptr_E_d_coo2(&E.d_coo2[0]);

  int size_smaller = thrust::count_if(thrust::device, ptr_E_d_val, ptr_E_d_val+size_E, is_less_equal(threshold));
  int size_bigger = size_E - size_smaller;

  E_leq.resize_and_set_num_edges(size_smaller);
  E_ge.resize_and_set_num_edges(size_bigger);

  E_leq.sync_hostToDevice();
  E_ge.sync_hostToDevice();

  thrust::device_ptr<int> ptr_E_leq_d_val(&E_leq.d_val[0]);
  thrust::device_ptr<int> ptr_E_leq_d_coo1(&E_leq.d_coo1[0]);
  thrust::device_ptr<int> ptr_E_leq_d_coo2(&E_leq.d_coo2[0]);

  thrust::copy_if(thrust::device, ptr_E_d_val, ptr_E_d_val+size_E, ptr_E_d_val, ptr_E_leq_d_val, is_less_equal(threshold));
  thrust::copy_if(thrust::device, ptr_E_d_coo1, ptr_E_d_coo1+size_E, ptr_E_d_val, ptr_E_leq_d_coo1, is_less_equal(threshold));
  thrust::copy_if(thrust::device, ptr_E_d_coo2, ptr_E_d_coo2+size_E, ptr_E_d_val, ptr_E_leq_d_coo2, is_less_equal(threshold));
  E_leq.sync_deviceToHost();

  thrust::device_ptr<int> ptr_E_ge_d_val(&E_ge.d_val[0]);
  thrust::device_ptr<int> ptr_E_ge_d_coo1(&E_ge.d_coo1[0]);
  thrust::device_ptr<int> ptr_E_ge_d_coo2(&E_ge.d_coo2[0]);

  thrust::copy_if(thrust::device, ptr_E_d_val, ptr_E_d_val+size_E, ptr_E_d_val, ptr_E_ge_d_val, is_greater(threshold));
  thrust::copy_if(thrust::device, ptr_E_d_coo1, ptr_E_d_coo1+size_E, ptr_E_d_val, ptr_E_ge_d_coo1, is_greater(threshold));
  thrust::copy_if(thrust::device, ptr_E_d_coo2, ptr_E_d_coo2+size_E, ptr_E_d_val, ptr_E_ge_d_coo2, is_greater(threshold));
  E_ge.sync_deviceToHost();

}



struct is_valid
{
  __host__ __device__
  bool operator()(const int &x)
  {
    return x;
  }
};



void filter_thrust(EdgeList &E, UnionFind &P)
{
  E.sync_hostToDevice();
  EdgeList E_new;

  size_t size = E.val.size();
  int num_bytes = E.val.size() * sizeof(int);

  thrust::device_ptr<int> ptr_E_d_val(&E.d_val[0]);
  thrust::device_ptr<int> ptr_E_d_coo1(&E.d_coo1[0]);
  thrust::device_ptr<int> ptr_E_d_coo2(&E.d_coo2[0]);

  int num_bytes_parents = P.parent.size() * sizeof(int);

  // allocate
  int *d_truth, *d_parents;
  cudaMalloc((void **)&d_truth, num_bytes);
  thrust::device_ptr<int> ptr_d_truth(&d_truth[0]);
  cudaMalloc((void **)&d_parents, num_bytes_parents);

  cudaMemcpy(d_parents, P.parent.data(), num_bytes_parents, cudaMemcpyHostToDevice);

  check_array_filter<<<GRIDSIZE, BLOCKSIZE>>>(d_parents, E.d_coo1, E.d_coo2, d_truth, size);

  int size_E_new = thrust::count_if(thrust::device, ptr_d_truth, ptr_d_truth+size, is_valid());

  // reserve some space here for leq and ge vectors
  E_new.resize_and_set_num_edges(size_E_new);
  E_new.set_owner(DEVICE);

  thrust::device_ptr<int> ptr_E_new_d_val(&E_new.d_val[0]);
  thrust::device_ptr<int> ptr_E_new_d_coo1(&E_new.d_coo1[0]);
  thrust::device_ptr<int> ptr_E_new_d_coo2(&E_new.d_coo2[0]);

  thrust::copy_if(thrust::device, ptr_E_d_val, ptr_E_d_val+size, ptr_d_truth, ptr_E_new_d_val, is_valid());
  thrust::copy_if(thrust::device, ptr_E_d_coo1, ptr_E_d_coo1+size, ptr_d_truth, ptr_E_new_d_coo1, is_valid());
  thrust::copy_if(thrust::device, ptr_E_d_coo2, ptr_E_d_coo2+size, ptr_d_truth, ptr_E_new_d_coo2, is_valid());

  cudaFree(d_truth);

  E = E_new;
}