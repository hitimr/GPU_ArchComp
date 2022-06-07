#include <common.hpp>
#include <sort.hpp>

#include <cassert>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

void sort_edgelist(EdgeList &E, int kernel)
{
  g_benchmarker.start("sort()");

  E.sync_deviceToHost();

  switch (kernel)
  {
  case SORT_KERNEL_GPU_BUBBLE_MULT:
    gpu_bubble_sort_mult(E);
    break;

  case SORT_KERNEL_MERGE_SORT:
    improved_mergesort_three(E);
    break;

  case SORT_KERNEL_THRUST:
    gpu_thrust_sort_three(E.val, E.coo1, E.coo2);
    break;

  default:
    throw std::invalid_argument("Unknown sort kernel");
  }

  g_benchmarker.stop("sort()");
}

void gpu_thrust_sort_three(std::vector<int> &vec, std::vector<int> &vec1, std::vector<int> &vec2)
{

  size_t size = vec.size();
  int num_bytes = vec.size() * sizeof(int);

  thrust::host_vector<int> h_vec = vec;
  thrust::device_vector<int> d_vec = h_vec;

  thrust::host_vector<int> h_ind_vec = vec;
  thrust::device_vector<int> d_ind_vec = h_ind_vec;
  thrust::sequence(h_ind_vec.begin(), h_ind_vec.end());

  thrust::copy(h_ind_vec.begin(), h_ind_vec.end(), d_ind_vec.begin());

  thrust::sort_by_key(d_vec.begin(), d_vec.end(), d_ind_vec.begin());

  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
  thrust::copy(h_vec.begin(), h_vec.end(), vec.begin());

  std::vector<int> indices(size);
  thrust::copy(d_ind_vec.begin(), d_ind_vec.end(), h_ind_vec.begin());
  thrust::copy(h_ind_vec.begin(), h_ind_vec.end(), indices.begin());

  std::vector<int> tmp_vec1, tmp_vec2;
  tmp_vec1 = vec1;
  tmp_vec2 = vec2;
  for (size_t i = 0; i < size; i++)
  {
    vec1[i] = tmp_vec1[indices[i]];
    vec2[i] = tmp_vec2[indices[i]];
  }
}

__global__ void gpu_merge_sort_thread_per_block_with_ind(int *input, int *output, int size,
                                                         int size_to_merge, int *input_ind,
                                                         int *output_ind)
{

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  int number_of_patches = (size / size_to_merge) + 1;

  for (int patch_id = thread_id; patch_id < number_of_patches; patch_id += num_threads)
  {
    int start_point = patch_id * size_to_merge;

    int mid_point = min(start_point + (size_to_merge / 2), size);
    int end_point = min(start_point + size_to_merge, size);

    int current_left_pointer = start_point;
    int current_right_pointer = mid_point;

    for (int position_in_merged = start_point; position_in_merged < end_point; position_in_merged++)
    {
      if ((current_left_pointer < mid_point) &&
          (current_right_pointer >= end_point ||
           input[current_left_pointer] < input[current_right_pointer]))
      {
        output[position_in_merged] = input[current_left_pointer];
        output_ind[position_in_merged] = input_ind[current_left_pointer];
        current_left_pointer++;
      }
      else
      {
        output[position_in_merged] = input[current_right_pointer];
        output_ind[position_in_merged] = input_ind[current_right_pointer];
        current_right_pointer++;
      }
    }
  }
}

// void improved_mergesort_three(std::vector<int> &vec, std::vector<int> &vec1, std::vector<int>
// &vec2)
void improved_mergesort_three(EdgeList &E)
{
  E.sync_hostToDevice();

  int size = E.size();
  int num_bytes = E.size() * sizeof(int);

  // allocate
  int *d_tmp;
  cudaMalloc((void **)&d_tmp, num_bytes);

  // copy
  cudaMemcpy(d_tmp, E.d_val, num_bytes, cudaMemcpyDeviceToDevice);

  int *ind_vec;
  int *ind_tmp;

  cudaMalloc((void **)&ind_vec, num_bytes);
  cudaMalloc((void **)&ind_tmp, num_bytes);

  std::vector<int> initial(size);
  std::iota(std::begin(initial), std::end(initial), 0);

  cudaMemcpy(ind_vec, initial.data(), num_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(ind_tmp, initial.data(), num_bytes, cudaMemcpyHostToDevice);

  int *input = d_tmp;
  int *output = E.d_val;

  int *input_ind = ind_tmp;
  int *output_ind = ind_vec;

  int *tmp;

  bool done = false;
  for (int size_to_merge = 2; done == false; size_to_merge *= 2)
  {
    tmp = input;
    input = output;
    output = tmp;

    tmp = input_ind;
    input_ind = output_ind;
    output_ind = tmp;

    // Actually call the kernel
    gpu_merge_sort_thread_per_block_with_ind<<<BLOCK_SIZE, GRID_SIZE>>>(
        input, output, size, size_to_merge, input_ind, output_ind);

    cudaDeviceSynchronize();

    if (size_to_merge >= size)
    {
      done = true;
    }
  }

  // Assembly on Host
  cudaMemcpy(E.val.data(), output, size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(initial.data(), output_ind, sizeof(int) * E.size(), cudaMemcpyDeviceToHost);
  E.set_owner(HOST);

  std::vector<int> tmp_vec1, tmp_vec2;
  tmp_vec1 = E.coo1;
  tmp_vec2 = E.coo2;
  for (int i = 0; i < size; i++)
  {
    E.coo1[i] = tmp_vec1[initial[i]];
    E.coo2[i] = tmp_vec2[initial[i]];
  }

  // Free the GPU memory
  cudaFree(d_tmp);
  cudaFree(ind_tmp);
  cudaFree(ind_vec);
}

__device__ void gpu_swap(int *vec, size_t i, size_t j)
{
  int temp = vec[i];
  vec[i] = vec[j];
  vec[j] = temp;
}

__global__ void gpu_odd_pass_mult(int *vec, size_t size, int *v2, int *v3)
{

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;

  for (size_t i = thread_id; i < size_t(size / 2 - 1); i += num_threads)
  {
    if (vec[2 * i + 1] > vec[2 * i + 2])
    {
      gpu_swap(vec, 2 * i + 1, 2 * i + 2);
      gpu_swap(v2, 2 * i + 1, 2 * i + 2);
      gpu_swap(v3, 2 * i + 1, 2 * i + 2);
    }
  }
}

//--------------------------------------
//       sort three arrays
//--------------------------------------
__global__ void gpu_even_pass_mult(int *vec, size_t size, int *v2, int *v3)
{

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;

  for (size_t i = thread_id; i < size_t(size / 2); i += num_threads)
  {
    if (vec[2 * i] > vec[2 * i + 1])
    {
      gpu_swap(vec, 2 * i, 2 * i + 1);
      gpu_swap(v2, 2 * i, 2 * i + 1);
      gpu_swap(v3, 2 * i, 2 * i + 1);
    }
  }
}

void gpu_bubble_sort_mult(EdgeList &E)
{
  E.sync_hostToDevice();

  size_t size = E.size();

  // sort
  for (size_t i = 0; i < size_t(size / 2); ++i)
  {
    gpu_even_pass_mult<<<BLOCK_SIZE, GRID_SIZE>>>(E.d_val, size, E.d_coo1, E.d_coo2);
    cudaDeviceSynchronize();
    gpu_odd_pass_mult<<<BLOCK_SIZE, GRID_SIZE>>>(E.d_val, size, E.d_coo1, E.d_coo2);
    cudaDeviceSynchronize();
  }
}