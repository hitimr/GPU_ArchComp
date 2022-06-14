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

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define SPLIT_SIZE 4

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

void sort_edgelist(EdgeList &E, int kernel)
{
  g_benchmarker.start("sort()");

  switch (kernel)
  {
  case SORT_KERNEL_GPU_BUBBLE_MULT:
    gpu_bubble_sort_mult(E);
    break;

  case SORT_KERNEL_MERGE_SORT:
    improved_mergesort_three(E);
    break;

  case SORT_KERNEL_THRUST:
    gpu_thrust_sort_three(E);
    break;

  case SORT_KERNEL_RADIX:
    radix_sort(E);
    break;

  default:
    throw std::invalid_argument("Unknown sort kernel");
  }

  g_benchmarker.stop("sort()");
}

__global__ void assemble_with_indices(int *d_coo1, int *d_coo2, int *tmp_coo1, int *tmp_coo2,
                                      int *sorted_indices, int size)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;

  for (size_t i = thread_id; i < size; i += num_threads)
  {
    d_coo1[i] = tmp_coo1[sorted_indices[i]];
    d_coo2[i] = tmp_coo2[sorted_indices[i]];
  }
}

void gpu_thrust_sort_three(EdgeList &E)
{
  E.sync_hostToDevice();

  size_t size = E.size();
  int num_bytes = E.size() * sizeof(int);

  thrust::device_vector<int> d_vec(E.d_val, E.d_val + E.size());
  thrust::device_vector<int> d_ind_vec(E.size());
  thrust::sequence(d_ind_vec.begin(), d_ind_vec.end());

  thrust::sort_by_key(d_vec.begin(), d_vec.end(), d_ind_vec.begin());

  // TODO: maybe perform assemlby on GPU..
  thrust::copy(d_vec.begin(), d_vec.end(), E.d_val);

  size_t bytes = E.size() * sizeof(int);
  int *tmp_coo1, *tmp_coo2;
  cudaMalloc(&tmp_coo1, bytes);
  cudaMalloc(&tmp_coo2, bytes);
  cudaMemcpy(tmp_coo1, E.d_coo1, bytes, cudaMemcpyDeviceToDevice);
  cudaMemcpy(tmp_coo2, E.d_coo2, bytes, cudaMemcpyDeviceToDevice);
  int *d_ind_ptr = thrust::raw_pointer_cast(d_ind_vec.data());

  assemble_with_indices<<<GRID_SIZE, BLOCK_SIZE>>>(E.d_coo1, E.d_coo2, tmp_coo1, tmp_coo2,
                                                   d_ind_ptr, size);
  cudaDeviceSynchronize();
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
  cudaMemcpy(E.val, output, size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(initial.data(), output_ind, sizeof(int) * E.size(), cudaMemcpyDeviceToHost);
  E.set_owner(HOST);

  std::vector<int> tmp_vec1(E.coo1, E.coo1 + E.size());
  std::vector<int> tmp_vec2(E.coo2, E.coo2 + E.size());
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

__global__ void final_sort(int *in, int *out, int size, int *in_ind, int *out_ind, int *prefixes,
                           int *block_sums, int bit_shift)
{
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
  {
    int mask = (in[i] >> bit_shift) & 3;
    int new_pos = block_sums[mask * gridDim.x + blockIdx.x] + prefixes[i];
    out[new_pos] = in[i];
    out_ind[new_pos] = in_ind[i];
  }
}

__global__ void add_prefix_inplace(int *in, int size, int *partial_sums)
{
  for (int i = 2 * blockIdx.x * blockDim.x + threadIdx.x; i < size; i += 2 * blockDim.x * gridDim.x)
  {
    in[i] = in[i] + partial_sums[blockIdx.x];
    if (i + blockDim.x < size)
      in[i + blockDim.x] = in[i + blockDim.x] + partial_sums[blockIdx.x];
  }
}

// https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
__global__ void gpu_prescan(int *out, int *in, int size, int *partial_sums)
{
  extern __shared__ int tmp[];

  int global_thread_id = 2 * blockDim.x * blockIdx.x + threadIdx.x;

  if (global_thread_id < blockDim.x * 2)
  {
    tmp[global_thread_id] = 0;
  }

  __syncthreads();

  if (global_thread_id < size)
  {
    tmp[threadIdx.x + CONFLICT_FREE_OFFSET(threadIdx.x)] = in[global_thread_id];
    if (global_thread_id + blockDim.x < size)
      tmp[threadIdx.x + blockDim.x + CONFLICT_FREE_OFFSET(threadIdx.x + blockDim.x)] =
          in[global_thread_id + blockDim.x];
  }

  int offset = 1;
  for (int d = 2 * blockDim.x >> 1; d > 0; d >>= 1)
  {
    __syncthreads();

    if (threadIdx.x < d)
    {
      int ai = offset * (threadIdx.x * 2 + 1) - 1;
      int bi = offset * (threadIdx.x * 2 + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      tmp[bi] += tmp[ai];
    }
    offset <<= 1;
  }

  if (threadIdx.x == 0)
  {
    // to be reused later
    partial_sums[blockIdx.x] = tmp[2 * blockDim.x - 1 + CONFLICT_FREE_OFFSET(2 * blockDim.x - 1)];
    tmp[2 * blockDim.x - 1 + CONFLICT_FREE_OFFSET(2 * blockDim.x - 1)] = 0;
  }
  for (int d = 1; d < 2 * blockDim.x; d <<= 1)
  {
    offset >>= 1;
    __syncthreads();

    if (threadIdx.x < d)
    {
      int ai = offset * ((threadIdx.x << 1) + 1) - 1;
      int bi = offset * ((threadIdx.x << 1) + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      int temp = tmp[ai];
      tmp[ai] = tmp[bi];
      tmp[bi] += temp;
    }
  }
  __syncthreads();
  if (global_thread_id < size)
  {
    out[global_thread_id] = tmp[threadIdx.x + CONFLICT_FREE_OFFSET(threadIdx.x)];
    if (global_thread_id + blockDim.x < size)
      out[global_thread_id + blockDim.x] =
          tmp[threadIdx.x + blockDim.x + CONFLICT_FREE_OFFSET(threadIdx.x + blockDim.x)];
  }
}

__global__ void first_stage(int *in, int *out, int size, int *in_ind, int *out_ind, int *prefixes,
                            int *partial_sums, int bit_shift)
{
  int thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  extern __shared__ int tmp[];

  int *masks = tmp + blockDim.x + 1;
  int *scan_output = masks + blockDim.x + 1;
  int *masks_sums = scan_output + blockDim.x + 1;
  int *scan_sums = masks_sums + SPLIT_SIZE + 1;
  int *tmp_ind = scan_sums + blockDim.x + 1;

  for (int i = thread_id; i < size; i += (blockDim.x * gridDim.x))
  {
    tmp[threadIdx.x] = in[i];
    tmp_ind[threadIdx.x] = in_ind[i];
  }

  int current_value = tmp[threadIdx.x];
  int current_ind = tmp_ind[threadIdx.x];
  int current_bits = (current_value >> bit_shift) & 3;

  for (int i = 0; i < SPLIT_SIZE; ++i)
  {
    __syncthreads();

    if (thread_id < size)
    {
      masks[threadIdx.x + 1] = current_bits == i;
    }
    else
    {
      masks[threadIdx.x + 1] = 0;
    }
    __syncthreads();

    int sum = 0;
    for (int d = 1; d < blockDim.x + 1; d *= 2)
    {
      if (threadIdx.x >= d)
      {
        sum = masks[threadIdx.x + 1] + masks[threadIdx.x + 1 - d];
      }
      else
      {
        sum = masks[threadIdx.x + 1];
      }
      __syncthreads();
      masks[threadIdx.x + 1] = sum;
      __syncthreads();
    }

    if (threadIdx.x == 0)
    {
      masks[0] = 0;
      masks_sums[i] = masks[blockDim.x];
      partial_sums[i * gridDim.x + blockIdx.x] = masks[blockDim.x];
    }

    if (current_bits == i && (thread_id < size))
    {
      scan_output[threadIdx.x] = masks[threadIdx.x];
    }
  }

  __syncthreads();

  if (threadIdx.x == 0)
  {
    int run_sum = 0;
    for (int i = 0; i < SPLIT_SIZE; ++i)
    {
      scan_sums[i] = run_sum;
      run_sum += masks_sums[i];
    }
  }

  __syncthreads();

  if (thread_id < size)
  {
    int tmp_scan = scan_output[threadIdx.x];
    int new_pos = scan_output[threadIdx.x] + scan_sums[current_bits];
    __syncthreads();
    tmp[new_pos] = current_value;
    tmp_ind[new_pos] = current_ind;
    scan_output[new_pos] = tmp_scan;
    __syncthreads();
    prefixes[thread_id] = scan_output[threadIdx.x];
    out[thread_id] = tmp[threadIdx.x];
    out_ind[thread_id] = tmp_ind[threadIdx.x];
  }
}

void scan(int *in, int *out, int size)
{
  // New grid size ( we want just to have as much as we need)
  int grid_size = (size / BLOCK_SIZE) + (size % BLOCK_SIZE != 0);
  // Calculate tmp size for the conflict free scan
  int tmp_size = BLOCK_SIZE + ((BLOCK_SIZE) >> LOG_NUM_BANKS);

  // tmp for current state
  int *partial_sums;
  cudaMalloc(&partial_sums, sizeof(int) * grid_size);

  gpu_prescan<<<grid_size, BLOCK_SIZE / 2, sizeof(int) * tmp_size>>>(out, in, size, partial_sums);

  if (grid_size > BLOCK_SIZE)
  {
    cudaMemcpy(in, partial_sums, sizeof(int) * grid_size, cudaMemcpyDeviceToDevice);
    scan(in, partial_sums, grid_size);
  }
  else
  {
    gpu_prescan<<<1, BLOCK_SIZE / 2, sizeof(int) * tmp_size>>>(partial_sums, partial_sums,
                                                               grid_size, in);
  }

  add_prefix_inplace<<<grid_size, BLOCK_SIZE / 2>>>(out, size, partial_sums);

  cudaFree(partial_sums);
}

void radix_sort(EdgeList &E)
{
  E.sync_hostToDevice();

  int size = E.size();

  // int *cuda_in;
  int *cuda_out;
  // cudaMalloc(&cuda_in, sizeof(int) * size);
  cudaMalloc(&cuda_out, sizeof(int) * size);
  // cudaMemcpy(cuda_in, vec.data(), sizeof(int) * size, cudaMemcpyHostToDevice);

  int *cuda_ind_vec;
  int *cuda_ind_tmp;

  cudaMalloc((void **)&cuda_ind_vec, sizeof(int) * size);
  cudaMalloc((void **)&cuda_ind_tmp, sizeof(int) * size);

  std::vector<int> initial(size);
  std::iota(std::begin(initial), std::end(initial), 0);

  cudaMemcpy(cuda_ind_vec, initial.data(), sizeof(int) * size, cudaMemcpyHostToDevice);

  int grid_size = size / BLOCK_SIZE;
  if (!size % BLOCK_SIZE == 0)
    grid_size += 1;

  int *cuda_prefix_sums;
  int *cuda_block_sums;
  int *cuda_scan_block_sums;
  cudaMalloc(&cuda_prefix_sums, sizeof(int) * size);
  cudaMemset(cuda_prefix_sums, 0, sizeof(int) * size);

  cudaMalloc(&cuda_block_sums, sizeof(int) * SPLIT_SIZE * grid_size);
  cudaMemset(cuda_block_sums, 0, sizeof(int) * SPLIT_SIZE * grid_size);
  cudaMalloc(&cuda_scan_block_sums, sizeof(int) * SPLIT_SIZE * grid_size);
  cudaMemset(cuda_scan_block_sums, 0, sizeof(int) * SPLIT_SIZE * grid_size);

  int memory_size = (6 * BLOCK_SIZE + 2 * SPLIT_SIZE) * sizeof(int);

  for (int current_bits = 0; current_bits <= 30; current_bits += 2)
  {
    first_stage<<<grid_size, BLOCK_SIZE, memory_size>>>(E.d_val, cuda_out, size, cuda_ind_vec,
                                                        cuda_ind_tmp, cuda_prefix_sums,
                                                        cuda_block_sums, current_bits);
    scan(cuda_block_sums, cuda_scan_block_sums, SPLIT_SIZE * grid_size);

    final_sort<<<grid_size, BLOCK_SIZE>>>(cuda_out, E.d_val, size, cuda_ind_tmp, cuda_ind_vec,
                                          cuda_prefix_sums, cuda_scan_block_sums, current_bits);
  }
  cudaMemcpy(E.val, E.d_val, sizeof(int) * size, cudaMemcpyDeviceToHost);
  cudaMemcpy(initial.data(), cuda_ind_tmp, size * sizeof(int), cudaMemcpyDeviceToHost);
  E.set_owner(HOST);

  std::vector<int> tmp_vec1(E.coo1, E.coo1 + E.size());
  std::vector<int> tmp_vec2(E.coo2, E.coo2 + E.size());
  for (int i = 0; i < size; i++)
  {
    E.coo1[i] = tmp_vec1[initial[i]];
    E.coo2[i] = tmp_vec2[initial[i]];
  }

  // cudaFree(cuda_in);
  cudaFree(cuda_out);
  cudaFree(cuda_ind_tmp);
  cudaFree(cuda_ind_vec);
  cudaFree(cuda_scan_block_sums);
  cudaFree(cuda_block_sums);
  cudaFree(cuda_prefix_sums);
}
