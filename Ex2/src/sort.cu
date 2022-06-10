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

#define MAX_BLOCK_SZ 128

#define MAX_BLOCK_SZ 128
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define SPLIT_SIZE 4

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

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

  case SORT_KERNEL_RADIX:
    radix_sort(E);
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
void improved_mergesort_three(EdgeList &E)
// &vec2)
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

__global__ void gpu_prescan(int *const d_out, const int *const d_in, int *const partial_sums,
                            const int len, const int tmp_sz, const int max_elems_per_block)
{
  // Allocated on invocation
  extern __shared__ int s_out[];

  int thid = threadIdx.x;
  int ai = thid;
  int bi = thid + blockDim.x;

  s_out[thid] = 0;
  s_out[thid + blockDim.x] = 0;

  s_out[thid + blockDim.x + (blockDim.x >> LOG_NUM_BANKS)] = 0;

  __syncthreads();

  int cpy_idx = max_elems_per_block * blockIdx.x + threadIdx.x;
  if (cpy_idx < len)
  {
    s_out[ai + CONFLICT_FREE_OFFSET(ai)] = d_in[cpy_idx];
    if (cpy_idx + blockDim.x < len)
      s_out[bi + CONFLICT_FREE_OFFSET(bi)] = d_in[cpy_idx + blockDim.x];
  }

  int offset = 1;
  for (int d = max_elems_per_block >> 1; d > 0; d >>= 1)
  {
    __syncthreads();

    if (thid < d)
    {
      int ai = offset * ((thid << 1) + 1) - 1;
      int bi = offset * ((thid << 1) + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      s_out[bi] += s_out[ai];
    }
    offset <<= 1;
  }

  if (thid == 0)
  {
    partial_sums[blockIdx.x] =
        s_out[max_elems_per_block - 1 + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)];
    s_out[max_elems_per_block - 1 + CONFLICT_FREE_OFFSET(max_elems_per_block - 1)] = 0;
  }

  for (int d = 1; d < max_elems_per_block; d <<= 1)
  {
    offset >>= 1;
    __syncthreads();

    if (thid < d)
    {
      int ai = offset * ((thid << 1) + 1) - 1;
      int bi = offset * ((thid << 1) + 2) - 1;
      ai += CONFLICT_FREE_OFFSET(ai);
      bi += CONFLICT_FREE_OFFSET(bi);

      int temp = s_out[ai];
      s_out[ai] = s_out[bi];
      s_out[bi] += temp;
    }
  }
  __syncthreads();

  // Copy contents of shared memory to global memory
  if (cpy_idx < len)
  {
    d_out[cpy_idx] = s_out[ai + CONFLICT_FREE_OFFSET(ai)];
    if (cpy_idx + blockDim.x < len)
      d_out[cpy_idx + blockDim.x] = s_out[bi + CONFLICT_FREE_OFFSET(bi)];
  }
}

void sum_scan_blelloch(int *const d_out, const int *const d_in, int size)
{
  // Zero out d_out
  cudaMemset(d_out, 0, size * sizeof(int));

  // Set up number of threads and blocks

  int block_sz = MAX_BLOCK_SZ / 2;
  int max_elems_per_block = 2 * block_sz; // due to binary tree nature of algorithm

  int grid_sz = ceilf(size / max_elems_per_block);

  if (size % max_elems_per_block != 0)
    grid_sz += 1;

  int tmp_sz = max_elems_per_block + ((max_elems_per_block) >> LOG_NUM_BANKS);

  int *partial_sums;
  cudaMalloc(&partial_sums, sizeof(int) * grid_sz);
  cudaMemset(partial_sums, 0, sizeof(int) * grid_sz);

  gpu_prescan<<<grid_sz, block_sz, sizeof(int) * tmp_sz>>>(d_out, d_in, partial_sums, size, tmp_sz,
                                                           max_elems_per_block);

  if (grid_sz <= max_elems_per_block)
  {
    int *d_dummy_blocks_sums;
    cudaMalloc(&d_dummy_blocks_sums, sizeof(int));
    cudaMemset(d_dummy_blocks_sums, 0, sizeof(int));
    gpu_prescan<<<1, block_sz, sizeof(int) * tmp_sz>>>(
        partial_sums, partial_sums, d_dummy_blocks_sums, grid_sz, tmp_sz, max_elems_per_block);
    cudaDeviceSynchronize();
    cudaFree(d_dummy_blocks_sums);
  }

  else
  {
    int *d_in_block_sums;
    cudaMalloc(&d_in_block_sums, sizeof(int) * grid_sz);
    cudaMemcpy(d_in_block_sums, partial_sums, sizeof(int) * grid_sz, cudaMemcpyDeviceToDevice);
    sum_scan_blelloch(partial_sums, d_in_block_sums, grid_sz);
    cudaFree(d_in_block_sums);
  }

  add_prefix_inplace<<<grid_sz, block_sz>>>(d_out, size, partial_sums);

  cudaFree(partial_sums);
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

  // Scan mask output sums
  // Just do a naive scan since the array is really small
  if (threadIdx.x == 0)
  {
    int run_sum = 0;
    for (int i = 0; i < 4; ++i)
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

void radix_sort(EdgeList &E)
{
  E.sync_hostToDevice();

  int size = E.size();
  int split_size = 4;

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

  cudaMalloc(&cuda_block_sums, sizeof(int) * split_size * grid_size);
  cudaMemset(cuda_block_sums, 0, sizeof(int) * split_size * grid_size);
  cudaMalloc(&cuda_scan_block_sums, sizeof(int) * split_size * grid_size);
  cudaMemset(cuda_scan_block_sums, 0, sizeof(int) * split_size * grid_size);

  int memory_size = (6 * BLOCK_SIZE + 2 * split_size) * sizeof(int);

  for (int current_bits = 0; current_bits <= 30; current_bits += 2)
  {
    first_stage<<<grid_size, BLOCK_SIZE, memory_size>>>(E.d_val, cuda_out, size, cuda_ind_vec,
                                                        cuda_ind_tmp, cuda_prefix_sums,
                                                        cuda_block_sums, current_bits);

    int current_size = split_size * grid_size;
    int block_sz = BLOCK_SIZE / 2;
    int max_elems_per_block = BLOCK_SIZE;

    int current_grid_size = current_size / BLOCK_SIZE;
    if (!current_size % BLOCK_SIZE == 0)
      current_grid_size += 1;

    int tmp_sz = max_elems_per_block + ((max_elems_per_block) >> LOG_NUM_BANKS);

    int *partial_sums;
    cudaMalloc(&partial_sums, sizeof(int) * current_grid_size);

    while (true)
    {
      cudaMemset(cuda_scan_block_sums, 0, current_size * sizeof(int));
      cudaMemset(partial_sums, 0, sizeof(int) * current_grid_size);

      gpu_prescan<<<current_grid_size, block_sz, sizeof(int) * tmp_sz>>>(
          cuda_scan_block_sums, cuda_block_sums, partial_sums, current_size, tmp_sz,
          max_elems_per_block);

      if (current_grid_size <= max_elems_per_block)
      {
        int *d_dummy_blocks_sums;
        cudaMalloc(&d_dummy_blocks_sums, sizeof(int));
        cudaMemset(d_dummy_blocks_sums, 0, sizeof(int));
        gpu_prescan<<<1, block_sz, sizeof(int) * tmp_sz>>>(partial_sums, partial_sums,
                                                           d_dummy_blocks_sums, current_grid_size,
                                                           tmp_sz, max_elems_per_block);
        cudaDeviceSynchronize();
        cudaFree(d_dummy_blocks_sums);
        break;
      }
      else
      {
        int *d_in_block_sums;
        cudaMalloc(&d_in_block_sums, sizeof(int) * current_grid_size);
        cudaMemcpy(d_in_block_sums, partial_sums, sizeof(int) * current_grid_size,
                   cudaMemcpyDeviceToDevice);
        sum_scan_blelloch(partial_sums, d_in_block_sums, current_grid_size);
        cudaFree(d_in_block_sums);
        break;
      }
    }

    add_prefix_inplace<<<current_grid_size, block_sz>>>(cuda_scan_block_sums, current_size,
                                                        partial_sums);

    cudaFree(partial_sums);

    final_sort<<<grid_size, BLOCK_SIZE>>>(cuda_out, E.d_val, size, cuda_ind_tmp, cuda_ind_vec,
                                          cuda_prefix_sums, cuda_scan_block_sums, current_bits);
  }
  cudaMemcpy(E.val.data(), E.d_val, sizeof(int) * size, cudaMemcpyDeviceToHost);
  cudaMemcpy(initial.data(), cuda_ind_tmp, size * sizeof(int), cudaMemcpyDeviceToHost);
  E.set_owner(HOST);

  std::vector<int> tmp_vec1, tmp_vec2;
  tmp_vec1 = E.coo1;
  tmp_vec2 = E.coo2;
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
