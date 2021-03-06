#include "common.hpp"
#include "union_find.hpp"
#include <vector>

void UnionFind::compress(int kernel)
{
  g_benchmarker.start("compress()");

  switch (kernel)
  {
  case COMPRESS_NOTHING:
    break;

  case COMPRESS_DURING_FIND:
    break;

  case COMPRESS_KERNEL_CPU:
    compress_cpu_naive(parent);
    break;

  case COMPRESS_KERNEL_GPU:
    compress_gpu(parent);
    break;

  default:
    throw std::invalid_argument("Unknown compress kernel");
  }

  g_benchmarker.stop("compress()");
}

int find_pc(std::vector<int> &parent, int i)
{
  if (parent[i] == i)
    return i;
  else
  {
    int root = find_pc(parent, parent[i]);
    parent[i] = root;
    return root;
  }
}

// path compression on cpu
void compress_cpu_naive(std::vector<int> &parent)
{
  for (int i = 0; i < (int)parent.size(); ++i)
  {
    // if more than one step is neccessary to find the root...
    if (parent[parent[i]] != int(i))
    {
      find_pc(parent, int(i));
    }
  }
}

// path compression on gpu
__global__ void compress_kernel(int *parent, int *result, int size)
{

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;
  int pot_root;
  for (int i = thread_id; i < size; i += num_threads)
  {
    pot_root = parent[i];

    while (true)
    {
      if (parent[pot_root] == pot_root)
        break;
      else
        pot_root = parent[pot_root];
    }
    result[i] = pot_root;
  }
}

// limited path compression on gpu, that only compresses to a certain depth
__global__ void compress_kernel_limited(int *parent, int *result, int size, int limit)
{

  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int num_threads = blockDim.x * gridDim.x;

  for (int i = thread_id; i < size; i += num_threads)
  {
    result[i] = parent[i];
    for (int ii = 0; ii < limit; ++ii)
    {
      result[i] = parent[result[i]];
    }
  }
}

void compress_gpu(std::vector<int> &parent)
{

  size_t size = parent.size();
  int num_bytes = size * sizeof(int);

  // allocate
  int *d_parent, *d_result;
  cudaMalloc((void **)&d_parent, num_bytes);
  cudaMalloc((void **)&d_result, num_bytes);

  // copy
  cudaMemcpy(d_parent, parent.data(), num_bytes, cudaMemcpyHostToDevice);

  // compress

  compress_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_parent, d_result, size);

  // copy back
  cudaMemcpy(parent.data(), d_result, num_bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_parent);
  cudaFree(d_result);
}
