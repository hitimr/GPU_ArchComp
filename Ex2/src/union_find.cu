#include "common.hpp"
#include "union_find.hpp"
#include <vector>

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
  for (int i = 0; i < parent.size(); ++i)
  {
    // if more than one step is neccessary to find the root...
    if (parent[parent[i]] != i)
    {
      find_pc(parent, i);
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

void UnionFind::compress(int kernel)
{
  switch (kernel)
  {
  case COMPRESS_KERNEL_CPU_NAIVE:
    compress_cpu_naive(parent);
    break;
  case COMPRESS_KERNEL_GPU:
    compress_gpu(parent);
    break;

  default:
    throw std::invalid_argument("Unknown compress kernel");
  }
}
