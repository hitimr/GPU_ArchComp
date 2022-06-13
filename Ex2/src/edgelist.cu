#include "common.hpp"
#include "edgelist.hpp"

void EdgeList::init_gpu() {}

void EdgeList::sync_hostToDevice()
{
  if (owner == DEVICE)
  {
    // Nothing to do
    return;
  }

  cudaDeviceSynchronize();

  cudaFree(d_coo1);
  cudaFree(d_coo2);
  cudaFree(d_val);

  // coo1
  size_t bytes = sizeof(int) * size();
  cudaMalloc(&d_coo1, bytes);
  cudaMemcpy(d_coo1, coo1, bytes, cudaMemcpyHostToDevice);

  // coo2
  bytes = sizeof(int) * size();
  cudaMalloc(&d_coo2, bytes);
  cudaMemcpy(d_coo2, coo2, bytes, cudaMemcpyHostToDevice);

  // val
  bytes = sizeof(int) * size();
  cudaMalloc(&d_val, bytes);
  cudaMemcpy(d_val, val, bytes, cudaMemcpyHostToDevice);

  owner = DEVICE;
}

void EdgeList::sync_deviceToHost()
{
  if (owner == HOST)
  {
    // Nothing to do
    return;
  }

  cudaDeviceSynchronize();

  // coo1
  size_t bytes = sizeof(int) * size();
  cudaMemcpy(coo1, d_coo1, bytes, cudaMemcpyDeviceToHost);

  // coo2
  bytes = sizeof(int) * size();
  cudaMemcpy(coo2, d_coo2, bytes, cudaMemcpyDeviceToHost);

  // val
  bytes = sizeof(int) * size();
  cudaMemcpy(val, d_val, bytes, cudaMemcpyDeviceToHost);
  
  owner = HOST;
}

void EdgeList::reserve(size_t new_size)
{
  assert(new_size >= size());

  // reserve on CPU
  size_t bytes = new_size * sizeof(int);
  if (g_options.count("pinned-memory"))
  {
    // Use pinned memory
    cudaMallocHost(&val, bytes);
    cudaMallocHost(&coo1, bytes);
    cudaMallocHost(&coo2, bytes);
  }
  else
  {
    // use regular memory
    val = new int[new_size];
    coo1 = new int[new_size];
    coo2 = new int[new_size];
  }

  // Reserve on GPU
  cudaMalloc(&d_coo1, bytes);
  cudaMalloc(&d_coo2, bytes);
  cudaMalloc(&d_val, bytes);
}

void EdgeList::resize_and_set_num_edges(size_t size)
{
  // reserve on CPU
  size_t bytes = size * sizeof(int);
  if (g_options.count("pinned-memory"))
  {
    // Use pinned memory
    cudaMallocHost(&val, bytes);
    cudaMallocHost(&coo1, bytes);
    cudaMallocHost(&coo2, bytes);
  }
  else
  {
    // use regular memory
    val = new int[size];
    coo1 = new int[size];
    coo2 = new int[size];
  }

  num_edges = size;

  cudaFree(d_coo1);
  cudaFree(d_coo2);
  cudaFree(d_val);

  cudaMalloc(&d_coo1, bytes);
  cudaMalloc(&d_coo2, bytes);
  cudaMalloc(&d_val, bytes);
}
