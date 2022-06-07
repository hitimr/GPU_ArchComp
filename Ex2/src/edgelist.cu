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

  // coo1
  size_t bytes = sizeof(int) * coo1.size();
  cudaMalloc(&d_coo1, bytes);
  cudaMemcpy(d_coo1, coo1.data(), bytes, cudaMemcpyHostToDevice);

  // coo2
  bytes = sizeof(int) * coo2.size();
  cudaMalloc(&d_coo2, bytes);
  cudaMemcpy(d_coo2, coo2.data(), bytes, cudaMemcpyHostToDevice);

  // val
  bytes = sizeof(int) * val.size();
  cudaMalloc(&d_val, bytes);
  cudaMemcpy(d_val, val.data(), bytes, cudaMemcpyHostToDevice);

  // constants
  bytes = sizeof(EdgeList);
  cudaMalloc(&gpu, bytes);
  cudaMemcpy(gpu, this, bytes, cudaMemcpyHostToDevice);

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
  size_t bytes = sizeof(int) * coo1.size();
  cudaMemcpy(coo1.data(), d_coo1, bytes, cudaMemcpyDeviceToHost);

  // coo2
  bytes = sizeof(int) * coo2.size();
  cudaMemcpy(coo2.data(), d_coo2, bytes, cudaMemcpyDeviceToHost);

  // val
  bytes = sizeof(int) * val.size();
  cudaMemcpy(val.data(), d_val, bytes, cudaMemcpyDeviceToHost);

  owner = HOST;
}

void EdgeList::reserve(size_t size)
{
  assert(size >= coo1.size());

  coo1.reserve(size);
  coo2.reserve(size);
  val.reserve(size);

  size_t bytes = size * sizeof(int);
  cudaMalloc(&d_coo1, bytes);
  cudaMalloc(&d_coo2, bytes);
  cudaMalloc(&d_val, bytes);
}

void EdgeList::resize_and_set_num_edges(size_t size)
{
  coo1.resize(size);
  coo2.resize(size);
  val.resize(size);
  num_edges = size;

  cudaFree(d_coo1);
  cudaFree(d_coo2);
  cudaFree(d_val);

  size_t bytes = size * sizeof(int);
  cudaMalloc(&d_coo1, bytes);
  cudaMalloc(&d_coo2, bytes);
  cudaMalloc(&d_val, bytes);
}
