#include "common.hpp"
#include "edgelist.hpp"

void EdgeList::init_gpu() {}

void EdgeList::sync_hostToDevice()
{
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

  bytes = sizeof(EdgeList);
  cudaMalloc(&gpu, bytes);
  cudaMemcpy(gpu, this, bytes, cudaMemcpyHostToDevice);
}

void EdgeList::sync_deviceToHost()
{
  // coo1
  size_t bytes = sizeof(int) * coo1.size();
  cudaMemcpy(coo1.data(), d_coo1, bytes, cudaMemcpyDeviceToHost);

  // coo2
  bytes = sizeof(int) * coo2.size();
  cudaMemcpy(coo2.data(), d_coo2, bytes, cudaMemcpyDeviceToHost);

  // val
  bytes = sizeof(int) * val.size();
  cudaMemcpy(val.data(), d_val, bytes, cudaMemcpyDeviceToHost);
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

