#include "timer.hpp"

#include <fstream>
#include <iostream>
#include <vector>

#define RGB_COLOR_RANGE 255

using IntVec = std::vector<int>;

__global__ void histogram_original(int *buckets, int *colors, size_t n_colors)
{

  size_t n_threads = blockDim.x * gridDim.x;
  size_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  for (size_t i = thread_id; i < n_colors; i += n_threads)
  {
    int c = colors[i];
    atomicAdd(&buckets[c], 1);
  }
}

/******************
 *
 * Add other Kernels here
 *
 ******************/

IntVec generate_random_array(size_t len)
{
  IntVec arr(len);
  for (size_t i = 0; i < len; i++)
  {
    arr[i] = rand() % RGB_COLOR_RANGE;
  }
  return arr;
}

/******************
 *
 * Add other Array generating functions here
 *
 ******************/

/**
 * @brief
 *
 * TODO: I will look into a way to pass different kernels as arguments. so we dont have to
 * constantly rewrite that whole block for othe4r kenels (Hiti)
 * TODO: multiple repeasts for each kernel
 * TODO: results verification. For now I have only checked the results with the debugger. But they
 * seem alrigth
 *
 * @param h_colors
 */
void benchmark_kernel(IntVec const &h_colors, size_t n_blocks = 256, size_t n_threads = 256)
{
  Timer timer;

  int *d_colors;
  size_t bytes_colors = sizeof(int) * h_colors.size();
  cudaMalloc(&d_colors, bytes_colors);
  cudaMemcpy(d_colors, h_colors.data(), bytes_colors, cudaMemcpyHostToDevice);

  IntVec h_buckets(RGB_COLOR_RANGE, 0);
  int *d_buckets;
  size_t bytes_buckets = sizeof(int) * RGB_COLOR_RANGE;
  cudaMalloc(&d_buckets, bytes_buckets);
  cudaMemcpy(d_buckets, h_buckets.data(), bytes_buckets, cudaMemcpyHostToDevice);

  // Benchmark start
  cudaDeviceSynchronize();
  timer.reset();

  histogram_original<<<n_blocks, n_threads>>>(d_buckets, d_colors, h_colors.size());

  cudaDeviceSynchronize();
  double time = timer.get();
  // Benchmark end

  cudaMemcpy(h_buckets.data(), d_buckets, bytes_buckets, cudaMemcpyDeviceToHost);

  std::cout << "Benchmark finished after " << time << "s" << std::endl;

  cudaFree(d_colors);
  cudaFree(d_buckets);
}

int main()
{
  srand(0);
  IntVec colors = generate_random_array((int)1e7);
  benchmark_kernel(colors);
}