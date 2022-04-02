#include "timer.hpp"
#include "input.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

#define RGB_COLOR_RANGE 255
#define GRID_SIZE 256
#define BLOCK_SIZE 256

using IntVec = std::vector<int>;

__global__ void histogram_noloop(int *buckets, int *colors, size_t n_colors)
{
  int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (global_idx < n_colors)
  {
    int c = colors[global_idx];
    atomicAdd(&buckets[c], 1);
  }
}

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

bool verifyOutput(IntVec const &output, IntVec const &gt){

  for(size_t i = 0; i < output.size(); ++i){
    if(output[i] != gt[i]){
      std::cout << "Wrong output" << std::endl;
      return false;
    }
  }
  
  return true;
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
template <typename KERNEL>
double benchmark_kernel(KERNEL kernel, IntVec const &h_colors, IntVec const &gt, std::string kernel_name = "", 
                        size_t grid_size = GRID_SIZE, size_t block_size = BLOCK_SIZE,
                        size_t repetitions = 10)
{
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
  double start_time;
  double median_time = 0;
  Timer timer;

  // vector of execution times to calculate median time
  std::vector<double> exec_times;
  timer.reset();

  for (size_t j = 0; j < repetitions; j++)
  {
    cudaDeviceSynchronize(); // make sure gpu is ready
    start_time = timer.get();

    kernel<<<grid_size, block_size>>>(d_buckets, d_colors, h_colors.size());

    cudaDeviceSynchronize(); // make sure gpu is done
    exec_times.push_back(timer.get() - start_time);
  }

  std::sort(exec_times.begin(), exec_times.end());
  median_time = exec_times[int(repetitions / 2)];

  // Benchmark end

  cudaMemcpy(h_buckets.data(), d_buckets, bytes_buckets, cudaMemcpyDeviceToHost);
  verifyOutput(h_buckets, gt);

  cudaFree(d_colors);
  cudaFree(d_buckets);

  // TODO: proper output in .csv format
  std::cout << "Kernel " << kernel_name << " finished with median time = " << median_time << "s."
            << std::endl;

  return median_time;
}

int main()
{
  srand(0);
  // auto input_pair = input::loadImageFromFile("./input_data/sample.png", input::image_type::GRAYSCALE);
  // auto input_pair = input::generateUniformlyDistributedArray(1e-10, 5);
  auto input_pair = input::generateRandomArray(1e-20, 5);
  auto image = input_pair.first;
  auto gt = input_pair.second;
  benchmark_kernel(histogram_original, image, gt, "original loops");
  benchmark_kernel(histogram_noloop, image, gt, "original no loops");
}
