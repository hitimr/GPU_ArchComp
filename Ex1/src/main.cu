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

__global__ void histogram_tlb(int* buckets, int* pixels, int num_pixels) // tlb: thread-local buckets
//__global__ void histogram_tlb(int* pixels, int num_pixels, int* buckets) // tlb: thread-local buckets
{
    int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    int loc_buc[RGB_COLOR_RANGE];

    for(int i = 0; i < RGB_COLOR_RANGE; ++i)
        loc_buc[i] = 0;

    int c;
    for(int i = global_idx; i < num_pixels; i += num_threads){
        c = pixels[i];
        loc_buc[c]++;
    }

    for(int i = 0; i < RGB_COLOR_RANGE; ++i)
        atomicAdd(&buckets[i],loc_buc[i]);
}


__global__ void histogram_tlb_blr(int* buckets, int* pixels, int num_pixels)  // tlb_blr: thread-local buckets, block-level reduction
//__global__ void histogram_tlb_blr(int* pixels, int num_pixels, int* buckets) // tlb_blr: thread-local buckets, block-level reduction
{ 
    int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int num_threads = gridDim.x * blockDim.x;
    int loc_buc[RGB_COLOR_RANGE];
    int c;

    for(int i = 0; i < RGB_COLOR_RANGE; ++i)
        loc_buc[i] = 0;

    // fill thread - local buckets
    for(int i = global_idx; i < num_pixels; i += num_threads){
        c = pixels[i];
        loc_buc[c]++;
    }

    // perform block level reduction for each color
    // ----------------------------------------------
    // this shared array can only hold one single color at once - so we recycle it once for each color (we do not have enough SM-local memory for a 256x256 array)
    __shared__ int block_buc[BLOCK_SIZE]; 
    for(int col_idx = 0; col_idx < RGB_COLOR_RANGE; ++col_idx){

        // initialize shared array with thread-local resuls
        block_buc[threadIdx.x] = loc_buc[col_idx];

        // do the block-level reduction
        for(int stride = blockDim.x/2; stride>0; stride/=2){
            __syncthreads();
            if (threadIdx.x < stride)
                block_buc[threadIdx.x] += block_buc[threadIdx.x + stride];
    	}

        // one atomicAdd() by the index-0-thread
        if(threadIdx.x == 0)
            atomicAdd(&buckets[col_idx], block_buc[0]);
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
