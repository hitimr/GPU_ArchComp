#include "timer.hpp"
#include "input.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#define RGB_COLOR_RANGE 256
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
    {
        size_t index = (i+threadIdx.x)%RGB_COLOR_RANGE;
        atomicAdd(&buckets[index],loc_buc[index]);
    }
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

__global__ void histogram_linear(int *buckets, int *colors, size_t n_colors)
{
  // shared memory within the block
  __shared__ int local_sums[BLOCK_SIZE];

  // thread local variable
  int sum = 0;

  for (unsigned int i = threadIdx.x; i < n_colors; i += blockDim.x)
  {
    if (colors[i] == blockIdx.x)
      sum += 1;
  }

  local_sums[threadIdx.x] = sum;
  
  for (unsigned int range = blockDim.x / 2; range > 0; range /= 2)
  {
    __syncthreads();
    if (threadIdx.x < range)
      local_sums[threadIdx.x] += local_sums[threadIdx.x + range];
  }

  if (threadIdx.x == 0)
    buckets[blockIdx.x] = local_sums[0];
  
}

__global__ void histogram_block_partition(int *buckets, int *colors, size_t n_colors)
{
  int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
  int num_threads = gridDim.x * blockDim.x;

  // shared memory within the block
  __shared__ int local_buckets[RGB_COLOR_RANGE];
  for(size_t i = threadIdx.x; i < RGB_COLOR_RANGE; i+=blockDim.x)
  {
      local_buckets[i] = 0;
  }

   __syncthreads();  
  for (unsigned int i = global_idx; i < n_colors; i += num_threads)
  {
    int c = colors[i];
    atomicAdd(&local_buckets[c], 1);
  }

  __syncthreads();
  {
    for(size_t i = threadIdx.x; i < RGB_COLOR_RANGE; i+=blockDim.x)
    {
      atomicAdd(&buckets[i], local_buckets[i]);
    }
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
double benchmark_kernel(KERNEL kernel, IntVec const &h_colors, IntVec const &gt, std::string filename, std::string kernel_name = "", std::string input_name = "",
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
    cudaMemset(d_buckets, 0, bytes_buckets);
    cudaDeviceSynchronize(); // make sure gpu is ready
    start_time = timer.get();
    
    kernel<<<grid_size, block_size>>>(d_buckets, d_colors, h_colors.size());

    cudaDeviceSynchronize(); // make sure gpu is done
    exec_times.push_back(timer.get() - start_time);
  }

  std::sort(exec_times.begin(), exec_times.end());
  median_time = exec_times[int(repetitions / 2)];

  // Benchmark end

  // TODO: proper output in .csv format
  std::cout << "Kernel " << kernel_name << " finished with median time = " << median_time << "s."
            << std::endl;

  cudaMemcpy(h_buckets.data(), d_buckets, bytes_buckets, cudaMemcpyDeviceToHost);
  verifyOutput(h_buckets, gt);

  cudaFree(d_colors);
  cudaFree(d_buckets);



  std::ofstream stream;
  stream.open(filename, std::ios::app);
  stream << "\n" << kernel_name << "," << input_name << "," << median_time;
  stream.close();

  return median_time;
}


void run_benchmark(const std::pair<std::vector<int>, std::vector<int>>& input_pair, std::string filename)
{
  std::ofstream stream;
  stream.open(filename);
  stream << "kernel,input,runtime";
  stream.close();

  auto image = input_pair.first;
  auto gt = input_pair.second;
  benchmark_kernel(histogram_original, image, gt, filename, "histogram_original", "random_1");
  // benchmark_kernel(histogram_noloop, image, gt, filename, "histogram_noloop", "random_1");
	benchmark_kernel(histogram_tlb, image, gt, filename, "histogram_tlb", "random_1");
  benchmark_kernel(histogram_tlb_blr, image, gt, filename, "histogram_tlb_blr", "random_1");
  benchmark_kernel(histogram_linear, image, gt, filename, "histogram_linear", "random_1");
  benchmark_kernel(histogram_block_partition, image, gt, filename, "histogram_block_partition", "random_1");

  image = input_pair.first;
  gt = input_pair.second;
  benchmark_kernel(histogram_original, image, gt, filename, "histogram_original", "random_2");
  // benchmark_kernel(histogram_noloop, image, gt, filename, "histogram_noloop", "random_2");
	benchmark_kernel(histogram_tlb, image, gt, filename, "histogram_tlb", "random_2");
  benchmark_kernel(histogram_tlb_blr, image, gt, filename, "histogram_tlb_blr", "random_2");
  benchmark_kernel(histogram_linear, image, gt, filename, "histogram_linear", "random_2");
  benchmark_kernel(histogram_block_partition, image, gt, filename, "histogram_block_partition", "random_2");

  image = input_pair.first;
  gt = input_pair.second;
  benchmark_kernel(histogram_original, image, gt, filename, "histogram_original", "random_5");
  // benchmark_kernel(histogram_noloop, image, gt, filename, "histogram_noloop", "random_5");
	benchmark_kernel(histogram_tlb, image, gt, filename, "histogram_tlb", "random_5");
  benchmark_kernel(histogram_tlb_blr, image, gt, filename, "histogram_tlb_blr", "random_5");
  benchmark_kernel(histogram_linear, image, gt, filename, "histogram_linear", "random_5");
  benchmark_kernel(histogram_block_partition, image, gt, filename, "histogram_block_partition", "random_5");

  image = input_pair.first;
  gt = input_pair.second;
  benchmark_kernel(histogram_original, image, gt, filename, "histogram_original", "random_256");
  // benchmark_kernel(histogram_noloop, image, gt, filename, "histogram_noloop", "random_256");
	benchmark_kernel(histogram_tlb, image, gt, filename, "histogram_tlb", "random_256");
  benchmark_kernel(histogram_tlb_blr, image, gt, filename, "histogram_tlb_blr", "random_256");
  benchmark_kernel(histogram_linear, image, gt, filename, "histogram_linear", "random_256");
  benchmark_kernel(histogram_block_partition, image, gt, filename, "histogram_block_partition", "random_256");
}

int main()
{
  // Random data 1 color
  {    
    std::cout << "Benchmarking with single color" << std::endl;
    std::vector<int> random_sizes({640*426, 1920*1080, 3840*2160});
    std::vector<std::string> fileNames({"output/random1_640x426.csv", "output/random1_1920x1080.csv", "output/random1_3840x2160.csv"});

    for (size_t i = 0; i < random_sizes.size(); i++)
    {
      auto input_pair = input::generateRandomArray(random_sizes[i], 1);
      run_benchmark(input_pair, fileNames[i]);
    }
  }

  // Random data 2 colors
  {    
    std::cout << "Benchmarking with 2 colors (approx. 50/50)" << std::endl;
    std::vector<int> random_sizes({640*426, 1920*1080, 3840*2160});
    std::vector<std::string> fileNames({"output/random2_640x426.csv", "output/random2_1920x1080.csv", "output/random2_3840x2160.csv"});

    for (size_t i = 0; i < random_sizes.size(); i++)
    {
      auto input_pair = input::generateRandomArray(random_sizes[i], 2);
      run_benchmark(input_pair, fileNames[i]);
    }
  }

  // Random data 256 colors
  {    
    std::cout << "Benchmarking with 256 colors, random distribution" << std::endl;
    std::vector<int> random_sizes({640*426, 1920*1080, 3840*2160});
    std::vector<std::string> fileNames({"output/random256_640x426.csv", "output/random256_1920x1080.csv", "output/random256_3840x2160.csv"});

    for (size_t i = 0; i < random_sizes.size(); i++)
    {
      auto input_pair = input::generateRandomArray(random_sizes[i], 2);
      run_benchmark(input_pair, fileNames[i]);
    }
  }

  // uniform distributed data 256 colors
  {    
    std::cout << "Benchmarking uniform disitribution 256 colors" << std::endl;
    std::vector<int> random_sizes({640*426, 1920*1080, 3840*2160});
    std::vector<std::string> fileNames({"output/uniform_640x426.csv", "output/uniform_1920x1080.csv", "output/uniform_3840x2160.csv"});

    for (size_t i = 0; i < random_sizes.size(); i++)
    {
      auto input_pair = input::generateUniformlyDistributedArray(random_sizes[i], RGB_COLOR_RANGE);
      run_benchmark(input_pair, fileNames[i]);
    }
  }

  // real images
  {
    std::cout << "Benchmarking real images" << std::endl;
    std::vector<std::string> infiles({"input_data/sample_640x426.png", "input_data/sample_1280x853.png", "input_data/sample_1920x1280.png", "input_data/sample_5184x3456.png"});
    std::vector<std::string> fileNames({"output/sample_640x426.csv", "output/sample_1280x853.csv", "output/sample_1920x1280.csv", "output/sample_5184x3456.csv"});

    for (size_t i = 0; i < infiles.size(); i++)
    {
      auto input_pair = input::loadImageFromFile(infiles[i].c_str(), input::image_type::GRAYSCALE);
      run_benchmark(input_pair, fileNames[i]);
    }
  }
}
