// first kernel for inclusive scan
// the whole array X to be scanned gets contigously distributed among the blocks.
// every block performs an inclusive scan on its array and writes the results to Y.
// Y then is partly inclusively scanned. what is still missing, is adding the end value of block
// i-1 to all elements in block i. This will later happen in a second kernel.
// all the blockoffsets get stored in carries.
__global__ void scan_kernel_1(int const *X, int *Y, int N, int *carries)
{
  __shared__ double shared_buffer[BLOCKSIZE];
  int my_value;

  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x * blockIdx.x;
  unsigned int block_stop = work_per_thread * blockDim.x * (blockIdx.x + 1);
  unsigned int block_offset = 0;

  // run scan on each section, this for loop is necessary if there are more elements in the array
  // than there are threads in total.
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
  {
    // load data:
    my_value = (i < N) ? X[i] : 0;

    // inclusive scan in shared buffer:
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
      __syncthreads();
      shared_buffer[threadIdx.x] = my_value;
      __syncthreads();
      if (threadIdx.x >= stride)
        my_value += shared_buffer[threadIdx.x - stride];
    }
    __syncthreads();
    shared_buffer[threadIdx.x] = my_value;
    __syncthreads();

    // write to output array
    if (i < N)
      Y[i] = block_offset + my_value;

    block_offset += shared_buffer[blockDim.x - 1];
  }

  // write carry:
  if (threadIdx.x == 0)
    carries[blockIdx.x] = block_offset;
}

// Y is partly inclusively scanned. Here the offsets of each block get added.
__global__ void scan_kernel_2(int *Y, int N, int const *carries)
{
  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x * blockIdx.x;
  unsigned int block_stop = work_per_thread * blockDim.x * (blockIdx.x + 1);

  __shared__ int shared_offset;
  __shared__ int shared_buffer[GRIDSIZE];

  // load data:
  int my_carry = carries[threadIdx.x];

  // inclusive scan in the carries array
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
  {
    __syncthreads();
    shared_buffer[threadIdx.x] = my_carry;
    __syncthreads();
    if (threadIdx.x >= stride)
      my_carry += shared_buffer[threadIdx.x - stride];
  }
  __syncthreads();
  shared_buffer[threadIdx.x] = my_carry;
  __syncthreads();

  if (threadIdx.x == 0)
    shared_offset = (blockIdx.x > 0) ? shared_buffer[blockIdx.x - 1] : 0;
  ;

  __syncthreads();

  // add offset to each element in the block:
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
    if (i < N)
      Y[i] += shared_offset;
}

// The general structure of the inclusive scan and most parts in the respective kernel codes are
// taken from Dr. Rupp's lecture "Computational Sciende on Many Core Architectures". We simplified
// it a bit to be restricted to work when GRIDSIZE == BLOCKSIZE. That saves us launching one
// intermediate kernel to perform an inclusive scan on the carries array.
void inclusive_scan(int const *input, int *output, int N)
{

  int *carries;
  cudaMalloc(&carries, sizeof(int) * GRIDSIZE);

  // First step: Scan within each thread group and write carries
  scan_kernel_1<<<GRIDSIZE, BLOCKSIZE>>>(input, output, N, carries);

  cudaDeviceSynchronize();

  // Second step: Offset each thread group accordingly
  scan_kernel_2<<<GRIDSIZE, BLOCKSIZE>>>(output, N, carries);

  cudaDeviceSynchronize();

  cudaFree(carries);
}