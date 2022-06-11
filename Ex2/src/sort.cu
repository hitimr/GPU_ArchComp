#include "timer.hpp"
#include <math.h>    
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>


#define BLOCK_SIZE 512
#define GRID_SIZE 512

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define SPLIT_SIZE 4


#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif


__global__ void final_sort(int* in, int* out, int size,
    int* in_ind, int* out_ind,
    int* prefixes,
    int* block_sums,
    int bit_shift)
{
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x)
    {
        int mask = (in[i] >> bit_shift) & 3;
        int new_pos = block_sums[mask * gridDim.x + blockIdx.x] + prefixes[i];
        out[new_pos] = in[i];
        out_ind[new_pos] = in_ind[i];
    }
}


__global__ void add_prefix_inplace(int* in, int size,
    int* partial_sums
    )
{
    for(int i = 2 * blockIdx.x * blockDim.x + threadIdx.x; i < size; i += 2*blockDim.x*gridDim.x)
    {
        in[i] = in[i] + partial_sums[blockIdx.x];
        if (i + blockDim.x < size)
            in[i + blockDim.x] = in[i + blockDim.x] + partial_sums[blockIdx.x];
    }

}

//https://www.eecs.umich.edu/courses/eecs570/hw/parprefix.pdf
__global__ void gpu_prescan(int* out, int* in, int size,
    int* partial_sums)
{
    extern __shared__ int tmp[];

    int global_thread_id = 2 * blockDim.x * blockIdx.x + threadIdx.x;

    if(global_thread_id < blockDim.x * 2){
        tmp[global_thread_id] = 0;
    }

    __syncthreads();

    if (global_thread_id < size)
    {
        tmp[threadIdx.x + CONFLICT_FREE_OFFSET(threadIdx.x)] = in[global_thread_id];
        if (global_thread_id + blockDim.x < size)
            tmp[threadIdx.x + blockDim.x + CONFLICT_FREE_OFFSET(threadIdx.x + blockDim.x)] = in[global_thread_id + blockDim.x];
    }

    int offset = 1;
    for (int d = 2 * blockDim.x >> 1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (threadIdx.x < d)
        {
            int ai = offset * (threadIdx.x * 2 + 1) - 1;
            int bi = offset * (threadIdx.x * 2 + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            tmp[bi] += tmp[ai];
        }
        offset <<= 1;
    }

    if (threadIdx.x == 0) 
    { 
        //to be reused later
        partial_sums[blockIdx.x] = tmp[2 * blockDim.x - 1 
            + CONFLICT_FREE_OFFSET(2 * blockDim.x - 1)];
        tmp[2 * blockDim.x - 1 
            + CONFLICT_FREE_OFFSET(2 * blockDim.x - 1)] = 0;
    }
    for (int d = 1; d < 2 * blockDim.x; d <<= 1)
    {
        offset >>= 1;
        __syncthreads();

        if (threadIdx.x < d)
        {
            int ai = offset * ((threadIdx.x << 1) + 1) - 1;
            int bi = offset * ((threadIdx.x << 1) + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            int temp = tmp[ai];
            tmp[ai] = tmp[bi];
            tmp[bi] += temp;
        }
    }
    __syncthreads();
    if (global_thread_id < size)
    {
        out[global_thread_id] = tmp[threadIdx.x + CONFLICT_FREE_OFFSET(threadIdx.x)];
        if (global_thread_id + blockDim.x < size)
            out[global_thread_id + blockDim.x] = tmp[threadIdx.x + blockDim.x + CONFLICT_FREE_OFFSET(threadIdx.x + blockDim.x)];
    }
}
 
void scan(int* in, int* out,
    int size)
{
    int grid_size = (size / BLOCK_SIZE) + (size % BLOCK_SIZE != 0);
    int tmp_size = BLOCK_SIZE + ((BLOCK_SIZE) >> LOG_NUM_BANKS);

    int* partial_sums;
    cudaMalloc(&partial_sums, sizeof(int) * grid_size);

    gpu_prescan<<<grid_size, BLOCK_SIZE/2, sizeof(int) * tmp_size>>>(out, 
        in, 
        size,
        partial_sums
    );


    if (grid_size > BLOCK_SIZE)
    {
        cudaMemcpy(in, partial_sums, sizeof(int) * grid_size, cudaMemcpyDeviceToDevice);
        scan(in, partial_sums, grid_size);
    }
    else
    {
        gpu_prescan<<<1, BLOCK_SIZE/2, sizeof(int) * tmp_size>>>(partial_sums, 
            partial_sums, 
            grid_size,
            in
        );
    }

    add_prefix_inplace<<<grid_size, BLOCK_SIZE/2>>>(out, size, partial_sums);

    cudaFree(partial_sums);
}

__global__ void first_stage(int* in, int* out, int size,
                            int *in_ind, int *out_ind,
                            int* prefixes,
                            int* partial_sums,
                            int bit_shift
    )
{
    int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    
    extern __shared__ int tmp[];

    int* masks = tmp + blockDim.x + 1;
    int* scan_output = masks + blockDim.x + 1;
    int* masks_sums = scan_output + blockDim.x + 1;
    int* scan_sums = masks_sums + SPLIT_SIZE + 1;
    int* tmp_ind = scan_sums + blockDim.x + 1;

    for(int i = thread_id; i < size; i += (blockDim.x * gridDim.x))
        {
            tmp[threadIdx.x] = in[i];
            tmp_ind[threadIdx.x] = in_ind[i];
        }

    int current_value = tmp[threadIdx.x];
    int current_ind = tmp_ind[threadIdx.x];
    int current_bits = (current_value >> bit_shift) & 3;

    for (int i = 0; i < SPLIT_SIZE; ++i)
    {
        __syncthreads();

        if (thread_id < size)
        {
            masks[threadIdx.x + 1] = current_bits == i;
        }
        else
        {
            masks[threadIdx.x + 1] = 0;
        }
        __syncthreads();

        int sum = 0;
        for (int d = 1; d < blockDim.x + 1; d*=2) {
            if (threadIdx.x >= d) {
                sum = masks[threadIdx.x + 1] + masks[threadIdx.x + 1 - d];
            }
            else {
                sum = masks[threadIdx.x + 1];
            }
            __syncthreads();
            masks[threadIdx.x + 1] = sum;
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            masks[0] = 0;
            masks_sums[i] = masks[blockDim.x];
            partial_sums[i * gridDim.x + blockIdx.x] = masks[blockDim.x];
        }

        if (current_bits == i && (thread_id < size))
        {
            scan_output[threadIdx.x] = masks[threadIdx.x];
        }

    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        int run_sum = 0;
        for (int i = 0; i < 4; ++i)
        {
            scan_sums[i] = run_sum;
            run_sum += masks_sums[i];
        }
    }

    __syncthreads();

    if (thread_id < size)
    {
        int tmp_scan = scan_output[threadIdx.x];
        int new_pos = scan_output[threadIdx.x] + scan_sums[current_bits];
        __syncthreads();
        tmp[new_pos] = current_value;
        tmp_ind[new_pos] = current_ind; 
        scan_output[new_pos] = tmp_scan;
        __syncthreads();
        prefixes[thread_id] = scan_output[threadIdx.x];
        out[thread_id] = tmp[threadIdx.x];
        out_ind[thread_id] = tmp_ind[threadIdx.x];
    }
}

void radix_sort(std::vector<int> &vec, std::vector<int> &vec1, std::vector<int> &vec2)
{
    int size = vec.size();
    int split_size = 4;

    int* cuda_in;
    int* cuda_out;
    cudaMalloc(&cuda_in, sizeof(int) * size);
    cudaMalloc(&cuda_out, sizeof(int) * size);
    cudaMemcpy(cuda_in, vec.data(), sizeof(int) * size, cudaMemcpyHostToDevice);

    int* cuda_ind_vec;
    int* cuda_ind_tmp;

    cudaMalloc((void**)&cuda_ind_vec, sizeof(int) * size);
    cudaMalloc((void**)&cuda_ind_tmp, sizeof(int) * size);
    
    std::vector<int> initial(size);
    std::iota (std::begin(initial), std::end(initial), 0);

    cudaMemcpy(cuda_ind_vec, initial.data(), sizeof(int) * size, cudaMemcpyHostToDevice);


    int grid_size = size / BLOCK_SIZE;
    if(!size%BLOCK_SIZE==0) grid_size += 1;

    int* cuda_prefix_sums;
    int* cuda_block_sums;
    int* cuda_scan_block_sums;
    cudaMalloc(&cuda_prefix_sums, sizeof(int) * size);
    cudaMemset(cuda_prefix_sums, 0, sizeof(int) * size);

    cudaMalloc(&cuda_block_sums, sizeof(int) * split_size * grid_size);
    cudaMemset(cuda_block_sums, 0, sizeof(int) * split_size * grid_size);
    cudaMalloc(&cuda_scan_block_sums, sizeof(int) * split_size * grid_size);
    cudaMemset(cuda_scan_block_sums, 0, sizeof(int) * split_size * grid_size);

    int memory_size = (6*BLOCK_SIZE + 2*split_size) * sizeof(int);

    for (int current_bits = 0; current_bits <= 30; current_bits += 2)
    {
        first_stage<<<grid_size, BLOCK_SIZE, memory_size>>>(
                                                            cuda_in, cuda_out, size,
                                                            cuda_ind_vec, cuda_ind_tmp,
                                                            cuda_prefix_sums, 
                                                            cuda_block_sums, 
                                                            current_bits
                                                        );
        scan(cuda_block_sums, cuda_scan_block_sums, split_size * grid_size);


        final_sort<<<grid_size, BLOCK_SIZE>>>(cuda_out, cuda_in, size,
                                            cuda_ind_tmp, cuda_ind_vec,
                                            cuda_prefix_sums, 
                                            cuda_scan_block_sums, 
                                            current_bits
                                        );
    }
    cudaMemcpy(vec.data(), cuda_in, sizeof(int) * size, cudaMemcpyDeviceToHost);
    cudaMemcpy(initial.data(), cuda_ind_tmp, size * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> tmp_vec1, tmp_vec2;
    tmp_vec1 = vec1;
    tmp_vec2 = vec2;
    for(int i = 0; i < size; i++){
        vec1[i] = tmp_vec1[initial[i]];
        vec2[i] = tmp_vec2[initial[i]];
    }

    
    cudaFree(cuda_in);
    cudaFree(cuda_out);
    cudaFree(cuda_ind_tmp);
    cudaFree(cuda_ind_vec);
    cudaFree(cuda_scan_block_sums);
    cudaFree(cuda_block_sums);
    cudaFree(cuda_prefix_sums);
}

__device__ void gpu_swap(int* vec, size_t i, size_t j){
    int temp = vec[i];
    vec[i] = vec[j];
    vec[j] = temp;
}

__global__ void gpu_even_pass(int* vec, size_t size){

    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(size_t i = thread_id; i < size_t(size/2); i += num_threads){
        if (vec[2*i] > vec[2*i+1])
            gpu_swap(vec, 2*i, 2*i+1);
    }
}

__global__ void gpu_odd_pass(int* vec, size_t size){

    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    for(size_t i = thread_id; i < size_t(size/2); i += num_threads){
        if (vec[2*i+1] > vec[2*(i+1)])
            gpu_swap(vec, 2*i+1, 2*(i+1));
    }
}

double gpu_bubble_sort(std::vector<int> &vec){

    size_t size = vec.size(); 
    int num_bytes = vec.size() * sizeof(int);

    int* d_vec;
    cudaMalloc((void**)&d_vec, num_bytes);

    cudaMemcpy(d_vec, vec.data(), num_bytes, cudaMemcpyHostToDevice);
    Timer timer;
    double time_measured = 0;
    timer.reset();
    for(size_t i = 0; i < size_t(size/2); ++i){
        gpu_even_pass<<<BLOCK_SIZE, GRID_SIZE>>>(d_vec, size - (size % 2));
        cudaDeviceSynchronize();            
        gpu_odd_pass<<<BLOCK_SIZE, GRID_SIZE>>>(d_vec, size - 1);
        cudaDeviceSynchronize();            
    }
    time_measured = timer.get();

    cudaMemcpy(vec.data(), d_vec, sizeof(int) * vec.size(), cudaMemcpyDeviceToHost);
    cudaFree(d_vec);

    return time_measured;
}

__global__ void gpu_merge_pass(int* vec, int* output, size_t size, size_t size_to_merge){

    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;

    int max_scope_to_merge = size / size_to_merge;
    if( size_to_merge % size > 0) max_scope_to_merge++; 
    max_scope_to_merge = max_scope_to_merge - (max_scope_to_merge % 2);

    for(size_t i = thread_id; i < size; i += num_threads){
        int my_value = vec[i];

        int position_in_current = (i % size_to_merge);
        int scope_id = i / size_to_merge;
        
        int side = (scope_id % 2) * 2 - 1; // -1 and 1 (left / rigt)
        int merge_to_scope = scope_id - side;
        if(merge_to_scope > size / size_to_merge) continue;
        

        int size_to_iterate = min(size_to_merge, size - (merge_to_scope*size_to_merge));
        int position_in_merged = 0;

        if(side == -1){
            position_in_merged = size_to_iterate;
            for(size_t comparison_pos = 0; comparison_pos < size_to_iterate; comparison_pos++){
                if(my_value <= vec[merge_to_scope*size_to_merge + comparison_pos]){
                    position_in_merged = comparison_pos;
                    break;
                }
            }
        }
        else{
            position_in_merged = 0;
            for(size_t comparison_pos = size_to_iterate; comparison_pos > 0; comparison_pos--){
                if(my_value >= vec[merge_to_scope*size_to_merge + comparison_pos - 1]){
                    position_in_merged = comparison_pos;
                    break;
                }
            }
        }

        output[min(merge_to_scope, scope_id)*size_to_merge + position_in_current + position_in_merged] = my_value;
    }

}


double gpu_merge_sort(std::vector<int> &vec){

    size_t size = vec.size(); 
    int num_bytes = vec.size() * sizeof(int);

    int* d_vec;
    int* d_tmp;

    cudaMalloc((void**)&d_vec, num_bytes);
    cudaMalloc((void**)&d_tmp, num_bytes);
    
    cudaMemcpy(d_vec, vec.data(), num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tmp, vec.data(), num_bytes, cudaMemcpyHostToDevice);

    Timer timer;
    double time_measured = 0;
    timer.reset();

    int *input = d_tmp;
    int *output = d_vec;    
    
    int *tmp;


    for(size_t size_to_merge = 1; size_to_merge < size; size_to_merge *= 2){
        tmp = input;
        input = output;
        output = tmp;


        gpu_merge_pass<<<1024, 1024>>>(input, output, size, size_to_merge);
        cudaDeviceSynchronize();                     
    }
    time_measured = timer.get();

    // copy back
    cudaMemcpy(vec.data(), output, sizeof(int) * vec.size(), cudaMemcpyDeviceToHost);
    

    cudaFree(d_vec);
    cudaFree(d_tmp);
    return time_measured;
}

double gpu_thrust_sort_three(std::vector<int> &vec, std::vector<int> &vec1, std::vector<int> &vec2){


    size_t size = vec.size(); 
    int num_bytes = vec.size() * sizeof(int);

    thrust::host_vector<int> h_vec = vec;
    thrust::device_vector<int> d_vec = h_vec;

    thrust::host_vector<int> h_ind_vec = vec;
    thrust::device_vector<int> ind_vec = h_ind_vec;
    thrust::sequence(h_ind_vec.begin(), h_ind_vec.end());
    
    thrust::copy(h_ind_vec.begin(), h_ind_vec.end(), ind_vec.begin());


    Timer timer;
    double time_measured = 0;
    timer.reset();
    
    thrust::sort_by_key(d_vec.begin(), d_vec.end(), ind_vec.begin());
    time_measured = timer.get();

    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    thrust::copy(h_vec.begin(), h_vec.end(), vec.begin());

    std::vector<int> indices(size);
    thrust::copy(ind_vec.begin(), ind_vec.end(), h_ind_vec.begin());
    thrust::copy(h_ind_vec.begin(), h_ind_vec.end(), indices.begin());
    

    std::vector<int> tmp_vec1, tmp_vec2;
    tmp_vec1 = vec1;
    tmp_vec2 = vec2;
    for(size_t i = 0; i < size; i++){
        vec1[i] = tmp_vec1[indices[i]];
        vec2[i] = tmp_vec2[indices[i]];
    }

    return time_measured;
}

double gpu_thrust_sort(std::vector<int> &vec){

    size_t size = vec.size(); 
    int num_bytes = vec.size() * sizeof(int);

    thrust::host_vector<int> h_vec = vec;
    thrust::device_vector<int> d_vec = h_vec;

    Timer timer;
    double time_measured = 0;
    timer.reset();
    thrust::sort(d_vec.begin(), d_vec.end());
    time_measured = timer.get();

    thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
    thrust::copy(h_vec.begin(), h_vec.end(), vec.begin());

    return time_measured;
}


__global__ void gpu_merge_sort_thread_per_block_with_ind(int* input, int* output, int size, int size_to_merge, int *input_ind, int* output_ind) {

    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    int number_of_patches = (size / size_to_merge) + 1;


    for(int patch_id = thread_id; patch_id < number_of_patches; patch_id += num_threads){
        int start_point = patch_id * size_to_merge;

        int mid_point = min(start_point + (size_to_merge / 2), size);
        int end_point = min(start_point + size_to_merge, size);

        int current_left_pointer = start_point;
        int current_right_pointer = mid_point;


        for (int position_in_merged = start_point; position_in_merged < end_point; position_in_merged++) {
            if ((current_left_pointer < mid_point) && 
                (current_right_pointer >= end_point || input[current_left_pointer] < input[current_right_pointer])){
                output[position_in_merged] = input[current_left_pointer];
                output_ind[position_in_merged] = input_ind[current_left_pointer];
                current_left_pointer++;
            }
            else {
                output[position_in_merged] = input[current_right_pointer];
                output_ind[position_in_merged] = input_ind[current_right_pointer];
                current_right_pointer++;
            }
        }
        
    }
}

__global__ void gpu_merge_sort_thread_per_block(int* input, int* output, int size, int size_to_merge) {

    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    int number_of_patches = (size / size_to_merge) + 1;


    for(int patch_id = thread_id; patch_id < number_of_patches; patch_id += num_threads){
        int start_point = patch_id * size_to_merge;

        int mid_point = min(start_point + (size_to_merge / 2), size);
        int end_point = min(start_point + size_to_merge, size);

        int current_left_pointer = start_point;
        int current_right_pointer = mid_point;


        for (int position_in_merged = start_point; position_in_merged < end_point; position_in_merged++) {
            if ((current_left_pointer < mid_point) && 
                (current_right_pointer >= end_point || input[current_left_pointer] < input[current_right_pointer])){
                output[position_in_merged] = input[current_left_pointer];
                current_left_pointer++;
            }
            else {
                output[position_in_merged] = input[current_right_pointer];
                current_right_pointer++;
            }
        }
        
    }
}


double improved_mergesort_three(std::vector<int> &vec, std::vector<int> &vec1, std::vector<int> &vec2) {

    int size = vec.size(); 
    int num_bytes = vec.size() * sizeof(int);

    int* d_vec;
    int* d_tmp;
    cudaMalloc((void**)&d_vec, num_bytes);
    cudaMalloc((void**)&d_tmp, num_bytes);

    cudaMemcpy(d_vec, vec.data(), num_bytes, cudaMemcpyHostToDevice);
    
    int* ind_vec;
    int* ind_tmp;

    cudaMalloc((void**)&ind_vec, num_bytes);
    cudaMalloc((void**)&ind_tmp, num_bytes);
    
    std::vector<int> initial(size);
    std::iota (std::begin(initial), std::end(initial), 0);

    cudaMemcpy(ind_vec, initial.data(), num_bytes, cudaMemcpyHostToDevice);
    
    Timer timer;
    double time_measured = 0;
    timer.reset();
 

    int *input = d_tmp;
    int *output = d_vec;

    int *input_ind = ind_tmp;
    int *output_ind = ind_vec; 

    int *tmp;
    
    bool done = false;
    for (int size_to_merge = 2; done == false; size_to_merge *= 2) {
        tmp = input;
        input = output;
        output = tmp;

        tmp = input_ind;
        input_ind = output_ind;
        output_ind = tmp;

        gpu_merge_sort_thread_per_block_with_ind<<<BLOCK_SIZE, GRID_SIZE>>>(input, output, size, size_to_merge, input_ind, output_ind);
        if(size_to_merge >= size) {done = true;}
    }
    cudaDeviceSynchronize();
    time_measured = timer.get();

    cudaMemcpy(vec.data(), output, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(initial.data(), output_ind, size * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> tmp_vec1, tmp_vec2;
    tmp_vec1 = vec1;
    tmp_vec2 = vec2;
    for(int i = 0; i < size; i++){
        vec1[i] = tmp_vec1[initial[i]];
        vec2[i] = tmp_vec2[initial[i]];
    }

    cudaFree(d_tmp);
    cudaFree(d_vec);
    cudaFree(ind_tmp);
    cudaFree(ind_vec);

    return time_measured;
}

double improved_mergesort(std::vector<int> &vec) {

    int size = vec.size(); 
    int num_bytes = vec.size() * sizeof(int);

    // allocate
    int* d_vec;
    int* d_tmp;
    cudaMalloc((void**)&d_vec, num_bytes);
    cudaMalloc((void**)&d_tmp, num_bytes);

    // copy
    cudaMemcpy(d_vec, vec.data(), num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tmp, vec.data(), num_bytes, cudaMemcpyHostToDevice);
    
    Timer timer;
    double time_measured = 0;
    timer.reset();

    int *input = d_tmp;
    int *output = d_vec;
    int *tmp;
    

    timer.reset();
    bool done = false;
    for (int size_to_merge = 2; done == false; size_to_merge *= 2) {
        tmp = input;
        input = output;
        output = tmp;
        gpu_merge_sort_thread_per_block<<<BLOCK_SIZE, GRID_SIZE>>>(input, output, size, size_to_merge);
        if(size_to_merge >= size) {done = true;}
    }

    cudaDeviceSynchronize();  

    time_measured = timer.get();

    cudaMemcpy(vec.data(), output, size * sizeof(int), cudaMemcpyDeviceToHost);
  
    cudaFree(d_tmp);
    cudaFree(d_vec);

    return time_measured;
}


int main()
{
    size_t N;
    size_t max_power = 30;
    size_t repetitions = 1;
    int min_range = 0;
    int max_range = 100000;

    


    for(size_t i = 4; i < max_power; i++){

        Timer timer;
        double time_cpu = 0.0;
        double time_bubble = 0.0;
        double time_merge = 0.0;
        double time_thrust = 0.0;
        double time_improved_merge = 0.0;
        double time_radix = 0.0;

        N = std::pow(2, i);
        
        for(size_t j = 0; j < repetitions; j++){
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 gen(rd()); // seed the generator
            std::uniform_int_distribution<> distr(min_range, max_range);
        
            std::vector<int> org_vector(N);
            
            // uniform noise
            for(size_t i = 0; i < org_vector.size(); ++i){
                org_vector[i] = distr(gen);
            }
            // descending order
            for(size_t i = 0; i < org_vector.size(); ++i){
                    org_vector[i] = org_vector.size() - i;
            }
        
        
            std::vector<int> cpu_vector = org_vector;
        
            timer.reset();
            std::stable_sort(cpu_vector.begin(), cpu_vector.end());
            time_cpu += timer.get();



            std::vector<int> bubble_vector = org_vector;
            timer.reset();
            gpu_bubble_sort(bubble_vector);
            time_bubble += timer.get();
        
            for(size_t i = 0; i < bubble_vector.size(); ++i){
                if(bubble_vector[i] != cpu_vector[i]){
                    std::cout << "ERROR IN Bubble !!!!!!!!!!!!!" << std::endl;
                    break;
                }
            }

            std::vector<int> merge_vector = org_vector;
            timer.reset();
            gpu_merge_sort(merge_vector);
            time_merge += timer.get();
        
            for(size_t i = 0; i < merge_vector.size(); ++i){
                if(merge_vector[i] != cpu_vector[i]){
                    std::cout << "ERROR IN Merge !!!!!!!!!!!!!" << std::endl;
                    break;
                }
            }



            std::vector<int> thrust_vector = org_vector;
            std::vector<int> thrust_vector1 = org_vector;
            std::vector<int> thrust_vector2 = org_vector;
        
            Timer timer;
            timer.reset();
            gpu_thrust_sort_three(thrust_vector, thrust_vector1, thrust_vector2);
            time_thrust += timer.get();
        
            for(size_t i = 0; i < thrust_vector.size(); ++i){
                if(thrust_vector[i] != cpu_vector[i] || thrust_vector1[i] != cpu_vector[i] || thrust_vector2[i] != cpu_vector[i]){
                    std::cout << "ERROR IN Thrust !!!!!!!!!!!!!" << std::endl;
                    break;
                }
            }
            

            std::vector<int> radix_vector = org_vector;
            std::vector<int> radix_vector1 = org_vector;
            std::vector<int> radix_vector2 = org_vector;


           

            timer.reset();

            radix_sort(radix_vector, radix_vector1, radix_vector2);
            cudaDeviceSynchronize();
            time_radix += timer.get();

            
            for (int i = 0; i < org_vector.size(); i++){
                if(radix_vector[i] != cpu_vector[i] || radix_vector1[i] != cpu_vector[i] || radix_vector2[i] != cpu_vector[i]){
                    
                    std::cout << "ERROR IN radix !!!!!!!!!!!!! " << radix_vector[i] << " " << radix_vector1[i] << " " << radix_vector2[i] << " " << cpu_vector[i] <<  std::endl;
                    break;
                }
            }

            

            std::vector<int> improved_merge_vector = org_vector;
            std::vector<int> improved_merge_vector1 = org_vector;
            std::vector<int> improved_merge_vector2 = org_vector;

            timer.reset();
            improved_mergesort_three(improved_merge_vector, improved_merge_vector1, improved_merge_vector2);
            cudaDeviceSynchronize();
            time_improved_merge += timer.get();
            
        
            for(size_t i = 0; i < improved_merge_vector.size(); ++i){
                if(improved_merge_vector[i] != cpu_vector[i] || improved_merge_vector1[i] != cpu_vector[i] || improved_merge_vector2[i] != cpu_vector[i]){
                    std::cout << "ERROR IN improved_merge !!!!!!!!!!!!!" << std::endl;
                    break;
                }
            }
        }

        time_cpu = time_cpu / repetitions;
        time_bubble = time_bubble / repetitions;
        time_merge = time_merge / repetitions;
        time_thrust = time_thrust / repetitions;
        time_improved_merge = time_improved_merge / repetitions;
        time_radix = time_radix / repetitions;

        std::cout << N << " " << time_bubble << " " << time_merge << " " << time_cpu  << " " << time_thrust << " " << time_improved_merge << " " << time_radix << std::endl;
   
    
    }
    return 0;

    
}