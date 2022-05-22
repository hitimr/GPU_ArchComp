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

#define BLOCK_SIZE 1024 
#define GRID_SIZE 1024

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

    // allocate
    int* d_vec;
    cudaMalloc((void**)&d_vec, num_bytes);

    // copy
    cudaMemcpy(d_vec, vec.data(), num_bytes, cudaMemcpyHostToDevice);
    Timer timer;
    double time_measured = 0;
    timer.reset();
    // sort
    for(size_t i = 0; i < size_t(size/2); ++i){
        gpu_even_pass<<<BLOCK_SIZE, GRID_SIZE>>>(d_vec, size - (size % 2));
        cudaDeviceSynchronize();            
        gpu_odd_pass<<<BLOCK_SIZE, GRID_SIZE>>>(d_vec, size - 1);
        cudaDeviceSynchronize();            
    }
    time_measured = timer.get();

    // copy back
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

// __device__ inline bool less(size_t &a, size_t &b){
//     return a < b;
// }

// __device__ inline bool greater(size_t &a, size_t &b){
//     return a > b;
// }

// __device__ inline bool greater_equal(int &a, int &b){
//     return a >= b;
// }

// __device__ inline bool less_equal(int &a, int &b){
//     return a <= b;
// }

// __global__ void gpu_merge_pass1(int* vec, int* output, size_t size, size_t size_to_merge){

//     int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
//     int num_threads = blockDim.x * gridDim.x;

//     int max_scope_to_merge = size / size_to_merge;
//     if( size_to_merge % size > 0) max_scope_to_merge++; 
//     max_scope_to_merge = max_scope_to_merge - (max_scope_to_merge % 2);

//     for(size_t i = thread_id; i < size; i += num_threads){
//         int my_value = vec[i];

//         int position_in_current = (i % size_to_merge);
//         int scope_id = i / size_to_merge;
        
//         int side = (scope_id % 2) * 2 - 1; // -1 and 1 (left / rigt)
//         int merge_to_scope = scope_id - side;
//         if(merge_to_scope > size / size_to_merge) continue;
        

//         int size_to_iterate = min(size_to_merge, size - (merge_to_scope*size_to_merge));
//         size_t position_in_merged = ((side - 1) / (-2)) * size_to_iterate;
//         size_t starting_position = ((side + 1) / (2)) * size_to_iterate;
//         int offset = -((side + 1) / 2);

//         auto comparator = side == -1 ? &less : &greater;
//         auto comparator_values = side == -1 ? &less_equal : &greater_equal;

//         for(size_t comparison_pos = starting_position; comparator(comparison_pos, position_in_merged); comparison_pos-=side){
//             if(comparator_values(my_value, vec[merge_to_scope*size_to_merge + comparison_pos + offset])){
//                 position_in_merged = comparison_pos;
//                 break;
//             }
//         }


//         output[min(merge_to_scope, scope_id)*size_to_merge + position_in_current + position_in_merged] = my_value;
//     }

// }

double gpu_merge_sort(std::vector<int> &vec){

    size_t size = vec.size(); 
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
    // sort
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

__global__ void gpu_merge_sort_thread_per_block(int* input, int* output, int size, int size_to_merge) {

    int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;


    for(int start_point = size_to_merge*thread_id; start_point < size; start_point += (num_threads*size_to_merge)){

        int mid_point = min(start_point + (size_to_merge / 2), size);
        int end_point = min(start_point + size_to_merge, size);

        int current_left_pointer = start_point;
        int current_right_pointer = mid_point;
        
        for (int position_in_merged = start_point; position_in_merged < end_point; position_in_merged++) {
            if ((current_left_pointer < mid_point) && 
                ((input[current_left_pointer] < input[current_right_pointer]) || current_right_pointer >= end_point)){
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
double mergesort(std::vector<int> &vec) {

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
    for (int size_to_merge = 2; size_to_merge < size; size_to_merge *= 2) {
        tmp = input;
        input = output;
        output = tmp;

        // Actually call the kernel
        gpu_merge_sort_thread_per_block<<<BLOCK_SIZE, GRID_SIZE>>>(input, output, size, size_to_merge);

    }
    time_measured = timer.get();

    cudaMemcpy(vec.data(), output, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(d_tmp);
    cudaFree(d_vec);
    return time_measured;
}



//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//




int main()
{
    std::cout << "Hello, World" << std::endl;
    size_t N;
    size_t max_power = 30;
    size_t repetitions = 1;
    int min_range = 0;
    int max_range = 10000;

    


    for(size_t i = 4; i < max_power; i++){

        Timer timer;
        double time_cpu = 0.0;
        double time_bubble = 0.0;
        double time_merge = 0.0;
        double time_thrust = 0.0;
        double time_improved_merge = 0.0;

        N = std::pow(2, i);
        
        for(size_t j = 0; j < repetitions; j++){
            std::random_device rd; // obtain a random number from hardware
            std::mt19937 gen(rd()); // seed the generator
            std::uniform_int_distribution<> distr(min_range, max_range);
        
            std::vector<int> org_vector(N);
            for(size_t i = 0; i < org_vector.size(); ++i){
                org_vector[i] = distr(gen);
            }
        
        
            std::vector<int> cpu_vector = org_vector;
        
            timer.reset();
            std::stable_sort(cpu_vector.begin(), cpu_vector.end());
            time_cpu += timer.get();
            for(size_t i = 0; i < cpu_vector.size(); ++i){
                // std::cout << cpu_vector[i] << " ";
            }
        
            std::vector<int> bubble_vector = org_vector;
        
            time_bubble += gpu_bubble_sort(bubble_vector);
        
        
            for(size_t i = 0; i < bubble_vector.size(); ++i){
                if(bubble_vector[i] != cpu_vector[i]){
                    std::cout << "ERROR IN BUBBLE !!!!!!!!!!!!!" << std::endl;
                    break;
                }
                // std::cout << bubble_vector[i] << " ";
            }
        
            std::vector<int> merge_vector = org_vector;
        
            time_merge += gpu_merge_sort(merge_vector);
        
        
            for(size_t i = 0; i < merge_vector.size(); ++i){
                if(merge_vector[i] != cpu_vector[i]){
                    std::cout << "ERROR IN MERGE !!!!!!!!!!!!!" << std::endl;
                    break;
                }
                // std::cout << merge_vector[i] << " ";
            }

            std::vector<int> thrust_vector = org_vector;

        
            time_thrust += gpu_thrust_sort(thrust_vector);
        
        
            for(size_t i = 0; i < thrust_vector.size(); ++i){
                if(thrust_vector[i] != cpu_vector[i]){
                    std::cout << "ERROR IN Thrust !!!!!!!!!!!!!" << std::endl;
                    break;
                }
                // std::cout << merge_vector[i] << " ";
            }

            std::vector<int> improved_merge_vector = org_vector;

        
            time_improved_merge += gpu_thrust_sort(improved_merge_vector);
        
        
            for(size_t i = 0; i < improved_merge_vector.size(); ++i){
                if(improved_merge_vector[i] != cpu_vector[i]){
                    std::cout << "ERROR IN improved_merge !!!!!!!!!!!!!" << std::endl;
                    break;
                }
                // std::cout << merge_vector[i] << " ";
            }


        }

        time_cpu = time_cpu / repetitions;
        time_bubble = time_bubble / repetitions;
        time_merge = time_merge / repetitions;
        time_thrust = time_thrust / repetitions;
        time_improved_merge = time_improved_merge / repetitions;

        std::cout << N << " " << time_cpu << " " << time_bubble << " " << time_merge << " " << time_thrust << " " << time_improved_merge << std::endl;
   
    
    }
    return 0;

    
}