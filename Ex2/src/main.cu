#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>

#include "partition_array.cu"

void print_vector(std::vector<int> &vec)
{
    for (auto i: vec)
        std::cout << i << ' ';

    std::cout << std::endl;
}

int main()
{
    // random number gen from https://stackoverflow.com/questions/21516575/fill-a-vector-with-random-numbers-c#:~:text=You%20can%20use%20std%3A%3A,specify%20the%20engine%20and%20distribution.
    // First create an instance of an engine.
    std::random_device rnd_device;


    // Specify the engine and distribution.
    std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    std::uniform_int_distribution<int> dist {0, 10};

    auto gen = [&dist, &mersenne_engine](){
                   return dist(mersenne_engine);
               };

    int size_vector = 10;
    std::vector<int> vec(size_vector), v2(size_vector), v3(size_vector), v4(size_vector), v5(size_vector);
    std::generate(begin(vec), end(vec), gen);

    size_t size = vec.size();
    int num_bytes = vec.size() * sizeof(int);

    // allocate
    int *d_vec, *d_v2, *d_v3, *d_v4, *d_v5;
    cudaMalloc((void**)&d_vec, num_bytes);
    cudaMalloc((void**)&d_v2, num_bytes);
    cudaMalloc((void**)&d_v3, num_bytes);
    cudaMalloc((void**)&d_v4, num_bytes);
    cudaMalloc((void**)&d_v5, num_bytes);

    cudaMemcpy(d_vec, vec.data(), num_bytes, cudaMemcpyHostToDevice);

    check_array<<<256, 256>>>(d_vec, d_v2, d_v3, size, 5);

    cudaMemcpy(v2.data(), d_v2, sizeof(int) * vec.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(v3.data(), d_v3, sizeof(int) * vec.size(), cudaMemcpyDeviceToHost);

    exclusive_scan(d_v2, d_v4, size);
    exclusive_scan(d_v3, d_v5, size);

    cudaMemcpy(v4.data(), d_v4, sizeof(int) * vec.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(v5.data(), d_v5, sizeof(int) * vec.size(), cudaMemcpyDeviceToHost);

    std::cout << std::endl << "Contents of original: " << std::endl;
    print_vector(vec);
    std::cout << std::endl << "smaller: " << std::endl;
    print_vector(v2);
    std::cout << "greater: " << std::endl;
    print_vector(v3);
    std::cout << std::endl << "Now comes the scanning" << std::endl;
    print_vector(v4);
    print_vector(v5);

    cudaFree(d_vec);
    cudaFree(d_v2);
    cudaFree(d_v3);
    cudaFree(d_v4);
    cudaFree(d_v5);



    return 0; //comment branch
}