#pragma once

// Program options
#include <boost/program_options.hpp>
namespace po = boost::program_options;
using OptionsT = boost::program_options::variables_map;
extern OptionsT g_options;

// Benchmarker
#include "benchmarker.hpp"
extern Benchmarker g_benchmarker;

#define PROJ_ROOT_FOLDER_NAME "Ex2"

// Options
#define DEFAULT_INPUT_FILE "input_data/example.csv"
// #define DEFAULT_INPUT_FILE "input_data/barabasi_25000_10.csv"

// Kernels
#define MST_KERNEL_REGULAR_KRUSKAL 0
#define MST_KERNEL_FILTER_KRUSKAL 1

#define SORT_KERNEL_GPU_BUBBLE_MULT 0

#define PARTITION_KERNEL_CPU_NAIVE 0
#define PARTITION_KERNEL_GPU 1

#define FILTER_KERNEL_CPU_NAIVE 0

// Defaults
#define MST_DEFAULT_KERNEL MST_KERNEL_FILTER_KRUSKAL        // MST
#define SORT_DEFAULT_KERNEL SORT_KERNEL_GPU_BUBBLE_MULT     // Sort
#define PARTITION_DEFAULT_KERNEL PARTITION_KERNEL_GPU // Partition
#define DEFAULT_FILTER_KERNEL FILTER_KERNEL_CPU_NAIVE       // Filter

// Misc
// TODO: change grid/block size to launch parameters
#define EXIT_SUCCESS 0
#define BLOCK_SIZE 256
#define GRID_SIZE 256
#define BLOCKSIZE BLOCK_SIZE
#define GRIDSIZE GRID_SIZE


#ifdef __CUDACC__
// Only define __hostdev__ when using NVIDIA CUDA compiler
#define __hostdev__ __host__ __device__
#else
#define __hostdev__
#endif