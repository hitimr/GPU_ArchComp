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

// Kernels
#define MST_KERNEL_REGULAR_KRUSKAL 0
#define MST_KERNEL_FILTER_KRUSKAL 1

#define SORT_KERNEL_GPU_BUBBLE_MULT 0

#define PARTITION_KERNEL_CPU_NAIVE 0


#define EXIT_SUCCESS 0
#define BLOCK_SIZE 256
#define GRID_SIZE 256
#define BLOCKSIZE BLOCK_SIZE
#define GRIDSIZE GRID_SIZE