#pragma once

// Program options
#include <boost/program_options.hpp>
namespace po = boost::program_options;
using OptionsT = boost::program_options::variables_map;

// Benchmarker
#include "benchmarker.hpp"

#define BLOCK_SIZE 256
#define GRID_SIZE 256

#define PROJ_ROOT_FOLDER_NAME "Ex2"

// Options
#define DEFAULT_INPUT_FILE "input_data/barabasi_200_3_mst_gt.csv"

// Kernels
enum SortKernel
{
  bubble = 1
};

#define EXIT_SUCCESS 0

// globals