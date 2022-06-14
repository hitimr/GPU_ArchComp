#include "common.hpp"
#include "edgelist.hpp"
#include "misc.hpp"

Benchmarker g_benchmarker;
OptionsT g_options;

__global__ void modify_1(int *val)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id == 0)
  {
    val[0] = -1;
  }
}

void test_transfer()
{
  EdgeList E(misc::get_example_file());
  assert(E.owner == HOST);

  E.sync_hostToDevice();
  assert(E.owner == DEVICE);

  // Modify on GPU
  modify_1<<<GRID_SIZE, BLOCK_SIZE>>>(E.d_val);
  cudaDeviceSynchronize();

  // check on CPU
  E.sync_deviceToHost();
  assert(E.owner == HOST);
  assert(E.val[0] == -1);
}


void test_assignment()
{
  EdgeList E(misc::get_example_file());
  int first_val = E.val[0];
  E.sync_hostToDevice();

  EdgeList E_new;
  assert(E_new.size() == 0);
  E_new = E;

  assert(E_new.val[0] == first_val);  
  assert(E.size() == E_new.size());

}

int main(int ac, char **av)
{

  // po::store(po::parse_command_line(ac, av), options);
  g_options = misc::parse_options(ac, av);

  test_transfer();
  test_assignment();

  return 0;
}