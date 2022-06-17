#pragma once
#include <cassert>
#include <stdexcept>
#include <vector>

class UnionFind
{
public:
  std::vector<int> parent;

  UnionFind(size_t size, int compress_level = 1)
  {
    if (compress_level == 0)
    {
      use_compression = false;
    }
    else
    {
      use_compression = true;
    }

    parent.resize(size);
    for (size_t i = 0; i < parent.size(); ++i)
      parent[i] = i; // TODO: maybe make parallel
  }

  int find(int i)
  {
    if (use_compression == true)
    {
      return find_and_compress(i);
    }
    else
    {
      if (parent[i] == i)
        return i;
      else
        return find(parent[i]);
    }
  }

  int find_and_compress(int i)
  {
    if (parent[i] == i)
      return i;
    else
    {
      int root = find_and_compress(parent[i]);
      parent[i] = root;
      return root;
    }
  }

  void link(int i, int j) { parent[i] = j; }

  void my_union(int i, int j)
  {
    if (find(i) != find(j))
      link(find(i), find(j));
  }

  int get_parent(int i) { return parent[i]; }

  void compress(int kernel);

  bool use_compression = true;
};

void compress_cpu_naive(std::vector<int> &parent);
void compress_gpu(std::vector<int> &parent);

// UnionFind with Path Compression according to the slides of Mario
class UnionFindPC : public UnionFind
{
public:
  UnionFindPC(size_t size) : UnionFind(size) {}

  int find(int i)
  {
    if (parent[i] == i)
      return i;
    else
    {
      int root = find(parent[i]);
      parent[i] = root;
      return root;
    }
  }
};