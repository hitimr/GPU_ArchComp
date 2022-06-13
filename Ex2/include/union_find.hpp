#pragma once
#include <cassert>
#include <vector>

class UnionFind
{
public:
  std::vector<int> parent;

  UnionFind(size_t size)
  {
    parent.resize(size);
    for (size_t i = 0; i < parent.size(); ++i)
      parent[i] = i; // TODO: maybe make parallel
  }

  int find(int i)
  {
    if (parent[i] == i)
      return i;
    else
      return find(parent[i]);
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
    if (find_and_compress(i) != find_and_compress(j))
      link(find_and_compress(i), find_and_compress(j));
  }

  int get_parent(int i) { return parent[i]; }

  void compress(int kernel);
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