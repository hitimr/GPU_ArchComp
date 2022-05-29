#pragma once
#include <vector>
#include <cassert>

class UnionFind
{
protected:
  std::vector<int> parent;

public:
  UnionFind(size_t size)
  {
    parent.resize(size);
    for (size_t i = 0; i < parent.size(); ++i)
      parent[i] = i; 
  }

  int find(int i)
  {
    if (parent[i] == i)
      return i;
    else
      return find(parent[i]);
  }

  void link(int i, int j)
  {
    assert(find(i) != find(j));
    parent[i] = j;
  }

  void my_union(int i, int j)
  {
    if (find(i) != find(j))
      link(find(i), find(j));
  }
};


// UnionFind with Path Compression
class UnionFindPC : public UnionFind
{
public:

  UnionFindPC(size_t size) : UnionFind(size){}

  int find(int i)
  {
    if (parent[i] == i)
      return i;
    else {
      int root = find(parent[i]);
      parent[i] = root;
      return root;
    }
  }
};