#pragma once
#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class Edge
{
public:
  __hostdev__ Edge(int source, int target, int weight)
      : source(source), target(target), weight(weight){};

  int source;
  int target;
  int weight;
};

class EdgeList
{
public:
  // Construct an edgelist from a given input file
  // format as specified in TUWEL
  EdgeList(std::string file_name)
  {
    load_from_file(file_name);
  };

  EdgeList()
  {
    num_edges = 0;
    num_nodes = 0;
  };

  // Allocate an EdgeList for [size] edges
  EdgeList(size_t size)
  {
    EdgeList();
    reserve(size);
  }

  ~EdgeList() {}

  // GPU Interface
  EdgeList *gpu; // Class holds a pointer to a copy of itself located on GPU memory
  void init_gpu();
  void sync_hostToDevice();
  void sync_deviceToHost();

  void load_from_file(std::string file_name)
  {
    // read header
    std::string filepath = std::string(file_name);
    std::fstream input_file;
    input_file.open(filepath, std::ios::in);
    if (!input_file.is_open())
    {
      throw std::runtime_error("Unable to open File " + file_name);
    }
    std::string current_line;

    size_t counter = 0;
    while (getline(input_file, current_line))
    {
      std::vector<std::string> row;
      std::string line, word;

      row.clear();

      std::stringstream str(current_line);

      while (getline(str, word, ';'))
      {
        row.push_back(word);
      }

      // Header
      if (row[0] == std::string("H"))
      {
        num_nodes = std::stoi(row[1]);
        num_edges = std::stoi(row[2]);
        direction = std::stoi(row[3]);

        val = new int[num_edges];
        coo1 = new int[num_edges];
        coo2 = new int[num_edges];
      }

      // Edges
      if (row[0] == std::string("E"))
      {
        coo1[counter] = std::stoi(row[1]);
        coo2[counter] = std::stoi(row[2]);
        val[counter] = std::stoi(row[3]);
        counter++;
      }
    }
    input_file.close();
  }

  void write_to_file(std::string file_name)
  {
    sync_deviceToHost();

    std::ofstream file;
    file.open(file_name);
    if (!file.is_open())
    {
      std::cout << "unable to write to " << file_name << std::endl;
      throw std::runtime_error("unable to write to " + file_name);
    }

    file << "H;" << size() + 1 << ";" << size() << ";1" << std::endl;
    for (size_t i = 0; i < size(); i++)
    {
      Edge edge = at(i);
      file << "E;" << edge.source << ";" << edge.target << ";" << edge.weight << std::endl;
    }

    file.close();
  }

  // reserve memory space for [size] nodes
  // useful for more efficient calls of append_edge()
  void reserve(size_t size);

  // append a new edge to the edgelist
  void append_edge(int source, int target, int weight)
  {
    // sync_deviceToHost();
    val[num_edges] = weight;
    coo1[num_edges] = source;
    coo2[num_edges] = target;
    num_edges += 1;
  }

  void resize_and_set_num_edges(size_t size);

  void set_owner(int new_owner)
  {
    assert((new_owner == HOST) || (new_owner == DEVICE));
    owner = new_owner;
  }

  int weigth()
  {
    int sum = 0;
    for (size_t i = 0; i < size(); i++)
    {
      sum += val[i];
    }
    return sum;
  }

  void append_edge(const Edge &e) { append_edge(e.source, e.target, e.weight); }
  void append_edge(Edge &&e) { append_edge(e.source, e.target, e.weight); }

  Edge operator[](int index) const { return Edge(coo1[index], coo2[index], val[index]); }
  Edge at(int index) const { return Edge(coo1[index], coo2[index], val[index]); }

  size_t size() const { return num_edges; }

  // TODO: make private and create getter functions
  size_t num_nodes = 0;
  size_t num_edges = 0;
  int direction;
  int owner = HOST;
  bool m_use_pinned_memory = false;

  // CPU Data
  int *coo1 = nullptr;
  int *coo2 = nullptr;
  int *val = nullptr;

  // GPU Data
  int *d_coo1 = nullptr;
  int *d_coo2 = nullptr;
  int *d_val = nullptr;
};