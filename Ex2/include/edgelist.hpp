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
  Edge(int source, int target, int weight) : source(source), target(target), weight(weight){};

  int source;
  int target;
  int weight;
};

class EdgeList
{
public:
  // Construct an edgelist from a given input file
  // format as specified in TUWEL
  EdgeList(std::string file_name) { load_from_file(file_name); };

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

  void load_from_file(std::string file_name)
  {
    // read header
    std::string filepath = std::string(file_name);
    std::fstream input_file;
    input_file.open(filepath, std::ios::in);
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

        coo1.resize(num_edges);
        coo2.resize(num_edges);
        val.resize(num_edges);
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

  // reserve memory space for [size] nodes
  // useful for more efficient calls of append_edge()
  void reserve(size_t size)
  {
    assert(size >= coo1.size());

    coo1.reserve(size);
    coo2.reserve(size);
    val.reserve(size);
  }

  // append a new edge to the edgelist
  void append_edge(int source, int target, int weight)
  {
    coo1.push_back(source);
    coo2.push_back(target);
    val.push_back(weight);
    num_edges += 1;
    // correctly update number of nodes
  }

  void append_edge(const Edge &e) { append_edge(e.source, e.target, e.weight); }
  void append_edge(Edge &&e) { append_edge(e.source, e.target, e.weight); }

  Edge operator[](int index) const { return Edge(coo1[index], coo2[index], val[index]); }

  size_t size() const { return num_edges; }

  // TODO: make private and create getter functions
  size_t num_nodes;
  size_t num_edges;
  int direction;

  std::vector<int> coo1;
  std::vector<int> coo2;
  std::vector<int> val;
};