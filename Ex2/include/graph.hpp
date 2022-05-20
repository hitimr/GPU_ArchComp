#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

class Graph
{
public:
  std::vector<std::vector<int>> adjacency_matrix;
  std::vector<std::vector<int>> mst_gt_adjacency_matrix;
  size_t size;

  Graph(const char *filename)
  {
    loadGraphFromFile(filename);
    // print_adj();
  }

  std::vector<int> getCoo1();
  std::vector<int> getCoo2();
  std::vector<int> getWeights();

  std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> getCSRRepresentation();

  std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>> getELLRepresentation();

  std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> getCOOReepresentation();

  bool checkMstSolution(std::vector<std::vector<int>> solution)
  {
    for (size_t i = 0; i < this->size; i++)
    {
      for (size_t j = i + 1; j < this->size; j++)
      {
        if (mst_gt_adjacency_matrix[i][j] != solution[i][j])
        {
          return false;
        }
      }
    }
    return true;
  }

  void print_adj()
  {
    for (size_t i = 0; i < this->size; i++)
    {
      for (size_t j = 0; j < this->size; j++)
      {
        std::cout << adjacency_matrix[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }

  void loadGraphFromFile(const char *filename)
  {
    {
      std::string filepath = std::string(filename);
      std::fstream input_file;
      input_file.open(filepath, std::ios::in);
      std::string current_line;
      // Header line

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
        if (row[0] == std::string("E"))
        {
          int start = std::stoi(row[1]);
          int end = std::stoi(row[2]);
          int weight = std::stoi(row[3]);
          adjacency_matrix[start][end] = weight;
        }
        if (row[0] == std::string("H"))
        {
          int size = std::stoi(row[1]);
          this->size = size;
          adjacency_matrix.assign(size, std::vector<int>(size, 0));
        }
      }
      input_file.close();
    }
    {
      std::string filepath = "../input_data/" + std::string("mst_gt_") + std::string(filename);
      std::fstream input_file;
      input_file.open(filepath, std::ios::in);
      std::string current_line;
      // Header line

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
        if (row[0] == std::string("E"))
        {
          int start = std::stoi(row[1]);
          int end = std::stoi(row[2]);
          int weight = std::stoi(row[3]);
          mst_gt_adjacency_matrix[start][end] = weight;
        }
        if (row[0] == std::string("H"))
        {
          int size = std::stoi(row[1]);
          mst_gt_adjacency_matrix.assign(size, std::vector<int>(size, 0));
        }
      }
      input_file.close();
    }
  }
};