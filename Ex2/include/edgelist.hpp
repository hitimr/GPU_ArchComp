#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class EdgeList
{
public:
  EdgeList(std::string file_name)
  {
      load_from_file(file_name);
  };

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

  size_t num_nodes;
  size_t num_edges;
  int direction;

  std::vector<int> coo1;
  std::vector<int> coo2;
  std::vector<int> val;
};