#include "graph.hpp"


// implementations of the next 3 functions are placeholder
// TODO: efficient implementation
std::vector<int> Graph::getWeights()
{
  std::vector<int> coo1, coo2, w;
  std::tie(coo1, coo2, w) = getCOOReepresentation();
  return w;
}

std::vector<int> Graph::getCoo1()
{
  std::vector<int> coo1, coo2, w;
  std::tie(coo1, coo2, w) = getCOOReepresentation();
  return coo1;
}


std::vector<int> Graph::getCoo2()
{
  std::vector<int> coo1, coo2, w;
  std::tie(coo1, coo2, w) = getCOOReepresentation();
  return coo2;
}


std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> Graph::getCSRRepresentation()
{
  std::vector<int> row_offset;
  std::vector<int> column_indices;
  std::vector<int> values;

  int offset = 0;
  for (size_t i = 0; i < this->size; i++)
  {
    if (i == 0)
    {
      row_offset.push_back(offset);
    }

    for (size_t j = 0; j < this->size; j++)
    {
      if (adjacency_matrix[i][j] > 0)
      {
        offset += 1;
        column_indices.push_back(j);
        values.push_back(adjacency_matrix[i][j]);
      }
    }
    row_offset.push_back(offset);
  }

  return std::make_tuple(row_offset, column_indices, values);
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> Graph::getCOOReepresentation()
{
  std::vector<int> row_offset;
  std::vector<int> column_indices;
  std::vector<int> values;

  for (size_t i = 0; i < this->size; i++)
  {
    for (size_t j = 0; j < this->size; j++)
    {
      if (adjacency_matrix[i][j] > 0)
      {
        row_offset.push_back(i);
        column_indices.push_back(j);
        values.push_back(adjacency_matrix[i][j]);
      }
    }
  }
  return std::make_tuple(row_offset, column_indices, values);
}

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>>
Graph::getELLRepresentation()
{
  std::vector<std::vector<int>> column_indices;
  std::vector<std::vector<int>> values;
  size_t max_size = 0;

  for (size_t i = 0; i < this->size; i++)
  {
    std::vector<int> current_column_indices;
    std::vector<int> current_values;
    for (size_t j = 0; j < this->size; j++)
    {
      if (adjacency_matrix[i][j] > 0)
      {
        current_column_indices.push_back(j);
        current_values.push_back(adjacency_matrix[i][j]);
      }
    }
    if (current_column_indices.size() > max_size)
    {
      max_size = current_column_indices.size();
    }
    column_indices.push_back(current_column_indices);
    values.push_back(current_values);
  }
  for (size_t i = 0; i < this->size; i++)
  {
    while (column_indices.size() < max_size)
    {
      column_indices[i].push_back(-1);
      values[i].push_back(-1);
    }
  }

  return std::make_tuple(column_indices, values);
}