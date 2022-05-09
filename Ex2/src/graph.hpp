#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>

class Graph{
    public:
        std::vector<std::vector<int>> adjacency_matrix;
        std::vector<std::vector<int>> mst_gt_adjacency_matrix;
        int size;

        Graph(const char* filename){
            loadGraphFromFile(filename);
            print_adj();
        }

        std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> getCSRRepresentation(){
            std::vector<int> row_offset;
            std::vector<int> column_indices;
            std::vector<int> values;
            
            int offset = 0;
            for(size_t i = 0; i < this->size; i++){
                if(i==0){
                    row_offset.push_back(offset);
                }

                for(size_t j = 0; j < this->size; j++){
                    if(adjacency_matrix[i][j] > 0){
                        offset += 1;
                        column_indices.push_back(j);
                        values.push_back(adjacency_matrix[i][j]);
                    }
                }
                row_offset.push_back(offset);

            }

            return std::make_tuple(row_offset, column_indices, values);
        }

        std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>> getELLRepresentation(){
            std::vector<std::vector<int>> column_indices;
            std::vector<std::vector<int>> values;
            int max_size = 0;

            for(size_t i = 0; i < this->size; i++){
                std::vector<int> current_column_indices;
                std::vector<int> current_values;
                for(size_t j = 0; j < this->size; j++){
                    if(adjacency_matrix[i][j] > 0){
                        current_column_indices.push_back(j);
                        current_values.push_back(adjacency_matrix[i][j]);
                    }
                }
                if(current_column_indices.size() > max_size){
                    max_size = current_column_indices.size();
                }
                column_indices.push_back(current_column_indices);
                values.push_back(current_values);
            }
            for(size_t i = 0; i < this->size; i++){
                while(column_indices.size() < max_size){
                    column_indices[i].push_back(-1);
                    values[i].push_back(-1);
                }
            }

            return std::make_tuple(column_indices, values);
        }

        std::tuple<std::vector<int>, std::vector<int>, std::vector<int>> getCOOReepresentation(){
            std::vector<int> row_offset;
            std::vector<int> column_indices;
            std::vector<int> values;

            for(size_t i = 0; i < this->size; i++){
                for(size_t j = 0; j < this->size; j++){
                    if(adjacency_matrix[i][j] > 0){
                        row_offset.push_back(i);
                        column_indices.push_back(j);
                        values.push_back(adjacency_matrix[i][j]);
                    }
                }
            }
            return std::make_tuple(row_offset, column_indices, values);
        }



        bool checkMstSolution(std::vector<std::vector<int>>solution){
            for(size_t i = 0; i < this->size; i++){
                for(size_t j = i+1; j < this->size; j++){
                    if(mst_gt_adjacency_matrix[i][j] != solution[i][j]){
                        return false;
                    }
                }
            }
            return true;
        }

        void print_adj(){
            for(size_t i = 0; i < this->size; i++){
                for(size_t j = 0; j < this->size; j++){
                    std::cout << adjacency_matrix[i][j] << " ";
                }
                std::cout << std::endl;
            }
            
        }


        void loadGraphFromFile(const char* filename){
            {
                std::string filepath = "../input_data/" + std::string(filename);
                std::fstream input_file;
                input_file.open(filepath,std::ios::in);
                std::string current_line;
                //Header line
            
                while(getline(input_file, current_line)){
                    std::vector<std::string> row;
                    std::string line, word;

                    row.clear();
        
                    std::stringstream str(current_line);
        
                    while(getline(str, word, ';')){
                        row.push_back(word);
                    }
                    if(row[0] == std::string("E")){
                        int start = std::stoi(row[1]);
                        int end = std::stoi(row[2]);
                        int weight = std::stoi(row[3]);
                        adjacency_matrix[start][end] = weight;

                    }
                    if(row[0] == std::string("H")){
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
                input_file.open(filepath,std::ios::in);
                std::string current_line;
                //Header line
            
                while(getline(input_file, current_line)){
                    std::vector<std::string> row;
                    std::string line, word;

                    row.clear();
        
                    std::stringstream str(current_line);
        
                    while(getline(str, word, ';')){
                        row.push_back(word);
                    }
                    if(row[0] == std::string("E")){
                        int start = std::stoi(row[1]);
                        int end = std::stoi(row[2]);
                        int weight = std::stoi(row[3]);
                        mst_gt_adjacency_matrix[start][end] = weight;
                    }
                    if(row[0] == std::string("H")){
                        int size = std::stoi(row[1]);
                        mst_gt_adjacency_matrix.assign(size, std::vector<int>(size, 0));
                    }
                        
                }
                input_file.close();
            }
        }
};