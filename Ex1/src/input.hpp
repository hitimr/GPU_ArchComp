#include <vector>
#include <iostream>
#include <assert.h>
#include "lodepng.h"

namespace input{
    enum image_type {GRAYSCALE, RGB, RGBA};

    std::pair<std::vector<int>, std::vector<int>> loadImageFromFile(const char* filepath, image_type mode = image_type::RGB){
        /*
        Output: Pair<Image, ground truth>
        Input: Filepath - const char*
            Load mode - image_type
        */
        std::vector<unsigned char> png;
        std::vector<unsigned char> image_tmp;
        unsigned width, height;

        unsigned error = lodepng::load_file(png, filepath);
        if(!error) error = lodepng::decode(image_tmp, width, height, png);

        if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        assert(!error);

        std::vector<int> histogram(256);
        std::vector<int> image;
        if (mode == image_type::RGBA){
            image.reserve(image_tmp.size());
            std::copy(image_tmp.begin(), image_tmp.end(), back_inserter(image));
            for(size_t i = 0; i < image_tmp.size(); ++i){
                histogram[image_tmp[i]] += 1;
            }
        }
        if (mode == image_type::GRAYSCALE){
            image.reserve(image_tmp.size()/4);
            for(size_t i = 0; i < image_tmp.size(); i+=4){
                int gray_value = 0.299*image_tmp[i] + 0.587*image_tmp[i+1]  + 0.114*image_tmp[i+2];
                image.push_back(gray_value);
                histogram[gray_value] += 1;
            }
        }
        if (mode == image_type::RGB){
            image.reserve((image_tmp.size()/4)*3);
            for(size_t i = 0; i < image_tmp.size(); i+=4){
                histogram[image_tmp[i]] += 1;
                histogram[image_tmp[i+1]] += 1;
                histogram[image_tmp[i+2]] += 1;
                image.push_back(image_tmp[i]);
                image.push_back(image_tmp[i+1]);
                image.push_back(image_tmp[i+2]);
            }
        }
        auto output = std::make_pair(image, histogram);
        return output;   
    }

    std::pair<std::vector<int>, std::vector<int>> generateUniformlyDistributedArray(size_t len, size_t number_of_colors = 2){
        std::vector<int> histogram(256);
        std::vector<int> image;
        image.reserve(len);
        for(size_t i = 0; i < len; ++i){
            int next_color = i%number_of_colors;
            histogram[next_color] += 1;
            image.push_back(next_color);
        }
        auto output = std::make_pair(image, histogram);
        return output;   
    }

    std::pair<std::vector<int>, std::vector<int>> generateRandomArray(size_t len, size_t number_of_colors = 2){
        std::vector<int> histogram(256);
        std::vector<int> image;
        image.reserve(len);
        for(size_t i = 0; i < len; ++i){
            int next_color = rand()%number_of_colors;
            histogram[next_color] += 1;
            image.push_back(next_color);
        }
        auto output = std::make_pair(image, histogram);
        return output;   
    }

}




// int main(){
//     const char* filename = "../sample.png";

//     auto pair_input = loadImageFromFile(filename);
//     auto image = pair_input.first;
//     auto gt = pair_input.second;
// }