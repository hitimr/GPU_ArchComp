#include "Graph.hpp"
#include "common.hpp"
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

namespace po = boost::program_options;
using OptionsT = boost::program_options::variables_map;

OptionsT parse_options(int ac, char **av)
{
    po::options_description desc("Allowed options");

    // clang-format off
    desc.add_options()
        ("help,h", 
        "produce help message")

        ("inputfile,i", 
        po::value<std::vector<std::string>>(), 
        "input file containing graph data");
    // clang-format on

    // Bioler-plate Boost options stuff
    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    // Produce help message
    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        exit(EXIT_SUCCESS);
    }


    return vm;
}

int main(int ac, char **av)
{
    OptionsT options = parse_options(ac, av);

    // I havent found a good way for adding defaults tring to boost::options so im doing it by hand
    std::string input_file = options.count("inputfile")
                                  ? options["inputfile"].as<std::vector<std::string>>()[0]
                                  : DEFAULT_INPUT_FILE;

    std::cout << "Loading " << input_file << std::endl;

    return 0;
}