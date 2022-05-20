#include <iostream>

#include <boost/program_options.hpp>

namespace po = boost::program_options;
using OptionsT = boost::program_options::variables_map;

OptionsT parse_options(int ac, char **av)
{
    po::options_description desc("Allowed options");

    // clang-format off
    desc.add_options()
        ("help,h", "produce help message");

    // clang-format on

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

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
    
    return 0;
}