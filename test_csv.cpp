#include "csv.hpp"

#include "utils.hpp"

#include <iostream>
#include <vector>

int main(int argc, char **argv) {
    char *file_name = argv[1];
    cle::CSV csv;

    std::vector<double> a;
    std::vector<double> b;

    argc = 0; // suppress compiler warning "unused variable"

    std::cout << "Reading from file \"" << file_name << "\"" << std::endl;

    csv.read_csv(file_name, a, b);

    cle::Utils::print_vector(a);
    cle::Utils::print_vector(b);

}
