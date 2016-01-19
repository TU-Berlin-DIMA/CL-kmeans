#include "csv.hpp"

#include "utils.hpp"
#include "timer.hpp"

#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>

int main(int argc, char **argv) {
    char *file_name = argv[1];
    cle::CSV csv;
    cle::Timer timer;
    uint64_t duration = 0;

    std::vector<double> a;
    std::vector<double> b;

    argc = 0; // suppress compiler warning "unused variable"

    std::cout << "Reading from file \"" << file_name << "\"" << std::endl;

    timer.start();
    csv.read_csv(file_name, a, b);
    duration = timer.stop<std::chrono::microseconds>();

    std::cout << "Variadic runtime: " << duration << " µs" << std::endl;

    // cle::Utils::print_vector(a);
    // cle::Utils::print_vector(b);

    std::array<std::vector<double>, 2> array;

    timer.start();
    csv.read_csv(file_name, array);
    duration = timer.stop<std::chrono::microseconds>();

    std::cout << "Array runtime: " << duration << " µs" << std::endl;

    // for (auto& v : array) {
    //     cle::Utils::print_vector(v);
    // }

}
