#include "csv.hpp"

#include "timer.hpp"
#include "common.hpp"

#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <algorithm>

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

    std::vector<std::vector<double>> array;

    timer.start();
    csv.read_csv(file_name, array);
    duration = timer.stop<std::chrono::microseconds>();

    std::cout << "Array runtime: " << duration << " µs" << std::endl;

    // for (auto& v : array) {
    //     cle::Utils::print_vector(v);
    // }

    if (
            not std::equal(a.cbegin(), a.cend(), array[0].cbegin())
            ||
            not std::equal(b.cbegin(), b.cend(), array[1].cbegin())
       ) {

        std::cout << "Mismatch between read_csv and read_csv_dynamic!!"
            << std::endl;
    }
}
