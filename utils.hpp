#ifndef UTILS_HPP
#define UTILS_HPP

#include <algorithm>
#include <vector>
#include <iostream>

namespace cle {

class Utils {

public:
    template <typename T>
    static void print_vector(std::vector<T> const& vec) {
        for_each(vec.begin(), vec.end(), [](T x){ std::cout << x << " "; });
        std::cout << std::endl;
    }

};

};

#endif /* UTILS_HPP */
