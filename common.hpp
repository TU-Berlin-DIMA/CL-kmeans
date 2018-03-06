/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <algorithm>
#include <vector>
#include <iostream>

#define CLE_STRINGIFY(name) #name

namespace cle {

class Utils {

public:
    template <typename T, typename Alloc>
    static void print_vector(std::vector<T, Alloc> const& vec) {
        for_each(vec.begin(), vec.end(), [](T x){ std::cout << x << " "; });
        std::cout << std::endl;
    }

};

};

#endif /* COMMON_HPP_ */
