/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef BINARY_FORMAT_HPP
#define BINARY_FORMAT_HPP

#include "matrix.hpp"

namespace Clustering {

class BinaryFormat {
public:
    template <typename FP, typename AllocFP, typename INT>
    int read(char const* file_name, cle::Matrix<FP, AllocFP, INT>& matrix);
};

}

extern template int Clustering::BinaryFormat::read(char const*, cle::Matrix<float, std::allocator<float>, uint32_t>&);
extern template int Clustering::BinaryFormat::read(char const*, cle::Matrix<float, std::allocator<float>, size_t>&);
extern template int Clustering::BinaryFormat::read(char const*, cle::Matrix<double, std::allocator<double>, size_t>&);

#endif /* BINARY_FORMAT_HPP */
