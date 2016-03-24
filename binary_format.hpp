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

#include <SystemConfig.h>

#ifdef USE_ALIGNED_ALLOCATOR
#include <boost/align/aligned_allocator.hpp>
#endif

namespace cle {

class BinaryFormat {
public:
    template <typename FP, typename AllocFP, typename INT>
    int read(char const* file_name, cle::Matrix<FP, AllocFP, INT>& matrix);
};

}

extern template int cle::BinaryFormat::read(char const*, cle::Matrix<float, std::allocator<float>, uint32_t>&);
extern template int cle::BinaryFormat::read(char const*, cle::Matrix<double, std::allocator<double>, uint64_t>&);

#ifdef USE_ALIGNED_ALLOCATOR
extern template int cle::BinaryFormat::read(char const*, cle::Matrix<float, boost::alignment::aligned_allocator<float, 32>, uint32_t>&);
extern template int cle::BinaryFormat::read(char const*, cle::Matrix<double, boost::alignment::aligned_allocator<double, 32>, uint64_t>&);
#endif

#endif /* BINARY_FORMAT_HPP */
