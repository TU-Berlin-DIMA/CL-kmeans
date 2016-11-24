/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "binary_format.hpp"

#include "matrix.hpp"

#include <cstdint>
#include <fstream>
#include <memory>
#include <cassert>

template <typename FP, typename AllocFP, typename INT>
int cle::BinaryFormat::read(char const* file_name, cle::Matrix<FP, AllocFP, INT>& matrix) {

    std::ifstream fh(file_name, std::fstream::binary);

    uint64_t num_features;
    uint64_t num_clusters;
    uint64_t num_points;

    fh.read((char*)&num_features, sizeof(num_features));
    fh.read((char*)&num_clusters, sizeof(num_clusters));
    fh.read((char*)&num_points, sizeof(num_points));

    // Importing ground-truth centroids not supported yet
    assert(num_clusters == 0);

    matrix.resize(num_points, num_features);

    for (uint64_t f = 0; f < num_features; ++f) {
        for (uint64_t p = 0; p < num_points; ++p) {
            float point;
            fh.read((char*)&point, sizeof(point));
            matrix(p, f) = point;
        }
    }

    return 1;
}

template int cle::BinaryFormat::read(char const*, cle::Matrix<float, std::allocator<float>, uint32_t>&);
template int cle::BinaryFormat::read(char const*, cle::Matrix<float, std::allocator<float>, size_t>&);
template int cle::BinaryFormat::read(char const*, cle::Matrix<double, std::allocator<double>, size_t>&);

#ifdef USE_ALIGNED_ALLOCATOR
template int cle::BinaryFormat::read(char const*, cle::Matrix<float, boost::alignment::aligned_allocator<float, 32>, uint32_t>&);
template int cle::BinaryFormat::read(char const*, cle::Matrix<double, boost::alignment::aligned_allocator<double, 32>, uint64_t>&);
#endif
