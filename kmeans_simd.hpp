/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef KMEANS_SIMD_HPP
#define KMEANS_SIMD_HPP

#include "kmeans_common.hpp"

#include <vector>

namespace cle {

class KmeansSIMD32 {
public:
    int initialize();
    int finalize();

    void operator() (
            uint32_t const max_iterations,
            std::vector<float, AlignedAllocatorFP32> const& points_x,
            std::vector<float, AlignedAllocatorFP32> const& points_y,
            std::vector<float, AlignedAllocatorFP32>& centroids_x,
            std::vector<float, AlignedAllocatorFP32>& centroids_y,
            std::vector<uint32_t, AlignedAllocatorINT32>& cluster_size,
            std::vector<uint32_t, AlignedAllocatorINT32>& memberships,
            KmeansStats& stats);
};

}

#endif /* KMEANS_SIMD_HPP */
