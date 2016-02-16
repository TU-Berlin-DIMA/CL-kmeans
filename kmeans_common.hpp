/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef KMEANS_COMMON_HPP
#define KMEANS_COMMON_HPP

#include <cstdint>
#include <memory> // std::allocator
#include <SystemConfig.h>

#ifdef USE_ALIGNED_ALLOCATOR
#include <boost/align/aligned_allocator.hpp>
#endif

namespace cle {

class KmeansStats {
public:
        uint32_t iterations;
};

#ifdef USE_ALIGNED_ALLOCATOR
using AlignedAllocatorFP32 =
    boost::alignment::aligned_allocator<float, 32>;
using AlignedAllocatorINT32 =
    boost::alignment::aligned_allocator<uint32_t, 32>;
using AlignedAllocatorFP64 =
    boost::alignment::aligned_allocator<double, 32>;
using AlignedAllocatorINT64 =
    boost::alignment::aligned_allocator<uint64_t, 32>;
#endif /* USE_ALIGNED_ALLOCATOR */

}

#endif /* KMEANS_COMMON_HPP */
