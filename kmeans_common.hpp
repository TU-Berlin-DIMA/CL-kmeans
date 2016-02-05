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

#include <boost/align/aligned_allocator.hpp>

namespace cle {

using KmeansStats = struct { uint32_t iterations; };
using AlignedAllocatorFP32 =
    boost::alignment::aligned_allocator<float, 256>;
using AlignedAllocatorINT32 =
    boost::alignment::aligned_allocator<uint32_t, 256>;
using AlignedAllocatorFP64 =
    boost::alignment::aligned_allocator<double, 256>;
using AlignedAllocatorINT64 =
    boost::alignment::aligned_allocator<uint64_t, 256>;

}

#endif /* KMEANS_COMMON_HPP */
