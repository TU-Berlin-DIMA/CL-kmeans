/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef KMEANS_GPU_ASSISTED_HPP
#define KMEANS_GPU_ASSISTED_HPP

#include "kmeans_cl_api.hpp"
#include "kmeans_common.hpp"
#include "matrix.hpp"

#include <cstdint>
#include <type_traits>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace cle {

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
class KmeansGPUAssisted {
public:
    KmeansGPUAssisted(
            cl::Context const& context,
            cl::CommandQueue const& queue
            );

    int initialize();
    int finalize();

    int operator() (
            uint32_t const max_iterations,
            cle::Matrix<FP, AllocFP, INT, true> const& points,
            cle::Matrix<FP, AllocFP, INT, true>& centroids,
            std::vector<INT, AllocINT>& cluster_size,
            std::vector<INT, AllocINT>& memberships,
            KmeansStats& stats
            );

private:
    using CL_FP = typename std::conditional<
        std::is_same<FP, float>::value, cl_float, cl_double>::type;
    using CL_INT = typename std::conditional<
        std::is_same<INT, uint32_t>::value, cl_uint, cl_ulong>::type;

    cle::Kmeans_With_Host_Kernel<CL_FP, CL_INT> kmeans_kernel_;
    cl::Context context_;
    cl::CommandQueue queue_;

    size_t max_work_group_size_;
    std::vector<size_t> max_work_item_sizes_;
};

using KmeansGPUAssisted32Aligned = KmeansGPUAssisted<float, uint32_t, AlignedAllocatorFP32, AlignedAllocatorINT32>;
using KmeansGPUAssisted64Aligned = KmeansGPUAssisted<double, uint64_t, AlignedAllocatorFP64, AlignedAllocatorINT64>;

}

extern template class cle::KmeansGPUAssisted<float, uint32_t, cle::AlignedAllocatorFP32, cle::AlignedAllocatorINT32>;
extern template class cle::KmeansGPUAssisted<double, uint64_t, cle::AlignedAllocatorFP64, cle::AlignedAllocatorINT64>;

#endif /* KMEANS_GPU_ASSISTED_HPP */
