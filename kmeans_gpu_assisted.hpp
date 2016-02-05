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

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace cle {

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
            std::vector<double> const& points_x,
            std::vector<double> const& points_y,
            std::vector<double>& centroids_x,
            std::vector<double>& centroids_y,
            std::vector<uint64_t>& cluster_size,
            std::vector<uint64_t>& memberships,
            KmeansStats& stats
            );

private:
    cle::Kmeans_With_Host_Kernel kmeans_kernel_;
    cl::Context context_;
    cl::CommandQueue queue_;

    size_t max_work_group_size_;
    std::vector<size_t> max_work_item_sizes_;
};

}

#endif /* KMEANS_GPU_ASSISTED_HPP */
