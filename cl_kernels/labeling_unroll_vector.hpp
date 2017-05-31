/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2017, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef LABELING_UNROLL_VECTOR_HPP
#define LABELING_UNROLL_VECTOR_HPP

#include "kernel_path.hpp"

#include "../utility.hpp"
#include "../labeling_configuration.hpp"
#include "../measurement/measurement.hpp"
#include "../allocator/readonly_allocator.hpp"

#include <iostream>
#include <stdexcept>
#include <cassert>
#include <string>
#include <type_traits>
#include <vector>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/memory/local_buffer.hpp>
#include <boost/compute/allocator/pinned_allocator.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, bool ColMajor>
class LabelingUnrollVector {
public:
    using Event = boost::compute::event;
    using Context = boost::compute::context;
    using Kernel = boost::compute::kernel;
    using Program = boost::compute::program;
    template <typename T>
    using LocalBuffer = boost::compute::local_buffer<T>;
    template <typename T>
    using Vector = boost::compute::vector<T>;
    template <typename T>
    using PinnedAllocator = boost::compute::pinned_allocator<T>;
    template <typename T>
    using PinnedVector = boost::compute::vector<T, PinnedAllocator<T>>;

    LabelingUnrollVector() :
        global_stride_kernel(Utility::log2(MAX_FEATURES)),
        local_stride_kernel(Utility::log2(MAX_FEATURES))
    {}

    void prepare(Context context, LabelingConfiguration config) {
        this->config = config;

        std::string defines;
        defines += " -DCL_INT=uint";
        defines += " -DCL_POINT=";
        defines += boost::compute::type_name<PointT>();
        defines += " -DCL_LABEL=";
        defines += boost::compute::type_name<LabelT>();
        if (std::is_same<float, PointT>::value) {
            defines += " -DCL_SINT=int";
            defines += " -DCL_POINT_MAX=FLT_MAX";
        }
        else if (std::is_same<double, PointT>::value) {
            defines += " -DCL_SINT=long";
            defines += " -DCL_POINT_MAX=DBL_MAX";
        }
        else {
            assert(false);
        }

        defines += " -DVEC_LEN="
            + std::to_string(this->config.vector_length);

        std::string ls_defines = " -DLOCAL_STRIDE";

        for (uint32_t d = 2; d <= MAX_FEATURES; d = d * 2) {

            std::string features =
                " -DNUM_FEATURES="
                + std::to_string(d);

            Program gs_program = Program::create_with_source_file(
                    PROGRAM_FILE,
                    context);
            Program ls_program = Program::create_with_source_file(
                    PROGRAM_FILE,
                    context);

            size_t kernel_index = Utility::log2(d) - 1;

            try {
                gs_program.build(defines + features);
                global_stride_kernel[kernel_index] =
                    gs_program.create_kernel(KERNEL_NAME);
            }
            catch (std::exception e) {
                std::cout << gs_program.build_log() << std::endl;
                throw e;
            }

            try {
                ls_program.build(defines + features + ls_defines);
                local_stride_kernel[kernel_index] =
                    ls_program.create_kernel(KERNEL_NAME);
            }
            catch (std::exception e) {
                std::cout << ls_program.build_log() << std::endl;
                throw e;
            }

        }
    }

    Event operator() (
            boost::compute::command_queue queue,
            size_t num_features,
            size_t num_points,
            size_t num_clusters,
            Vector<PointT>& points,
            Vector<PointT>& centroids,
            PinnedVector<LabelT>& labels,
            Measurement::DataPoint& datapoint,
            boost::compute::wait_list const& events) {

        assert(num_features <= MAX_FEATURES);

        datapoint.set_name("LabelingUnrollVector");

        LocalBuffer<PointT> local_points(
                this->config.local_size[0]
                * this->config.vector_length
                * num_features
                );

        boost::compute::vector<PointT, readonly_allocator<PointT>> ro_centroids(
                num_clusters * num_features,
                queue.get_context()
                );
        boost::compute::copy(
                centroids.begin(),
                centroids.begin() + num_clusters * num_features,
                ro_centroids.begin(),
                queue
                );

        boost::compute::device device = queue.get_device();
        auto& kernel = (device.type() == device.cpu)
            ? this->local_stride_kernel
            : this->global_stride_kernel
            ;
        size_t kernel_index = Utility::log2(num_features) - 1;

        kernel[kernel_index].set_args(
                points,
                ro_centroids,
                labels,
                local_points,
                (cl_uint) num_points,
                (cl_uint) num_clusters);

        size_t work_offset[3] = {0, 0, 0};

        Event event;
        event = queue.enqueue_nd_range_kernel(
                kernel[kernel_index],
                1,
                work_offset,
                this->config.global_size,
                this->config.local_size,
                events);

        datapoint.add_event() = event;
        return event;
    }

private:
    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("lloyd_labeling_vp_clcp.cl");
    static constexpr const char* KERNEL_NAME = "lloyd_labeling_vp_clcp";
    static constexpr const size_t MAX_FEATURES = 1024;

    std::vector<Kernel> global_stride_kernel;
    std::vector<Kernel> local_stride_kernel;
    LabelingConfiguration config;

};

}

#endif /* LABELING_UNROLL_VECTOR_HPP */
