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

    LabelingUnrollVector() :
        kernel(Utility::log2(MAX_FEATURES))
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

        for (uint32_t d = 2; d <= MAX_FEATURES; d = d * 2) {

            std::string features =
                " -DNUM_FEATURES="
                + std::to_string(d);

            Program program = Program::create_with_source_file(
                    PROGRAM_FILE,
                    context);

            try {
                size_t kernel_index = Utility::log2(d) - 1;
                program.build(defines + features);
                this->kernel[kernel_index] =
                    program.create_kernel(KERNEL_NAME);
            }
            catch (std::exception e) {
                std::cout << program.build_log() << std::endl;
                throw e;
            }

        }
    }

    Event operator() (
            boost::compute::command_queue queue,
            size_t num_features,
            size_t num_points,
            size_t num_clusters,
            boost::compute::vector<PointT>& points,
            boost::compute::vector<PointT>& centroids,
            boost::compute::vector<LabelT>& labels,
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

        size_t kernel_index = Utility::log2(num_features) - 1;
        this->kernel[kernel_index].set_args(
                points,
                ro_centroids,
                labels,
                local_points,
                (cl_uint) num_points,
                (cl_uint) num_clusters);

        size_t work_offset[3] = {0, 0, 0};

        Event event;
        event = queue.enqueue_nd_range_kernel(
                this->kernel[kernel_index],
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

    std::vector<Kernel> kernel;
    LabelingConfiguration config;

};

}

#endif /* LABELING_UNROLL_VECTOR_HPP */
