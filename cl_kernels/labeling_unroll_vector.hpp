/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>
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
#include <utility> // std::move

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
    using ReadonlyVector = boost::compute::vector<T, readonly_allocator<T>>;
    template <typename T>
    using PinnedAllocator = boost::compute::pinned_allocator<T>;
    template <typename T>
    using PinnedVector = boost::compute::vector<T, PinnedAllocator<T>>;

    LabelingUnrollVector() :
        g_stride_g_mem_kernel(Utility::log2(MAX_FEATURES)),
        g_stride_l_mem_kernel(Utility::log2(MAX_FEATURES)),
        l_stride_g_mem_kernel(Utility::log2(MAX_FEATURES)),
        local_points(1)
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

        std::string l_stride_defines = " -DLOCAL_STRIDE";
        std::string g_mem_defines = " -DGLOBAL_MEM";

        for (uint32_t d = 2; d <= MAX_FEATURES; d = d * 2) {

            std::string features =
                " -DNUM_FEATURES="
                + std::to_string(d);

            Program g_stride_g_mem_program = Program::create_with_source_file(
                    PROGRAM_FILE,
                    context);
            Program g_stride_l_mem_program = Program::create_with_source_file(
                    PROGRAM_FILE,
                    context);
            Program l_stride_g_mem_program = Program::create_with_source_file(
                    PROGRAM_FILE,
                    context);

            size_t kernel_index = Utility::log2(d) - 1;

            try {
                g_stride_g_mem_program.build(defines + features + g_mem_defines);
                g_stride_g_mem_kernel[kernel_index] =
                    g_stride_g_mem_program.create_kernel(KERNEL_NAME);
            }
            catch (std::exception e) {
                std::cout << g_stride_g_mem_program.build_log() << std::endl;
                throw e;
            }

            try {
                g_stride_l_mem_program.build(defines + features);
                g_stride_l_mem_kernel[kernel_index] =
                    g_stride_l_mem_program.create_kernel(KERNEL_NAME);
            }
            catch (std::exception e) {
                std::cout << g_stride_l_mem_program.build_log() << std::endl;
                throw e;
            }

            try {
                l_stride_g_mem_program.build(
                        defines + features + l_stride_defines + g_mem_defines
                        );
                l_stride_g_mem_kernel[kernel_index] =
                    l_stride_g_mem_program.create_kernel(KERNEL_NAME);
            }
            catch (std::exception e) {
                std::cout << l_stride_g_mem_program.build_log() << std::endl;
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
            boost::compute::wait_list const& events
            )
    {
        return (*this)(
                queue,
                num_features,
                num_points,
                num_clusters,
                points.begin(),
                points.end(),
                centroids.begin(),
                centroids.end(),
                labels.begin(),
                labels.end(),
                datapoint,
                events
                );
    }

    Event operator() (
            boost::compute::command_queue queue,
            size_t num_features,
            size_t num_points,
            size_t num_clusters,
            boost::compute::buffer_iterator<PointT> points_begin,
            boost::compute::buffer_iterator<PointT> points_end,
            boost::compute::buffer_iterator<PointT> centroids_begin,
            boost::compute::buffer_iterator<PointT> centroids_end,
            boost::compute::buffer_iterator<LabelT> labels_begin,
            boost::compute::buffer_iterator<LabelT> labels_end,
            Measurement::DataPoint& datapoint,
            boost::compute::wait_list const& events
            )
    {

        assert(num_features <= MAX_FEATURES);
        assert(points_end - points_begin == (long) (num_points * num_features));
        assert(centroids_end - centroids_begin == (long) (num_clusters * num_features));
        assert(labels_end - labels_begin == (long) num_points);
        assert(points_begin.get_index() == 0u);
        assert(centroids_begin.get_index() == 0u);
        assert(labels_begin.get_index() == 0u);

        datapoint.set_name("LabelingUnrollVector");

        size_t const local_points_size =
            this->config.local_size[0]
            * this->config.vector_length
            * num_features
            ;
        if (this->local_points.size() != local_points_size) {
            this->local_points = std::move(
                    LocalBuffer<PointT>(
                        local_points_size
                        ));
        }

        size_t const min_ro_centroids_size = num_clusters * num_features;
        if (this->ro_centroids.size() < min_ro_centroids_size) {
            this->ro_centroids = std::move(
                    ReadonlyVector<PointT>(
                        min_ro_centroids_size,
                        queue.get_context()
                        ));
        }
        boost::compute::copy(
                centroids_begin,
                centroids_begin + num_clusters * num_features,
                this->ro_centroids.begin(),
                queue
                );

        boost::compute::device device = queue.get_device();
        bool use_local_stride =
            device.type() == device.cpu ||
            device.type() == device.accelerator
            ;
        bool use_local_memory =
            device.type() == device.gpu &&
            device.local_memory_size() >
            local_points_size * sizeof(PointT)
            ;
        auto& kernel = (use_local_stride)
            ? this->l_stride_g_mem_kernel
            : (
                    use_local_memory
              )
            ? this->g_stride_l_mem_kernel
            : this->g_stride_g_mem_kernel
            ;
        size_t kernel_index = Utility::log2(num_features) - 1;

        if (use_local_memory) {
            kernel[kernel_index].set_args(
                    points_begin.get_buffer(),
                    this->ro_centroids,
                    labels_begin.get_buffer(),
                    this->local_points,
                    (cl_uint) num_points,
                    (cl_uint) num_clusters);
        }
        else {
            kernel[kernel_index].set_args(
                    points_begin.get_buffer(),
                    this->ro_centroids,
                    labels_begin.get_buffer(),
                    (cl_uint) num_points,
                    (cl_uint) num_clusters);
        }

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

    std::vector<Kernel> g_stride_g_mem_kernel;
    std::vector<Kernel> g_stride_l_mem_kernel;
    std::vector<Kernel> l_stride_g_mem_kernel;
    ReadonlyVector<PointT> ro_centroids;
    LocalBuffer<PointT> local_points;
    LabelingConfiguration config;

};

}

#endif /* LABELING_UNROLL_VECTOR_HPP */
