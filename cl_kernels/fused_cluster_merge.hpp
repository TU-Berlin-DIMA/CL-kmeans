/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2017, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef FUSED_CLUSTER_MERGE_HPP
#define FUSED_CLUSTER_MERGE_HPP

#include "kernel_path.hpp"

#include "reduce_vector_parcol.hpp"
#include "matrix_binary_op.hpp"

#include "../fused_configuration.hpp"
#include "../measurement/measurement.hpp"
#include "../allocator/readonly_allocator.hpp"
#include "../utility.hpp"

#include <cassert>
#include <string>
#include <type_traits>
#include <vector>
#include <stdexcept>
#include <iostream>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/memory/local_buffer.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
class FusedClusterMerge {
public:
    using Event = boost::compute::event;
    using Context = boost::compute::context;
    using Kernel = boost::compute::kernel;
    using Program = boost::compute::program;
    template <typename T>
    using Vector = boost::compute::vector<T>;
    template <typename T>
    using ReadonlyVector = boost::compute::vector<T, readonly_allocator<T>>;
    template <typename T>
    using LocalBuffer = boost::compute::local_buffer<T>;

    FusedClusterMerge() :
        g_stride_g_mem_kernel(Utility::log2(MAX_FEATURES)),
        g_stride_l_mem_kernel(Utility::log2(MAX_FEATURES)),
        l_stride_g_mem_kernel(Utility::log2(MAX_FEATURES))
    {}

    void prepare(
            Context context,
            FusedConfiguration config
            )
    {
        static_assert(boost::compute::is_fundamental<PointT>(),
                "PointT must be a boost compute fundamental type");
        static_assert(boost::compute::is_fundamental<LabelT>(),
                "LabelT must be a boost compute fundamental type");
        static_assert(boost::compute::is_fundamental<MassT>(),
                "MassT must be a boost compute fundamental type");
        static_assert(std::is_same<float, PointT>::value
                or std::is_same<double, PointT>::value,
                "PointT must be float or double");

        this->config = config;

        std::string defines;
        defines += " -DCL_INT=uint";
        defines += " -DCL_POINT=";
        defines += boost::compute::type_name<PointT>();
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
        defines += " -DCL_LABEL=";
        defines += boost::compute::type_name<LabelT>();
        defines += " -DCL_MASS=";
        defines += boost::compute::type_name<MassT>();
        defines += " -DVEC_LEN=";
        defines += std::to_string(this->config.vector_length);

        std::string l_stride_defines = " -DLOCAL_STRIDE";
        std::string g_mem_defines = " -DGLOBAL_MEM";

        for (size_t d = 2; d <= MAX_FEATURES; d = d * 2) {

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
                std::cerr << g_stride_g_mem_program.build_log() << std::endl;
                throw e;
            }

            try {
                g_stride_l_mem_program.build(defines + features);
                g_stride_l_mem_kernel[kernel_index] =
                    g_stride_l_mem_program.create_kernel(KERNEL_NAME);
            }
            catch (std::exception e) {
                std::cerr << g_stride_l_mem_program.build_log() << std::endl;
                throw e;
            }

            try {
                l_stride_g_mem_program.build(defines + features + l_stride_defines + g_mem_defines);
                l_stride_g_mem_kernel[kernel_index] =
                    l_stride_g_mem_program.create_kernel(KERNEL_NAME);
            }
            catch (std::exception e) {
                std::cerr << l_stride_g_mem_program.build_log() << std::endl;
                throw e;
            }
        }

        reduce_centroids.prepare(context);
        reduce_masses.prepare(context);
        divide_matrix.prepare(context, divide_matrix.Divide);
    }

    Event operator() (
            boost::compute::command_queue queue,
            size_t num_features,
            size_t num_points,
            size_t num_clusters,
            Vector<PointT>& points,
            Vector<PointT>& centroids,
            Vector<LabelT>& labels,
            Vector<MassT>& masses,
            Measurement::DataPoint& datapoint,
            boost::compute::wait_list const& events
            )
    {
        assert(points.size() == num_points * num_features);
        assert(centroids.size() == num_clusters * num_features);
        assert(labels.size() == num_points);
        assert(masses.size() == num_clusters);
        assert(num_features <= MAX_FEATURES);

        datapoint.set_name("FusedClusterMerge");

        size_t const min_centroids_size =
            this->config.global_size[0]
            * num_clusters
            * num_features;
        Vector<PointT> new_centroids(
                min_centroids_size,
                queue.get_context()
                );

        size_t const min_masses_size =
            this->config.global_size[0]
            * num_clusters;
        Vector<MassT> new_masses(
                min_masses_size,
                queue.get_context()
                );

        LocalBuffer<PointT> local_points(
                this->config.local_size[0]
                * this->config.vector_length
                * num_features
                );
        LocalBuffer<PointT> local_new_centroids(
                this->config.local_size[0]
                * num_clusters
                * num_features
                );
        LocalBuffer<MassT> local_masses(
                this->config.local_size[0] * num_clusters
                );

        ReadonlyVector<PointT> ro_centroids(
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
        boost::compute::device device = queue.get_device();
        bool use_local_stride =
            device.type() == device.cpu ||
            device.type() == device.accelerator
            ;
        bool use_local_memory =
            device.type() == device.gpu &&
            device.local_memory_size() >
            (
             local_points.size() +
             local_new_centroids.size() +
             local_masses.size()
            )
            ;
        auto& kernel = (use_local_stride)
            ? l_stride_g_mem_kernel[kernel_index]
            : (
                    use_local_memory
              )
            ? g_stride_l_mem_kernel[kernel_index]
            : g_stride_g_mem_kernel[kernel_index]
            ;

        if (use_local_memory) {
            kernel.set_args(
                    points,
                    ro_centroids,
                    new_centroids,
                    new_masses,
                    labels,
                    local_points,
                    local_new_centroids,
                    local_masses,
                    (cl_uint)num_points,
                    (cl_uint)num_clusters);
        }
        else {
            kernel.set_args(
                    points,
                    ro_centroids,
                    new_centroids,
                    new_masses,
                    labels,
                    (cl_uint)num_points,
                    (cl_uint)num_clusters);
        }

        size_t work_offset[3] = {0, 0, 0};

        Event event;
        event = queue.enqueue_nd_range_kernel(
                kernel,
                1,
                work_offset,
                this->config.global_size,
                this->config.local_size,
                events);

        datapoint.add_event() = event;

        boost::compute::wait_list wait_list;
        wait_list.insert(event);

        event = reduce_centroids(
                queue,
                this->config.global_size[0],
                num_clusters * num_features,
                new_centroids,
                datapoint.create_child(),
                wait_list
                );

        boost::compute::copy_async(
                new_centroids.begin(),
                new_centroids.begin()
                + num_features * num_clusters,
                centroids.begin(),
                queue
                );

        wait_list.insert(event);

        event = reduce_masses(
                queue,
                this->config.global_size[0],
                num_clusters,
                new_masses,
                datapoint.create_child(),
                wait_list
                );

        boost::compute::copy_async(
                new_masses.begin(),
                new_masses.begin() + num_clusters,
                masses.begin(),
                queue
                );

        wait_list.insert(event);

        event = divide_matrix.row(
                queue,
                num_features,
                num_clusters,
                centroids,
                masses,
                datapoint.create_child(),
                wait_list
                );

        return event;
    }


private:
    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("lloyd_fused_cluster_merge.cl");
    static constexpr const char* KERNEL_NAME = "lloyd_fused_cluster_merge";
    static constexpr const size_t MAX_FEATURES = 1024;

    std::vector<Kernel> g_stride_g_mem_kernel;
    std::vector<Kernel> g_stride_l_mem_kernel;
    std::vector<Kernel> l_stride_g_mem_kernel;
    FusedConfiguration config;
    ReduceVectorParcol<PointT> reduce_centroids;
    ReduceVectorParcol<MassT> reduce_masses;
    MatrixBinaryOp<PointT, MassT> divide_matrix;
};

}


#endif /* FUSED_CLUSTER_MERGE_HPP */
