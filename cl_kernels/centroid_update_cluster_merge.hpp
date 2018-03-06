/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifndef CENTROID_UPDATE_CLUSTER_MERGE_HPP
#define CENTROID_UPDATE_CLUSTER_MERGE_HPP

#include "kernel_path.hpp"

#include "reduce_vector_parcol.hpp"
#include "matrix_binary_op.hpp"

#include "../centroid_update_configuration.hpp"
#include "../measurement/measurement.hpp"

#include <cassert>
#include <string>
#include <type_traits>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/memory/local_buffer.hpp>
#include <boost/compute/allocator/pinned_allocator.hpp>
#include <boost/compute/algorithm/copy.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
class CentroidUpdateClusterMerge {
public:
    using Event = boost::compute::event;
    using Context = boost::compute::context;
    using Kernel = boost::compute::kernel;
    using Program = boost::compute::program;
    template <typename T>
    using Vector = boost::compute::vector<T>;
    template <typename T>
    using PinnedAllocator = boost::compute::pinned_allocator<T>;
    template <typename T>
    using PinnedVector = boost::compute::vector<T, PinnedAllocator<T>>;
    template <typename T>
    using LocalBuffer = boost::compute::local_buffer<T>;

    void prepare(
            Context context,
            CentroidUpdateConfiguration config
            )
    {
        assert(boost::compute::is_fundamental<PointT>());
        assert(boost::compute::is_fundamental<LabelT>());
        assert(boost::compute::is_fundamental<MassT>());

        this->config = config;

        std::string defines;
        defines += " -DCL_INT=uint";
        defines += " -DCL_POINT=";
        defines += boost::compute::type_name<PointT>();
        defines += " -DCL_LABEL=";
        defines += boost::compute::type_name<LabelT>();
        defines += " -DVEC_LEN=";
        defines += std::to_string(this->config.vector_length);

        std::string l_stride_defines = " -DLOCAL_STRIDE";
        std::string g_mem_defines = " -DGLOBAL_MEM";

        Program g_stride_g_mem_program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);
        try {
            g_stride_g_mem_program.build(defines + g_mem_defines);
        }
        catch (std::exception e) {
            std::cerr << g_stride_g_mem_program.build_log() << std::endl;
        }
        g_stride_g_mem_kernel = g_stride_g_mem_program.create_kernel(KERNEL_NAME);

        Program g_stride_l_mem_program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);
        try {
            g_stride_l_mem_program.build(defines);
        }
        catch (std::exception e) {
            std::cerr << g_stride_l_mem_program.build_log() << std::endl;
        }
        g_stride_l_mem_kernel = g_stride_l_mem_program.create_kernel(KERNEL_NAME);

        Program l_stride_g_mem_program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);
        try {
        l_stride_g_mem_program.build(defines + l_stride_defines + g_mem_defines);
        }
        catch (std::exception e) {
            std::cerr << l_stride_g_mem_program.build_log() << std::endl;
        }
        l_stride_g_mem_kernel = l_stride_g_mem_program.create_kernel(KERNEL_NAME);

        reduce.prepare(context);
        matrix_add.prepare(context, matrix_add.Add);
    }

    Event operator() (
            boost::compute::command_queue queue,
            size_t num_features,
            size_t num_points,
            size_t num_clusters,
            Vector<PointT>& points,
            Vector<PointT>& centroids,
            PinnedVector<LabelT>& labels,
            Vector<MassT>& masses,
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
                masses.begin(),
                masses.end(),
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
            boost::compute::buffer_iterator<MassT> masses_begin,
            boost::compute::buffer_iterator<MassT> masses_end,
            Measurement::DataPoint& datapoint,
            boost::compute::wait_list const& events
            )
    {
        assert(points_end - points_begin == (long) (num_points * num_features));
        assert(centroids_end - centroids_begin == (long) (num_clusters * num_features));
        assert(labels_end - labels_begin == (long) num_points);
        assert(masses_end - masses_begin == (long) num_clusters);
        assert(points_begin.get_index() == 0u);
        assert(centroids_begin.get_index() == 0u);
        assert(labels_begin.get_index() == 0u);
        assert(masses_begin.get_index() == 0u);

        datapoint.set_name("CentroidUpdateClusterMerge");

        size_t min_centroids_size =
            this->config.global_size[0]
            * num_clusters
            * num_features;
        Vector<PointT> tmp_centroids(min_centroids_size, queue.get_context());

        LocalBuffer<PointT> local_centroids(
                this->config.local_size[0]
                * num_features
                * num_clusters
                );

        boost::compute::device device = queue.get_device();
        bool use_local_stride =
            device.type() == device.cpu ||
            device.type() == device.accelerator
            ;
        bool use_local_memory =
            device.type() == device.gpu &&
            device.local_memory_size() >
            local_centroids.size() * sizeof(PointT)
            ;
        Kernel& kernel = (use_local_stride)
            ? l_stride_g_mem_kernel
            : (use_local_memory)
            ? g_stride_l_mem_kernel
            : g_stride_g_mem_kernel
            ;

        if (use_local_memory) {
            kernel.set_args(
                    points_begin.get_buffer(),
                    tmp_centroids,
                    labels_begin.get_buffer(),
                    local_centroids,
                    (cl_uint)num_features,
                    (cl_uint)num_points,
                    (cl_uint)num_clusters);
        }
        else {
            kernel.set_args(
                    points_begin.get_buffer(),
                    tmp_centroids,
                    labels_begin.get_buffer(),
                    (cl_uint)num_features,
                    (cl_uint)num_points,
                    (cl_uint)num_clusters);
        }

        Event event;
        event = queue.enqueue_1d_range_kernel(
                kernel,
                0,
                this->config.global_size[0],
                this->config.local_size[0],
                events);

        datapoint.add_event() = event;

        boost::compute::wait_list wait_list;
        wait_list.insert(event);

        event = reduce(
                queue,
                this->config.global_size[0],
                num_clusters * num_features,
                tmp_centroids,
                datapoint.create_child(),
                wait_list
                );

        wait_list.insert(event);

        event = matrix_add.matrix(
                queue,
                num_features,
                num_clusters,
                centroids_begin,
                centroids_end,
                tmp_centroids.begin(),
                tmp_centroids.begin() + num_clusters * num_features,
                datapoint.create_child(),
                wait_list
                );

        return event;
    }


private:
    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("lloyd_cluster_merge.cl");
    static constexpr const char* KERNEL_NAME = "lloyd_cluster_merge";

    Kernel g_stride_g_mem_kernel;
    Kernel g_stride_l_mem_kernel;
    Kernel l_stride_g_mem_kernel;
    CentroidUpdateConfiguration config;
    ReduceVectorParcol<PointT> reduce;
    MatrixBinaryOp<PointT, PointT> matrix_add;
};

}


#endif /* CENTROID_UPDATE_CLUSTER_MERGE_HPP */
