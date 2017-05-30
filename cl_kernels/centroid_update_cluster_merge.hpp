/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
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

        Program gs_program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);
        gs_program.build(defines);
        this->global_stride_kernel = gs_program.create_kernel(KERNEL_NAME);

        defines += " -DLOCAL_STRIDE";
        Program ls_program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);
        ls_program.build(defines);
        this->local_stride_kernel = ls_program.create_kernel(KERNEL_NAME);

        reduce.prepare(context);
        divide_matrix.prepare(context, divide_matrix.Divide);
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
        assert(points.size() == num_points * num_features);
        assert(labels.size() == num_points);
        assert(masses.size() >= num_clusters);

        datapoint.set_name("CentroidUpdateClusterMerge");

        size_t min_centroids_size =
            this->config.global_size[0]
            * num_clusters
            * num_features;
        if (centroids.size() < min_centroids_size) {
            centroids.resize(min_centroids_size);
        }

        LocalBuffer<PointT> local_centroids(
                this->config.local_size[0]
                * num_features
                * num_clusters
                );

        boost::compute::device device = queue.get_device();
        Kernel& kernel = (
                device.type() == device.cpu ||
                device.type() == device.accelerator
                )
            ? this->local_stride_kernel
            : this->global_stride_kernel
            ;

        kernel.set_args(
                points,
                centroids,
                labels,
                local_centroids,
                (cl_uint)num_features,
                (cl_uint)num_points,
                (cl_uint)num_clusters);

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
                centroids,
                datapoint.create_child(),
                wait_list
                );

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
    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("lloyd_cluster_merge.cl");
    static constexpr const char* KERNEL_NAME = "lloyd_cluster_merge";

    Kernel global_stride_kernel;
    Kernel local_stride_kernel;
    CentroidUpdateConfiguration config;
    ReduceVectorParcol<PointT> reduce;
    MatrixBinaryOp<PointT, MassT> divide_matrix;
};

}


#endif /* CENTROID_UPDATE_CLUSTER_MERGE_HPP */
