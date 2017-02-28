/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef FUSED_FEATURE_SUM_HPP
#define FUSED_FEATURE_SUM_HPP

#include "kernel_path.hpp"

#include "reduce_vector_parcol.hpp"
#include "matrix_binary_op.hpp"

#include "../fused_configuration.hpp"
#include "../measurement/measurement.hpp"

#include <cassert>
#include <string>
#include <type_traits>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/memory/local_buffer.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
class FusedFeatureSum {
public:
    using Event = boost::compute::event;
    using Context = boost::compute::context;
    using Kernel = boost::compute::kernel;
    using Program = boost::compute::program;
    template <typename T>
    using Vector = boost::compute::vector<T>;
    template <typename T>
    using LocalBuffer = boost::compute::local_buffer<T>;

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
            defines += " -DCL_POINT_MAX=FLT_MAX";
        }
        else if (std::is_same<double, PointT>::value) {
            defines += " -DCL_POINT_MAX=DBL_MAX";
        }
        defines += " -DCL_LABEL=";
        defines += boost::compute::type_name<LabelT>();
        defines += " -DCL_MASS=";
        defines += boost::compute::type_name<MassT>();
        defines += " -DNUM_THREAD_FEATURES=1";

        Program program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);

        program.build(defines);

        this->kernel = program.create_kernel(KERNEL_NAME);

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

        datapoint.set_name("FusedFeatureSum");

        size_t const min_centroids_size =
            this->config.global_size[0]
            * num_clusters
            // * config.num_thread_features
            ;
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
                this->config.local_size[0] * num_features
                );
        LocalBuffer<PointT> local_old_centroids(
                num_clusters
                * num_features
                );
        LocalBuffer<PointT> local_new_centroids(
                this->config.local_size[0]
                // * config.num_thread_features,
                * num_clusters
                );
        LocalBuffer<MassT> local_masses(
                this->config.local_size[0] * num_clusters
                );
        LocalBuffer<LabelT> local_labels(
                this->config.local_size[0]
                );

        this->kernel.set_args(
                points,
                centroids,
                new_centroids,
                new_masses,
                labels,
                local_points,
                local_old_centroids,
                local_new_centroids,
                local_masses,
                local_labels,
                (cl_uint)num_features,
                (cl_uint)num_points,
                (cl_uint)num_clusters);

        size_t work_offset[3] = {0, 0, 0};

        Event event;
        event = queue.enqueue_nd_range_kernel(
                this->kernel,
                1,
                work_offset,
                this->config.global_size,
                this->config.local_size,
                events);

        datapoint.add_event() = event;

        boost::compute::wait_list wait_list;
        wait_list.insert(event);

        size_t num_tiles =
            this->config.global_size[0]
             / num_features
            // * config.num_thread_features
            ;

        event = reduce_centroids(
                queue,
                num_tiles,
                num_clusters * num_features,
                new_centroids,
                datapoint.create_child(),
                wait_list
                );

        boost::compute::copy_async(
                new_centroids.begin(),
                new_centroids.begin()
                + num_clusters * num_features,
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
    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("lloyd_fused_feature_sum.cl");
    static constexpr const char* KERNEL_NAME = "lloyd_fused_feature_sum";

    Kernel kernel;
    FusedConfiguration config;
    ReduceVectorParcol<PointT> reduce_centroids;
    ReduceVectorParcol<MassT> reduce_masses;
    MatrixBinaryOp<PointT, MassT> divide_matrix;
};

}


#endif /* FUSED_FEATURE_SUM_HPP */
