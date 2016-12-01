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

#include "../centroid_update_configuration.hpp"
#include "../measurement/measurement.hpp"

#include <cassert>
#include <string>
#include <type_traits>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/memory/local_buffer.hpp>

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
        defines += " -DCL_MASS=";
        defines += boost::compute::type_name<MassT>();

        Program program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);

        program.build(defines);

        this->kernel = program.create_kernel(KERNEL_NAME);

        reduce.prepare(context);
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

        this->kernel.set_args(
                points,
                centroids,
                masses,
                labels,
                local_centroids,
                (cl_uint)num_features,
                (cl_uint)num_points,
                (cl_uint)num_clusters);

        size_t work_offset[3] = {0, 0, 0};

        Event event;
        event = queue.enqueue_nd_range_kernel(
                this->kernel,
                2,
                work_offset,
                this->config.global_size,
                this->config.local_size,
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

        return event;
    }


private:
    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("lloyd_cluster_merge.cl");
    static constexpr const char* KERNEL_NAME = "lloyd_cluster_merge";

    Kernel kernel;
    CentroidUpdateConfiguration config;
    ReduceVectorParcol<PointT> reduce;
};

}


#endif /* CENTROID_UPDATE_CLUSTER_MERGE_HPP */
