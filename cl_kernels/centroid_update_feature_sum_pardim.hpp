/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef CENTROID_UPDATE_FEATURE_SUM_PARDIM_HPP
#define CENTROID_UPDATE_FEATURE_SUM_PARDIM_HPP

#include "kernel_path.hpp"

#include "reduce_vector_parcol.hpp"

#include "../centroid_update_configuration.hpp"
#include "../measurement/measurement.hpp"
#include "../utility.hpp"

#include <cassert>
#include <string>
#include <type_traits>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/memory/local_buffer.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
class CentroidUpdateFeatureSumPardim {
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
        assert(config.thread_features > 0);
        assert(Utility::is_power_of_two(config.thread_features));
        assert(config.local_features > 0);
        assert(Utility::is_power_of_two(config.local_features));
        assert(config.local_size[0] % config.local_features == 0);

        this->config = config;

        std::string defines;
        if (std::is_same<float, PointT>::value) {
            defines = "-DTYPE32";
        }
        else if (std::is_same<double, PointT>::value) {
            defines = "-DTYPE64";
        }
        else {
            assert(false);
        }
        defines += " -DNUM_THREAD_FEATURES=";
        defines += std::to_string(config.thread_features);

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
        size_t const num_feature_tiles =
            num_features / this->config.thread_features;

        assert(num_feature_tiles > 0);
        assert(points.size() == num_points * num_features);
        assert(labels.size() == num_points);
        assert(masses.size() >= num_clusters);
        assert(num_features
                % (this->config.thread_features
                    * this->config.local_features)
                == 0);

        datapoint.set_name("CentroidUpdateFeatureSumPardim");

        size_t global_size[3] = {
            this->config.global_size[0] / num_feature_tiles,
            num_feature_tiles,
            1
        };

        size_t local_size[3];
        if (num_feature_tiles == 1) {
            local_size[0] = this->config.local_size[0];
            local_size[1] = this->config.local_size[1];
            local_size[2] = 1;
        }
        else {
            local_size[0] = this->config.local_size[0]
                / this->config.local_features;
            local_size[1] = this->config.local_features;
            local_size[2] = 1;
        }

        size_t min_centroids_size =
            global_size[0] * num_clusters * num_features;
        if (centroids.size() < min_centroids_size) {
            centroids.resize(min_centroids_size);
        }

        LocalBuffer<PointT> local_centroids(
                local_size[0] * local_size[1]
                * this->config.thread_features
                * num_clusters
                );

        this->kernel.set_args(
                points,
                centroids,
                masses,
                labels,
                local_centroids,
                (LabelT)num_features,
                (LabelT)num_points,
                (LabelT)num_clusters);

        size_t work_offset[3] = {0, 0, 0};

        Event event;
        event = queue.enqueue_nd_range_kernel(
                this->kernel,
                2,
                work_offset,
                global_size,
                local_size,
                events);

        datapoint.add_event() = event;

        boost::compute::wait_list wait_list;
        wait_list.insert(event);

        event = reduce(
                queue,
                global_size[0],
                num_clusters * num_features,
                centroids,
                datapoint.create_child(),
                wait_list
                );

        return event;
    }


private:
    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("lloyd_feature_sum_pardim.cl");
    static constexpr const char* KERNEL_NAME = "lloyd_feature_sum_pardim";

    Kernel kernel;
    CentroidUpdateConfiguration config;
    ReduceVectorParcol<PointT> reduce;
};

}

#endif /* CENTROID_UPDATE_FEATURE_SUM_PARDIM_HPP */
