/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2017, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef FUSED_FACTORY_HPP
#define FUSED_FACTORY_HPP

#include "fused_configuration.hpp"

#include "cl_kernels/fused_cluster_merge.hpp"
#include "cl_kernels/fused_feature_sum.hpp"

#include <functional>
#include <string>
#include <stdexcept>

#include <boost/compute/core.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
class FusedFactory {
public:

    template <typename T>
    using BufferIterator = boost::compute::buffer_iterator<T>;

    using FusedFunction = std::function<
        boost::compute::event(
                boost::compute::command_queue queue,
                size_t num_features,
                size_t num_points,
                size_t num_clusters,
                BufferIterator<PointT> points_begin,
                BufferIterator<PointT> points_end,
                BufferIterator<PointT> old_centroids_begin,
                BufferIterator<PointT> old_centroids_end,
                BufferIterator<PointT> new_centroids_begin,
                BufferIterator<PointT> new_centroids_end,
                BufferIterator<LabelT> labels_begin,
                BufferIterator<LabelT> labels_end,
                BufferIterator<MassT> masses_begin,
                BufferIterator<MassT> masses_end,
                Measurement::DataPoint& datapoint,
                boost::compute::wait_list const& events
                )
        >;

    FusedFunction create(
            boost::compute::context context,
            FusedConfiguration config,
            Measurement::Measurement& measurement)
    {
        measurement.set_parameter(
                "FusedGlobalSize",
                std::to_string(config.global_size[0])
                );
        measurement.set_parameter(
                "FusedLocalSize",
                std::to_string(config.local_size[0])
                );
        measurement.set_parameter(
                "FusedVectorLength",
                std::to_string(config.vector_length)
                );

        if (config.strategy == "cluster_merge") {
            FusedClusterMerge<PointT, LabelT, MassT, ColMajor> strategy;
            strategy.prepare(context, config);
            return strategy;
        }
        else if (config.strategy == "feature_sum") {
            FusedFeatureSum<PointT, LabelT, MassT, ColMajor> strategy;
            strategy.prepare(context, config);
            return strategy;
        }
        else {
            throw std::invalid_argument(config.strategy);
        }
    }
};

}

#endif /* FUSED_FACTORY_HPP */
