/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef CENTROID_UPDATE_FACTORY_HPP
#define CENTROID_UPDATE_FACTORY_HPP

#include "centroid_update_configuration.hpp"

#include "measurement/measurement.hpp"

#include "cl_kernels/centroid_update_feature_sum.hpp"
#include "cl_kernels/centroid_update_feature_sum_pardim.hpp"
#include "cl_kernels/centroid_update_cluster_merge.hpp"

#include <functional>
#include <string>
#include <stdexcept>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/allocator/pinned_allocator.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
class CentroidUpdateFactory {
public:

    template <typename T>
    using Vector = boost::compute::vector<T>;
    template <typename T>
    using PinnedAllocator = boost::compute::pinned_allocator<T>;
    template <typename T>
    using PinnedVector = boost::compute::vector<T, PinnedAllocator<T>>;

    using CentroidUpdateFunction = std::function<
        boost::compute::event(
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
        >;

    CentroidUpdateFunction create(
            boost::compute::context context,
            CentroidUpdateConfiguration config,
            Measurement::Measurement& measurement
            )
    {
        measurement.set_parameter(
                "CentroidUpdateGlobalSize",
                std::to_string(config.global_size[0])
                );
        measurement.set_parameter(
                "CentroidUpdateLocalSize",
                std::to_string(config.local_size[0])
                );

        if (
                config.strategy == "cluster_merge" ||
                config.strategy == "feature_sum_pardim"
           )
        {
            measurement.set_parameter(
                    "CentroidUpdateVectorLength",
                    std::to_string(config.vector_length)
                    );
        }

        if (config.strategy == "feature_sum") {
            CentroidUpdateFeatureSum<
                PointT,
                LabelT,
                MassT,
                ColMajor>
                    strategy;
            strategy.prepare(context, config);
            return strategy;
        }
        else if (config.strategy == "feature_sum_pardim") {
            measurement.set_parameter(
                    "CentroidUpdateLocalFeatures",
                    std::to_string(config.local_features)
                    );
            measurement.set_parameter(
                    "CentroidUpdateThreadFeatures",
                    std::to_string(config.thread_features)
                    );

            CentroidUpdateFeatureSumPardim<
                PointT,
                LabelT,
                MassT,
                ColMajor>
                    strategy;
            strategy.prepare(context, config);
            return strategy;
        }
        else if (config.strategy == "cluster_merge") {
            CentroidUpdateClusterMerge<
                PointT,
                LabelT,
                MassT,
                ColMajor>
                    strategy;
            strategy.prepare(context, config);
            return strategy;
        }
        else {
            throw std::invalid_argument(config.strategy);
        }

    }
};

}

#endif /* CENTROID_UPDATE_FACTORY_HPP */
