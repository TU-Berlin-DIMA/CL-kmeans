/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef FUSED_FACTORY_HPP
#define FUSED_FACTORY_HPP

#include "fused_configuration.hpp"

#include "cl_kernels/fused_cluster_merge.hpp"

#include <functional>
#include <string>
#include <stdexcept>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
class FusedFactory {
public:

    template <typename T>
    using Vector = boost::compute::vector<T>;

    using FusedFunction = std::function<
        boost::compute::event(
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
        >;

    FusedFunction create(
            boost::compute::context context,
            FusedConfiguration config)
    {
        if (config.strategy == "cluster_merge") {
            FusedClusterMerge<PointT, LabelT, MassT, ColMajor> strategy;
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
