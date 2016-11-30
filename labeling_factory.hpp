/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef LABELING_FACTORY_HPP
#define LABELING_FACTORY_HPP

#include "labeling_configuration.hpp"
#include "measurement/measurement.hpp"

#include "cl_kernels/labeling_unroll_vector.hpp"

#include <functional>
#include <string>
#include <stdexcept>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, bool ColMajor>
class LabelingFactory {
public:
    template <typename T>
    using Vector = boost::compute::vector<T>;

    using LabelingFunction = std::function<
        boost::compute::event(
                boost::compute::command_queue queue,
                size_t num_features,
                size_t num_points,
                size_t num_clusters,
                Vector<char>& did_changes,
                Vector<PointT>& points,
                Vector<PointT>& centroids,
                Vector<LabelT>& labels,
                Measurement::DataPoint& datapoint,
                boost::compute::wait_list const& events
            )
        >;

    LabelingFunction create(
            boost::compute::context context,
            LabelingConfiguration config) {

        if (config.strategy == "unroll_vector") {
            LabelingUnrollVector<PointT, LabelT, ColMajor> strategy;
            strategy.prepare(context, config);
            return strategy;
        }
        else {
            throw std::invalid_argument(config.strategy);
        }
    }
};

}

#endif /* LABELING_FACTORY_HPP */
