/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
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
#include <boost/compute/iterator/buffer_iterator.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, bool ColMajor>
class LabelingFactory {
public:
    template <typename T>
    using BufferIterator = boost::compute::buffer_iterator<T>;

    using LabelingFunction = std::function<
        boost::compute::event(
                boost::compute::command_queue queue,
                size_t num_features,
                size_t num_points,
                size_t num_clusters,
                BufferIterator<PointT> points_begin,
                BufferIterator<PointT> points_end,
                BufferIterator<PointT> centroids_begin,
                BufferIterator<PointT> centroids_end,
                BufferIterator<LabelT> labels_begin,
                BufferIterator<LabelT> labels_end,
                Measurement::DataPoint& datapoint,
                boost::compute::wait_list const& events
            )
        >;

    LabelingFunction create(
            boost::compute::context context,
            LabelingConfiguration config,
            Measurement::Measurement& measurement) {

        measurement.set_parameter(
                "LabelingGlobalSize",
                std::to_string(config.global_size[0])
                );
        measurement.set_parameter(
                "LabelingLocalSize",
                std::to_string(config.local_size[0])
                );

        if (config.strategy == "unroll_vector") {
            measurement.set_parameter(
                    "LabelingVectorLength",
                    std::to_string(config.vector_length)
                    );
            measurement.set_parameter(
                    "LabelingUnrollClustersLength",
                    std::to_string(config.unroll_clusters_length)
                    );
            measurement.set_parameter(
                    "LabelingUnrollFeaturesLength",
                    std::to_string(config.unroll_features_length)
                    );

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
