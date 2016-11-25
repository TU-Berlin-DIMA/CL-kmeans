#ifndef CENTROID_UPDATE_FACTORY_HPP
#define CENTROID_UPDATE_FACTORY_HPP

#include "centroid_update_configuration.hpp"

#include "measurement/measurement.hpp"

#include "cl_kernels/centroid_update_feature_sum.hpp"

#include <functional>
#include <string>
#include <stdexcept>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, typename MassT, bool ColMajor>
class CentroidUpdateFactory {
public:

    template <typename T>
    using Vector = boost::compute::vector<T>;

    using CentroidUpdateFunction = std::function<
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

    CentroidUpdateFunction create(boost::compute::context context, CentroidUpdateConfiguration config) {

        if (config.strategy == "feature_sum") {
            CentroidUpdateFeatureSum<PointT, LabelT, MassT, ColMajor> strategy;
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
