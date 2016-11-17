#ifndef CENTROID_UPDATE_FACTORY_HPP
#define CENTROID_UPDATE_FACTORY_HPP

#include "temp.hpp"

#include "centroid_update_configuration.hpp"

#include "cl_kernels/centroid_update_feature_sum.hpp"

#include <functional>
#include <string>

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
                MeasurementLogger& logger,
                boost::compute::wait_list const& events
                )
        >;

    CentroidUpdateFunction create(std::string flavor, boost::compute::context context, CentroidUpdateConfiguration config) {

        if (flavor.compare("feature_sum")) {
            CentroidUpdateFeatureSum<PointT, LabelT, MassT, ColMajor> flavor;
            flavor.prepare(context, config);
            return flavor;
        }

    }
};

}

#endif /* CENTROID_UPDATE_FACTORY_HPP */
