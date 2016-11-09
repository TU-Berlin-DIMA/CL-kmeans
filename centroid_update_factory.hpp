#ifndef CENTROID_UPDATE_FACTORY_HPP
#define CENTROID_UPDATE_FACTORY_HPP

#include "temp.hpp"

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
                Vector<const PointT>& points,
                Vector<PointT>& centroids,
                Vector<LabelT>& labels,
                Vector<MassT>& masses,
                MeasurementLogger& logger,
                boost::compute::wait_list const& events
                )
        >;

    CentroidUpdateFunction create(std::string flavor, CentroidUpdateConfiguration config) {

    }
};

}

#endif /* CENTROID_UPDATE_FACTORY_HPP */
