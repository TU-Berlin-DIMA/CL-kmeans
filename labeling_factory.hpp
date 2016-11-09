#ifndef LABELING_FACTORY_HPP
#define LABELING_FACTORY_HPP

#include "temp.hpp"

#include "cl_kernels/lloyd_labeling_api.hpp"

#include <functional>
#include <string>

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
                Vector<int>& did_changes,
                Vector<const PointT>& points,
                Vector<PointT>& centroids,
                Vector<LabelT>& labels,
                MeasurementLogger& logger,
                boost::compute::wait_list const& events
            )
        >;

    LabelingFunction create(std::string flavor, LabelingConfiguration config) {

        if (flavor.compare("normal")) {
        }
    }
};

}

#endif /* LABELING_FACTORY_HPP */
