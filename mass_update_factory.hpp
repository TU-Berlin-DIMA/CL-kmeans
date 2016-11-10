#ifndef MASS_UPDATE_FACTORY_HPP
#define MASS_UPDATE_FACTORY_HPP

#include "temp.hpp"

#include "mass_update_configuration.hpp"

#include "cl_kernels/mass_update_global_atomic.hpp"

#include <functional>
#include <string>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>

namespace Clustering {
template <typename LabelT, typename MassT>
class MassUpdateFactory {
public:
    template <typename T>
    using Vector = boost::compute::vector<T>;

    using MassUpdateFunction = std::function<
        boost::compute::event(
                boost::compute::command_queue queue,
                size_t num_points,
                size_t num_clusters,
                Vector<LabelT>& labels,
                Vector<MassT>& masses,
                MeasurementLogger& logger,
                boost::compute::wait_list const& events
                )
        >;

    MassUpdateFunction create(std::string flavor, boost::compute::context context, MassUpdateConfiguration config) {

        if (flavor.compare("global_atomic")) {
            MassUpdateGlobalAtomic<LabelT, MassT> flavor;
            flavor.prepare(context, config);
            return flavor;
        }
    }

};

}


#endif /* MASS_UPDATE_FACTORY_HPP */
