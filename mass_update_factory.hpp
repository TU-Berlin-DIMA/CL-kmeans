#ifndef MASS_UPDATE_FACTORY_HPP
#define MASS_UPDATE_FACTORY_HPP

#include "temp.hpp"

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
                Vector<LabelT>& labels,
                Vector<MassT>& masses,
                MeasurementLogger& logger,
                boost::compute::wait_list const& events
                )
        >;

    MassUpdateFunction create(std::string flavor, MassUpdateConfiguration config) {
    }

};

}


#endif /* MASS_UPDATE_FACTORY_HPP */
