/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef MASS_UPDATE_FACTORY_HPP
#define MASS_UPDATE_FACTORY_HPP

#include "mass_update_configuration.hpp"

#include "measurement/measurement.hpp"

#include "cl_kernels/mass_update_global_atomic.hpp"
#include "cl_kernels/mass_update_part_global.hpp"

#include <functional>
#include <string>
#include <stdexcept>

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
                Measurement::DataPoint& datapoint,
                boost::compute::wait_list const& events
                )
        >;

    MassUpdateFunction create(
            boost::compute::context context,
            MassUpdateConfiguration config) {

        if (config.strategy == "global_atomic") {
            MassUpdateGlobalAtomic<LabelT, MassT> strategy;
            strategy.prepare(context, config);
            return strategy;
        }
        else if (config.strategy == "part_global") {
            MassUpdatePartGlobal<LabelT, MassT> strategy;
            strategy.prepare(context, config);
            return strategy;
        }
        else {
            throw std::invalid_argument(config.strategy);
        }
    }

};

}


#endif /* MASS_UPDATE_FACTORY_HPP */
