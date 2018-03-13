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
#include "cl_kernels/mass_update_part_local.hpp"
#include "cl_kernels/mass_update_part_private.hpp"

#include <functional>
#include <string>
#include <stdexcept>

#include <boost/compute/core.hpp>
#include <boost/compute/iterator/buffer_iterator.hpp>

namespace Clustering {
template <typename LabelT, typename MassT>
class MassUpdateFactory {
public:
    template <typename T>
    using BufferIterator = boost::compute::buffer_iterator<T>;

    using MassUpdateFunction = std::function<
        boost::compute::event(
                boost::compute::command_queue queue,
                size_t num_points,
                size_t num_clusters,
                BufferIterator<LabelT> labels_begin,
                BufferIterator<LabelT> labels_end,
                BufferIterator<MassT> masses_begin,
                BufferIterator<MassT> masses_end,
                Measurement::DataPoint& datapoint,
                boost::compute::wait_list const& events
                )
        >;

    MassUpdateFunction create(
            boost::compute::context context,
            MassUpdateConfiguration config,
            Measurement::Measurement& measurement) {

        measurement.set_parameter(
                "MassUpdateGlobalSize",
                std::to_string(config.global_size[0])
                );
        measurement.set_parameter(
                "MassUpdateLocalSize",
                std::to_string(config.local_size[0])
                );

        if (config.strategy == "part_private") {
            measurement.set_parameter(
                    "MassUpdateVectorLength",
                    std::to_string(config.vector_length)
                    );
        }

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
        else if (config.strategy == "part_local") {
            MassUpdatePartLocal<LabelT, MassT> strategy;
            strategy.prepare(context, config);
            return strategy;
        }
        else if (config.strategy == "part_private") {
            MassUpdatePartPrivate<LabelT, MassT> strategy;
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
