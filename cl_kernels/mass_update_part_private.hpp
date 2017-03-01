/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2017, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef MASS_UPDATE_PART_PRIVATE_HPP
#define MASS_UPDATE_PART_PRIVATE_HPP

#include "kernel_path.hpp"

#include "reduce_vector_parcol.hpp"

#include "../mass_update_configuration.hpp"
#include "../measurement/measurement.hpp"

#include <cassert>
#include <string>
#include <type_traits>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/memory/local_buffer.hpp>

namespace Clustering {

template <typename LabelT, typename MassT>
class MassUpdatePartPrivate {
public:
    using Event = boost::compute::event;
    using Context = boost::compute::context;
    using Kernel = boost::compute::kernel;
    using Program = boost::compute::program;
    template <typename T>
    using Vector = boost::compute::vector<T>;

    void prepare(
            Context context,
            MassUpdateConfiguration config)
    {
        this->config = config;

        std::string defines;
        defines += " -DCL_INT=uint";
        defines += " -DCL_TYPE_IN=";
        defines += boost::compute::type_name<LabelT>();
        defines += " -DCL_TYPE_OUT=";
        defines += boost::compute::type_name<MassT>();

        Program program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);
        program.build(defines);
        this->kernel = program.create_kernel(KERNEL_NAME);

        reduce.prepare(context);
    }

    Event operator() (
            boost::compute::command_queue queue,
            size_t num_points,
            size_t num_clusters,
            Vector<LabelT>& labels,
            Vector<MassT>& masses,
            Measurement::DataPoint& datapoint,
            boost::compute::wait_list const& events
            )
    {
        assert(labels.size() == num_points);
        assert(masses.size() >= num_clusters);

        datapoint.set_name("MassUpdatePartPrivate");

        size_t const buffer_size =
            num_clusters * this->config.global_size[0];

        if (masses.size() < buffer_size) {
            masses.resize(buffer_size);
        }

        boost::compute::local_buffer<MassT> local_masses(
                num_clusters * this->config.local_size[0]
                );

        this->kernel.set_args(
                labels,
                masses,
                local_masses,
                (uint32_t) num_points,
                (uint32_t) num_clusters
                );

        Event event;
        event = queue.enqueue_1d_range_kernel(
                this->kernel,
                0,
                this->config.global_size[0],
                this->config.local_size[0],
                events);
        datapoint.add_event() = event;

        boost::compute::wait_list wait_list;
        wait_list.insert(event);

        event = reduce(
                queue,
                this->config.global_size[0],
                num_clusters,
                masses,
                datapoint.create_child(),
                wait_list);

        return event;
    }

private:
    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("histogram_part_private.cl");
    static constexpr const char* KERNEL_NAME = "histogram_part_private";

    Kernel kernel;
    MassUpdateConfiguration config;
    ReduceVectorParcol<MassT> reduce;
};
}

#endif /* MASS_UPDATE_PART_PRIVATE_HPP */
