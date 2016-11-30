/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef MASS_UPDATE_PART_GLOBAL_HPP
#define MASS_UPDATE_PART_GLOBAL_HPP

#include "kernel_path.hpp"

#include "../mass_update_configuration.hpp"
#include "../measurement/measurement.hpp"

#include "reduce_vector_parcol.hpp"

#include <cassert>
#include <string>
#include <type_traits>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>

namespace Clustering {

template <typename LabelT, typename MassT>
class MassUpdatePartGlobal {
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
        if (std::is_same<uint32_t, LabelT>::value) {
            defines = "-DTYPE32";
        }
        else if (std::is_same<uint64_t, LabelT>::value) {
            defines = "-DTYPE64";
        }
        else {
            assert(false);
        }

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
        size_t const buffer_size =
            num_clusters *
            (this->config.global_size[0]
             / this->config.local_size[0]);

        assert(labels.size() == num_points);
        // assert(masses.size() >= buffer_size);

        datapoint.set_name("MassUpdatePartGlobal");

        size_t num_work_groups =
            this->config.global_size[0] / this->config.local_size[0];

        if (masses.size() < buffer_size) {
            masses.resize(buffer_size);
        }

        this->kernel.set_args(
                labels,
                masses,
                (LabelT) num_points,
                (LabelT) num_clusters);

        size_t work_offset[3] = {0, 0, 0};

        Event event;
        event = queue.enqueue_nd_range_kernel(
                this->kernel,
                1,
                work_offset,
                this->config.global_size,
                this->config.local_size,
                events);
        datapoint.add_event() = event;

        boost::compute::wait_list wait_list;
        wait_list.insert(event);

        event = reduce(
                queue,
                num_work_groups,
                num_clusters,
                masses,
                datapoint,
                wait_list);

        return event;
    }

private:
    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("histogram_part_global.cl");
    static constexpr const char* KERNEL_NAME = "histogram_part_global";

    Kernel kernel;
    MassUpdateConfiguration config;
    Clustering::ReduceVectorParcol<MassT> reduce;
};
}

#endif /* MASS_UPDATE_PART_GLOBAL_HPP */
