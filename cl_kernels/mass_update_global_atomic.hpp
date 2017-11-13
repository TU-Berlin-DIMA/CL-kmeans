/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2017, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef MASS_UPDATE_GLOBAL_ATOMIC_HPP
#define MASS_UPDATE_GLOBAL_ATOMIC_HPP

#include "kernel_path.hpp"

#include "../mass_update_configuration.hpp"
#include "../measurement/measurement.hpp"

#include <cassert>
#include <string>
#include <type_traits>
#include <stdexcept>
#include <iostream>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/algorithm/fill.hpp>
#include <boost/compute/allocator/pinned_allocator.hpp>

namespace Clustering {

template <typename LabelT, typename MassT>
class MassUpdateGlobalAtomic {
public:
    using Event = boost::compute::event;
    using Future = boost::compute::future<void>;
    using Context = boost::compute::context;
    using Kernel = boost::compute::kernel;
    using Program = boost::compute::program;
    template <typename T>
    using Vector = boost::compute::vector<T>;
    template <typename T>
    using PinnedAllocator = boost::compute::pinned_allocator<T>;
    template <typename T>
    using PinnedVector = boost::compute::vector<T, PinnedAllocator<T>>;

    void prepare(
            Context context,
            MassUpdateConfiguration config) {
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

        Program gs_program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);
        try {
            gs_program.build(defines);
            this->global_stride_kernel = gs_program.create_kernel(KERNEL_NAME);
        }
        catch (std::exception e) {
          std::cerr << gs_program.build_log() << std::endl;
          throw e;
        }

        defines += " -DLOCAL_STRIDE";
        Program ls_program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);
        try {
            ls_program.build(defines);
            this->local_stride_kernel = ls_program.create_kernel(KERNEL_NAME);
        }
        catch (std::exception e) {
          std::cerr << ls_program.build_log() << std::endl;
          throw e;
        }
    }

    Event operator() (
            boost::compute::command_queue queue,
            size_t num_points,
            size_t num_clusters,
            PinnedVector<LabelT>& labels,
            Vector<MassT>& masses,
            Measurement::DataPoint& datapoint,
            boost::compute::wait_list const& wait_list
            )
    {
        return (*this)(
                queue,
                num_points,
                num_clusters,
                labels.begin(),
                labels.end(),
                masses.begin(),
                masses.end(),
                datapoint,
                wait_list
                );
    }

    Event operator() (
            boost::compute::command_queue queue,
            size_t num_points,
            size_t num_clusters,
            boost::compute::buffer_iterator<LabelT> labels_begin,
            boost::compute::buffer_iterator<LabelT> labels_end,
            boost::compute::buffer_iterator<MassT> masses_begin,
            boost::compute::buffer_iterator<MassT> masses_end,
            Measurement::DataPoint& datapoint,
            boost::compute::wait_list const& wait_list
            )
    {
        assert(labels_end - labels_begin == (long) num_points);
        assert(masses_end - masses_begin == (long) num_clusters);
        assert(labels_begin.get_index() == 0u);
        assert(masses_begin.get_index() == 0u);

        datapoint.set_name("MassUpdateGlobalAtomic");

        Future future = boost::compute::fill_async(
                masses_begin,
                masses_end,
                0,
                queue
                );
        datapoint.add_event() = future.get_event();

        boost::compute::device device = queue.get_device();
        Kernel& kernel = (
                device.type() == device.cpu ||
                device.type() == device.accelerator
                )
            ? this->local_stride_kernel
            : this->global_stride_kernel
            ;

        kernel.set_args(
                labels_begin.get_buffer(),
                masses_begin.get_buffer(),
                (LabelT) num_points,
                (LabelT) num_clusters);

        size_t work_offset[3] = {0, 0, 0};

        Event event;
        event = queue.enqueue_nd_range_kernel(
                kernel,
                1,
                work_offset,
                this->config.global_size,
                this->config.local_size,
                wait_list);
        datapoint.add_event() = event;
        return event;
    }

private:
    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("histogram_global.cl");
    static constexpr const char* KERNEL_NAME = "histogram_global";

    Kernel global_stride_kernel;
    Kernel local_stride_kernel;
    MassUpdateConfiguration config;
};


}

#endif /* MASS_UPDATE_GLOBAL_ATOMIC_HPP */
