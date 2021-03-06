/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef MASS_UPDATE_PART_GLOBAL_HPP
#define MASS_UPDATE_PART_GLOBAL_HPP

#include "kernel_path.hpp"

#include "../mass_update_configuration.hpp"
#include "../measurement/measurement.hpp"

#include "reduce_vector_parcol.hpp"
#include "matrix_binary_op.hpp"

#include <cassert>
#include <string>
#include <type_traits>
#include <utility> // std::move

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/allocator/pinned_allocator.hpp>
#include <boost/compute/algorithm/copy.hpp>

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
    template <typename T>
    using PinnedAllocator = boost::compute::pinned_allocator<T>;
    template <typename T>
    using PinnedVector = boost::compute::vector<T, PinnedAllocator<T>>;

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

        Program gs_program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);
        gs_program.build(defines);
        this->global_stride_kernel = gs_program.create_kernel(KERNEL_NAME);

        defines += " -DLOCAL_STRIDE";
        Program ls_program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);
        ls_program.build(defines);
        this->local_stride_kernel = ls_program.create_kernel(KERNEL_NAME);

        reduce.prepare(context);
        matrix_add.prepare(context, decltype(matrix_add)::Add);
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

        datapoint.set_name("MassUpdatePartGlobal");

        size_t num_work_groups =
            this->config.global_size[0] / this->config.local_size[0];

        size_t const buffer_size = num_clusters * num_work_groups;

        if (this->tmp_masses.size() < buffer_size) {
            this->tmp_masses = std::move(
                    Vector<MassT>(
                        buffer_size,
                        queue.get_context()
                        ));
        }

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
                this->tmp_masses,
                (uint32_t) num_points,
                (uint32_t) num_clusters);

        size_t work_offset[3] = {0, 0, 0};

        Event event;
        event = queue.enqueue_1d_range_kernel(
                kernel,
                work_offset[0],
                this->config.global_size[0],
                this->config.local_size[0],
                wait_list);
        datapoint.add_event() = event;

        boost::compute::wait_list wait_list_i;
        wait_list_i.insert(event);

        event = reduce(
                queue,
                num_work_groups,
                num_clusters,
                this->tmp_masses.begin(),
                this->tmp_masses.begin() + buffer_size,
                datapoint.create_child(),
                wait_list_i
                );

        wait_list_i.insert(event);
        event = matrix_add.matrix(
                queue,
                1,
                num_clusters,
                masses_begin,
                masses_end,
                this->tmp_masses.begin(),
                this->tmp_masses.begin() + num_clusters,
                datapoint.create_child(),
                wait_list_i
                );

        return event;
    }

private:
    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("histogram_part_global.cl");
    static constexpr const char* KERNEL_NAME = "histogram_part_global";

    Kernel global_stride_kernel;
    Kernel local_stride_kernel;
    Vector<MassT> tmp_masses;
    MassUpdateConfiguration config;
    ReduceVectorParcol<MassT> reduce;
    MatrixBinaryOp<MassT, MassT> matrix_add;
};
}

#endif /* MASS_UPDATE_PART_GLOBAL_HPP */
