/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2017-2018, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef MASS_UPDATE_PART_PRIVATE_HPP
#define MASS_UPDATE_PART_PRIVATE_HPP

#include "kernel_path.hpp"

#include "reduce_vector_parcol.hpp"
#include "matrix_binary_op.hpp"

#include "../mass_update_configuration.hpp"
#include "../measurement/measurement.hpp"

#include <cassert>
#include <string>
#include <type_traits>
#include <utility> // std::move

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/memory/local_buffer.hpp>
#include <boost/compute/allocator/pinned_allocator.hpp>
#include <boost/compute/algorithm/copy.hpp>

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
    template <typename T>
    using PinnedAllocator = boost::compute::pinned_allocator<T>;
    template <typename T>
    using PinnedVector = boost::compute::vector<T, PinnedAllocator<T>>;
    template <typename T>
    using LocalBuffer = boost::compute::local_buffer<T>;

    MassUpdatePartPrivate() :
        local_masses(1)
    {}

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
        defines += " -DVEC_LEN=";
        defines += std::to_string(this->config.vector_length);

        std::string l_stride_defines = " -DLOCAL_STRIDE";
        std::string g_mem_defines = " -DGLOBAL_MEM";

        Program g_stride_g_mem_program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);
        try {
            g_stride_g_mem_program.build(defines + g_mem_defines);
            g_stride_g_mem_kernel =
                g_stride_g_mem_program.create_kernel(KERNEL_NAME);
        }
        catch (std::exception e) {
            std::cerr << g_stride_g_mem_program.build_log() << std::endl;
            throw e;
        }

        Program g_stride_l_mem_program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);
        try {
            g_stride_l_mem_program.build(defines);
            g_stride_l_mem_kernel =
                g_stride_l_mem_program.create_kernel(KERNEL_NAME);
        }
        catch (std::exception e) {
            std::cerr << g_stride_l_mem_program.build_log() << std::endl;
            throw e;
        }

        Program l_stride_g_mem_program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);
        try {
            l_stride_g_mem_program.build(defines + l_stride_defines + g_mem_defines);
            l_stride_g_mem_kernel =
                l_stride_g_mem_program.create_kernel(KERNEL_NAME);
        }
        catch (std::exception e) {
            std::cerr << l_stride_g_mem_program.build_log() << std::endl;
            throw e;
        }

        reduce.prepare(context);
        matrix_add.prepare(context, matrix_add.Add);
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

        datapoint.set_name("MassUpdatePartPrivate");

        size_t const buffer_size =
            num_clusters * this->config.global_size[0];

        Vector<MassT> tmp_masses(buffer_size, queue.get_context());
        if (this->tmp_masses.size() < buffer_size) {
            this->tmp_masses = std::move(
                    Vector<MassT>(
                        buffer_size,
                        queue.get_context()
                        ));
        }

        size_t const local_masses_size =
            num_clusters
            * this->config.local_size[0]
            ;
        if (this->local_masses.size() != local_masses_size) {
            this->local_masses = std::move(
                    LocalBuffer<MassT>(
                        local_masses_size
                        ));
        }

        boost::compute::device device = queue.get_device();
        bool use_local_stride =
            device.type() == device.cpu ||
            device.type() == device.accelerator
            ;
        bool use_local_memory =
            device.type() == device.gpu &&
            device.local_memory_size() > local_masses_size * sizeof(MassT)
            ;
        auto& kernel = (use_local_stride)
            ? l_stride_g_mem_kernel
            : (use_local_memory)
            ? g_stride_l_mem_kernel
            : g_stride_g_mem_kernel
            ;

        if (use_local_memory) {
            kernel.set_args(
                    labels_begin.get_buffer(),
                    this->tmp_masses,
                    this->local_masses,
                    (uint32_t) num_points,
                    (uint32_t) num_clusters
                    );
        }
        else {
            kernel.set_args(
                    labels_begin.get_buffer(),
                    this->tmp_masses,
                    (uint32_t) num_points,
                    (uint32_t) num_clusters
                    );
             }

        Event event;
        event = queue.enqueue_1d_range_kernel(
                kernel,
                0,
                this->config.global_size[0],
                this->config.local_size[0],
                wait_list);
        datapoint.add_event() = event;

        boost::compute::wait_list wait_list_i;
        wait_list_i.insert(event);

        event = reduce(
                queue,
                this->config.global_size[0],
                num_clusters,
                this->tmp_masses.begin(),
                this->tmp_masses.begin() + buffer_size,
                datapoint.create_child(),
                wait_list_i);

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
    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("histogram_part_private.cl");
    static constexpr const char* KERNEL_NAME = "histogram_part_private";

    Kernel g_stride_g_mem_kernel;
    Kernel g_stride_l_mem_kernel;
    Kernel l_stride_g_mem_kernel;
    Vector<MassT> tmp_masses;
    LocalBuffer<MassT> local_masses;
    MassUpdateConfiguration config;
    ReduceVectorParcol<MassT> reduce;
    MatrixBinaryOp<MassT, MassT> matrix_add;
};
}

#endif /* MASS_UPDATE_PART_PRIVATE_HPP */
