/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifndef REDUCE_VECTOR_PARCOL_HPP
#define REDUCE_VECTOR_PARCOL_HPP

#include "kernel_path.hpp"

#include "../measurement/measurement.hpp"

#include <cassert>
#include <string>
#include <type_traits>
#include <cstdint>

#include <boost/compute/core.hpp>
#include <boost/compute/types.hpp>
#include <boost/compute/type_traits.hpp>
#include <boost/compute/container/vector.hpp>

namespace Clustering {

template <typename T>
class ReduceVectorParcol {
public:
    using Event = boost::compute::event;
    using Context = boost::compute::context;
    using Kernel = boost::compute::kernel;
    using Program = boost::compute::program;
    template <typename Q>
    using Vector = boost::compute::vector<Q>;

    void prepare(Context context)
    {
        static_assert(
                boost::compute::is_fundamental<T>::value == true,
                "T must be boost compute fundamental type");

        std::string defines;
        defines += " -DCL_TYPE=";
        defines += boost::compute::type_name<T>();
        defines += " -DCL_INT=uint";
        defines += " -DWORKGROUP_SIZE=";
        defines += std::to_string(WORKGROUP_SIZE);

        Program program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);

        program.build(defines);

        this->kernel_compact = program
            .create_kernel(COMPACT_KERNEL_NAME);
        this->kernel_inner = program
            .create_kernel(INNER_KERNEL_NAME);
    }

    Event operator() (
            boost::compute::command_queue queue,
            size_t num_cols,
            size_t result_rows,
            Vector<T>& data,
            Measurement::DataPoint& datapoint,
            boost::compute::wait_list const& events
            ) {

        assert(data.size() >= num_cols * result_rows);

        datapoint.set_name("ReduceVectorParcol");

        Event event;
        boost::compute::wait_list wait_list;
        size_t work_offset = 0;
        uint32_t round = 0;
        size_t global_size = data.size() / 2;
        size_t data_size = data.size();

        while (
                data_size > result_rows
                && data_size > 2 * WORKGROUP_SIZE
              )
        {
            assert(global_size * 2 == data_size);
            assert(global_size % result_rows == 0);

            this->kernel_compact.set_args(
                    data,
                    (cl_uint) data_size);

            event = queue.enqueue_1d_range_kernel(
                    this->kernel_compact,
                    work_offset,
                    global_size,
                    0,
                    events);
            datapoint.add_event() = event;

            wait_list.clear();
            wait_list.insert(event);

            ++round;
            global_size /= 2;
            data_size = global_size * 2;
        }

        if (
                data_size != result_rows
                )
        {
            assert(WORKGROUP_SIZE % result_rows == 0);

            this->kernel_inner.set_args(
                    data,
                    (cl_uint) num_cols,
                    (cl_uint) result_rows);

            event = queue.enqueue_1d_range_kernel(
                    this->kernel_inner,
                    work_offset,
                    (cl_int) data_size / 2,
                    (cl_int) data_size / 2,
                    wait_list);
            datapoint.add_event() = event;
        }

        return event;
    }

private:
    static constexpr size_t WORKGROUP_SIZE = 256;

    static constexpr const char *PROGRAM_FILE =
        CL_KERNEL_FILE_PATH("reduce_vector_parcol.cl");
    static constexpr const char *COMPACT_KERNEL_NAME = "reduce_vector_parcol_compact";
    static constexpr const char *INNER_KERNEL_NAME = "reduce_vector_parcol_inner";

    Kernel kernel_compact;
    Kernel kernel_inner;
};
}

#endif /* REDUCE_VECTOR_PARCOL_HPP */
