/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef MATRIX_BINARY_OP_HPP
#define MATRIX_BINARY_OP_HPP

#include "kernel_path.hpp"

#include "../measurement/measurement.hpp"

#include <cassert>
#include <string>
#include <type_traits>
#include <stdexcept>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>

namespace Clustering {

template <typename T1, typename T2>
class MatrixBinaryOp {
public:
    enum BinaryOp { Add, Subtract, Multiply, Divide };

    using Event = boost::compute::event;
    using Context = boost::compute::context;
    using Kernel = boost::compute::kernel;
    using Program = boost::compute::program;
    template <typename ComputeVecT>
    using Vector = boost::compute::vector<ComputeVecT>;

    void prepare(Context context, BinaryOp op) {
        static_assert(boost::compute::is_fundamental<T1>(),
                "T1 must be a boost compute fundamental type");
        static_assert(boost::compute::is_fundamental<T2>(),
                "T2 must be a boost compute fundamental type");

        std::string defines;
        defines += " -DCL_INT=uint";
        defines += " -DCL_TYPE_1=";
        defines += boost::compute::type_name<T1>();
        defines += " -DCL_TYPE_2=";
        defines += boost::compute::type_name<T2>();
        defines += " -DBINARY_OP=";
        defines += op_to_str(op);

        Program program = Program::create_with_source_file(
                PROGRAM_FILE,
                context);

        program.build(defines);

        this->scalar_kernel = program.create_kernel(SCALAR_KERNEL_NAME);
        this->row_kernel = program.create_kernel(ROW_KERNEL_NAME);
        this->col_kernel = program.create_kernel(COL_KERNEL_NAME);
    }

    /*
     * Apply scalar to all elements of matrix
     */
    Event scalar(
            boost::compute::command_queue /* queue */,
            size_t /* num_cols */,
            size_t /* num_rows */,
            boost::compute::buffer_iterator<T1> /* matrix_begin */,
            boost::compute::buffer_iterator<T1> /* matrix_end */,
            T2 /* scalar */,
            Measurement::DataPoint& /* datapoint */,
            boost::compute::wait_list const& /* events */
            )
    {
        assert(false /* Not implemented! */);
        // assert(matrix_end - matrix_begin == num_cols * num_rows);
    }

    /*
     * Apply elements of vector to rows of matrix
     */
    Event row(
            boost::compute::command_queue queue,
            size_t num_cols,
            size_t num_rows,
            boost::compute::buffer_iterator<T1> matrix_begin,
            boost::compute::buffer_iterator<T1> matrix_end,
            boost::compute::buffer_iterator<T2> vector_begin,
            boost::compute::buffer_iterator<T2> vector_end,
            Measurement::DataPoint& datapoint,
            boost::compute::wait_list const& events
            )
    {
        assert(matrix_end - matrix_begin == (long) (num_cols * num_rows));
        assert(vector_end - vector_begin == (long) num_rows);
        assert(matrix_begin.get_index() == 0u);
        assert(vector_begin.get_index() == 0u);

        datapoint.set_name("MatrixBinaryOpRow");

        this->row_kernel.set_args(
                matrix_begin.get_buffer(),
                vector_begin.get_buffer(),
                (cl_uint)num_cols,
                (cl_uint)num_rows);

        size_t work_offset[3] = {0, 0, 0};
        size_t global_size[3] = {num_rows, num_cols, 1};

        Event event;
        event = queue.enqueue_nd_range_kernel(
                this->row_kernel,
                2,
                work_offset,
                global_size,
                0,
                events);

        datapoint.add_event() = event;

        return event;
    }

    /*
     * Apply elements of vector to columns of matrix
     */
    Event col(
            boost::compute::command_queue /* queue */,
            size_t /* num_cols */,
            size_t /* num_rows */,
            boost::compute::buffer_iterator<T1> /* matrix_begin */,
            boost::compute::buffer_iterator<T1> /* matrix_end */,
            boost::compute::buffer_iterator<T2> /* vector_begin */,
            boost::compute::buffer_iterator<T2> /* vector_end */,
            Measurement::DataPoint& /* datapoint */,
            boost::compute::wait_list const& /* wait_list */
            )
    {
        assert(false /* Not implemented! */);
        // assert(matrix_end - matrix_begin == num_cols * num_rows);
        // assert(vector_end - vector_begin == num_cols);
    }

private:
    static std::string op_to_str(BinaryOp op) {
        switch(op) {
            case BinaryOp::Add:
                return "+";
            case BinaryOp::Subtract:
                return "-";
            case BinaryOp::Multiply:
                return "*";
            case BinaryOp::Divide:
                return "/";
            default:
                throw std::invalid_argument("BinaryOp");
        }
    }

    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("matrix_binary_op.cl");
    static constexpr const char* SCALAR_KERNEL_NAME = "matrix_scalar";
    static constexpr const char* ROW_KERNEL_NAME = "matrix_row_vector";
    static constexpr const char* COL_KERNEL_NAME = "matrix_col_vector";

    Kernel scalar_kernel;
    Kernel col_kernel;
    Kernel row_kernel;
};
}

#endif /* MATRIX_BINARY_OP_HPP */
