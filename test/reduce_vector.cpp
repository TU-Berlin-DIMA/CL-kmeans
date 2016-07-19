/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include <cl_kernels/reduce_vector_parcol_api.hpp>
#include <matrix.hpp>

#include <cstdint>
#include <memory>
#include <vector>
#include <algorithm>

#include <gtest/gtest.h>

#include "data_generator.hpp"
#include "opencl_setup.hpp"

void reduce_vector_verify(
        cle::Matrix<uint32_t, std::allocator<uint32_t>, uint32_t> const& data,
        std::vector<uint32_t>& reduced
        ) {

    reduced.resize(data.rows());
    std::fill(reduced.begin(), reduced.end(), 0);

    for (uint32_t col = 0; col < data.cols(); ++col) {
        for (uint32_t row = 0; row < data.rows(); ++row) {
            reduced[row] += data(row, col);
        }
    }
}

void reduce_vector_run(
        cl::Context context,
        cl::CommandQueue queue,
        cle::Matrix<uint32_t, std::allocator<uint32_t>, uint32_t> const& data,
        std::vector<uint32_t>& reduced
        ) {

    reduced.resize(data.rows());
    cle::TypedBuffer<cl_uint> d_buffer(context, CL_MEM_READ_WRITE, data.size());

    cle::ReduceVectorParcolAPI<cl_uint, cl_uint> reducevector;
    reducevector.initialize(context);
    cl::Event event;

    queue.enqueueWriteBuffer(
        d_buffer,
        CL_FALSE,
        0,
        d_buffer.bytes(),
        data.data(),
        NULL,
        NULL);

    reducevector(
        cl::EnqueueArgs(
            queue,
            cl::NDRange(0),
            cl::NDRange(0)
            ),
        data.cols(),
        data.rows(),
        d_buffer,
        event);

    queue.enqueueReadBuffer(
        d_buffer,
        CL_TRUE,
        0,
        reduced.size() * sizeof(uint32_t),
        reduced.data(),
        NULL,
        NULL);
}

TEST(ReduceVectorParcol, Ascending) {

    auto const& data = dgen.ascending();
    std::vector<uint32_t> test_output;
    std::vector<uint32_t> verify_output;

    reduce_vector_run(clenv->context, clenv->queue, data, test_output);
    reduce_vector_verify(data, verify_output);

    EXPECT_TRUE(std::equal(test_output.begin(), test_output.end(), verify_output.begin()));
}

TEST(ReduceVectorParcol, Large) {

    auto const& data = dgen.large();
    std::vector<uint32_t> test_output;
    std::vector<uint32_t> verify_output;

    reduce_vector_run(clenv->context, clenv->queue, data, test_output);
    reduce_vector_verify(data, verify_output);

    EXPECT_TRUE(std::equal(test_output.begin(), test_output.end(), verify_output.begin()));
}

TEST(ReduceVectorParcol, Random) {

    auto const& data = dgen.random();
    std::vector<uint32_t> test_output;
    std::vector<uint32_t> verify_output;

    reduce_vector_run(clenv->context, clenv->queue, data, test_output);
    reduce_vector_verify(data, verify_output);

    EXPECT_TRUE(std::equal(test_output.begin(), test_output.end(), verify_output.begin()));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  clenv = new CLEnvironment;
  ::testing::AddGlobalTestEnvironment(clenv);
  return RUN_ALL_TESTS();
}
