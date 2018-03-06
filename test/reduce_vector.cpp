/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#include <cl_kernels/reduce_vector_parcol.hpp>

#include <cstdint>
#include <memory>
#include <vector>
#include <algorithm>

#include <gtest/gtest.h>

#include "data_generator.hpp"
#include "opencl_setup.hpp"

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>

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
        boost::compute::context context,
        boost::compute::command_queue queue,
        cle::Matrix<uint32_t, std::allocator<uint32_t>, uint32_t> const& data,
        std::vector<uint32_t>& reduced,
        Measurement::Measurement& measurement
        ) {

    reduced.resize(data.rows());
    boost::compute::vector<uint32_t> d_buffer(data.get_data(), queue);

    Clustering::ReduceVectorParcol<uint32_t> reducevector;
    reducevector.prepare(context);
    boost::compute::wait_list wait_list;

    auto& dp = measurement.add_datapoint();
    dp.set_name("ReduceVectorParcol");

    reducevector(
            queue,
            data.cols(),
            data.rows(),
            d_buffer,
            dp,
            wait_list);

    boost::compute::copy(
            d_buffer.begin(),
            d_buffer.begin() + reduced.size(),
            reduced.begin(),
            queue
            );
}

TEST(ReduceVectorParcol, Ascending) {

    auto const& data = dgen.ascending();
    std::vector<uint32_t> test_output;
    std::vector<uint32_t> verify_output;
    Measurement::Measurement measurement;

    reduce_vector_run(
            clenv->context,
            clenv->queue,
            data,
            test_output,
            measurement
            );
    reduce_vector_verify(data, verify_output);

    EXPECT_TRUE(std::equal(test_output.begin(), test_output.end(), verify_output.begin()));
}

TEST(ReduceVectorParcol, Large) {

    auto const& data = dgen.large();
    std::vector<uint32_t> test_output;
    std::vector<uint32_t> verify_output;
    Measurement::Measurement measurement;

    reduce_vector_run(
            clenv->context,
            clenv->queue,
            data,
            test_output,
            measurement
            );
    reduce_vector_verify(data, verify_output);

    EXPECT_TRUE(std::equal(test_output.begin(), test_output.end(), verify_output.begin()));
}

TEST(ReduceVectorParcol, Random) {

    auto const& data = dgen.random();
    std::vector<uint32_t> test_output;
    std::vector<uint32_t> verify_output;
    Measurement::Measurement measurement;

    reduce_vector_run(
            clenv->context,
            clenv->queue,
            data,
            test_output,
            measurement
            );
    reduce_vector_verify(data, verify_output);

    EXPECT_TRUE(std::equal(test_output.begin(), test_output.end(), verify_output.begin()));
}

TEST(ReduceVectorParcol, BigRandom) {

    auto const& data = dgen.def_size(4, 2048 * 32);
    std::vector<uint32_t> test_output;
    std::vector<uint32_t> verify_output;
    Measurement::Measurement measurement;

    reduce_vector_run(
            clenv->context,
            clenv->queue,
            data,
            test_output,
            measurement
            );
    reduce_vector_verify(data, verify_output);

    EXPECT_TRUE(std::equal(
                test_output.begin(),
                test_output.end(),
                verify_output.begin()
                ));
}

TEST(ReduceVectorParcol, ManyRows) {

    auto const& data = dgen.def_size(2048, 2048 * 32);
    std::vector<uint32_t> test_output;
    std::vector<uint32_t> verify_output;
    Measurement::Measurement measurement;

    reduce_vector_run(
            clenv->context,
            clenv->queue,
            data,
            test_output,
            measurement
            );
    reduce_vector_verify(data, verify_output);

    EXPECT_TRUE(std::equal(
                test_output.begin(),
                test_output.end(),
                verify_output.begin()
                ));
}

TEST(ReduceVectorParcol, TeslaSize) {

    auto const& data = dgen.def_size(256, 90 * 32 * 16);
    std::vector<uint32_t> test_output;
    std::vector<uint32_t> verify_output;
    Measurement::Measurement measurement;

    reduce_vector_run(
            clenv->context,
            clenv->queue,
            data,
            test_output,
            measurement
            );
    reduce_vector_verify(data, verify_output);

    EXPECT_TRUE(std::equal(
                test_output.begin(),
                test_output.end(),
                verify_output.begin()
                ));
}

TEST(ReduceVectorParcol, FurySize) {

    auto const& data = dgen.def_size(4, 56 * 64 * 16);
    std::vector<uint32_t> test_output;
    std::vector<uint32_t> verify_output;
    Measurement::Measurement measurement;

    reduce_vector_run(
            clenv->context,
            clenv->queue,
            data,
            test_output,
            measurement
            );
    reduce_vector_verify(data, verify_output);

    EXPECT_TRUE(std::equal(
                test_output.begin(),
                test_output.end(),
                verify_output.begin()
                ));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  clenv = new CLEnvironment;
  ::testing::AddGlobalTestEnvironment(clenv);
  return RUN_ALL_TESTS();
}
