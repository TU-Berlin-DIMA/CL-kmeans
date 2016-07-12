#include <cl_kernels/histogram_part_local_api.hpp>
#include <cl_kernels/histogram_part_global_api.hpp>
#include <cl_kernels/reduce_vector_parcol_api.hpp>
#include <matrix.hpp>
#include <measurement/measurement.hpp>

#include <cstdint>
#include <memory>
#include <vector>
#include <algorithm>
#include <random>

#include <gtest/gtest.h>

#include "data_generator.hpp"
#include "opencl_setup.hpp"

void histogram_verify(
        std::vector<uint32_t> const& data,
        std::vector<uint32_t>& histogram
        ) {

    std::fill(histogram.begin(), histogram.end(), 0);

    for_each(data.begin(), data.end(), [&](uint32_t x){ ++histogram[x]; });
}

void histogram_part_local_run(
        cl::Context context,
        cl::CommandQueue queue,
        uint32_t work_group_size,
        std::vector<uint32_t> const& data,
        std::vector<uint32_t>& histogram,
        cl::Event& event
        ) {

    uint32_t num_work_groups = data.size() / work_group_size;

    cle::TypedBuffer<uint32_t> d_data(context, CL_MEM_READ_WRITE, data.size());
    cle::TypedBuffer<uint32_t> d_histogram(context, CL_MEM_READ_WRITE, histogram.size() * num_work_groups);

    cle::HistogramPartLocalAPI<cl_uint> kernel;
    kernel.initialize(context);

    cle::ReduceVectorParcolAPI<cl_uint, cl_uint> reduce;
    reduce.initialize(context);
    cl::Event reduce_event;

    queue.enqueueWriteBuffer(
            d_data,
            CL_FALSE,
            0,
            d_data.bytes(),
            data.data(),
            NULL,
            NULL);

    kernel(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(data.size()),
                cl::NDRange(work_group_size)),
            data.size(),
            histogram.size(),
            d_data,
            d_histogram,
            event
            );

    reduce(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(histogram.size() * num_work_groups),
                cl::NullRange),
            num_work_groups,
            histogram.size(),
            d_histogram,
            reduce_event
            );

    queue.enqueueReadBuffer(
            d_histogram,
            CL_TRUE,
            0,
            histogram.size() * sizeof(uint32_t),
            histogram.data(),
            NULL,
            NULL);
}

void histogram_part_local_performance(
        cl::Context context,
        cl::CommandQueue queue,
        uint32_t work_group_size,
        std::vector<uint32_t> const& data,
        uint32_t NUM_BUCKETS,
        Measurement::Measurement& measurement,
        int num_runs
        ) {

    uint32_t num_work_groups = data.size() / work_group_size;

    cle::TypedBuffer<uint32_t> d_data(context, CL_MEM_READ_WRITE, data.size());
    cle::TypedBuffer<uint32_t> d_histogram(context, CL_MEM_READ_WRITE, NUM_BUCKETS * num_work_groups);

    cle::HistogramPartLocalAPI<cl_uint> kernel;
    kernel.initialize(context);

    cle::ReduceVectorParcolAPI<cl_uint, cl_uint> reduce;
    reduce.initialize(context);
    cl::Event reduce_event;

    queue.enqueueWriteBuffer(
            d_data,
            CL_FALSE,
            0,
            d_data.bytes(),
            data.data(),
            NULL,
            NULL);

    measurement.start();
    for (int r = 0; r < num_runs; ++r) {

        kernel(
                cl::EnqueueArgs(
                    queue,
                    cl::NDRange(data.size()),
                    cl::NDRange(work_group_size)),
                data.size(),
                NUM_BUCKETS,
                d_data,
                d_histogram,
                measurement.add_datapoint(Measurement::DataPointType::LloydMassSumMerge, r).add_opencl_event()
                );

    }
    measurement.end();
}

void histogram_part_global_run(
        cl::Context context,
        cl::CommandQueue queue,
        uint32_t work_group_size,
        std::vector<uint32_t> const& data,
        std::vector<uint32_t>& histogram,
        cl::Event& event
        ) {

    uint32_t num_work_groups = data.size() / work_group_size;

    cle::TypedBuffer<uint32_t> d_data(context, CL_MEM_READ_WRITE, data.size());
    cle::TypedBuffer<uint32_t> d_histogram(context, CL_MEM_READ_WRITE, histogram.size() * num_work_groups);

    cle::HistogramPartGlobalAPI<cl_uint> kernel;
    kernel.initialize(context);

    cle::ReduceVectorParcolAPI<cl_uint, cl_uint> reduce;
    reduce.initialize(context);
    cl::Event reduce_event;

    queue.enqueueWriteBuffer(
            d_data,
            CL_FALSE,
            0,
            d_data.bytes(),
            data.data(),
            NULL,
            NULL);

    kernel(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(data.size()),
                cl::NDRange(work_group_size)),
            data.size(),
            histogram.size(),
            d_data,
            d_histogram,
            event
            );

    reduce(
            cl::EnqueueArgs(
                queue,
                cl::NDRange(histogram.size() * num_work_groups),
                cl::NullRange),
            num_work_groups,
            histogram.size(),
            d_histogram,
            reduce_event
            );

    queue.enqueueReadBuffer(
            d_histogram,
            CL_TRUE,
            0,
            histogram.size() * sizeof(uint32_t),
            histogram.data(),
            NULL,
            NULL);
}

void histogram_part_global_performance(
        cl::Context context,
        cl::CommandQueue queue,
        uint32_t work_group_size,
        std::vector<uint32_t> const& data,
        uint32_t NUM_BUCKETS,
        Measurement::Measurement& measurement,
        int num_runs
        ) {

    uint32_t num_work_groups = data.size() / work_group_size;

    cle::TypedBuffer<uint32_t> d_data(context, CL_MEM_READ_WRITE, data.size());
    cle::TypedBuffer<uint32_t> d_histogram(context, CL_MEM_READ_WRITE, NUM_BUCKETS * num_work_groups);

    cle::HistogramPartGlobalAPI<cl_uint> kernel;
    kernel.initialize(context);

    cle::ReduceVectorParcolAPI<cl_uint, cl_uint> reduce;
    reduce.initialize(context);
    cl::Event reduce_event;

    queue.enqueueWriteBuffer(
            d_data,
            CL_FALSE,
            0,
            d_data.bytes(),
            data.data(),
            NULL,
            NULL);

    measurement.start();
    for (int r = 0; r < num_runs; ++r) {

        kernel(
                cl::EnqueueArgs(
                    queue,
                    cl::NDRange(data.size()),
                    cl::NDRange(work_group_size)),
                data.size(),
                NUM_BUCKETS,
                d_data,
                d_histogram,
                measurement.add_datapoint(Measurement::DataPointType::HistogramPartGlobal, r).add_opencl_event()
                );

    }
    measurement.end();
}

TEST(HistogramPartLocal, UniformDistribution) {

    const size_t NUM_DATA = 1024 * 1024;
    const size_t NUM_BUCKETS = 16;
    const uint32_t WORK_GROUP_SIZE = 32;

    std::vector<uint32_t> data(NUM_DATA);
    std::vector<uint32_t> test_output(NUM_BUCKETS), verify_output(NUM_BUCKETS);

    std::default_random_engine rgen;
    std::uniform_int_distribution<uint32_t> uniform(0, NUM_BUCKETS - 1);
    std::generate(
            data.begin(),
            data.end(),
            [&](){ return uniform(rgen); }
            );

    cl::Event event;
    histogram_part_local_run(clenv->context, clenv->queue, WORK_GROUP_SIZE, data, test_output, event);
    histogram_verify(data, verify_output);

    EXPECT_TRUE(std::equal(test_output.begin(), test_output.end(), verify_output.begin()));
}

TEST(HistogramPartGlobal, UniformDistribution) {

    const size_t NUM_DATA = 1024 * 1024;
    const size_t NUM_BUCKETS = 16;
    const uint32_t WORK_GROUP_SIZE = 32;

    std::vector<uint32_t> data(NUM_DATA);
    std::vector<uint32_t> test_output(NUM_BUCKETS), verify_output(NUM_BUCKETS);

    std::default_random_engine rgen;
    std::uniform_int_distribution<uint32_t> uniform(0, NUM_BUCKETS - 1);
    std::generate(
            data.begin(),
            data.end(),
            [&](){ return uniform(rgen); }
            );

    cl::Event event;
    histogram_part_global_run(clenv->context, clenv->queue, WORK_GROUP_SIZE, data, test_output, event);
    histogram_verify(data, verify_output);

    EXPECT_TRUE(std::equal(test_output.begin(), test_output.end(), verify_output.begin()));
}

TEST(HistogramPartLocal, UniformDistributionPerformance) {

    const size_t NUM_DATA = 128 * 1024 * 1024;
    const size_t NUM_BUCKETS = 16;
    const uint32_t WORK_GROUP_SIZE = 32;
    const int NUM_RUNS = 5;

    std::vector<uint32_t> data(NUM_DATA);

    std::default_random_engine rgen;
    std::uniform_int_distribution<uint32_t> uniform(0, NUM_BUCKETS - 1);
    std::generate(
            data.begin(),
            data.end(),
            [&](){ return uniform(rgen); }
            );

    Measurement::Measurement measurement;
    measurement_setup(measurement, clenv->device, NUM_RUNS);

    histogram_part_local_performance(clenv->context, clenv->queue, WORK_GROUP_SIZE, data, NUM_BUCKETS, measurement, NUM_RUNS);

    measurement.write_csv("histogram_part_local.csv");

    EXPECT_TRUE(true);
}

TEST(HistogramPartGlobal, UniformDistributionPerformance) {

    const size_t NUM_DATA = 128 * 1024 * 1024;
    const size_t NUM_BUCKETS = 16;
    const uint32_t WORK_GROUP_SIZE = 32;
    const int NUM_RUNS = 5;

    std::vector<uint32_t> data(NUM_DATA);

    std::default_random_engine rgen;
    std::uniform_int_distribution<uint32_t> uniform(0, NUM_BUCKETS - 1);
    std::generate(
            data.begin(),
            data.end(),
            [&](){ return uniform(rgen); }
            );

    Measurement::Measurement measurement;
    measurement_setup(measurement, clenv->device, NUM_RUNS);

    histogram_part_global_performance(clenv->context, clenv->queue, WORK_GROUP_SIZE, data, NUM_BUCKETS, measurement, NUM_RUNS);

    measurement.write_csv("histogram_part_global.csv");

    EXPECT_TRUE(true);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    clenv = new CLEnvironment;
    ::testing::AddGlobalTestEnvironment(clenv);
    return RUN_ALL_TESTS();
}
