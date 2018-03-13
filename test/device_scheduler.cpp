/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2017-2018, Lutz, Clemens <lutzcle@cml.li>
 */

#include <simple_buffer_cache.hpp>
#include <single_device_scheduler.hpp>

#include <chrono>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>

#include <gtest/gtest.h>
#include <boost/compute/core.hpp>

#include <measurement/measurement.hpp>

constexpr size_t MAX_PRINT_FAILURES = 3;
constexpr std::chrono::milliseconds WAIT_DURATION(900);
constexpr size_t BUFFER_SIZE = 16ul << 20;  //  16 MB
constexpr size_t POOL_SIZE = 128ul << 20;   // 128 MB
constexpr size_t OBJECT_SIZE = 256ul << 20; // 256 MB
constexpr size_t GLOBAL_SIZE = 2048ul;
constexpr size_t LOCAL_SIZE  = 64;

namespace bc = boost::compute;

constexpr char zero_source[] =
R"ENDSTR(
__kernel void zero(__global int * const restrict buffer, uint size)
{
    for (uint i = get_global_id(0); i < size; i += get_global_size(0)) {
        buffer[i] = 0;
    }
}
)ENDSTR";

constexpr char increment_source[] =
R"ENDSTR(
__kernel void inc(__global int * const restrict buffer, uint size)
{
    for (uint i = get_global_id(0); i < size; i += get_global_size(0)) {
        buffer[i] = buffer[i] + 1;
    }
}
)ENDSTR";

constexpr char copy_source[] =
R"ENDSTR(
__kernel void copy(__global int * const restrict dst, __global int * const restrict src, uint size)
{
    for (uint i = get_global_id(0); i < size; i += get_global_size(0)) {
        dst[i] = src[i];
    }
}
)ENDSTR";

constexpr char reduce_source[] =
R"ENDSTR(
__kernel void reduce(__global int * const restrict dst, __global int * const restrict src, uint src_size)
{
    for (uint i = get_global_id(0); i < src_size / 2; i += get_global_size(0)) {
        dst[i] = src[i] + src[src_size / 2 + i];
    }
}
)ENDSTR";

class DeviceSchedulerEnvironment : public ::testing::Environment
{
public:
    void SetUp()
    {
        device = boost::compute::system::default_device();
        queue = boost::compute::system::default_queue();

        bc::program zero_program = bc::program::build_with_source(
                zero_source,
                queue.get_context()
                );

        zero_f = [zero_program](
                bc::command_queue queue,
                size_t cl_offset,
                size_t size,
                bc::buffer buffer,
                Measurement::DataPoint& dp
                )
        {
            dp.set_name("zero");
            bc::kernel kernel = zero_program.create_kernel("zero");
            kernel.set_args(buffer, (cl_uint) (size / sizeof(cl_int)));
            bc::event event;
            event = queue.enqueue_1d_range_kernel(
                    kernel,
                    cl_offset / sizeof(cl_int),
                    GLOBAL_SIZE,
                    LOCAL_SIZE
                    );
            dp.add_event() = event;
            return event;
        };

        bc::program inc_program = bc::program::build_with_source(
                    increment_source,
                    queue.get_context()
                    );

        increment_f = [inc_program](
                bc::command_queue queue,
                size_t cl_offset,
                size_t size,
                bc::buffer buffer,
                Measurement::DataPoint& dp
                )
        {
            dp.set_name("inc");
            bc::kernel kernel = inc_program.create_kernel("inc");
            kernel.set_args(buffer, (cl_uint) (size / sizeof(cl_int)));
            bc::event event;
            event = queue.enqueue_1d_range_kernel(
                    kernel,
                    cl_offset / sizeof(cl_int),
                    GLOBAL_SIZE,
                    LOCAL_SIZE
                    );
            dp.add_event() = event;
            return event;
        };

        bc::program copy_program = bc::program::build_with_source(
                copy_source,
                queue.get_context()
                );

        copy_f = [copy_program](
                bc::command_queue queue,
                size_t cl_offset,
                size_t dst_size,
                size_t /* src_size */,
                bc::buffer dst,
                bc::buffer src,
                Measurement::DataPoint& dp
                )
        {
            dp.set_name("copy");
            bc::kernel kernel = copy_program.create_kernel("copy");
            kernel.set_args(dst, src, (cl_uint) (dst_size / sizeof(cl_int)));
            bc::event event;
            event = queue.enqueue_1d_range_kernel(
                    kernel,
                    cl_offset / sizeof(cl_int),
                    GLOBAL_SIZE,
                    LOCAL_SIZE
                    );
            dp.add_event() = event;
            return event;
        };

        bc::program reduce_program = bc::program::build_with_source(
                reduce_source,
                queue.get_context()
                );

        reduce_f = [reduce_program](
                bc::command_queue queue,
                size_t cl_offset,
                size_t dst_size,
                size_t src_size,
                bc::buffer dst,
                bc::buffer src,
                Measurement::DataPoint& dp
                )
        {
            assert(dst_size * 2 >= src_size);

            dp.set_name("reduce");
            bc::kernel kernel = reduce_program.create_kernel("reduce");
            kernel.set_args(dst, src, (cl_uint) (src_size / sizeof(cl_int)));
            bc::event event;
            event = queue.enqueue_1d_range_kernel(
                    kernel,
                    cl_offset / sizeof(cl_int),
                    GLOBAL_SIZE,
                    LOCAL_SIZE
                    );
            dp.add_event() = event;
            return event;
        };
    }

    void TearDown()
    {}

    boost::compute::device device;
    boost::compute::command_queue queue;
    Clustering::DeviceScheduler::FunUnary zero_f;
    Clustering::DeviceScheduler::FunUnary increment_f;
    Clustering::DeviceScheduler::FunBinary copy_f;
    Clustering::DeviceScheduler::FunBinary reduce_f;
} *dsenv = nullptr;

class SingleDeviceScheduler : public ::testing::Test {
public:
    SingleDeviceScheduler() :
        buffer_size(BUFFER_SIZE),
        buffer_ints(BUFFER_SIZE / sizeof(int)),
        pool_size(POOL_SIZE),
        fst_object_size(OBJECT_SIZE),
        snd_object_size(OBJECT_SIZE),
        fst_object_id(0),
        snd_object_id(0),
        fst_data_object(OBJECT_SIZE / sizeof(int), 0),
        snd_data_object(OBJECT_SIZE / sizeof(int), 0)
    {
    }

    void SetUp()
    {
        scheduler = std::make_shared<Clustering::SingleDeviceScheduler>();
        buffer_cache = std::make_shared<Clustering::SimpleBufferCache>(BUFFER_SIZE);

        buffer_cache->add_device(
                dsenv->queue.get_context(),
                dsenv->device, pool_size
                );
        fst_object_id = buffer_cache->add_object(
                fst_data_object.data(),
                fst_data_object.size() * sizeof(int),
                Clustering::ObjectMode::Mutable
                );
        snd_object_id = buffer_cache->add_object(
                snd_data_object.data(),
                snd_data_object.size() * sizeof(int),
                Clustering::ObjectMode::Mutable
                );

        scheduler->add_buffer_cache(buffer_cache);
        scheduler->add_device(dsenv->queue.get_context(), dsenv->device);

        size_t i = 0;
        for (auto& obj : fst_data_object) {
            obj = i;
            ++i;
        }

        i = 0;
        for (auto& obj : snd_data_object) {
            obj = i;
            ++i;
        }
    }

    void TearDown()
    {
        scheduler.reset();
        buffer_cache.reset();
    }

    size_t const buffer_size, buffer_ints, pool_size, fst_object_size, snd_object_size;
    uint32_t fst_object_id, snd_object_id;
    std::vector<uint32_t> fst_data_object, snd_data_object;
    std::shared_ptr<Clustering::BufferCache> buffer_cache;
    std::shared_ptr<Clustering::DeviceScheduler> scheduler;
};

TEST_F(SingleDeviceScheduler, EnqueueUnaryKernel)
{
    int ret = 0;
    std::future<std::deque<bc::event>> fevents;
    Measurement::Measurement measurement;

    ret = scheduler->enqueue(dsenv->zero_f, fst_object_id, buffer_size, fevents, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    ASSERT_TRUE(fevents.valid());

    ret = scheduler->run();
    ASSERT_EQ(true, ret);

    auto status = fevents.wait_for(WAIT_DURATION);
    fevents.wait();
    ASSERT_EQ(std::future_status::ready, status);

    auto events = fevents.get();
    EXPECT_FALSE(events.empty());
}

TEST_F(SingleDeviceScheduler, RunUnaryAndRead)
{
    int ret = 0;
    std::future<std::deque<bc::event>> zero_fevents, inc_fevents;
    Measurement::Measurement measurement;
    bc::wait_list dummy_wait_list;

    ret = scheduler->enqueue(dsenv->zero_f, fst_object_id, buffer_size, zero_fevents, measurement.add_datapoint());
    ASSERT_EQ(true, ret);

    ret = scheduler->enqueue(dsenv->increment_f, fst_object_id, buffer_size, inc_fevents, measurement.add_datapoint());
    ASSERT_EQ(true, ret);

    ret = scheduler->run();
    ASSERT_EQ(true, ret);

    bc::event read_event;
    for (size_t offset = 0; offset < fst_data_object.size(); offset += buffer_ints) {
        size_t num_ints = (offset + buffer_ints > fst_data_object.size())
            ? fst_data_object.size() - offset
            : buffer_ints
            ;
        ret = buffer_cache->read(
                dsenv->queue,
                fst_object_id,
                &fst_data_object[offset],
                &fst_data_object[offset + num_ints],
                read_event,
                dummy_wait_list,
                measurement.add_datapoint()
                );
        ASSERT_EQ(true, ret);
    }
    dsenv->queue.finish();

    size_t failed_fields = 0;
    for (size_t i = 0; i < buffer_ints; ++i) {
        if (fst_data_object[i] != 1u) {
            ++failed_fields;
        }
        if (failed_fields <= MAX_PRINT_FAILURES) {
            EXPECT_EQ(1u, fst_data_object[i]) << "Object differs at index " << i;
        }
    }
    EXPECT_EQ(0ul, failed_fields);
}

TEST_F(SingleDeviceScheduler, RunBinaryAndRead)
{
    int ret = 0;
    std::future<std::deque<bc::event>> copy_fevents;
    Measurement::Measurement measurement;
    bc::wait_list dummy_wait_list;

    for (auto& obj : fst_data_object) {
        obj = 0x0EADBEEF;
    }

    ret = scheduler->enqueue(dsenv->copy_f, fst_object_id, snd_object_id, buffer_size, buffer_size, copy_fevents, measurement.add_datapoint());
    ASSERT_EQ(true, ret);

    ret = scheduler->run();
    ASSERT_EQ(true, ret);

    bc::event read_event;
    for (size_t offset = 0; offset < fst_data_object.size(); offset += buffer_ints) {
        size_t num_ints = (offset + buffer_ints > fst_data_object.size())
            ? fst_data_object.size() - offset
            : buffer_ints
            ;
        ret = buffer_cache->read(
                dsenv->queue,
                fst_object_id,
                &fst_data_object[offset],
                &fst_data_object[offset + num_ints],
                read_event,
                dummy_wait_list,
                measurement.add_datapoint()
                );
        ASSERT_EQ(true, ret);
    }
    dsenv->queue.finish();

    size_t failed_fields = 0;
    for (size_t i = 0; i < fst_data_object.size(); ++i) {
        if (fst_data_object[i] != i) {
            ++failed_fields;
        }
        if (failed_fields <= MAX_PRINT_FAILURES) {
            EXPECT_EQ(i, fst_data_object[i]) << "Object differs at index " << i;
        }
    }
    EXPECT_EQ(0ul, failed_fields);
}

TEST_F(SingleDeviceScheduler, RunBinaryWithSteps)
{
    int ret = 0;
    std::future<std::deque<bc::event>> copy_fevents;
    Measurement::Measurement measurement;
    bc::wait_list dummy_wait_list;

    decltype(snd_data_object) dst_object(snd_data_object.size() / 2);
    auto dst_object_id = buffer_cache->add_object(
            dst_object.data(),
            dst_object.size() * sizeof(decltype(dst_object)::value_type),
            Clustering::ObjectMode::Mutable
            );

    ret = scheduler->enqueue(dsenv->copy_f, dst_object_id, snd_object_id, buffer_size / 2, buffer_size, copy_fevents, measurement.add_datapoint());
    ASSERT_EQ(true, ret);

    ret = scheduler->run();
    ASSERT_EQ(true, ret);

    bc::event read_event;
    for (size_t offset = 0; offset < dst_object.size(); offset += buffer_ints / 2) {
        size_t num_ints = (offset + buffer_ints / 2 > dst_object.size())
            ? dst_object.size() - offset
            : buffer_ints / 2
            ;
        ret = buffer_cache->read(
                dsenv->queue,
                dst_object_id,
                &dst_object[offset],
                &dst_object[offset + num_ints],
                read_event,
                dummy_wait_list,
                measurement.add_datapoint()
                );
        ASSERT_EQ(true, ret);
    }
    dsenv->queue.finish();

    size_t failed_fields = 0;
    for (size_t i = 0; i < dst_object.size(); i += buffer_ints / 2) {
        for (size_t j = 0; j < buffer_ints / 2; ++j) {
            if (dst_object[i + j] != i * 2 + j) {
                ++failed_fields;
            }
            if (failed_fields <= MAX_PRINT_FAILURES) {
                EXPECT_EQ(i * 2 + j , dst_object[i + j]) << "Object differs at index " << i;
            }
        }
    }
    EXPECT_EQ(0ul, failed_fields);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    dsenv = new DeviceSchedulerEnvironment;
    AddGlobalTestEnvironment(dsenv);

    return RUN_ALL_TESTS();
}
