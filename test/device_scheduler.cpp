/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2017, Lutz, Clemens <lutzcle@cml.li>
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
                size_t size,
                size_t offset,
                bc::buffer buffer
                )
        {
            bc::kernel kernel = zero_program.create_kernel("zero");
            kernel.set_args(buffer, (cl_uint) (size / sizeof(cl_int)));
            return queue.enqueue_1d_range_kernel(
                    kernel,
                    offset / sizeof(cl_int),
                    GLOBAL_SIZE,
                    LOCAL_SIZE
                    );
        };

        bc::program inc_program = bc::program::build_with_source(
                    increment_source,
                    queue.get_context()
                    );

        increment_f = [inc_program](
                bc::command_queue queue,
                size_t size,
                size_t offset,
                bc::buffer buffer
                )
        {
            bc::kernel kernel = inc_program.create_kernel("inc");
            kernel.set_args(buffer, (cl_uint) (size / sizeof(cl_int)));
            return queue.enqueue_1d_range_kernel(
                    kernel,
                    offset / sizeof(cl_int),
                    GLOBAL_SIZE,
                    LOCAL_SIZE
                    );
        };
    }

    void TearDown()
    {}

    boost::compute::device device;
    boost::compute::command_queue queue;
    Clustering::DeviceScheduler::FunUnary zero_f;
    Clustering::DeviceScheduler::FunUnary increment_f;
} *dsenv = nullptr;

class SingleDeviceScheduler : public ::testing::Test {
public:
    SingleDeviceScheduler() :
        buffer_size(BUFFER_SIZE),
        buffer_ints(BUFFER_SIZE / sizeof(int)),
        pool_size(POOL_SIZE),
        object_size(OBJECT_SIZE),
        object_id(0),
        data_object(OBJECT_SIZE / sizeof(int), 0),
        buffer_cache(std::make_shared<decltype(buffer_cache)::element_type>(BUFFER_SIZE))
    {
        buffer_cache->add_device(dsenv->queue.get_context(), dsenv->device, pool_size);
        object_id = buffer_cache->add_object(
                data_object.data(),
                data_object.size() * sizeof(int),
                Clustering::ObjectMode::Mutable
                );

        scheduler.add_buffer_cache(buffer_cache);
        scheduler.add_device(dsenv->queue.get_context(), dsenv->device);
    }

    void SetUp()
    {
        int i = 0;
        for (auto& obj : data_object) {
            obj = i;
            ++i;
        }
    }

    void TearDown()
    {}

    size_t const buffer_size, buffer_ints, pool_size, object_size;
    uint32_t object_id;
    std::vector<uint32_t> data_object;
    std::shared_ptr<Clustering::SimpleBufferCache> buffer_cache;
    Clustering::SingleDeviceScheduler scheduler;
};

TEST_F(SingleDeviceScheduler, EnqueueUnaryKernel)
{
    int ret = 0;
    std::future<std::deque<bc::event>> fevents;

    ret = scheduler.enqueue(dsenv->zero_f, object_id, fevents);
    ASSERT_EQ(true, ret);
    ASSERT_TRUE(fevents.valid());

    ret = scheduler.run();
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

    ret = scheduler.enqueue(dsenv->zero_f, object_id, zero_fevents);
    ASSERT_EQ(true, ret);

    ret = scheduler.enqueue(dsenv->increment_f, object_id, inc_fevents);
    ASSERT_EQ(true, ret);

    ret = scheduler.run();
    ASSERT_EQ(true, ret);

    bc::event read_event;
    for (size_t offset = 0; offset < data_object.size(); offset += buffer_ints) {
        size_t num_ints = (offset + buffer_ints > data_object.size())
            ? data_object.size() - offset
            : buffer_ints
            ;
        ret = buffer_cache->read(
                dsenv->queue,
                object_id,
                &data_object[offset],
                &data_object[offset + num_ints],
                read_event
                );
        ASSERT_EQ(true, ret);
    }
    dsenv->queue.finish();

    size_t failed_fields = 0;
    for (size_t i = 0; i < buffer_ints; ++i) {
        if (data_object[i] != 1u) {
            ++failed_fields;
        }
        if (failed_fields <= MAX_PRINT_FAILURES) {
            EXPECT_EQ(1u, data_object[i]) << "Object differs at index " << i;
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
