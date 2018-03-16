/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2017-2018, Lutz, Clemens <lutzcle@cml.li>
 */

#include <measurement/measurement.hpp>
#include <simple_buffer_cache.hpp>

#include <gtest/gtest.h>
#include <boost/compute/core.hpp>

#include <vector>

#define MAX_PRINT_FAILURES 3

namespace bc = boost::compute;

class SimpleBufferCache : public ::testing::Test {
public:

    SimpleBufferCache() :
        buffer_size(16 << 20),
        buffer_ints((16 << 20) / sizeof(int)),
        pool_size(128 << 20),
        object_size(256 << 20),
        object_id(0),
        data_object((256 << 20) / sizeof(int), 0),
        buffer_cache(16 << 20)
    {
        device = boost::compute::system::default_device();
        queue = boost::compute::system::default_queue();

        buffer_cache.add_device(queue.get_context(), device, pool_size);
        object_id = buffer_cache.add_object(data_object.data(), data_object.size() * sizeof(int), Clustering::ObjectMode::ReadWrite);
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
    Clustering::SimpleBufferCache buffer_cache;
    boost::compute::device device;
    boost::compute::command_queue queue;
};

TEST_F(SimpleBufferCache, RetrieveObject)
{
    void *ret_object_ptr = nullptr;
    size_t ret_object_size = 0;

    buffer_cache.object(object_id, ret_object_ptr, ret_object_size);

    EXPECT_EQ((void*)&data_object[0], ret_object_ptr);
    EXPECT_EQ(object_size, ret_object_size);
}

TEST_F(SimpleBufferCache, GetSize)
{
    EXPECT_EQ(buffer_cache.buffer_size(), buffer_size);
    EXPECT_EQ(buffer_cache.pool_size(device), pool_size);
}

TEST_F(SimpleBufferCache, WriteAndGetCheckBuffer)
{
    boost::compute::event event;
    boost::compute::wait_list wait_list;
    Measurement::Measurement measurement;
    Clustering::BufferCache::BufferList buffers;
    int ret = 0;

    ret = buffer_cache.write_and_get(queue, object_id, &data_object[0], &data_object[buffer_ints], buffers, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);

    for (auto& bufdesc : buffers) {
        EXPECT_LT(0u, bufdesc.content_length);
    }
}

TEST_F(SimpleBufferCache, ReadNotCached)
{
    boost::compute::event event;
    boost::compute::wait_list wait_list;
    Measurement::Measurement measurement;
    int ret = 0;
    uint32_t *begin = &data_object[0];
    uint32_t *end = &data_object[buffer_ints];

    // flush cache or ensure read segment not in cache

    ret = buffer_cache.read(queue, object_id, begin, end, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
}

TEST_F(SimpleBufferCache, WriteReadBasic)
{
    boost::compute::event event;
    boost::compute::wait_list wait_list;
    Measurement::Measurement measurement;
    Clustering::BufferCache::BufferList buffers;
    int ret = 0;
    uint32_t *begin = &data_object[0];
    uint32_t *end = &data_object[buffer_ints];

    ret = buffer_cache.write_and_get(queue, object_id, begin, end, buffers, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    ret = buffer_cache.read(queue, object_id, begin, end, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    event.wait();
    uint32_t failed_fields = 0;
    for (uint32_t i = 0; i < buffer_ints; ++i) {
        if (data_object[i] != i) {
            ++failed_fields;
        }
        if (failed_fields < MAX_PRINT_FAILURES) {
            EXPECT_EQ(i, data_object[i]) << "Object differs at index " << i;
        }
    }
    EXPECT_EQ(0u, failed_fields);
    ret = buffer_cache.unlock(queue, object_id, buffers, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    event.wait();
}

TEST_F(SimpleBufferCache, WriteReadDeadBeefAligned)
{
    boost::compute::event event;
    boost::compute::wait_list wait_list;
    Measurement::Measurement measurement;
    Clustering::BufferCache::BufferList buffers;
    int ret = 0;
    uint32_t *begin = &data_object[buffer_ints];
    uint32_t *end = &data_object[2 * buffer_ints];

    for (auto& obj : data_object) {
        obj = 0xDEADBEEFu;
    }
    ret = buffer_cache.write_and_get(queue, object_id, begin, end, buffers, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    event.wait();
    for (auto& obj : data_object) {
        obj = 0xCAFED00Du;
    }
    ret = buffer_cache.read(queue, object_id, begin, end, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    event.wait();
    uint32_t failed_fields = 0;
    for (auto iter = begin; iter != end; ++iter) {
        if (*iter != 0xDEADBEEFu){
            ++failed_fields;
        }
        if (failed_fields < MAX_PRINT_FAILURES) {
            EXPECT_EQ(0xDEADBEEFu, *iter) << "Object differs at index " << iter - begin;
        }
    }
    EXPECT_EQ(0u, failed_fields);
    ret = buffer_cache.unlock(queue, object_id, buffers, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    event.wait();
}

TEST_F(SimpleBufferCache, WriteReadDeadBeefNonAligned)
{
    boost::compute::event event;
    boost::compute::wait_list wait_list;
    Measurement::Measurement measurement;
    Clustering::BufferCache::BufferList buffers;
    int ret = 0;
    const uint32_t offset = buffer_ints / 2;
    const uint32_t length = buffer_ints - 4;
    uint32_t *begin = &data_object[offset];
    uint32_t *end = &data_object[offset + length];
    auto canary_before = begin - 1u;
    auto canary_after = end;

    for (auto& obj : data_object) {
        obj = 0xDEADBEEFu;
    }
    *canary_before = 0x13371337u;
    *canary_after = 0x13371337u;

    ret = buffer_cache.write_and_get(queue, object_id, begin, end, buffers, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    EXPECT_EQ(length * sizeof(decltype(data_object)::value_type), buffers.front().content_length);
    event.wait();
    for (auto iter = begin; iter != end; ++iter) {
        *iter = 0xCAFED00Du;
    }
    ret = buffer_cache.read(queue, object_id, begin, end, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    event.wait();
    uint32_t failed_fields = 0;
    for (auto iter = begin; iter != end; ++iter) {
        if (*iter != 0xDEADBEEFu){
            ++failed_fields;
        }
        if (failed_fields < MAX_PRINT_FAILURES) {
            EXPECT_EQ(0xDEADBEEFu, *iter) << "Object differs at index " << iter - begin;
        }
    }
    EXPECT_EQ(0u, failed_fields);
    EXPECT_EQ(0x13371337u, *canary_before);
    EXPECT_EQ(0x13371337u, *canary_after);
    ret = buffer_cache.unlock(queue, object_id, buffers, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    event.wait();
}

TEST_F(SimpleBufferCache, GetReadBasic)
{
    boost::compute::event event;
    boost::compute::wait_list wait_list;
    Measurement::Measurement measurement;
    Clustering::BufferCache::BufferList buffers;
    int ret = 0;
    uint32_t *begin = &data_object[0];
    uint32_t *end = &data_object[buffer_ints];

    ret = buffer_cache.get(queue, object_id, begin, end, buffers, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    ret = buffer_cache.read(queue, object_id, begin, end, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    event.wait();
    uint32_t failed_fields = 0;
    for (uint32_t i = 0; i < buffer_ints; ++i) {
        if (data_object[i] != i) {
            ++failed_fields;
        }
        if (failed_fields < MAX_PRINT_FAILURES) {
            EXPECT_EQ(i, data_object[i]) << "Object differs at index " << i;
        }
    }
    EXPECT_EQ(0u, failed_fields);
    ret = buffer_cache.unlock(queue, object_id, buffers, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    event.wait();
}

TEST_F(SimpleBufferCache, GetTwice)
{
    boost::compute::event event;
    boost::compute::wait_list wait_list;
    Measurement::Measurement measurement;
    Clustering::BufferCache::BufferList buffers;
    int ret = 0;
    uint32_t *begin = &data_object[0];
    uint32_t *end = &data_object[buffer_ints];

    ret = buffer_cache.get(queue, object_id, begin, end, buffers, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);

    // Test without unlocking, should fail
    ret = buffer_cache.get(queue, object_id, begin, end, buffers, event, wait_list, measurement.add_datapoint());
    ASSERT_GT(0, ret);

    // Unlock, then try to get again
    ret = buffer_cache.unlock(queue, object_id, buffers, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    ret = buffer_cache.get(queue, object_id, begin, end, buffers, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    event.wait();
}

TEST_F(SimpleBufferCache, GetReadDeadBeef)
{
    boost::compute::event event;
    boost::compute::wait_list wait_list;
    Measurement::Measurement measurement;
    Clustering::BufferCache::BufferList buffers;
    int ret = 0;
    uint32_t *begin = &data_object[0];
    uint32_t *end = &data_object[buffer_ints];

    for (auto& obj : data_object) {
        obj = 0xDEADBEEFu;
    }
    ret = buffer_cache.get(queue, object_id, begin, end, buffers, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    event.wait();
    for (auto& obj : data_object) {
        obj = 0xCAFED00Du;
    }
    ret = buffer_cache.read(queue, object_id, begin, end, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    event.wait();
    uint32_t failed_fields = 0;
    for (uint32_t i = 0; i < buffer_ints; ++i) {
        if (data_object[i] != 0xDEADBEEFu){
            ++failed_fields;
        }
        if (failed_fields < MAX_PRINT_FAILURES) {
            EXPECT_EQ(0xDEADBEEFu, data_object[i]) << "Object differs at index " << i;
        }
    }
    EXPECT_EQ(0u, failed_fields);
    ret = buffer_cache.unlock(queue, object_id, buffers, event, wait_list, measurement.add_datapoint());
    ASSERT_EQ(true, ret);
    event.wait();
}

TEST_F(SimpleBufferCache, ParallelWrites)
{
    constexpr int DUAL_QUEUE = 2;
    auto context = bc::system::default_context();
    bc::command_queue dualq[DUAL_QUEUE]
    {
        bc::command_queue(context, device),
        bc::command_queue(context, device)
    };
    uint32_t *start_ptr[DUAL_QUEUE], *end_ptr[DUAL_QUEUE];
    Clustering::BufferCache::BufferList buffers[DUAL_QUEUE];
    bc::event event[DUAL_QUEUE];
    boost::compute::wait_list wait_list[DUAL_QUEUE];
    Measurement::Measurement measurement;

    for (auto& obj : data_object) {
        obj = 0xDEADBEEFu;
    }

    for (size_t offset = 0; offset < data_object.size(); offset += DUAL_QUEUE * buffer_ints)
    {
        start_ptr[0] = &data_object[offset];
        start_ptr[1] = &data_object[
            (offset + buffer_ints > data_object.size())
                ? data_object.size()
                : offset + buffer_ints
        ];
        end_ptr[0] = start_ptr[1];
        end_ptr[1] = &data_object[
            (offset + 2 * buffer_ints > data_object.size())
                ? data_object.size()
                : offset + 2 * buffer_ints
        ];

        ASSERT_EQ(
                true,
                buffer_cache.write_and_get(
                    dualq[0],
                    object_id,
                    start_ptr[0],
                    end_ptr[0],
                    buffers[0],
                    event[0],
                    wait_list[0],
                    measurement.add_datapoint()
                    ));

        if (start_ptr[1] < end_ptr[1]) {
            ASSERT_EQ(
                    true,
                    buffer_cache.write_and_get(
                        dualq[1],
                        object_id,
                        start_ptr[1],
                        end_ptr[1],
                        buffers[1],
                        event[1],
                        wait_list[1],
                        measurement.add_datapoint()
                        ));

            event[1].wait();
            for (auto iter = start_ptr[1]; iter < end_ptr[1]; ++iter) {
                *iter = 0xCAFED00Du;
            }

            ASSERT_EQ(
                    true,
                    buffer_cache.unlock(
                        dualq[1],
                        object_id,
                        buffers[1],
                        event[1],
                        wait_list[1],
                        measurement.add_datapoint()
                        ));
            event[1].wait();
        }

        event[0].wait();
        for (auto iter = start_ptr[0]; iter < end_ptr[0]; ++iter) {
            *iter = 0xCAFED00Du;
        }

        ASSERT_EQ(
            true,
            buffer_cache.unlock(
                dualq[0],
                object_id,
                buffers[0],
                event[0],
                wait_list[0],
                measurement.add_datapoint()
                ));
        event[0].wait();
    }

    event[0].wait();
    event[1].wait();

    for (size_t offset = 0; offset < data_object.size(); offset += buffer_ints) {
        start_ptr[0] = &data_object[offset];
        end_ptr[0] = &data_object[
            (offset + buffer_ints > data_object.size())
                ? data_object.size()
                : offset + buffer_ints
        ];

        ASSERT_EQ(
                true,
                buffer_cache.read(
                    dualq[0],
                    object_id,
                    start_ptr[0],
                    end_ptr[0],
                    event[0],
                    wait_list[0],
                    measurement.add_datapoint()
                    ));
    }

    event[0].wait();

    uint32_t failed_fields = 0;
    for (size_t i = 0; i < data_object.size(); ++i) {
        if (data_object[i] != 0xDEADBEEFu){
            ++failed_fields;
        }
        if (failed_fields < MAX_PRINT_FAILURES) {
            EXPECT_EQ(0xDEADBEEFu, data_object[i]) << "Object differs at index " << i;
        }
    }
    EXPECT_EQ(0u, failed_fields);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    // ::testing::AddGlobalTestEnvironment(boost_env);
    return RUN_ALL_TESTS();
}
