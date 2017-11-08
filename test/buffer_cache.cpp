/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2017, Lutz, Clemens <lutzcle@cml.li>
 */

#include <simple_buffer_cache.hpp>

#include <gtest/gtest.h>
#include <boost/compute/core.hpp>

#include <vector>

#define MAX_PRINT_FAILURES 3

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
        object_id = buffer_cache.add_object(data_object.data(), data_object.size() * sizeof(int));
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

TEST_F(SimpleBufferCache, BufferPointerConversion)
{
    uint32_t bid = 0;
    uint32_t bid_ret = 0;
    void *offset = nullptr;
    void *offset_ret = nullptr;

    bid = 0;
    offset_ret = buffer_cache.buffer2pointer(object_id, bid);
    EXPECT_EQ((void*)data_object.data(), offset_ret);

    offset = data_object.data();
    bid_ret = buffer_cache.pointer2buffer(object_id, offset);
    EXPECT_EQ(0u, bid_ret);

    bid = 5;
    offset_ret = buffer_cache.buffer2pointer(object_id, bid);
    EXPECT_EQ((void*) &data_object[bid * buffer_size / sizeof(int)], offset_ret);

    offset = &data_object[bid * buffer_size / sizeof(int)];
    bid_ret = buffer_cache.pointer2buffer(object_id, offset);
    EXPECT_EQ(bid, bid_ret);

    offset = &data_object[bid * buffer_size / sizeof(int) - 1];
    bid_ret = buffer_cache.pointer2buffer(object_id, offset);
    EXPECT_EQ(bid - 1, bid_ret);
}

TEST_F(SimpleBufferCache, GetSize)
{
    EXPECT_EQ(buffer_cache.buffer_size(), buffer_size);
    EXPECT_EQ(buffer_cache.pool_size(device), pool_size);
}

TEST_F(SimpleBufferCache, WriteAndGetCheckBuffer)
{
    boost::compute::event event;
    Clustering::BufferCache::BufferList buffers;
    int ret = 0;

    ret = buffer_cache.write_and_get(queue, object_id, &data_object[0], &data_object[buffer_ints], buffers, event);
    ASSERT_EQ(true, ret);

    for (auto& bufdesc : buffers) {
        EXPECT_LT(0u, bufdesc.content_length);
    }
}

TEST_F(SimpleBufferCache, ReadNotCached)
{
    boost::compute::event event;
    int ret = 0;
    uint32_t *begin = &data_object[0];
    uint32_t *end = &data_object[buffer_ints];

    // flush cache or ensure read segment not in cache

    ret = buffer_cache.read(queue, object_id, begin, end, event);
    ASSERT_EQ(true, ret);
}

TEST_F(SimpleBufferCache, WriteReadBasic)
{
    boost::compute::event event;
    Clustering::BufferCache::BufferList buffers;
    int ret = 0;
    uint32_t *begin = &data_object[0];
    uint32_t *end = &data_object[buffer_ints];

    ret = buffer_cache.write_and_get(queue, object_id, begin, end, buffers, event);
    ASSERT_EQ(true, ret);
    ret = buffer_cache.read(queue, object_id, begin, end, event);
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
    ret = buffer_cache.unlock(device, object_id, begin, end);
    ASSERT_EQ(true, ret);
}

TEST_F(SimpleBufferCache, WriteReadDeadBeef)
{
    boost::compute::event event;
    Clustering::BufferCache::BufferList buffers;
    int ret = 0;
    uint32_t *begin = &data_object[0];
    uint32_t *end = &data_object[buffer_ints];

    for (auto& obj : data_object) {
        obj = 0xDEADBEEFu;
    }
    ret = buffer_cache.write_and_get(queue, object_id, begin, end, buffers, event);
    ASSERT_EQ(true, ret);
    event.wait();
    for (auto& obj : data_object) {
        obj = 0xCAFED00Du;
    }
    ret = buffer_cache.read(queue, object_id, begin, end, event);
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
    ret = buffer_cache.unlock(device, object_id, begin, end);
    ASSERT_EQ(true, ret);
}

TEST_F(SimpleBufferCache, GetReadBasic)
{
    boost::compute::event event;
    Clustering::BufferCache::BufferList buffers;
    int ret = 0;
    uint32_t *begin = &data_object[0];
    uint32_t *end = &data_object[buffer_ints];

    ret = buffer_cache.get(queue, object_id, begin, end, buffers, event);
    ASSERT_EQ(true, ret);
    ret = buffer_cache.read(queue, object_id, begin, end, event);
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
    ret = buffer_cache.unlock(device, object_id, begin, end);
    ASSERT_EQ(true, ret);
}

TEST_F(SimpleBufferCache, GetTwice)
{
    boost::compute::event event;
    Clustering::BufferCache::BufferList buffers;
    int ret = 0;
    uint32_t *begin = &data_object[0];
    uint32_t *end = &data_object[buffer_ints];

    ret = buffer_cache.get(queue, object_id, begin, end, buffers, event);
    ASSERT_EQ(true, ret);

    // Test without unlocking, should fail
    ret = buffer_cache.get(queue, object_id, begin, end, buffers, event);
    ASSERT_GT(0, ret);

    // Unlock, then try to get again
    ret = buffer_cache.unlock(queue.get_device(), object_id, begin, end);
    ASSERT_EQ(true, ret);
    ret = buffer_cache.get(queue, object_id, begin, end, buffers, event);
    ASSERT_EQ(true, ret);
}

TEST_F(SimpleBufferCache, GetReadDeadBeef)
{
    boost::compute::event event;
    Clustering::BufferCache::BufferList buffers;
    int ret = 0;
    uint32_t *begin = &data_object[0];
    uint32_t *end = &data_object[buffer_ints];

    for (auto& obj : data_object) {
        obj = 0xDEADBEEFu;
    }
    ret = buffer_cache.get(queue, object_id, begin, end, buffers, event);
    ASSERT_EQ(true, ret);
    event.wait();
    for (auto& obj : data_object) {
        obj = 0xCAFED00Du;
    }
    ret = buffer_cache.read(queue, object_id, begin, end, event);
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
    ret = buffer_cache.unlock(device, object_id, begin, end);
    ASSERT_EQ(true, ret);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    // ::testing::AddGlobalTestEnvironment(boost_env);
    return RUN_ALL_TESTS();
}
