/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2017, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef SIMPLE_BUFFER_CACHE_HPP
#define SIMPLE_BUFFER_CACHE_HPP

#include <buffer_cache.hpp>

#include <cstdint>
#include <vector>

#include <boost/compute/buffer.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/event.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/utility/wait_list.hpp>

namespace Clustering {

class SimpleBufferCache : public BufferCache {
public:

    using Buffer = boost::compute::buffer;
    using BufferList = typename BufferCache::BufferList;
    using Device = boost::compute::device;
    using Event = boost::compute::event;
    using Queue = boost::compute::command_queue;
    using WaitList = boost::compute::wait_list;

    SimpleBufferCache(size_t buffer_size);

    size_t pool_size(Device device);
    int add_device(Queue queue, Device device, size_t pool_size);
    uint32_t add_object(void *data_object, size_t length);
    int get(Queue queue, uint32_t oid, void *begin, void *end, BufferList& buffer, Event& event, WaitList const& wait_list = WaitList());
    int write_and_get(Queue queue, uint32_t oid, void *begin, void *end, BufferList& buffer, Event& event, WaitList const& wait_list = WaitList());
    int read(Queue queue, uint32_t oid, void *begin, void *end, Event& event, WaitList const& wait_list = WaitList());
    int sync_and_get(Queue, Queue, uint32_t, void*, void*, Event&, WaitList const&) { return -1; /* not supported */ };
    int unlock(Device device, uint32_t oid, void *begin, void *end);
    void* buffer2pointer(uint32_t oid, uint32_t buffer_id);
    uint32_t pointer2buffer(uint32_t oid, void *ptr);

private:

    uint32_t static constexpr DoubleBuffering = 2u;

    struct DeviceInfo {
        Device device;
        size_t pool_size;
        size_t num_buffers;
        std::vector<int> buffer_lock;
        std::vector<int64_t> cached_object_id;
        std::vector<int64_t> cached_buffer_id;
        std::vector<Buffer> device_buffer;
        std::vector<Buffer> host_buffer;
        std::vector<void*> host_ptr;
    };

    struct ObjectInfo {
        void* ptr;
        size_t size;
    };

    std::vector<DeviceInfo> device_info_i;
    std::vector<ObjectInfo> object_info_i;

    int try_lock(uint32_t device_id, uint32_t oid, void *begin, void *end);
    int64_t find_device_id(Device device);
    int64_t find_buffer_id(uint32_t device_id, uint32_t oid, void *ptr);
    int64_t find_cache_slot(uint32_t device_id, uint32_t oid, uint32_t buffer_id);
    int64_t assign_cache_slot(uint32_t device_id, uint32_t oid, uint32_t bid);
};

} // namespace Clustering

#endif /* SIMPLE_BUFFER_CACHE_HPP */