/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2017, Lutz, Clemens <lutzcle@cml.li>
 */

#include "simple_buffer_cache.hpp"

#include <cstdint>
#include <cstring>

using namespace Clustering;

SimpleBufferCache::SimpleBufferCache(size_t buffer_size)
    :
        BufferCache(buffer_size)
{
    // Invalidate object ID == 0
    object_info_i.emplace_back();
    ObjectInfo& obj = object_info_i[0];
    obj.ptr = nullptr;
    obj.size = 0;
}

size_t SimpleBufferCache::pool_size(Device device)
{
    int64_t did = find_device_id(device);
    if (did < 0) {
        return -1;
    }

    DeviceInfo& dev = device_info_i[did];
    return dev.pool_size;
}

int SimpleBufferCache::add_device(Context context, Device device, size_t pool_size)
{
    if (pool_size <= buffer_size_i * DoubleBuffering) {
        return -1;
    }

    device_info_i.emplace_back();
    DeviceInfo& info = device_info_i.back();

    info.context = context;
    info.device = device;
    info.pool_size = pool_size;
    info.num_buffers = DoubleBuffering;
    info.buffer_lock.resize(DoubleBuffering, 0);
    info.cached_object_id.resize(DoubleBuffering, -1);
    info.cached_buffer_id.resize(DoubleBuffering, -1);
    info.device_buffer.resize(DoubleBuffering);
    info.host_buffer.resize(DoubleBuffering);
    info.host_ptr.resize(DoubleBuffering, nullptr);

    auto queue = Queue(context, device);

    for (auto& buf : info.device_buffer) {
        buf = Buffer(context, buffer_size_i);
    }

    for (size_t i = 0; i < info.host_buffer.size(); ++i) {
        auto& buf = info.host_buffer[i];
        buf = Buffer(
                context,
                buffer_size_i,
                Buffer::read_write | Buffer::alloc_host_ptr
                );
        info.host_ptr[i] = queue.enqueue_map_buffer(
                buf,
                Queue::map_write_invalidate_region,
                0,
                buffer_size_i
                );
    }

    return 1;
}

uint32_t SimpleBufferCache::add_object(void *data_object, size_t size)
{
    uint32_t oid = object_info_i.size();
    object_info_i.emplace_back();
    ObjectInfo& obj = object_info_i[oid];
    obj.ptr = data_object;
    obj.size = size;

    return oid;
}

int SimpleBufferCache::get(Queue queue, uint32_t oid, void *begin, void *end, BufferList& buffers, Event& event, WaitList const& wait_list)
{
    char *cbegin = (char*) begin, *cend = (char*) end;
    size_t size = cend - cbegin;

    if (size > buffer_size_i) {
        return -1;
    }

    auto device_id = find_device_id(queue.get_device());
    if (device_id < 0) {
        return device_id;
    }
    auto buffer_id = find_buffer_id(device_id, oid, begin);
    if (buffer_id < 0) {
        return buffer_id;
    }

    auto cache_slot = find_cache_slot(device_id, oid, buffer_id);
    if (cache_slot == -2) {
        // Case: not yet in cache
        return write_and_get(queue, oid, begin, end, buffers, event, wait_list);
    }
    else if (cache_slot < 0) {
        // Case: other error
        return cache_slot;
    }

    // Case: in cache
    auto& device_info = device_info_i[device_id];
    device_info.buffer_lock[cache_slot] = 1;
    buffers.push_back({device_info.device_buffer[cache_slot], size});

    return 1;
}

int SimpleBufferCache::write_and_get(Queue queue, uint32_t oid, void *begin, void *end, BufferList& buffers, Event& event, WaitList const& wait_list)
{
    char *cbegin = (char*) begin, *cend = (char*) end;
    size_t size = cend - cbegin;

    if (size > buffer_size_i) {
        return -1;
    }

    auto device_id = find_device_id(queue.get_device());
    if (device_id < 0) {
        return -1;
    }
    auto buffer_id = find_buffer_id(device_id, oid, begin);
    if (buffer_id < 0) {
        return -1;
    }
    auto cache_slot = assign_cache_slot(device_id, oid, buffer_id);
    if (cache_slot < 0) {
        return -1;
    }
    auto& device_info = device_info_i[device_id];
    void *host_ptr = device_info.host_ptr[cache_slot];
    auto& device_buffer = device_info.device_buffer[cache_slot];

    auto locked = try_lock(device_id, oid, begin, end);
    if (locked != 1) {
        return -1;
    }

    if (wait_list.empty()) {
        queue.finish();
    }
    else {
        wait_list.wait();
    }
    std::memcpy(host_ptr, begin, size);
    event = queue.enqueue_write_buffer_async(
            device_buffer,
            0,
            size,
            host_ptr
            );

    buffers.push_back({device_buffer, size});

    device_info.cached_object_id[cache_slot] = oid;
    device_info.cached_buffer_id[cache_slot] = buffer_id;

    return 1;
}

int SimpleBufferCache::read(Queue queue, uint32_t oid, void *begin, void *end, Event& event, WaitList const& wait_list)
{
    char *cbegin = (char*) begin, *cend = (char*) end;
    size_t size = cend - cbegin;

    if (size > buffer_size_i) {
        return -1;
    }

    auto device_id = find_device_id(queue.get_device());
    if (device_id < 0) {
        return device_id;
    }
    auto buffer_id = find_buffer_id(device_id, oid, begin);
    if (buffer_id < 0) {
        return buffer_id;
    }
    auto cache_slot = find_cache_slot(device_id, oid, buffer_id);
    if (cache_slot < 0) {
        return cache_slot;
    }

    auto& device_info = device_info_i[device_id];
    void *host_ptr = device_info.host_ptr[cache_slot];
    auto& device_buffer = device_info.device_buffer[cache_slot];

    Event read_event;
    read_event = queue.enqueue_read_buffer_async(
            device_buffer,
            0,
            size,
            host_ptr,
            wait_list
            );
    read_event.wait();
    std::memcpy(begin, host_ptr, size);

    event = read_event;

    return 1;
}

int SimpleBufferCache::try_lock(uint32_t device_id, uint32_t oid, void *begin, void * /* end */)
{
    DeviceInfo& dev = device_info_i[device_id];
    int64_t buf_id = find_buffer_id(device_id, oid, begin);
    if (buf_id < 0) {
        return -1;
    }

    auto& lock = dev.buffer_lock[buf_id];
    if (not lock) {
        lock = 1;
        return 1;
    }
    else {
        return -1;
    }
}

int SimpleBufferCache::unlock(Device device, uint32_t oid, void *begin, void * /* end */)
{
    // TODO: support unlocking multiple buffers

    int64_t dev_id = find_device_id(device);
    if (dev_id < 0) {
        return -1;
    }
    DeviceInfo& dev = device_info_i[dev_id];
    int64_t buf_id = find_buffer_id(dev_id, oid, begin);
    if (buf_id < 0) {
        return -1;
    }
    dev.buffer_lock[buf_id] = 0;

    // TODO: Delete buffer from device

    return 1;
}

void* SimpleBufferCache::buffer2pointer(uint32_t oid, uint32_t buffer_id)
{
    if (oid >= object_info_i.size()) {
        return nullptr;
    }

    return ((char*)object_info_i[oid].ptr) + buffer_size_i * buffer_id;
}

uint32_t SimpleBufferCache::pointer2buffer(uint32_t oid, void *ptr)
{
    if (oid >= object_info_i.size()) {
        return 0;
    }

    return ((char*)ptr - (char*)object_info_i[oid].ptr) / buffer_size_i;
}

int64_t SimpleBufferCache::find_device_id(Device device)
{
    for (uint32_t did = 0; did < device_info_i.size(); ++did) {
        if (device_info_i[did].device == device) {
            return did;
        }
    }

    return -1;
}

int64_t SimpleBufferCache::find_buffer_id(uint32_t device_id, uint32_t oid, void *ptr)
{
    if (device_id >= device_info_i.size()) {
        return -1;
    }
    if (oid >= object_info_i.size()) {
        return -1;
    }
    ObjectInfo& oinfo = object_info_i[oid];
    if (not ptr or ptr < oinfo.ptr or ptr >= &((char*)oinfo.ptr)[oinfo.size]) {
        return -1;
    }

    size_t bid = ((char*)ptr - (char*)object_info_i[oid].ptr) / buffer_size_i;

    return bid;
}

int64_t SimpleBufferCache::find_cache_slot(uint32_t device_id, uint32_t oid, uint32_t buffer_id)
{
    if (device_id >= device_info_i.size()) {
        return -1;
    }
    if (oid >= object_info_i.size()) {
        return -1;
    }
    auto& device_info = device_info_i[device_id];
    if (buffer_id >= device_info.num_buffers) {
        return -1;
    }

    for (size_t i = 0; i < device_info.cached_buffer_id.size(); ++i) {
        auto& coi = device_info.cached_object_id[i];
        auto& cbi = device_info.cached_buffer_id[i];

        if (coi == oid and cbi == buffer_id) return i;
    }

    return -2;
}

int64_t SimpleBufferCache::assign_cache_slot(uint32_t device_id, uint32_t oid, uint32_t bid)
{
    if (device_id >= device_info_i.size()) {
        return -1;
    }
    if (oid >= object_info_i.size()) {
        return -1;
    }

    return bid % DoubleBuffering;
}
