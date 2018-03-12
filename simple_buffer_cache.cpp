/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#include "timer.hpp"
#include "simple_buffer_cache.hpp"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>

#define VERBOSE false

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

    size_t num_cache_slots = pool_size / buffer_size_i;

    info.context = context;
    info.device = device;
    info.pool_size = pool_size;
    info.num_slots = num_cache_slots;
    info.slot_lock.resize(num_cache_slots, {});
    info.cached_object_id.resize(num_cache_slots, -1);
    info.cached_buffer_id.resize(num_cache_slots, -1);
    info.cached_ptr.resize(num_cache_slots, nullptr);
    info.cached_content_length.resize(num_cache_slots, 0);
    info.device_buffer.resize(num_cache_slots);
    info.host_buffer.resize(num_cache_slots);
    info.host_ptr.resize(num_cache_slots, nullptr);

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

uint32_t SimpleBufferCache::add_object(void *data_object, size_t size, ObjectMode mode)
{
    uint32_t oid = object_info_i.size();
    object_info_i.emplace_back();
    ObjectInfo& obj = object_info_i[oid];
    obj.ptr = data_object;
    obj.size = size;
    obj.mode = mode;

    return oid;
}

void SimpleBufferCache::object(uint32_t oid, void *& data_object, size_t& length)
{
    if (oid == 0 or oid >= object_info_i.size()) {
        data_object = nullptr;
        length = 0;
        return;
    }

    ObjectInfo& obj = object_info_i[oid];
    data_object = obj.ptr;
    length = obj.size;
}

int SimpleBufferCache::get(Queue queue, uint32_t oid, void *begin, void *end, BufferList& buffers, Event& event, WaitList const& wait_list, Measurement::DataPoint& datapoint)
{
    datapoint.set_name("BufferCache::get");

    char *cbegin = (char*) begin, *cend = (char*) end;
    size_t size = cend - cbegin;


    if (size > buffer_size_i) {
        std::cerr << "get: ranges > buffer_size not supported" << std::endl;
        return -1;
    }

    auto device_id = find_device_id(queue.get_device());
    if (device_id < 0) {
        std::cerr << "get: bad device" << std::endl;
        return device_id;
    }
    auto buffer_id = find_buffer_id(device_id, oid, begin);
    if (buffer_id < 0) {
        std::cerr << "get: bad begin ptr" << std::endl;
        return buffer_id;
    }

    if (VERBOSE) {
        std::cerr << "get: OID " << oid << " BID " << buffer_id << " DID " << device_id << std::endl;
    }

    auto cache_slot = find_cache_slot(device_id, oid, buffer_id);
    if (cache_slot == -2) {
        // Case: not yet in cache
        return write_and_get(queue, oid, begin, end, buffers, event, wait_list, datapoint.create_child());
    }
    else if (cache_slot < 0) {
        // Case: other error
        std::cerr << "get: find_cache_slot error" << std::endl;
        return cache_slot;
    }

    // Case: in cache
    if(try_read_lock(device_id, cache_slot) < 0) {
        std::cerr << "get: try_read_lock error" << std::endl;
        return -1;
    }
    auto& device_info = device_info_i[device_id];
    buffers.clear();
    buffers.push_back({device_info.device_buffer[cache_slot], size, buffer_id});

    return 1;
}

int SimpleBufferCache::write_and_get(Queue queue, uint32_t oid, void *begin, void *end, BufferList& buffers, Event& event, WaitList const& wait_list, Measurement::DataPoint& datapoint)
{
    datapoint.set_name("BufferCache::write_and_get");

    char *cbegin = (char*) begin, *cend = (char*) end;
    size_t size = cend - cbegin;

    if (size > buffer_size_i) {
        std::cerr << "write_and_get: ranges > buffer_size not supported" << std::endl;
        return -1;
    }

    auto device_id = find_device_id(queue.get_device());
    if (device_id < 0) {
        std::cerr << "write_and_get: bad device" << std::endl;
        return -1;
    }
    auto buffer_id = find_buffer_id(device_id, oid, begin);
    if (buffer_id < 0) {
        std::cerr << "write_and_get: bad begin ptr" << std::endl;
        return -1;
    }
    auto cache_slot = assign_cache_slot(device_id, oid, buffer_id);
    if (cache_slot < 0) {
        std::cerr << "write_and_get: no free cache slot" << std::endl;
        return -1;
    }
    auto& device_info = device_info_i[device_id];
    void *host_ptr = device_info.host_ptr[cache_slot];
    auto& device_buffer = device_info.device_buffer[cache_slot];

    auto locked = try_write_lock(device_id, cache_slot);
    if (locked != 1) {
        std::cerr << "write_and_get: cannot lock cache slot " << cache_slot << std::endl;
        return -1;
    }

    if (evict_cache_slot(queue, device_id, cache_slot, event, wait_list, datapoint.create_child()) < 0) {
        std::cerr << "write_and_get: cannot evict cache slot " << cache_slot << std::endl;
        return -1;
    }

    if (wait_list.empty()) {
        queue.finish();
    }
    else {
        wait_list.wait();
    }

    Timer::Timer memcpy_timer;
    memcpy_timer.start();
    std::memcpy(host_ptr, begin, size);
    uint64_t memcpy_time = memcpy_timer
        .stop<std::chrono::nanoseconds>();
    datapoint.add_value() = memcpy_time;

    event = queue.enqueue_write_buffer_async(
            device_buffer,
            0,
            size,
            host_ptr
            );
    datapoint.add_event() = event;

    buffers.clear();
    buffers.push_back({device_buffer, size, buffer_id});

    device_info.cached_object_id[cache_slot] = oid;
    device_info.cached_buffer_id[cache_slot] = buffer_id;
    device_info.cached_ptr[cache_slot] = begin;
    device_info.cached_content_length[cache_slot] = size;

    return 1;
}

int SimpleBufferCache::read(Queue queue, uint32_t oid, void *begin, void *end, Event&, WaitList const& wait_list, Measurement::DataPoint& datapoint)
{
    datapoint.set_name("BufferCache::read");

    char *cbegin = (char*) begin, *cend = (char*) end;
    size_t size = cend - cbegin;

    if (size > buffer_size_i) {
        std::cerr << "read: multi-buffer read not supported" << std::endl;
        return -1;
    }

    auto device_id = find_device_id(queue.get_device());
    if (device_id < 0) {
        std::cerr << "read: find_device_id error" << std::endl;
        return device_id;
    }
    auto buffer_id = find_buffer_id(device_id, oid, begin);
    if (buffer_id < 0) {
        std::cerr << "read: find_buffer_id error" << std::endl;
        return buffer_id;
    }
    if (object_info_i[oid].mode == ObjectMode::Immutable) {
        // Assume object didn't change, don't need to do anything
        return 1;
    }
    auto cache_slot = find_cache_slot(device_id, oid, buffer_id);
    if (cache_slot == -2) {
        // Case: not in device cache
        return 1;
    }
    else if (cache_slot < 0) {
        // Case: error getting cache slot
        std::cerr << "read: find_cache_slot error" << std::endl;
        return cache_slot;
    }
    // else Case: in device cache, must read back

    auto& device_info = device_info_i[device_id];
    void *host_ptr = device_info.host_ptr[cache_slot];
    auto& device_buffer = device_info.device_buffer[cache_slot];

    if (VERBOSE) {
        std::cerr << "read: OID " << oid << " BID " << buffer_id << " destination " << begin << " length " << size << " DID " << device_id << std::endl;
    }

    if (device_info.cached_content_length[cache_slot] > size) {
        std::cerr << "read: cannot read more than length of cached content " << std::endl;
        return -1;
    }

    Event read_event;
    read_event = queue.enqueue_read_buffer_async(
            device_buffer,
            0,
            size,
            host_ptr,
            wait_list
            );
    datapoint.add_event() = read_event;
    read_event.wait();

    Timer::Timer memcpy_timer;
    memcpy_timer.start();
    std::memcpy(begin, host_ptr, size);
    uint64_t memcpy_time = memcpy_timer
        .stop<std::chrono::nanoseconds>();
    datapoint.add_value() = memcpy_time;

    // TODO: make asynchronous and set event = read_event instead of calling wait()
    // event = read_event;

    return 1;
}

int SimpleBufferCache::evict_cache_slot(Queue queue, uint32_t device_id, uint32_t cache_slot, Event& event, WaitList const& wait_list, Measurement::DataPoint& datapoint)
{
    datapoint.set_name("SimpleBufferCache::evict_cache_slot");

    DeviceInfo& devinfo = device_info_i[device_id];
    int64_t& object_id = devinfo.cached_object_id[cache_slot];
    int64_t& buffer_id = devinfo.cached_buffer_id[cache_slot];
    void*& cached_ptr = devinfo.cached_ptr[cache_slot];
    size_t& content_length = devinfo.cached_content_length[cache_slot];
    ObjectMode mode = object_info_i[object_id].mode;

    if (object_id == -1 and buffer_id == -1) {
        // Case: cache slot is empty
        return 1;
    }
    else if (mode == ObjectMode::Immutable) {
        // Case: object is immutable, can trivially be evicted
        object_id = -1;
        buffer_id = -1;
        cached_ptr = nullptr;
        content_length = 0;

        return 1;
    }
    // Case: object is mutable and presumed dirty, must write back to evict
    int ret = read(
            queue,
            object_id,
            cached_ptr,
            ((char*)cached_ptr) + content_length,
            event,
            wait_list,
            datapoint.create_child()
        );
    if (ret < 0) {
        std::cerr << "evict_cache_slot: write-back error" << std::endl;
        return -1;
    }

    object_id = -1;
    buffer_id = -1;
    cached_ptr = nullptr;
    content_length = 0;

    return 1;
}

int SimpleBufferCache::try_read_lock(uint32_t device_id, uint32_t cache_slot)
{
    DeviceInfo& dev = device_info_i[device_id];
    if (cache_slot >= dev.num_slots) {
        std::cerr << "try_read_lock: invalid cache slot " << cache_slot << std::endl;
        return -1;
    }

    if (VERBOSE) {
        std::cerr << "try_read_lock: DID " << device_id << " SlotID " << cache_slot << std::endl;
    }

    auto& lock = dev.slot_lock[cache_slot];
    if (lock.status == DeviceInfo::SlotLock::Free) {
        lock.status = DeviceInfo::SlotLock::ReadLock;
        lock.count = 1;

        return 1;
    }
    else if (lock.status == DeviceInfo::SlotLock::ReadLock) {
        ++lock.count;

        return 1;
    }
    else {
        return -1;
    }
}

int SimpleBufferCache::try_write_lock(uint32_t device_id, uint32_t cache_slot)
{
    DeviceInfo& dev = device_info_i[device_id];
    if (cache_slot >= dev.num_slots) {
        std::cerr << "try_write_lock: invalid cache slot " << cache_slot << std::endl;
        return -1;
    }

    if (VERBOSE) {
        std::cerr << "try_write_lock: DID " << device_id << " SlotID " << cache_slot << std::endl;
    }

    auto& lock = dev.slot_lock[cache_slot];
    if (lock.status == DeviceInfo::SlotLock::Free) {
        lock.status = DeviceInfo::SlotLock::WriteLock;
        lock.count = 1;

        return 1;
    }
    else {
        return -1;
    }
}

int SimpleBufferCache::unlock(Queue queue, uint32_t oid, BufferList const& buffers, Event&, WaitList const&, Measurement::DataPoint& datapoint)
{
    datapoint.set_name("BufferCache::unlock");

    Device device = queue.get_device();
    int64_t dev_id = find_device_id(device);
    if (dev_id < 0) {
        return -1;
    }
    DeviceInfo& dev = device_info_i[dev_id];
    if (buffers.empty()) {
        std::cerr << "unlock: buffers list is empty" << std::endl;
        return -1;
    }
    else if (buffers.size() > 1) {
        std::cerr << "unlock: multiple buffers not supported" << std::endl;
        return -1;
    }
    int64_t buf_id = buffers.front().buffer_id;
    if (buf_id < 0) {
        return -1;
    }
    int64_t slot_id = find_cache_slot(dev_id, oid, buf_id);
    if (slot_id < 0) {
        std::cerr << "unlock: find_cache_slot error" << std::endl;
        return -1;
    }

    if (VERBOSE) {
        std::cerr << "unlock: OID " << oid << " BID " << buf_id << " DID " << dev_id << " SlotID " << slot_id << std::endl;
    }

    auto& lock = dev.slot_lock[slot_id];
    if (lock.status == DeviceInfo::SlotLock::ReadLock) {
        if (lock.count == 1) {
            lock.status = DeviceInfo::SlotLock::Free;
            lock.count = 0;
        }
        else {
            --lock.count;
        }
    }
    else if (lock.status == DeviceInfo::SlotLock::WriteLock) {
        lock.status = DeviceInfo::SlotLock::Free;
        lock.count = 0;
    }
    else {
        std::cerr << "unlock: invalid lock state error" << std::endl;
        return -1;
    }

    return 1;
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
        std::cerr << "find_buffer_id: invalid DID " << device_id << std::endl;
        return -1;
    }
    if (oid == 0 or oid >= object_info_i.size()) {
        std::cerr << "find_buffer_id: invalid OID " << oid << std::endl;
        return -1;
    }
    ObjectInfo& oinfo = object_info_i[oid];
    if (not ptr or ptr < oinfo.ptr or ptr >= &((char*)oinfo.ptr)[oinfo.size]) {
        std::cerr << "find_buffer_id: invalid pointer " << ptr << std::endl;
        return -1;
    }

    size_t bid = ((char*)ptr - (char*)oinfo.ptr);

    return bid;
}

int64_t SimpleBufferCache::find_cache_slot(uint32_t device_id, uint32_t oid, uint32_t buffer_id)
{
    if (device_id >= device_info_i.size()) {
        std::cerr << "find_cache_slot: invalid DID " << device_id << std::endl;
        return -1;
    }
    if (oid == 0 or oid >= object_info_i.size()) {
        std::cerr << "find_cache_slot: invalid OID " << oid << std::endl;
        return -1;
    }
    auto& device_info = device_info_i[device_id];
    if (buffer_id >= (object_info_i[oid].size + buffer_size_i - 1)) {
        std::cerr << "find_cache_slot: invalid BID " << buffer_id << std::endl;
        return -1;
    }

    for (size_t i = 0; i < device_info.cached_buffer_id.size(); ++i) {
        auto& coi = device_info.cached_object_id[i];
        auto& cbi = device_info.cached_buffer_id[i];

        if (coi == oid and cbi == buffer_id) return i;
    }

    return -2;
}

int64_t SimpleBufferCache::assign_cache_slot(uint32_t device_id, uint32_t oid, uint32_t /* bid */)
{
    if (device_id >= device_info_i.size()) {
        return -1;
    }
    if (oid >= object_info_i.size()) {
        return -1;
    }

    auto& dev = device_info_i[device_id];
    uint32_t base_slot = (oid - 1) * DoubleBuffering;
    uint32_t slot = (dev.slot_lock[base_slot].status == DeviceInfo::SlotLock::Free) ? base_slot : base_slot + 1;

    if (dev.slot_lock[slot].status != DeviceInfo::SlotLock::Free) {
        std::cerr << "assign_cache_slot: cannot find free cache slot" << std::endl;
        return -1;
    }

    return slot;
}
