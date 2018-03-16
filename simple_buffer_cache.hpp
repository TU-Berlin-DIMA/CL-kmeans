/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2017-2018, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef SIMPLE_BUFFER_CACHE_HPP
#define SIMPLE_BUFFER_CACHE_HPP

#include <buffer_cache.hpp>

#include <cstdint>
#include <vector>
#include <list>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <boost/compute/buffer.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/event.hpp>
#include <boost/compute/user_event.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/utility/wait_list.hpp>

namespace Clustering {

class SimpleBufferCache : public BufferCache {
public:

    using Buffer = boost::compute::buffer;
    using BufferList = typename BufferCache::BufferList;
    using Context = boost::compute::context;
    using Device = boost::compute::device;
    using Event = boost::compute::event;
    using Queue = boost::compute::command_queue;
    using WaitList = boost::compute::wait_list;

    SimpleBufferCache(size_t buffer_size);
    ~SimpleBufferCache();

    // TODO: return multiple OpenCL events in read / write / etc

    size_t pool_size(Device device);
    int add_device(Context context, Device device, size_t pool_size);
    uint32_t add_object(void *data_object, size_t length, ObjectMode mode = ObjectMode::ReadOnly);
    void object(uint32_t object_id, void *& data_object, size_t& length);
    int get(Queue queue, uint32_t oid, void *begin, void *end, BufferList& buffer, Event& event, WaitList const& wait_list, Measurement::DataPoint& datapoint);
    int write_and_get(Queue queue, uint32_t oid, void *begin, void *end, BufferList& buffer, Event& event, WaitList const& wait_list, Measurement::DataPoint& datapoint);
    int read(Queue queue, uint32_t oid, void *begin, void *end, Event& event, WaitList const& wait_list, Measurement::DataPoint& datapoint);
    int sync_and_get(Queue, Queue, uint32_t, void*, void*, Event&, WaitList const&, Measurement::DataPoint&) { return -1; /* not supported */ };
    int unlock(Queue queue, uint32_t oid, BufferList const& buffers, Event& event, WaitList const& wait_list, Measurement::DataPoint& datapoint);

private:

    uint32_t static constexpr DoubleBuffering = 2u;

    struct DeviceInfo {
        struct SlotLock {
            enum SlotLockStatus { Free = 0, ReadLock, WriteLock };
            SlotLockStatus status;
            uint32_t count;
        };

        Context context;
        Device device;
        size_t pool_size;
        size_t num_slots;
        std::vector<SlotLock> slot_lock;
        std::vector<int64_t> cached_object_id;
        std::vector<int64_t> cached_buffer_id;
        std::vector<void*> cached_ptr;
        std::vector<size_t> cached_content_length;
        std::vector<Buffer> device_buffer;
        std::vector<Buffer> host_buffer;
        std::vector<void*> host_ptr;
    };

    struct ObjectInfo {
        void* ptr;
        size_t size;
        ObjectMode mode;
    };

    struct AsyncTask;

    class IOThread {
    public:
        void launch();
        void join();
        static void work(IOThread *io_thread);
        void push_back(AsyncTask *task);

    private:
        std::thread thread;
        std::list<AsyncTask*> tasks;
        std::mutex queue_mutex;
        std::condition_variable queue_cv;
        int queue_locked;

        AsyncTask* pop_front();
        static void async_memcpy(AsyncTask& task);
    };

    struct AsyncTask {
        IOThread *io_thread;
        void *src_ptr;
        void *dst_ptr;
        size_t size;
        WaitList wait_list;
        boost::compute::user_event finish_event;
        Measurement::DataPoint *datapoint;
    };

    std::vector<DeviceInfo> device_info_i;
    std::vector<ObjectInfo> object_info_i;
    IOThread io_thread;

    int evict_cache_slot(Queue queue, uint32_t device_id, uint32_t cache_slot, Event& event, WaitList const& wait_list, Measurement::DataPoint& datapoint);
    int try_read_lock(uint32_t device_id, uint32_t cache_slot);
    int try_write_lock(uint32_t device_id, uint32_t cache_slot);
    int64_t find_device_id(Device device);
    int64_t find_buffer_id(uint32_t device_id, uint32_t oid, void *ptr);
    int64_t find_cache_slot(uint32_t device_id, uint32_t oid, uint32_t buffer_id);
    int64_t assign_cache_slot(uint32_t device_id, uint32_t oid, uint32_t bid);
};

} // namespace Clustering

#endif /* SIMPLE_BUFFER_CACHE_HPP */
