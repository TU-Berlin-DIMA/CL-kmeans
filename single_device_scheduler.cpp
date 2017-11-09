/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2017, Lutz, Clemens <lutzcle@cml.li>
 */

#include <single_device_scheduler.hpp>

#include <future>
#include <deque>
#include <iostream>

#define VERBOSE false

using namespace Clustering;

using sds = SingleDeviceScheduler;

int sds::add_buffer_cache(std::shared_ptr<BufferCache> buffer_cache)
{
    buffer_cache_i = buffer_cache;

    return 1;
}

int sds::add_device(Context context, Device device)
{
    device_info_i = {{{Queue(context, device), Queue(context, device)}}};

    return 1;
}

int sds::enqueue(
        FunUnary kernel_function,
        uint32_t object_id,
        std::future<std::deque<Event>>& kernel_events
        )
{
    int ret = 0;

    std::promise<std::deque<Event>> promise;
    kernel_events = promise.get_future();
    std::deque<Event> events;

    void *object_vptr = nullptr;
    char *object_ptr = nullptr;
    size_t object_size = 0;
    buffer_cache_i->object(object_id, object_vptr, object_size);
    object_ptr = (char*) object_vptr;

    size_t buffer_size = buffer_cache_i->buffer_size();

    BufferCache::BufferList buffers;
    uint32_t next_queue = 0;
    for (size_t offset = 0; offset < object_size; offset += buffer_size) {

        Queue& queue = device_info_i.qpair[next_queue];
        size_t end_offset = (offset + buffer_size < object_size)
            ? offset + buffer_size
            : object_size
            ;

        if (VERBOSE) {
            std::cout << "objectptr: " << (uint64_t)object_ptr << " offset " << offset << " endoffset " << end_offset << std::endl;
        }

        events.emplace_back();
        Event& transfer_event = events.back();
        ret = buffer_cache_i->get(
                queue,
                object_id,
                object_ptr + offset,
                object_ptr + end_offset,
                buffers,
                transfer_event
                );
        if (ret < 0) {
            return -1;
        }
        auto& bdesc = buffers.front();
        events.emplace_back();
        events.back() = kernel_function(
                queue,
                bdesc.content_length,
                0,
                bdesc.buffer
                );

        events.emplace_back();
        Event& unlock_event = events.back();
        buffer_cache_i->unlock(
                queue,
                object_id,
                buffers,
                unlock_event
                );

        // TODO: dual-queue scheduling
        // next_queue = next_queue % device_info_i.qpair.size();
        buffers.clear();
    }

    promise.set_value(std::move(events));

    return 1;
}

int sds::enqueue(
        FunBinary function,
        uint32_t fst_object_id,
        uint32_t snd_object_id,
        std::future<std::deque<Event>>& kernel_events
        )
{
    return 0;
}

int sds::enqueue_barrier()
{
    return 0;
}
