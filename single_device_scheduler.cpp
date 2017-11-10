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

    auto runnable = std::make_unique<UnaryRunnable>();
    runnable->kernel_function = kernel_function;
    runnable->object_id = object_id;

    kernel_events = runnable->events_promise.get_future();

    run_queue_i.push_back(std::move(runnable));

    return 1;
}

int sds::enqueue(
        FunBinary function,
        uint32_t fst_object_id,
        uint32_t snd_object_id,
        std::future<std::deque<Event>>& kernel_events
        )
{
    return -1;
}

int sds::enqueue_barrier()
{
    return -1;
}

int sds::run()
{
    uint32_t num_buffers = 0;
    for (auto& runnable : run_queue_i) {
        auto n = runnable->register_buffers(*buffer_cache_i);

        if (num_buffers == 0) {
            num_buffers = n;
        }
        else if (n == 0) {
            // barrier -> skip
        }
        else if (n != num_buffers) {
            std::cerr << "[Run] number of requested iterations differs" << std::endl;
            return -1;
        }
        else {
            // n == num_buffers -> ok
        }
    }

    uint32_t current_queue = 0;
    for (uint32_t current_index = 0u; current_index < num_buffers; ++current_index) {

        Queue& queue = device_info_i.qpair[current_queue];

        for (auto& runnable : run_queue_i) {
            auto ret = runnable->run(queue, *buffer_cache_i, current_index);
            if (ret < 0) {
                return -1;
            }
        }

    //     // TODO: dual-queue scheduling
        // current_queue = (current_queue + 1) % device_info_i.qpair.size();
    }

    for (auto& runnable : run_queue_i) {
        auto ret = runnable->finish();
        if (ret < 0) {
            return -1;
        }
    }

    for (auto& queue : device_info_i.qpair) {
        queue.finish();
    }

    run_queue_i.clear();

    return 1;
}

int sds::UnaryRunnable::run(Queue queue, BufferCache& buffer_cache, uint32_t index)
{
    int ret = 0;

    void *object_vptr = nullptr;
    char *object_ptr = nullptr;
    size_t object_size = 0;
    buffer_cache.object(object_id, object_vptr, object_size);
    object_ptr = (char*) object_vptr;

    size_t buffer_size = buffer_cache.buffer_size();

    BufferCache::BufferList buffers;
    size_t offset = buffer_size * index;

    size_t end_offset = (offset + buffer_size < object_size)
        ? offset + buffer_size
        : object_size
        ;

    if (VERBOSE) {
        std::cout << "[UnaryRunnable::run] objectptr: " << (uint64_t)object_ptr << " offset " << offset << " endoffset " << end_offset << std::endl;
    }

    events.emplace_back();
    Event& transfer_event = events.back();

    ret = buffer_cache.get(
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
            0,
            bdesc.content_length,
            bdesc.buffer
            );

    events.emplace_back();
    Event& unlock_event = events.back();
    buffer_cache.unlock(
            queue,
            object_id,
            buffers,
            unlock_event
            );

    return 1;
}

uint32_t sds::UnaryRunnable::register_buffers(BufferCache& buffer_cache)
{
    auto buffer_size = buffer_cache.buffer_size();
    size_t object_size = 0;
    {
        void *ptr = nullptr;
        buffer_cache.object(object_id, ptr, object_size);
    }

    return (object_size + buffer_size - 1) / buffer_size;
}

int sds::UnaryRunnable::finish()
{
    events_promise.set_value(std::move(events));

    return 1;
}

uint32_t sds::BinaryRunnable::register_buffers(BufferCache& buffer_cache)
{
    return -1;
}

int sds::BinaryRunnable::run(Queue queue, BufferCache& buffer_cache, uint32_t index)
{
    return -1;
}

int sds::BinaryRunnable::finish()
{
    return -1;
}
