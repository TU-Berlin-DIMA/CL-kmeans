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

sds::SingleDeviceScheduler()
    :
        DeviceScheduler()
{
}

sds::SingleDeviceScheduler(SingleDeviceScheduler const& other)
    :
        DeviceScheduler(other),
        buffer_cache_i(other.buffer_cache_i),
        run_queue_i()
{
}

int sds::add_buffer_cache(std::shared_ptr<BufferCache> buffer_cache)
{
    buffer_cache_i = buffer_cache;

    return 1;
}

int sds::add_device(Context context, Device device)
{
    device_info_i = {{{
        Queue(context, device, Queue::enable_profiling),
        Queue(context, device, Queue::enable_profiling)
    }}};

    return 1;
}

int sds::enqueue(
        FunUnary kernel_function,
        uint32_t object_id,
        size_t step,
        std::future<std::deque<Event>>& kernel_events
        )
{

    auto runnable = std::make_unique<UnaryRunnable>();
    runnable->kernel_function = kernel_function;
    runnable->object_id = object_id;
    runnable->step = step;

    kernel_events = runnable->events_promise.get_future();

    run_queue_i.push_back(std::move(runnable));

    return 1;
}

int sds::enqueue(
        FunBinary kernel_function,
        uint32_t fst_object_id,
        uint32_t snd_object_id,
        size_t fst_step,
        size_t snd_step,
        std::future<std::deque<Event>>& kernel_events
        )
{
    auto runnable = std::make_unique<BinaryRunnable>();
    runnable->kernel_function = kernel_function;
    runnable->fst_object_id = fst_object_id;
    runnable->snd_object_id = snd_object_id;
    runnable->fst_step = fst_step;
    runnable->snd_step = snd_step;

    kernel_events = runnable->events_promise.get_future();

    run_queue_i.push_back(std::move(runnable));

    return 1;
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

        if (n < 0) {
            std::cerr << "[Run] error in registering buffers; maybe lengths of n-ary function's buffers do not match?" << std::endl;
            return -1;
        }
        else if (n == 0) {
            // barrier -> skip
        }
        else if (num_buffers == 0) {
            num_buffers = (uint32_t)n;
        }
        else if ((uint32_t)n != num_buffers) {
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

    BufferCache::BufferList buffers;
    size_t offset = step * index;

    size_t end_offset = (offset + step < object_size)
        ? offset + step
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
    ret = buffer_cache.unlock(
            queue,
            object_id,
            buffers,
            unlock_event
            );
    if (ret < 0) {
        return -1;
    }

    return 1;
}

int64_t sds::UnaryRunnable::register_buffers(BufferCache& buffer_cache)
{
    size_t object_size = 0;
    {
        void *ptr = nullptr;
        buffer_cache.object(object_id, ptr, object_size);
    }

    return (object_size + step - 1) / step;
}

int sds::UnaryRunnable::finish()
{
    events_promise.set_value(std::move(events));

    return 1;
}

int64_t sds::BinaryRunnable::register_buffers(BufferCache& buffer_cache)
{
    size_t fst_object_size = 0, snd_object_size = 0;
    {
        void *ptr = nullptr;
        buffer_cache.object(fst_object_id, ptr, fst_object_size);
        buffer_cache.object(snd_object_id, ptr, snd_object_size);
    }

    auto fst_num = (fst_object_size + fst_step - 1) / fst_step;
    auto snd_num = (snd_object_size + snd_step - 1) / snd_step;

    return (fst_num == snd_num) ? fst_num : -1;
}

int sds::BinaryRunnable::run(Queue queue, BufferCache& buffer_cache, uint32_t index)
{
    int ret = 0;

    void *fst_object_vptr = nullptr, *snd_object_vptr = nullptr;
    char *fst_object_ptr = nullptr, *snd_object_ptr = nullptr;
    size_t fst_object_size = 0, snd_object_size = 0;
    buffer_cache.object(fst_object_id, fst_object_vptr, fst_object_size);
    buffer_cache.object(snd_object_id, snd_object_vptr, snd_object_size);
    fst_object_ptr = (char*) fst_object_vptr;
    snd_object_ptr = (char*) snd_object_vptr;

    BufferCache::BufferList fst_buffers, snd_buffers;
    auto fst_offset = fst_step * index;
    auto snd_offset = snd_step * index;

    auto fst_end_offset = (fst_offset + fst_step < fst_object_size)
        ? fst_offset + fst_step
        : fst_object_size
        ;
    auto snd_end_offset = (snd_offset + snd_step < snd_object_size)
        ? snd_offset + snd_step
        : snd_object_size
        ;

    if (VERBOSE) {
        std::cout << "[BinaryRunnable::run] fst_object_id: " << fst_object_id << " snd_object_id " << snd_object_id << " fst_offset " << fst_offset << " fst_step " << fst_end_offset - fst_offset << " snd_offset " << snd_offset << " snd_step " << snd_end_offset - snd_offset << std::endl;
    }

    events.emplace_back();
    Event& fst_transfer_event = events.back();
    ret = buffer_cache.get(
            queue,
            fst_object_id,
            fst_object_ptr + fst_offset,
            fst_object_ptr + fst_end_offset,
            fst_buffers,
            fst_transfer_event
            );
    if (ret < 0) {
        return -1;
    }

    events.emplace_back();
    Event& snd_transfer_event = events.back();
    ret = buffer_cache.get(
            queue,
            snd_object_id,
            snd_object_ptr + snd_offset,
            snd_object_ptr + snd_end_offset,
            snd_buffers,
            snd_transfer_event
            );
    if (ret < 0) {
        return -1;
    }

    auto& fst_bdesc = fst_buffers.front();
    auto& snd_bdesc = snd_buffers.front();
    events.emplace_back();
    events.back() = kernel_function(
            queue,
            0,
            fst_bdesc.content_length,
            snd_bdesc.content_length,
            fst_bdesc.buffer,
            snd_bdesc.buffer
            );

    events.emplace_back();
    Event& fst_unlock_event = events.back();
    ret = buffer_cache.unlock(
            queue,
            fst_object_id,
            fst_buffers,
            fst_unlock_event
            );
    if (ret < 0) {
        return -1;
    }

    events.emplace_back();
    Event& snd_unlock_event = events.back();
    ret = buffer_cache.unlock(
            queue,
            snd_object_id,
            snd_buffers,
            snd_unlock_event
            );
    if (ret < 0) {
        return -1;
    }

    return 1;
}

int sds::BinaryRunnable::finish()
{
    events_promise.set_value(std::move(events));

    return 1;
}
