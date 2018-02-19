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

#include <boost/compute/wait_list.hpp>

#define VERBOSE false

using namespace Clustering;
namespace bc = boost::compute;

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
        std::future<std::deque<Event>>& kernel_events,
        Measurement::DataPoint& datapoint
        )
{

    auto runnable = std::make_unique<UnaryRunnable>();
    runnable->kernel_function = kernel_function;
    runnable->object_id = object_id;
    runnable->step = step;
    runnable->datapoint = &datapoint;

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
        std::future<std::deque<Event>>& kernel_events,
        Measurement::DataPoint& datapoint
        )
{
    auto runnable = std::make_unique<BinaryRunnable>();
    runnable->kernel_function = kernel_function;
    runnable->fst_object_id = fst_object_id;
    runnable->snd_object_id = snd_object_id;
    runnable->fst_step = fst_step;
    runnable->snd_step = snd_step;
    runnable->datapoint = &datapoint;

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
    std::deque<std::unique_ptr<RState>> active_rstates;
    for (uint32_t current_index = 0u; current_index < num_buffers; ++current_index) {

        Queue& queue = device_info_i.qpair[current_queue];

        for (auto& runnable : run_queue_i) {
            if (VERBOSE) {
                std::cout << "[Run] Schedule job on queue " << current_queue << std::endl;
            }

            Event run_event;
            std::unique_ptr<RState> rstate_ptr = runnable->create_rstate(queue);

            // TODO: run a small pipeline of jobs and manage cache
            //
            // for (3 runnables per queue)
            // runnable->run()
            //
            // for (first runnable on each queue)
            // runnable->finish()
            // if (next_runnable)
            // next_runnable->run()
            //
            if (runnable->run(rstate_ptr.get(), *buffer_cache_i, current_index, run_event) < 0) {
                return -1;
            }
            if (runnable->finish(rstate_ptr.get(), *buffer_cache_i) < 0) {
                return -1;
            }

            active_rstates.push_back(std::move(rstate_ptr));
        }

    //     // TODO: dual-queue scheduling
        // current_queue = (current_queue + 1) % device_info_i.qpair.size();
    }

    for (auto& runnable : run_queue_i) {
        if (runnable->teardown() < 0) {
            return -1;
        }
    }

    for (auto& queue : device_info_i.qpair) {
        queue.finish();
    }

    run_queue_i.clear();

    return 1;
}

int sds::UnaryRunnable::run(RState *rstate, BufferCache& buffer_cache, uint32_t index, Event& last_event)
{
    int ret = 0;
    bc::wait_list dummy_wait_list;

    UnaryRState *urstate = dynamic_cast<UnaryRState*>(rstate);
    if (not this->datapoint) {
        std::cerr << "[UnaryRunnable::run] nullptr error converting RState" << std::endl;
        return -1;
    }

    void *object_vptr = nullptr;
    char *object_ptr = nullptr;
    size_t object_size = 0;
    buffer_cache.object(this->object_id, object_vptr, object_size);
    object_ptr = (char*) object_vptr;

    size_t offset = step * index;

    size_t end_offset = (offset + step < object_size)
        ? offset + step
        : object_size
        ;

    if (VERBOSE) {
        std::cout << "[UnaryRunnable::run] objectptr: " << (uint64_t)object_ptr << " offset " << offset << " endoffset " << end_offset << std::endl;
    }

    if (not this->datapoint) {
        std::cerr << "[Run] error running UnaryRunnable; datapoint is NULL" << std::endl;
        return -1;
    }

    events.emplace_back();
    Event& transfer_event = events.back();
    ret = buffer_cache.get(
            urstate->queue,
            this->object_id,
            object_ptr + offset,
            object_ptr + end_offset,
            urstate->active_buffers,
            transfer_event,
            dummy_wait_list,
            this->datapoint->create_child()
            );
    if (ret < 0) {
        return -1;
    }
    auto& bdesc = urstate->active_buffers.front();
    last_event = kernel_function(
            urstate->queue,
            0,
            bdesc.content_length,
            bdesc.buffer,
            *this->datapoint
            );
    this->datapoint->add_event() = last_event;
    events.push_back(last_event);

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

std::unique_ptr<sds::RState> sds::UnaryRunnable::create_rstate(Queue queue)
{
    auto urstate_ptr = new UnaryRState;
    urstate_ptr->queue = queue;
    std::unique_ptr<RState> rstate_ptr(urstate_ptr);

    return rstate_ptr;
}

int sds::UnaryRunnable::finish(RState *rstate, BufferCache& buffer_cache)
{
    int ret = 0;
    bc::wait_list dummy_wait_list;

    UnaryRState *urstate = dynamic_cast<UnaryRState*>(rstate);
    if (not this->datapoint) {
        std::cerr << "[UnaryRunnable::finish] nullptr error converting RState" << std::endl;
        return -1;
    }

    events.emplace_back();
    Event& unlock_event = events.back();
    ret = buffer_cache.unlock(
            urstate->queue,
            this->object_id,
            urstate->active_buffers,
            unlock_event,
            dummy_wait_list,
            this->datapoint->create_child()
            );

    if (ret < 0) {
        return -1;
    }

    return 1;
}

int sds::UnaryRunnable::teardown()
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

std::unique_ptr<sds::RState> sds::BinaryRunnable::create_rstate(Queue queue)
{
    auto brstate_ptr = new BinaryRState;
    brstate_ptr->queue = queue;
    std::unique_ptr<RState> rstate_ptr(brstate_ptr);

    return rstate_ptr;
}

int sds::BinaryRunnable::run(RState *rstate, BufferCache& buffer_cache, uint32_t index, Event& last_event)
{
    int ret = 0;
    bc::wait_list dummy_wait_list;

    BinaryRState *brstate = dynamic_cast<BinaryRState*>(rstate);
    if (not this->datapoint) {
        std::cerr << "[BinaryRunnable::run] nullptr error converting RState" << std::endl;
        return -1;
    }

    void *fst_object_vptr = nullptr, *snd_object_vptr = nullptr;
    char *fst_object_ptr = nullptr, *snd_object_ptr = nullptr;
    size_t fst_object_size = 0, snd_object_size = 0;
    buffer_cache.object(this->fst_object_id, fst_object_vptr, fst_object_size);
    buffer_cache.object(this->snd_object_id, snd_object_vptr, snd_object_size);
    fst_object_ptr = (char*) fst_object_vptr;
    snd_object_ptr = (char*) snd_object_vptr;

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

    if (not this->datapoint) {
        std::cerr << "[Run] error running BinaryRunnable; datapoint is NULL" << std::endl;
        return -1;
    }

    events.emplace_back();
    Event& fst_transfer_event = events.back();
    ret = buffer_cache.get(
            brstate->queue,
            this->fst_object_id,
            fst_object_ptr + fst_offset,
            fst_object_ptr + fst_end_offset,
            brstate->fst_active_buffers,
            fst_transfer_event,
            dummy_wait_list,
            this->datapoint->create_child()
            );
    if (ret < 0) {
        return -1;
    }

    events.emplace_back();
    Event& snd_transfer_event = events.back();
    ret = buffer_cache.get(
            brstate->queue,
            this->snd_object_id,
            snd_object_ptr + snd_offset,
            snd_object_ptr + snd_end_offset,
            brstate->snd_active_buffers,
            snd_transfer_event,
            dummy_wait_list,
            this->datapoint->create_child()
            );
    if (ret < 0) {
        return -1;
    }

    auto& fst_bdesc = brstate->fst_active_buffers.front();
    auto& snd_bdesc = brstate->snd_active_buffers.front();
    last_event = kernel_function(
            brstate->queue,
            0,
            fst_bdesc.content_length,
            snd_bdesc.content_length,
            fst_bdesc.buffer,
            snd_bdesc.buffer,
            *this->datapoint
            );
    this->datapoint->add_event() = last_event;
    events.push_back(last_event);

    return 1;
}

int sds::BinaryRunnable::finish(RState *rstate, BufferCache& buffer_cache)
{
    int ret = 0;
    bc::wait_list dummy_wait_list;

    BinaryRState *brstate = dynamic_cast<BinaryRState*>(rstate);
    if (not this->datapoint) {
        std::cerr << "[BinaryRunnable::finish] nullptr error converting RState" << std::endl;
        return -1;
    }

    events.emplace_back();
    Event& fst_unlock_event = events.back();
    ret = buffer_cache.unlock(
            brstate->queue,
            this->fst_object_id,
            brstate->fst_active_buffers,
            fst_unlock_event,
            dummy_wait_list,
            this->datapoint->create_child()
            );
    if (ret < 0) {
        return -1;
    }

    events.emplace_back();
    Event& snd_unlock_event = events.back();
    ret = buffer_cache.unlock(
            brstate->queue,
            this->snd_object_id,
            brstate->snd_active_buffers,
            snd_unlock_event,
            dummy_wait_list,
            this->datapoint->create_child()
            );
    if (ret < 0) {
        return -1;
    }

    return 1;
}

int sds::BinaryRunnable::teardown()
{
    events_promise.set_value(std::move(events));

    return 1;
}
