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
    std::deque<RState> active_rstates;
    Event trans_loop_run_event;
    for (uint32_t current_index = 0u; current_index < num_buffers; ++current_index) {

        Event run_event;
        Event activate_event;
        Event deactivate_event;
        Event const empty_event;

        if (active_rstates.size() >= device_info_i.qpair.size()) {

            auto& first_rstate = active_rstates.front();
            WaitList deactivate_wait_list(first_rstate.last_event());
            for (auto& runnable : run_queue_i) {
                int ret = 0;
                runnable->deactivate_buffers(
                        first_rstate,
                        *buffer_cache_i,
                        deactivate_wait_list,
                        deactivate_event
                        );
                if (ret < 0) {
                    return -1;
                }

                deactivate_wait_list = WaitList(deactivate_event);
            }
            active_rstates.pop_front();
        }

        Queue& queue = device_info_i.qpair[current_queue];
        RState active_rstate(queue);

        // Prepare buffers for runnables
        WaitList activate_wait_list;
        if (deactivate_event != empty_event) {
            activate_wait_list.insert(deactivate_event);
        }
        for (auto& runnable : run_queue_i) {
            int ret = 0;
            ret = runnable->activate_buffers(
                    active_rstate,
                    *buffer_cache_i,
                    current_index,
                    activate_wait_list,
                    activate_event
                    );
            if (ret < 0) {
                return -1;
            }

            // activate_event is empty when buffer is in cache
            // in that case, don't modify the wait list
            // only if we have a new event, then create a new wait list
            if (activate_event != empty_event) {
                activate_wait_list = WaitList(activate_event);
            }
        }

        WaitList run_wait_list = activate_wait_list;
        if (trans_loop_run_event != empty_event) {
            run_wait_list.insert(trans_loop_run_event);
        }
        for (auto& runnable : run_queue_i) {
            if (VERBOSE) {
                std::cout << "[Run] Schedule job on queue " << current_queue << std::endl;
            }

            if (runnable->run(
                        active_rstate,
                        *buffer_cache_i,
                        current_index,
                        run_wait_list,
                        run_event
                        ) < 0)
            {
                return -1;
            }

            run_wait_list = WaitList(run_event);
        }

        trans_loop_run_event = run_event;
        active_rstate.last_event(run_event);
        active_rstates.push_back(std::move(active_rstate));
        current_queue = (current_queue + 1) % device_info_i.qpair.size();
    }

    while (not active_rstates.empty()) {

        auto& first_rstate = active_rstates.front();
        Event deactivate_event;
        WaitList deactivate_wait_list(first_rstate.last_event());

        for (auto& runnable : run_queue_i) {
            int ret = 0;
            ret = runnable->deactivate_buffers(
                    first_rstate,
                    *buffer_cache_i,
                    deactivate_wait_list,
                    deactivate_event
                    );

            if (ret < 0) {
                return -1;
            }

            deactivate_wait_list = WaitList(deactivate_event);
        }
        active_rstates.pop_front();
    }

    for (auto& runnable : run_queue_i) {
        if (runnable->finish() < 0) {
            return -1;
        }
    }

    for (auto& queue : device_info_i.qpair) {
        queue.finish();
    }

    run_queue_i.clear();

    return 1;
}

sds::RState::RState(Queue queue)
    : queue_i(queue)
{}

sds::Queue sds::RState::queue()
{
    return this->queue_i;
}

void sds::RState::last_event(Event event) {
    this->last_event_i = event;
}

sds::Event sds::RState::last_event() {
    return this->last_event_i;
}

BufferCache::BufferList& sds::RState::active_buffers(uint32_t object_id)
{
    return this->active_buffers_i.at(object_id);
}

int sds::RState::activate_buffers(uint32_t object_id, size_t runnable_step, BufferCache& buffer_cache, uint32_t index, WaitList wait_list, std::deque<Event>& events, Event& last_event, Measurement::DataPoint& datapoint)
{
    void *object_vptr = nullptr;
    char *object_ptr = nullptr;
    size_t object_size = 0;
    buffer_cache.object(object_id, object_vptr, object_size);
    object_ptr = (char*) object_vptr;

    size_t offset = runnable_step * index;

    size_t end_offset = (offset + runnable_step < object_size)
        ? offset + runnable_step
        : object_size
        ;

    if (VERBOSE) {
        std::cout << "[RState::activate_buffers] objectptr: " << (uint64_t)object_ptr << " offset " << offset << " endoffset " << end_offset << std::endl;
    }

    auto& ab = this->active_buffers_i;
    auto it = ab.find(object_id);
    if (it == ab.end()) {
        auto& buffers = ab[object_id];

        events.emplace_back();
        Event& transfer_event = events.back();
        int ret = buffer_cache.get(
                this->queue_i,
                object_id,
                object_ptr + offset,
                object_ptr + end_offset,
                buffers,
                transfer_event,
                wait_list,
                datapoint
                );
        last_event = transfer_event;

        if (ret < 0) {
            return -1;
        }
    }

    return 1;
}

int sds::RState::deactivate_buffers(uint32_t object_id, BufferCache& buffer_cache, WaitList wait_list, std::deque<Event>& events, Event& last_event, Measurement::DataPoint& datapoint)
{
    auto& ab = this->active_buffers_i;
    auto it = ab.find(object_id);
    if (it != ab.end()) {
        auto& buffers = it->second;

        events.emplace_back();
        Event& unlock_event = events.back();
        int ret = buffer_cache.unlock(
                this->queue_i,
                object_id,
                buffers,
                unlock_event,
                wait_list,
                datapoint
                );
        last_event = unlock_event;

        if (ret < 0) {
            return -1;
        }

        ab.erase(it);
    }

    return 1;
}

int sds::UnaryRunnable::activate_buffers(RState& rstate, BufferCache& buffer_cache, uint32_t index, WaitList wait_list, Event& last_event)
{
    if (not this->datapoint) {
        std::cerr << "[UnaryRunnable::activate_buffers] error: datapoint is NULL" << std::endl;
        return -1;
    }

    return rstate.activate_buffers(
            this->object_id,
            this->step,
            buffer_cache,
            index,
            wait_list,
            this->events,
            last_event,
            this->datapoint->create_child()
            );
}

int sds::UnaryRunnable::deactivate_buffers(RState& rstate, BufferCache& buffer_cache, WaitList wait_list, Event& last_event)
{
    return rstate.deactivate_buffers(
            this->object_id,
            buffer_cache,
            wait_list,
            this->events,
            last_event,
            this->datapoint->create_child()
            );
}

int sds::UnaryRunnable::run(RState& rstate, BufferCache&, uint32_t, WaitList wait_list, Event& last_event)
{
    if (not this->datapoint) {
        std::cerr << "[UnaryRunnable::run] error: datapoint is NULL" << std::endl;
        return -1;
    }

    auto& bdesc = rstate.active_buffers(this->object_id).front();
    last_event = kernel_function(
            rstate.queue(),
            0,
            bdesc.content_length,
            bdesc.buffer,
            wait_list,
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

int sds::BinaryRunnable::activate_buffers(RState& rstate, BufferCache& buffer_cache, uint32_t index, WaitList wait_list, Event& last_event)
{
    int ret = 0;

    if (not this->datapoint) {
        std::cerr << "[BinaryRunnable::activate_buffers] error: datapoint is NULL" << std::endl;
        return -1;
    }

    Event fst_event;
    ret = rstate.activate_buffers(
            this->fst_object_id,
            this->fst_step,
            buffer_cache,
            index,
            wait_list,
            this->events,
            fst_event,
            this->datapoint->create_child()
            );
    if (ret < 0) {
        std::cerr << "[BinaryRunnable::activate_buffers] error: could not activate fst buffer" << std::endl;
        return -1;
    }

    WaitList snd_wait_list(fst_event);
    ret = rstate.activate_buffers(
            this->snd_object_id,
            this->snd_step,
            buffer_cache,
            index,
            snd_wait_list,
            this->events,
            last_event,
            this->datapoint->create_child()
            );
    if (ret < 0) {
        std::cerr << "[BinaryRunnable::activate_buffers] error: could not activate snd buffer" << std::endl;
        return -1;
    }

    return 1;
}

int sds::BinaryRunnable::deactivate_buffers(RState& rstate, BufferCache& buffer_cache, WaitList wait_list, Event& last_event)
{
    int ret = 0;

    Event fst_event;
    ret = rstate.deactivate_buffers(
            this->fst_object_id,
            buffer_cache,
            wait_list,
            this->events,
            fst_event,
            this->datapoint->create_child()
            );
    if (ret < 0) {
        std::cerr << "[BinaryRunnable::deactivate_buffers] error: could not deactivate fst buffer" << std::endl;
        return -1;
    }

    WaitList snd_wait_list(fst_event);
    ret = rstate.deactivate_buffers(
            this->snd_object_id,
            buffer_cache,
            snd_wait_list,
            this->events,
            last_event,
            this->datapoint->create_child()
            );
    if (ret < 0) {
        std::cerr << "[BinaryRunnable::deactivate_buffers] error: could not deactivate snd buffer" << std::endl;
        return -1;
    }

    return 1;
}

int sds::BinaryRunnable::run(RState& rstate, BufferCache&, uint32_t, WaitList wait_list, Event& last_event)
{
    if (not this->datapoint) {
        std::cerr << "[Run] error running BinaryRunnable; datapoint is NULL" << std::endl;
        return -1;
    }

    auto& fst_bdesc = rstate.active_buffers(this->fst_object_id).front();
    auto& snd_bdesc = rstate.active_buffers(this->snd_object_id).front();
    last_event = kernel_function(
            rstate.queue(),
            0,
            fst_bdesc.content_length,
            snd_bdesc.content_length,
            fst_bdesc.buffer,
            snd_bdesc.buffer,
            wait_list,
            *this->datapoint
            );
    this->datapoint->add_event() = last_event;
    events.push_back(last_event);

    return 1;
}

int sds::BinaryRunnable::finish()
{
    events_promise.set_value(std::move(events));

    return 1;
}
