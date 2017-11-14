/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2017, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef SINGLE_DEVICE_SCHEDULER_HPP
#define SINGLE_DEVICE_SCHEDULER_HPP

#include <buffer_cache.hpp>
#include <device_scheduler.hpp>

#include <array>
#include <functional>
#include <future>
#include <deque>
#include <memory>
#include <cstdint>

#include <boost/compute/buffer.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/event.hpp>

namespace Clustering {
    class SingleDeviceScheduler : public DeviceScheduler {
    public:

        using Buffer = boost::compute::buffer;
        using Context = boost::compute::context;
        using Device = boost::compute::device;
        using Event = boost::compute::event;
        using FunUnary = typename DeviceScheduler::FunUnary;
        using FunBinary = typename DeviceScheduler::FunBinary;
        using Queue = boost::compute::command_queue;

        int add_buffer_cache(std::shared_ptr<BufferCache> buffer_cache);
        int add_device(Context context, Device device);

        int run();

        int enqueue(
                FunUnary kernel_function,
                uint32_t object_id,
                size_t step,
                std::future<std::deque<Event>>& kernel_events
                );
        int enqueue(
                FunBinary kernel_function,
                uint32_t fst_object_id,
                uint32_t snd_object_id,
                size_t fst_step,
                size_t snd_step,
                std::future<std::deque<Event>>& kernel_events
                );
        int enqueue_barrier();

    private:

        struct DeviceInfo {
            std::array<Queue, 2> qpair;
        } device_info_i;

        struct Runnable {
            virtual int64_t register_buffers(BufferCache& buffer_cache) = 0;
            virtual int run(Queue queue, BufferCache& buffer_cache, uint32_t index) = 0;
            virtual int finish() = 0;
        };

        struct UnaryRunnable : public Runnable {
            int64_t register_buffers(BufferCache& buffer_cache);
            int run(Queue queue, BufferCache& buffer_cache, uint32_t index);
            int finish();
            FunUnary kernel_function;
            uint32_t object_id;
            std::deque<Event> events;
            std::promise<std::deque<Event>> events_promise;

        };

        struct BinaryRunnable : public Runnable {
            int64_t register_buffers(BufferCache& buffer_cache);
            int run(Queue queue, BufferCache& buffer_cache, uint32_t index);
            int finish();
            FunBinary kernel_function;
            uint32_t fst_object_id;
            uint32_t snd_object_id;
            std::deque<Event> events;
            std::promise<std::deque<Event>> events_promise;
        };

        std::shared_ptr<BufferCache> buffer_cache_i;
        std::deque<std::unique_ptr<Runnable>> run_queue_i;
    };
} // namespace Clustering

#endif /* SINGLE_DEVICE_SCHEDULER_HPP */
