/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2017, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef DEVICE_SCHEDULER_HPP
#define DEVICE_SCHEDULER_HPP

#include <buffer_cache.hpp>

#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>

#include <boost/compute/buffer.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/context.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/event.hpp>

namespace Clustering {
    class DeviceScheduler {
    public:

        using Buffer = boost::compute::buffer;
        using Context = boost::compute::context;
        using Device = boost::compute::device;
        using Event = boost::compute::event;
        using Queue = boost::compute::command_queue;

        /*
         * Function signatures of enqueable functions.
         *
         * The parameters are:
         * 1. Boost::Compute CommandQueue
         * 2. OpenCL offset in bytes
         * 3. Buffer size(s) in bytes
         * 4. Boost::Compute Buffer(s)
         *
         * The function shall return a Boost::Compute Event.
         *
         * Enqueable functions must be reenterable.
         */
        using FunUnary = std::function<Event(Queue, size_t, size_t, Buffer)>;
        using FunBinary = std::function<Event(Queue, size_t, size_t, size_t, Buffer, Buffer)>;

        virtual ~DeviceScheduler() {};

        virtual int add_buffer_cache(std::shared_ptr<BufferCache> buffer_cache) = 0;

        virtual int add_device(Context context, Device device) = 0;

        virtual int run() = 0;

        virtual int enqueue(FunUnary kernel_function, uint32_t object_id, std::future<std::deque<Event>>& kernel_events) = 0;
        virtual int enqueue(FunBinary function, uint32_t fst_object_id, uint32_t snd_object_id, std::future<std::deque<Event>>& kernel_events) = 0;
        virtual int enqueue_barrier() = 0;
    };
}

#endif /* DEVICE_SCHEDULER_HPP */
