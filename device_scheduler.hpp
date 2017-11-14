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
         * 3. Buffer content length(s) in bytes
         * 4. Boost::Compute Buffer(s)
         *
         * The function shall return a Boost::Compute Event.
         *
         * Enqueable functions must be reenterable.
         */
        using FunUnary = std::function<Event(Queue, size_t, size_t, Buffer)>;
        using FunBinary = std::function<Event(Queue, size_t, size_t, size_t, Buffer, Buffer)>;

        virtual ~DeviceScheduler() {};

        /*
         * Add a BufferCache.
         * At least one BufferCache is required.
         * Some DeviceScheduler implementations may support multiple BufferCache
         * objects.
         */
        virtual int add_buffer_cache(std::shared_ptr<BufferCache> buffer_cache) = 0;

        /*
         * Add a OpenCL device.
         * At least one device is required.
         * Some DeviceScheduler implementations may support multiple devices.
         */
        virtual int add_device(Context context, Device device) = 0;

        /*
         * Run all enqueued functions.
         * Blocking with eager evaluation.
         */
        virtual int run() = 0;

        /*
         * Enqueue a function.
         * Lazy and returns immediately.
         *
         * Given function will be passed buffer(s) from object(s) with the given
         * object_id(s). Buffers will have maximum content_length of size "step"
         * bytes. Object(s) will be processed exactly once.
         *
         * Note that for n-ary functions with n > 1, "step" must be defined such
         * that all objects consist of an equal number of "step"-sized buffers.
         * Buffers with corresponding indices will be passed at the same time,
         * as in: map[f(fst(x), snd(x)), with x = zip(fst_object, snd_object)]
         */
        virtual int enqueue(
                FunUnary kernel_function,
                uint32_t object_id,
                size_t step,
                std::future<std::deque<Event>>& kernel_events
                ) = 0;
        virtual int enqueue(
                FunBinary kernel_function,
                uint32_t fst_object_id,
                uint32_t snd_object_id,
                size_t fst_step,
                size_t snd_step,
                std::future<std::deque<Event>>& kernel_events
                ) = 0;

        /*
         * Enqueue a barrier.
         * Lazy and returns immediately.
         *
         * The barrier blocks processing of the next function in the queue until
         * all instances of the previous function have finished processing.
         */
        virtual int enqueue_barrier() = 0;
    };
}

#endif /* DEVICE_SCHEDULER_HPP */
