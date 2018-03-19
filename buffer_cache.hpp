/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2017-2018, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef BUFFER_CACHE_HPP
#define BUFFER_CACHE_HPP

#include <cstdint>
#include <vector>

#include <boost/compute/buffer.hpp>
#include <boost/compute/device.hpp>
#include <boost/compute/event.hpp>
#include <boost/compute/command_queue.hpp>
#include <boost/compute/utility/wait_list.hpp>

#include "measurement/measurement.hpp"

namespace Clustering {

/*
 * Modes define how objects are treated on access and cache eviction:
 *
 * ReadWrite: Copied to device on access and copied from device on eviction.
 * ReadOnly: Copied to device on access and dropped on eviction.
 * Transient: Instantiated on access and dropped on eviction.
 *
 * Note: Transient mode useful for e.g. pipelines while object is locked in cache.
 */
enum class ObjectMode {
    ReadWrite,
    ReadOnly,
    Transient
};

class BufferCache {
public:

    class BufferDesc;

    using Buffer = boost::compute::buffer;
    using BufferList = std::vector<BufferDesc>;
    using Context = boost::compute::context;
    using Device = boost::compute::device;
    using Event = boost::compute::event;
    using Queue = boost::compute::command_queue;
    using WaitList = boost::compute::wait_list;

    class BufferDesc {
    public:
        Buffer buffer;
        size_t content_length;
        size_t buffer_id; /* internal data */
    };

    /*
     * Construct BufferCache with buffer_size in bytes.
     *
     * buffer_size must not be larger than the smallest pool_size.
     */
    BufferCache(size_t buffer_size) :
        buffer_size_i(buffer_size)
    {}
    virtual ~BufferCache()
    {}

    /*
     * Returns buffer size in bytes.
     */
    size_t buffer_size() { return buffer_size_i; }

    /*
     * Returns pool size of the device in bytes.
     */
    virtual size_t pool_size(Device device) = 0;

    /*
     * Add device for BufferCache to manage with pool_size in bytes.
     * pool_size must fit into device memory.
     *
     * Returns 1 if successful, negative value if unsuccessful.
     */
    virtual int add_device(Context context, Device device, size_t pool_size) = 0;

    /*
     * Add data object for buffer cache to manage.
     * Note that BufferCache captures the object, but does not manage
     * its life-cycle.
     *
     * Returns new object id (oid).
     */
    virtual uint32_t add_object(void *data_object, size_t length, ObjectMode mode = ObjectMode::ReadOnly) = 0;

    /*
     * Get pointer to previously added data object.
     * Does not transfer ownership of object.
     */
    virtual void object(uint32_t object_id, void *& data_object, size_t& length) = 0;

    /*
     * Returns a list devices on which the given range is currently in device memory. Default length is one buffer.
     */
    virtual std::vector<Device> where_is(uint32_t /* object_id */, void * /* begin */) { return std::vector<Device>(); };
    virtual std::vector<Device> where_is(uint32_t /* object_id */, void * /* begin */, void * /* end */) { return std::vector<Device>(); };

    /*
     * Get locked device buffer of object at location of pointer.
     * Forces asynchronous write if buffer not cached on device.
     * User shall unlock buffer after use.
     *
     * Returns 1 if successful, negative value if unsuccessful.
     */
    virtual int get(Queue queue, uint32_t object_id, void *begin, void *end, BufferList& buffers, Event& event, WaitList const& wait_list, Measurement::DataPoint& datapoint) = 0;

    /*
     * Asynchronously write buffer at offset from host to device and get locked buffer at location of pointer.
     * User shall unlock buffer after use.
     *
     * Returns 1 if successful, negative value if unsucessful.
     */
    virtual int write_and_get(Queue queue, uint32_t object_id, void *begin, void *end, BufferList& buffers, Event& event, WaitList const& wait_list, Measurement::DataPoint& datapoint) = 0;

    /*
     * Asynchronously read buffer at location of pointer from device to host.
     *
     * Returns 1 if successful, negative value if unsuccessful.
     */
    virtual int read(Queue queue, uint32_t object_id, void *begin, void *end, Event& event, WaitList const& wait_list, Measurement::DataPoint& datapoint) = 0;

    /*
     * Asynchronously write buffer at location of pointer from src device to dst device and get locked buffer at offset.
     * dst and src must not be on same device.
     *
     * Returns 1 if successful, negative value if unsuccessful.
     */
    virtual int sync_and_get(Queue dst, Queue src, uint32_t object_id, void *begin, void *end, Event& event, WaitList const& wait_list, Measurement::DataPoint& datapoint) = 0;

    /*
     * Locking prevents eviction of buffer at location of pointer on device. Necessary during kernel execution.
     */
    virtual int unlock(Queue queue, uint32_t object_id, BufferList const& buffers, Event& event, WaitList const& wait_list, Measurement::DataPoint& datapoint) = 0;

protected:

    size_t buffer_size_i;
};

} // namespace Clustering

#endif /* BUFFER_CACHE_HPP */
