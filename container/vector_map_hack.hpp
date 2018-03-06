/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifndef VECTOR_MAP_HACK_HPP
#define VECTOR_MAP_HACK_HPP

#include <cstdint>

#include <boost/compute/container/vector.hpp>
#include <boost/compute/allocator/pinned_allocator.hpp>

/*
 * Map and transport vector from one OpenCL vendor context to another
 *
 * OpenCL buffers are constrained to one context. This is a problem
 * when using distinct contexts for both a GPU and and a CPU, because
 * reading / writing the buffer involves an additional memcopy (i.e.
 * from GPU pinned buffer to CPU buffer). Boost::Compute provides
 * mapped_view, which is only a partial solution because it requires
 * code duplication throughout the entire program to distinguish between
 * mapped_view and vector.
 *
 * This function avoids the memcopy by injecting a pointer pointing
 * to GPU pinned memory into a vector. The vector can then be used on
 * the host CPU.
 *
 * Warning: Do not use the host vector after freeing the device vector!
 */

#define CLUSTERING_BOOST_VECTOR_MAP_HACK(TYPE)                          \
template<> template<>                                                   \
void ::boost::compute::vector<                                          \
    TYPE,                                                               \
    boost::compute::pinned_allocator<TYPE>                              \
>::assign(boost::compute::buffer buf, boost::compute::buffer)           \
{                                                                       \
    this->m_allocator.deallocate(this->m_data, this->m_size);           \
    boost::compute::detail::device_ptr<TYPE> ptr(buf);                  \
    clRetainMemObject(buf.get());                                       \
    this->m_data = ptr;                                                 \
    this->m_size = buf.size() / sizeof(TYPE);                           \
}

CLUSTERING_BOOST_VECTOR_MAP_HACK(uint32_t)
CLUSTERING_BOOST_VECTOR_MAP_HACK(uint64_t)

#endif /* VECTOR_MAP_HACK_HPP */
