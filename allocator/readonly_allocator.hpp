/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifndef READONLY_ALLOCATOR_HPP
#define READONLY_ALLOCATOR_HPP

#include <boost/compute/allocator/buffer_allocator.hpp>

namespace Clustering {

template <typename T>
class readonly_allocator : public boost::compute::buffer_allocator<T>
{
public:
    explicit readonly_allocator(const boost::compute::context &context)
        : boost::compute::buffer_allocator<T>(context)
    {
        boost::compute::buffer_allocator<T>::set_mem_flags(
            boost::compute::buffer::read_only
        );
    }

    readonly_allocator(const readonly_allocator<T> &other)
        : boost::compute::buffer_allocator<T>(other)
    {
    }

    readonly_allocator<T>& operator=(const readonly_allocator<T> &other)
    {
        if(this != &other){
            boost::compute::buffer_allocator<T>::operator=(other);
        }

        return *this;
    }

    ~readonly_allocator() {}
};

} // Clustering

#endif /* READONLY_ALLOCATOR_HPP */
