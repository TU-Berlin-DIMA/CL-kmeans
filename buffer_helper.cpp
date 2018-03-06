/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#include "buffer_helper.hpp"

#include <cstddef>
#include <cstring>
#include <iostream>

int Clustering::BufferHelper::partition_matrix(
        void const *src,
        void *dst,
        size_t size,
        size_t num_dims,
        size_t buffer_size
        )
{
    size_t dim_size = size / num_dims;
    if (size % num_dims != 0) {
        std::cerr
            << "BufferHelper::two_dim_to_buffers:"
            << " source array dimension mismatch"
            << std::endl;
        return -1;
    }

    size_t buf_dim_size = buffer_size / num_dims;
    if (buffer_size % num_dims != 0) {
        std::cerr
            << "BufferHelper::two_dim_to_buffers:"
            << " buffer dimension mismatch"
            << std::endl;
        return -1;
    }

    size_t num_bufs = (size + buffer_size - 1) / buffer_size;
    char *c_src = (char*) src;
    char *c_dst = (char*) dst;

#ifdef VERBOSE
    std::cout
        << "size " << size
        << "\nbuffer_size " << buffer_size
        << "\nnum_dims " << num_dims
        << "\ndim_size " << dim_size
        << "\nbuf_dim_size " << buf_dim_size
        << "\nnum_bufs " << num_bufs
        << std::endl
        ;
#endif

    for (size_t b = 0; b < num_bufs; ++b) {
        for (size_t v = 0; v < num_dims; ++v) {

            size_t real_buf_dim_size =
                (buf_dim_size * (b + 1) > dim_size)
                ? dim_size - b * buf_dim_size
                : buf_dim_size
                ;

#ifdef VERBOSE
            std::cout
                << "b " << b
                << "  v " << v
                << "  c_dst " << b * buffer_size + v * real_buf_dim_size
                << "  c_src " << v * dim_size + b * buf_dim_size
                << "  length " << real_buf_dim_size
                << std::endl
                ;
#endif

            std::memcpy(
                    &c_dst[b * buffer_size + v * real_buf_dim_size],
                    &c_src[v * dim_size + b * buf_dim_size],
                    real_buf_dim_size
                    );
        }
    }

    return num_bufs;
}

