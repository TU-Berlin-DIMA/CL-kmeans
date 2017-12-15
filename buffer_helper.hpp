/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2017, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef BUFFER_HELPER_HPP
#define BUFFER_HELPER_HPP

#include <cstddef>

namespace Clustering {

class BufferHelper {
public:
    static int partition_matrix(
            void const *src,
            void *dst,
            size_t size,
            size_t num_dims,
            size_t buffer_size
            );
};

}

#endif /* BUFFER_HELPER_HPP */
