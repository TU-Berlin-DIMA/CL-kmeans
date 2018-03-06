/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifndef BENCHMARK_CONFIGURATION_HPP
#define BENCHMARK_CONFIGURATION_HPP

#include <cstddef>

namespace Clustering {

struct BenchmarkConfiguration {
    size_t runs;
    bool verify;
};

}

#endif /* BENCHMARK_CONFIGURATION_HPP */
