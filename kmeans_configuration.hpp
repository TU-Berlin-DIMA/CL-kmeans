/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifndef KMEANS_CONFIGURATION_HPP
#define KMEANS_CONFIGURATION_HPP

#include <cstddef>
#include <string>

namespace Clustering {

struct KmeansConfiguration {
    size_t clusters;
    std::string pipeline;
    size_t iterations;
    bool converge;
    std::string point_type;
    std::string label_type;
    std::string mass_type;
};

}


#endif /* KMEANS_CONFIGURATION_HPP */
