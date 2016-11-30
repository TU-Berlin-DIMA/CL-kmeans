/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef MASS_UPDATE_CONFIGURATION_HPP
#define MASS_UPDATE_CONFIGURATION_HPP

#include <cstddef>
#include <string>

namespace Clustering {

struct MassUpdateConfiguration {
    size_t platform;
    size_t device;
    std::string strategy;
    size_t global_size[3];
    size_t local_size[3];
};

}

#endif /* MASS_UPDATE_CONFIGURATION_HPP */
