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
