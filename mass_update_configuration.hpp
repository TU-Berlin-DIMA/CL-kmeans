#ifndef MASS_UPDATE_CONFIGURATION_HPP
#define MASS_UPDATE_CONFIGURATION_HPP

#include <cstddef>

namespace Clustering {

struct MassUpdateConfiguration {
    size_t global_size[3];
    size_t local_size[3];
};

}

#endif /* MASS_UPDATE_CONFIGURATION_HPP */
