#ifndef CENTROID_UPDATE_CONFIGURATION_HPP
#define CENTROID_UPDATE_CONFIGURATION_HPP

#include <cstddef>
#include <string>

namespace Clustering {

struct CentroidUpdateConfiguration {
    size_t platform;
    size_t device;
    std::string strategy;
    size_t global_size[3];
    size_t local_size[3];
};

}
#endif /* CENTROID_UPDATE_CONFIGURATION_HPP */
