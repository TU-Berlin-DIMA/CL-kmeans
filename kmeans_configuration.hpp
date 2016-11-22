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
