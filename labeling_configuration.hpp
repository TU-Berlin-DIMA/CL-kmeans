#ifndef LABELING_CONFIGURATION_HPP
#define LABELING_CONFIGURATION_HPP

#include <cstddef>

namespace Clustering {

struct LabelingConfiguration {
    size_t global_size[3];
    size_t local_size[3];
    size_t vector_length;
    size_t unroll_clusters_length;
    size_t unroll_features_length;
};

}

#endif /* LABELING_CONFIGURATION_HPP */
