/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifndef CLUSTER_GENERATOR_HPP
#define CLUSTER_GENERATOR_HPP

#include "matrix.hpp"

#include <cstdint>
#include <vector>

namespace cle {
class ClusterGenerator {
public:
    void num_features(uint64_t features);
    void num_clusters(uint64_t clusters);
    void cluster_radius(float radius);
    void domain(float min, float max);
    void total_size(uint64_t bytes);
    void point_multiple(uint64_t multiple);

    void generate_matrix(
        Matrix<float, std::allocator<float>, uint32_t>& points,
        Matrix<float, std::allocator<float>, uint32_t>& centroids,
        std::vector<uint32_t>& labels
        );

    void generate_csv(char const* file_name);
    void generate_bin(char const* file_name);

private:
    uint64_t features_;
    uint64_t clusters_;
    float radius_;
    float domain_min_;
    float domain_max_;
    uint64_t bytes_;
    uint64_t multiple_;
};
}
#endif /* CLUSTER_GENERATOR_HPP */
