/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "cluster_generator.hpp"

#include <cstdint>
#include <random>
#include <fstream>

void cle::ClusterGenerator::num_features(uint64_t features) {
    features_ = features;
}

void cle::ClusterGenerator::num_clusters(uint64_t clusters) {
    clusters_ = clusters;
}

void cle::ClusterGenerator::cluster_radius(float radius) {
    radius_ = radius;
}

void cle::ClusterGenerator::domain(float min, float max) {
    domain_min_ = min;
    domain_max_ = max;
}

void cle::ClusterGenerator::total_size(uint64_t bytes) {
    bytes_ = bytes;
}

void cle::ClusterGenerator::point_multiple(uint64_t multiple) {
    multiple_ = multiple;
}

void cle::ClusterGenerator::generate(char const* file_name) {

    uint64_t size = bytes_ / sizeof(float);
    uint64_t num_points = size / features_;
    uint64_t points_per_cluster = num_points / clusters_;
    num_points = points_per_cluster * clusters_;
    uint64_t remainder = num_points % multiple_;
    num_points = num_points - remainder;

    std::default_random_engine rgen;
    std::uniform_real_distribution<float> uniform(domain_min_, domain_max_);
    std::normal_distribution<float> gaussian(-radius_, radius_);

    std::ofstream fh(file_name, std::fstream::trunc);

    for (uint64_t c = 0; c < clusters_; ++c) {
        float centroid = uniform(rgen);

        uint64_t start = 0;
        if (remainder != 0 && c != 0) {
            start = (clusters_ + remainder - 2) / (clusters_ - 1);
            remainder = remainder - start;
        }

        for (uint64_t p = start; p < points_per_cluster; ++p) {

            if (not (c == 0 and p == 0)) {
                fh << '\n';
            }

            for (uint64_t f = 0; f < features_; ++f) {
                float point = centroid + gaussian(rgen);

                if (f != 0) {
                    fh << ',';
                }

                fh << point;
            }
        }
    }
}
