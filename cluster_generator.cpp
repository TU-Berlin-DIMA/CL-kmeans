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

/*
 * Generate binary file
 * File format:
 *
 * uint64_t num_features
 * uint64_t num_clusters
 * uint64_t num_points
 * float clusters[0 ... num_clusters-1], column major
 * float points[0 ... num_points-1], column major
 */
void cle::ClusterGenerator::generate_matrix(
        Matrix<float, std::allocator<float>, uint32_t>& points,
        Matrix<float, std::allocator<float>, uint32_t>& centroids,
        std::vector<uint32_t>& labels
        ) {
    uint64_t size = bytes_ / sizeof(float);
    uint64_t num_points = size / features_;
    uint64_t points_per_cluster = num_points / clusters_;
    num_points = points_per_cluster * clusters_;
    uint64_t remainder = num_points % multiple_;
    num_points = num_points - remainder;

    std::default_random_engine rgen;
    std::uniform_real_distribution<float> uniform(domain_min_, domain_max_);
    std::normal_distribution<float> gaussian(-radius_, radius_);

    points.resize(num_points, features_);
    centroids.resize(clusters_, features_);
    labels.resize(num_points);

    for (uint64_t f = 0; f < features_; ++f) {
        uint64_t tmp_remainder = remainder;

        for (uint64_t c = 0; c < clusters_; ++c) {
            uint64_t row = 0;
            float centroid = uniform(rgen);
            centroids(c, f) = centroid;

            uint64_t start = 0;
            if (tmp_remainder != 0 && c != 0) {
                start = (clusters_ + tmp_remainder - 2) / (clusters_ - 1);
                tmp_remainder = tmp_remainder - start;
            }

            for (uint64_t p = start; p < points_per_cluster; ++p) {
                float point = centroid + gaussian(rgen);
                points(row, f) = point;

                if (f == 0) {
                    labels[row] = c;
                }

                ++row;
            }
        }
    }
}

void cle::ClusterGenerator::generate_csv(char const* file_name) {

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

/*
 * Generate binary file
 * File format:
 *
 * uint64_t num_features
 * uint64_t num_clusters
 * uint64_t num_points
 * float clusters[0 ... num_clusters-1], column major
 * float points[0 ... num_points-1], column major
 */
void cle::ClusterGenerator::generate_bin(char const* file_name) {

    uint64_t size = bytes_ / sizeof(float);
    uint64_t num_points = size / features_;
    uint64_t points_per_cluster = num_points / clusters_;
    num_points = points_per_cluster * clusters_;
    uint64_t remainder = num_points % multiple_;
    num_points = num_points - remainder;

    std::default_random_engine rgen;
    std::uniform_real_distribution<float> uniform(domain_min_, domain_max_);
    std::normal_distribution<float> gaussian(-radius_, radius_);

    std::ofstream fh(file_name, std::fstream::binary | std::fstream::trunc);

    uint64_t features = features_;
    fh.write((char*)&features, sizeof(features));
    uint64_t num_clusters = 0;
    fh.write((char*)&num_clusters, sizeof(num_clusters));
    fh.write((char*)&num_points, sizeof(num_points));

    for (uint64_t f = 0; f < features_; ++f) {
        uint64_t tmp_remainder = remainder;

        for (uint64_t c = 0; c < clusters_; ++c) {
            float centroid = uniform(rgen);

            uint64_t start = 0;
            if (tmp_remainder != 0 && c != 0) {
                start = (clusters_ + tmp_remainder - 2) / (clusters_ - 1);
                tmp_remainder = tmp_remainder - start;
            }

            for (uint64_t p = start; p < points_per_cluster; ++p) {
                float point = centroid + gaussian(rgen);
                fh.write((char*)&point, sizeof(point));
            }
        }
    }
}
