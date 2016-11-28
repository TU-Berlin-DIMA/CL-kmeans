/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "kmeans_naive.hpp"

#include <cassert>
#include <algorithm>
#include <limits>
#include <string>

template <typename PointT, typename LabelT, typename MassT>
char const* cle::KmeansNaive<PointT, LabelT, MassT>::name() const {

    return "Lloyd_Naive";
}


template <typename PointT, typename LabelT, typename MassT>
int cle::KmeansNaive<PointT, LabelT, MassT>::initialize() { return 1; }

template <typename PointT, typename LabelT, typename MassT>
int cle::KmeansNaive<PointT, LabelT, MassT>::finalize() { return 1; }

template <typename PointT, typename LabelT, typename MassT>
void cle::KmeansNaive<PointT, LabelT, MassT>::operator() (
        uint32_t const max_iterations,
        cle::Matrix<PointT, std::allocator<PointT>, size_t, true> const& points,
        cle::Matrix<PointT, std::allocator<PointT>, size_t, true>& centroids,
        std::vector<MassT>& cluster_mass,
        std::vector<LabelT>& labels,
        Measurement::Measurement& stats) {

    assert(labels.size() == points.rows());
    assert(cluster_mass.size() == centroids.rows());

    uint32_t iterations = 0;
    bool did_changes = true;
    while (did_changes == true && iterations < max_iterations) {
        did_changes = false;

        // Phase 1: assign points to clusters
        for (size_t p = 0; p != points.rows(); ++p) {
            PointT min_distance = std::numeric_limits<PointT>::max();
            LabelT min_centroid = 0;

            for (size_t c = 0; c != centroids.rows(); ++c) {
                PointT distance = 0;
                for (size_t d = 0; d < points.cols(); ++d) {
                    PointT t = points(p, d) - centroids(c, d);
                    distance += t * t;
                }

                if (distance < min_distance) {
                    min_distance = distance;
                    min_centroid = c;
                }
            }

            if (min_centroid != labels[p]) {
                labels[p] = min_centroid;
                did_changes = true;
            }
        }

        // Phase 2: calculate new clusters
        // Arithmetic mean of all points assigned to cluster
        std::fill(cluster_mass.begin(), cluster_mass.end(), 0);
        std::fill(centroids.begin(), centroids.end(), 0);

        Matrix<PointT, std::allocator<PointT>, size_t, true> compensation;
        compensation.resize(centroids.rows(), centroids.cols());
        std::fill(compensation.begin(), compensation.end(), 0);

        for (size_t p = 0; p < points.rows(); ++p) {
            LabelT c = labels[p];

            cluster_mass[c] += 1;
            for (size_t d = 0; d < points.cols(); ++d) {
                // Kahan sum of centroids(c, d) += points(p, d);
                PointT const& point = points(p, d);
                PointT& comp = compensation(c, d);
                PointT& sum = centroids(c, d);
                PointT y = point - comp;
                PointT t = sum + y;
                comp = (t - sum) - y;
                sum = t;
            }
        }

        for (size_t f = 0; f < centroids.cols(); ++f) {
            for (size_t c = 0; c < centroids.rows(); ++c) {
                centroids(c, f) = centroids(c, f) / cluster_mass[c];
            }
        }

        ++iterations;
    }

    stats.set_parameter(
            Measurement::ParameterType::NumIterations,
            std::to_string(iterations)
            );
}

template class cle::KmeansNaive<float, uint32_t, uint32_t>;
template class cle::KmeansNaive<double, uint64_t, uint64_t>;
