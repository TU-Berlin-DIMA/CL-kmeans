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

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
char const* cle::KmeansNaive<FP, INT, AllocFP, AllocINT>::name() const {

    return "Lloyd_Naive";
}


template <typename FP, typename INT, typename AllocFP, typename AllocINT>
int cle::KmeansNaive<FP, INT, AllocFP, AllocINT>::initialize() { return 1; }

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
int cle::KmeansNaive<FP, INT, AllocFP, AllocINT>::finalize() { return 1; }

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
void cle::KmeansNaive<FP, INT, AllocFP, AllocINT>::operator() (
        uint32_t const max_iterations,
        cle::Matrix<FP, AllocFP, INT, true> const& points,
        cle::Matrix<FP, AllocFP, INT, true>& centroids,
        std::vector<INT, AllocINT>& cluster_mass,
        std::vector<INT, AllocINT>& labels,
        Measurement::Measurement& stats) {

    assert(labels.size() == points.rows());
    assert(cluster_mass.size() == centroids.rows());

    stats.start();

    uint32_t iterations = 0;
    bool did_changes = true;
    while (did_changes == true && iterations < max_iterations) {
        did_changes = false;

        // Phase 1: assign points to clusters
        for (INT p = 0; p != points.rows(); ++p) {
            FP min_distance = std::numeric_limits<FP>::max();
            INT min_centroid = 0;

            for (INT c = 0; c != centroids.rows(); ++c) {
                FP distance = 0;
                for (INT d = 0; d < points.cols(); ++d) {
                    FP t = points(p, d) - centroids(c, d);
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

        for (INT p = 0; p < points.rows(); ++p) {
            INT c = labels[p];

            cluster_mass[c] += 1;
            for (INT d = 0; d < points.cols(); ++d) {
                centroids(c, d) += points(p, d);
            }
        }

        for (INT f = 0; f < centroids.cols(); ++f) {
            for (INT c = 0; c < centroids.rows(); ++c) {
                centroids(c, f) = centroids(c, f) / cluster_mass[c];
            }
        }

        ++iterations;
    }

    stats.end();
    stats.set_parameter(
            Measurement::ParameterType::NumIterations,
            std::to_string(iterations)
            );
}

template class cle::KmeansNaive<float, uint32_t, std::allocator<float>, std::allocator<uint32_t>>;
template class cle::KmeansNaive<double, uint64_t, std::allocator<double>, std::allocator<uint64_t>>;
#ifdef USE_ALIGNED_ALLOCATOR
template class cle::KmeansNaive<float, uint32_t, cle::AlignedAllocatorFP32, cle::AlignedAllocatorINT32>;
template class cle::KmeansNaive<double, uint64_t, cle::AlignedAllocatorFP64, cle::AlignedAllocatorINT64>;
#endif
