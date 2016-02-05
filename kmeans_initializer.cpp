#include "kmeans.hpp"
/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */


#include <random>

template <typename FP, typename Alloc>
void cle::KmeansInitializer<FP, Alloc>::forgy(
        std::vector<FP, Alloc> const& points_x,
        std::vector<FP, Alloc> const& points_y,
        std::vector<FP, Alloc>& centroids_x,
        std::vector<FP, Alloc>& centroids_y) {

    std::random_device rand;
    const size_t num_points = points_x.size();
    const size_t num_clusters = centroids_x.size();


    for (size_t c = 0; c != num_clusters; ++c) {
        size_t random_point = rand() % num_points;
        centroids_x[c] = points_x[random_point];
        centroids_y[c] = points_y[random_point];
    }
}

template <typename FP, typename Alloc>
void cle::KmeansInitializer<FP, Alloc>::first_x(
        std::vector<FP, Alloc> const& points_x,
        std::vector<FP, Alloc> const& points_y,
        std::vector<FP, Alloc>& centroids_x,
        std::vector<FP, Alloc>& centroids_y) {

    const size_t num_clusters = centroids_x.size();
    const size_t num_points = points_x.size();

    for (size_t c = 0; c != num_clusters; ++c) {
        centroids_x[c] = points_x[c % num_points];
        centroids_y[c] = points_y[c % num_points];
    }
}

template class cle::KmeansInitializer<float, std::allocator<float>>;
template class cle::KmeansInitializer<double, std::allocator<double>>;
template class cle::KmeansInitializer<float, cle::AlignedAllocatorFP32>;
template class cle::KmeansInitializer<double, cle::AlignedAllocatorFP64>;
