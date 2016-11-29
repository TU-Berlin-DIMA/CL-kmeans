/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "kmeans_initializer.hpp"

#include <random>

template <typename PointT>
void Clustering::KmeansInitializer<PointT>::forgy(
        cle::Matrix<PointT, std::allocator<PointT>, size_t, true> const& points,
        cle::Matrix<PointT, std::allocator<PointT>, size_t, true>& centroids) {

    std::random_device rand;

    for (size_t c = 0; c != centroids.rows(); ++c) {
        size_t random_point = rand() % points.rows();
        for (size_t d = 0; d < centroids.cols(); ++d) {
            centroids(c, d) = points(random_point, d);
        }
    }
}

template <typename PointT>
void Clustering::KmeansInitializer<PointT>::first_x(
        cle::Matrix<PointT, std::allocator<PointT>, size_t, true> const& points,
        cle::Matrix<PointT, std::allocator<PointT>, size_t, true>& centroids) {

    for (size_t d = 0; d < centroids.cols(); ++d) {
        for (size_t c = 0; c != centroids.rows(); ++c) {
            centroids(c, d) = points(c % points.rows(), d);
        }
    }
}

template class Clustering::KmeansInitializer<float>;
template class Clustering::KmeansInitializer<double>;
