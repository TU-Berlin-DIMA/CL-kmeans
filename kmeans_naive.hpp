/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifndef KMEANS_NAIVE_HPP
#define KMEANS_NAIVE_HPP

#include "kmeans_common.hpp"
#include "matrix.hpp"
#include "measurement/measurement.hpp"

#include <vector>
#include <memory>

namespace Clustering {

template <typename PointT, typename LabelT, typename MassT>
class KmeansNaive {
public:
    char const* name() const;

    int initialize();
    int finalize();

    std::shared_ptr<Measurement::Measurement> operator() (
            uint32_t const max_iterations,
            cle::Matrix<PointT, std::allocator<PointT>, size_t, true> const& points,
            cle::Matrix<PointT, std::allocator<PointT>, size_t, true>& centroids,
            std::vector<MassT>& cluster_mass,
            std::vector<LabelT>& labels
            );
};

using KmeansNaive32 =
    KmeansNaive<float, uint32_t, uint32_t>;
using KmeansNaive64 =
    KmeansNaive<double, uint64_t, uint64_t>;

}

extern template class Clustering::KmeansNaive<float, uint32_t, uint32_t>;
extern template class Clustering::KmeansNaive<double, uint64_t, uint64_t>;

#endif /* KMEANS_NAIVE_HPP */
