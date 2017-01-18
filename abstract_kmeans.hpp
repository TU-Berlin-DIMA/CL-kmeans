/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef ABSTRACT_KMEANS_HPP
#define ABSTRACT_KMEANS_HPP

#include "measurement/measurement.hpp"

#include <functional>
#include <cstdint>
#include <memory>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, typename MassT, bool ColMajor = true>
class AbstractKmeans {
public:
    using MeasurementPtr = std::shared_ptr<Measurement::Measurement>;
    template <typename T>
    using Vector = boost::compute::vector<T>;
    template <typename T>
    using VectorPtr = std::shared_ptr<Vector<T>>;
    template <typename T>
    using HostVector = std::vector<T>;
    template <typename T>
    using HostVectorPtr = std::shared_ptr<std::vector<T>>;

    using InitCentroidsFunction = std::function<
        void(
                Vector<PointT>& points,
                Vector<PointT> & centroids
                )
        >;

    AbstractKmeans() :
        num_features(0),
        num_points(0),
        num_clusters(0),
        host_points(nullptr),
        host_centroids(nullptr),
        host_masses(nullptr),
        host_labels(nullptr),
        measurement(new Measurement::Measurement)
    {}

    virtual ~AbstractKmeans() {}

    virtual void set_max_iterations(size_t i) {
        this->max_iterations = i;
    }

    virtual void set_points(std::shared_ptr<const std::vector<PointT>> p) {
        this->host_points = p;

        if (this->num_features != 0) {
            this->num_points = p->size() / this->num_features;
        }
    }

    virtual void set_features(size_t f) {
        this->num_features = f;
        this->num_points = this->host_points->size() / f;
    }

    virtual void set_clusters(size_t c) {
        this->num_clusters = c;
    }

    virtual void set_initializer(InitCentroidsFunction f) {
        this->centroids_initializer = f;
    }

    virtual void set_measurement(
            std::shared_ptr<Measurement::Measurement> measurement
            ) {
        this->measurement = measurement;
    }

    virtual HostVector<PointT> const& get_centroids() const {
        return *this->host_centroids;
    }

    virtual HostVector<LabelT> const& get_labels() const {
        return *this->host_labels;
    }

    virtual HostVector<MassT> const& get_cluster_masses() const {
        return *this->host_masses;
    }

    virtual Measurement::Measurement const& get_measurement() const {
        return *this->measurement;
    }

    virtual void run() = 0;

    virtual MeasurementPtr operator() (
            size_t max_iterations,
            size_t num_features,
            std::shared_ptr<const std::vector<PointT>> points,
            HostVectorPtr<PointT> centroids,
            HostVectorPtr<MassT> masses,
            HostVectorPtr<LabelT> labels
            ) {

        this->max_iterations = max_iterations;
        this->num_features = num_features;
        this->num_points = points->size() / num_features;
        this->num_clusters = centroids->size() / num_features;
        this->host_points = points;
        this->host_centroids = centroids;
        this->host_masses = masses;
        this->host_labels = labels;

        this->run();

        return this->measurement;
    }

protected:
    size_t max_iterations;
    size_t num_features;
    size_t num_points;
    size_t num_clusters;
    std::shared_ptr<const std::vector<PointT>> host_points;
    HostVectorPtr<PointT> host_centroids;
    HostVectorPtr<MassT> host_masses;
    HostVectorPtr<LabelT> host_labels;

    InitCentroidsFunction centroids_initializer;
    std::shared_ptr<Measurement::Measurement> measurement;
};

} // Clustering

#endif /* ABSTRACT_KMEANS_HPP */
