/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef KMEANS_SINGLE_STAGE_HPP
#define KMEANS_SINGLE_STAGE_HPP

#include "abstract_kmeans.hpp"
#include "fused_factory.hpp"

#include "measurement/measurement.hpp"

#include <functional>
#include <algorithm>
#include <vector>
#include <memory>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/async/wait.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, typename MassT, bool ColMajor = true>
class KmeansSingleStage :
    public AbstractKmeans<PointT, LabelT, MassT, ColMajor>
{
public:
    template <typename T>
    using Vector = boost::compute::vector<T>;
    template <typename T>
    using VectorPtr = std::shared_ptr<Vector<T>>;
    template <typename T>
    using HostVectorPtr = std::shared_ptr<std::vector<T>>;
    using Event = boost::compute::event;
    using Future = boost::compute::future<void>;
    using WaitList = boost::compute::wait_list;

    using FusedFunction = typename FusedFactory<PointT, LabelT, MassT, ColMajor>::FusedFunction;

    KmeansSingleStage() :
        AbstractKmeans<PointT, LabelT, MassT, ColMajor>()
    {}

    void run() {

        Event fu_event;
        WaitList fu_wait_list;

        buffer_manager.set_queue(this->queue);
        buffer_manager.set_context(this->context);
        buffer_manager.set_parameters(
                this->num_features,
                this->num_points,
                this->num_clusters);
        buffer_manager.set_points_buffer(
                this->host_points,
                this->measurement->add_datapoint());
        buffer_manager.set_centroids_buffer(
                this->host_centroids,
                this->measurement->add_datapoint());
        buffer_manager.set_labels_buffer();
        buffer_manager.set_masses_buffer();

        // If centroids initializer function is callable, then call
        if (this->centroids_initializer) {
            this->centroids_initializer(
                    buffer_manager.get_points(),
                    buffer_manager.get_centroids());
        }

        for (
                uint32_t iteration = 0;
                iteration < this->max_iterations;
                ++iteration)
        {
            // execute fused variant
            fu_event = this->f_fused(
                    this->queue,
                    this->num_features,
                    this->num_points,
                    this->num_clusters,
                    buffer_manager.get_points(),
                    buffer_manager.get_centroids(),
                    buffer_manager.get_labels(),
                    buffer_manager.get_masses(),
                    this->measurement->add_datapoint(iteration),
                    fu_wait_list);

            fu_wait_list.insert(fu_event);
        }

        // Wait for all to finish
        fu_wait_list.wait();

        // Copy centroids and labels to host
        buffer_manager.get_centroids(
                this->host_centroids,
                this->measurement->add_datapoint());
        buffer_manager.get_labels(
                this->host_labels,
                this->measurement->add_datapoint());
        buffer_manager.get_masses(
                this->host_masses,
                this->measurement->add_datapoint());
    }

    void set_fused(FusedConfiguration config) {
        FusedFactory<PointT, LabelT, MassT, ColMajor> factory;
        f_fused = factory.create(
                this->context,
                config);
    }

    void set_context(boost::compute::context c) {
        context = c;
    }

    void set_queue(boost::compute::command_queue q) {
        queue = q;
    }

private:
    FusedFunction f_fused;

    boost::compute::context context;
    boost::compute::command_queue queue;

    struct BufferManager {

        void set_queue(boost::compute::command_queue q) {
            queue = q;
        }

        void set_context(boost::compute::context c) {
            context = c;
        }

        void set_parameters(size_t num_features, size_t num_points, size_t num_clusters) {
            this->num_features = num_features;
            this->num_points = num_points;
            this->num_clusters = num_clusters;
        }

        void set_points_buffer(
                std::shared_ptr<const std::vector<PointT>> buf,
                Measurement::DataPoint& dp)
        {
            dp.set_name("PointsH2D");

            points = std::make_shared<Vector<PointT>>(
                    buf->size(),
                    context);

            Future future = boost::compute::copy_async(
                    buf->begin(),
                    buf->end(),
                    points->begin(),
                    queue);

            dp.add_event() = future.get_event();
            future.wait();
        }

        void set_centroids_buffer(
                HostVectorPtr<PointT> buf,
                Measurement::DataPoint& dp)
        {
            dp.set_name("CentroidsH2D");

            centroids = std::make_shared<Vector<PointT>>(
                    buf->size(),
                    context);

            Future future = boost::compute::copy_async(
                    buf->begin(),
                    buf->end(),
                    centroids->begin(),
                    queue);

            dp.add_event() = future.get_event();
            future.wait();
        }

        void set_labels_buffer()
        {
            labels = std::make_shared<Vector<LabelT>>(
                    num_points,
                    0,
                    queue);
        }

        void set_masses_buffer()
        {
            masses = std::make_shared<Vector<MassT>>(
                        num_clusters,
                        0,
                        queue);
        }

        Vector<PointT>& get_points() {
            return *points;
        }

        Vector<PointT>& get_centroids() {
            return *centroids;
        }

        Vector<LabelT>& get_labels() {
            return *labels;
        }

        Vector<MassT>& get_masses() {
            return *masses;
        }

        void get_centroids(
                HostVectorPtr<PointT> buf,
                Measurement::DataPoint& dp)
        {
            assert(buf->size() >= num_clusters * num_features);

            dp.set_name("CentroidsD2H");

            Future future = boost::compute::copy_async(
                    centroids->begin(),
                    centroids->begin()
                    + num_clusters * num_features,
                    buf->begin(),
                    queue);

            dp.add_event() = future.get_event();
            future.wait();
        }

        void get_labels(
                HostVectorPtr<LabelT> buf,
                Measurement::DataPoint& dp
                )
        {
            assert(buf->size() >= num_points);

            dp.set_name("LabelsD2H");

            Future future = boost::compute::copy_async(
                    labels->begin(),
                    labels->begin() + num_points,
                    buf->begin(),
                    queue);

            dp.add_event() = future.get_event();
            future.wait();
        }

        void get_masses(
                HostVectorPtr<MassT> buf,
                Measurement::DataPoint& dp
                )
        {
            assert(buf->size() >= num_clusters);

            dp.set_name("MassesD2H");

            Future future = boost::compute::copy_async(
                    masses->begin(),
                    masses->begin() + num_clusters,
                    buf->begin(),
                    queue);

            dp.add_event() = future.get_event();
            future.wait();
        }

        size_t num_features;
        size_t num_points;
        size_t num_clusters;
        boost::compute::context context;
        boost::compute::command_queue queue;
        VectorPtr<PointT> points;
        VectorPtr<PointT> centroids;
        VectorPtr<LabelT> labels;
        VectorPtr<MassT> masses;
    } buffer_manager;
};

}

#endif /* KMEANS_SINGLE_STAGE_HPP */
