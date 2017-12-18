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
#include "timer.hpp"

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
        buffer_manager.set_new_centroids_buffer();
        buffer_manager.set_labels_buffer();
        buffer_manager.set_masses_buffer();

        this->matrix_divide.prepare(
                this->queue.get_context(),
                matrix_divide.Divide
                );

        // If centroids initializer function is callable, then call
        if (this->centroids_initializer) {
            this->centroids_initializer(
                    buffer_manager.get_points(),
                    buffer_manager.get_centroids());
        }

        // Wait for all preprocessing steps to finish before
        // starting timer
        this->queue.finish();

        Timer::Timer total_timer;
        total_timer.start();

        for (
                uint32_t iteration = 0;
                iteration < this->max_iterations;
                ++iteration)
        {
            boost::compute::event fill_masses_event =
                boost::compute::fill_async(
                        buffer_manager.get_masses().begin(),
                        buffer_manager.get_masses().end(),
                        0,
                        this->queue
                        )
                .get_event();
            boost::compute::event fill_centroids_event =
                boost::compute::fill_async(
                        buffer_manager.get_new_centroids().begin(),
                        buffer_manager.get_new_centroids().end(),
                        0,
                        this->queue
                        )
                .get_event();

            // execute fused variant
            fu_event = this->f_fused(
                    this->queue,
                    this->num_features,
                    this->num_points,
                    this->num_clusters,
                    buffer_manager.get_points().begin(),
                    buffer_manager.get_points().end(),
                    buffer_manager.get_centroids().begin(),
                    buffer_manager.get_centroids().end(),
                    buffer_manager.get_new_centroids().begin(),
                    buffer_manager.get_new_centroids().end(),
                    buffer_manager.get_labels().begin(),
                    buffer_manager.get_labels().end(),
                    buffer_manager.get_masses().begin(),
                    buffer_manager.get_masses().end(),
                    this->measurement->add_datapoint(iteration),
                    fu_wait_list);

            boost::compute::wait_list division_wait_list;
            matrix_divide.row(
                    this->queue,
                    this->num_features,
                    this->num_clusters,
                    buffer_manager.get_new_centroids().begin(),
                    buffer_manager.get_new_centroids().end(),
                    buffer_manager.get_masses().begin(),
                    buffer_manager.get_masses().end(),
                    this->measurement->add_datapoint(iteration),
                    division_wait_list
                    );

            std::swap(
                    buffer_manager.get_centroids(),
                    buffer_manager.get_new_centroids());

            fu_wait_list.insert(fu_event);
        }

        // Wait for all to finish
        this->queue.finish();

        uint64_t total_time = total_timer
            .stop<std::chrono::nanoseconds>();
        this->measurement->add_datapoint()
            .set_name("TotalTime")
            .add_value() = total_time;

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
                config,
                *this->measurement);
    }

    void set_context(boost::compute::context c) {
        context = c;
    }

    void set_queue(boost::compute::command_queue q) {
        queue = q;

        auto device = q.get_device();
        this->measurement->set_parameter(
                "FusedPlatform",
                device.platform().name()
                );
        this->measurement->set_parameter(
                "FusedDevice",
                device.name()
                );
    }

private:
    FusedFunction f_fused;
    MatrixBinaryOp<PointT, MassT> matrix_divide;

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

            if (not points || points->size() != buf->size()) {
                points.reset();
                points = std::make_shared<Vector<PointT>>(
                        buf->size(),
                        context);
            }

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

        void set_new_centroids_buffer()
        {
            new_centroids = std::make_shared<Vector<PointT>>(
                    num_clusters * num_features,
                    0,
                    queue);
        }

        void set_labels_buffer()
        {
            if (not labels || labels->size() != num_points) {
                labels.reset();
                labels = std::make_shared<Vector<LabelT>>(
                        num_points,
                        0,
                        queue);
            }
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

        Vector<PointT>& get_new_centroids() {
            return *new_centroids;
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
        VectorPtr<PointT> new_centroids;
        VectorPtr<LabelT> labels;
        VectorPtr<MassT> masses;
    } buffer_manager;
};

}

#endif /* KMEANS_SINGLE_STAGE_HPP */
