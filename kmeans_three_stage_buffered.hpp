/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2017, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef KMEANS_THREE_STAGE_BUFFERED_HPP
#define KMEANS_THREE_STAGE_BUFFERED_HPP

#include "abstract_kmeans.hpp"
#include "labeling_factory.hpp"
#include "mass_update_factory.hpp"
#include "centroid_update_factory.hpp"
#include "simple_buffer_cache.hpp"
#include "single_device_scheduler.hpp"
#include "buffer_helper.hpp"
#include "cl_kernels/matrix_binary_op.hpp"

#include "measurement/measurement.hpp"
#include "timer.hpp"

#include <boost/compute/core.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/fill.hpp>
#include <boost/compute/async/wait.hpp>
#include <boost/compute/container/vector.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, typename MassT, bool ColMajor = true>
class KmeansThreeStageBuffered :
    public AbstractKmeans<PointT, LabelT, MassT, ColMajor>
{
public:

    using LabelingFunction = typename LabelingFactory<PointT, LabelT, ColMajor>::LabelingFunction;
    using MassUpdateFunction = typename MassUpdateFactory<LabelT, MassT>::MassUpdateFunction;
    using CentroidUpdateFunction = typename CentroidUpdateFactory<PointT, LabelT, MassT, ColMajor>::CentroidUpdateFunction;

    KmeansThreeStageBuffered() :
        AbstractKmeans<PointT, LabelT, MassT, ColMajor>()
    {
        buffer_cache = std::make_shared<SimpleBufferCache>(
                size_t(buffer_size)
                );
        this->scheduler.add_buffer_cache(buffer_cache);
    }

    void run() {

        this->matrix_divide.prepare(
                this->context,
                matrix_divide.Divide
                );

        this->host_points_partitioned.resize(this->host_points->size());
        BufferHelper::partition_matrix(
                this->host_points->data(),
                &this->host_points_partitioned[0],
                this->host_points->size() * sizeof(PointT),
                this->num_features,
                this->buffer_cache->buffer_size()
                );

        device_old_centroids = decltype(device_old_centroids)(
                this->num_clusters * this->num_features,
                this->queue.get_context()
                );
        device_new_centroids = decltype(device_new_centroids)(
                this->num_clusters * this->num_features,
                this->queue.get_context()
                );
        boost::compute::event centroids_event = boost::compute::copy_async(
                this->host_centroids->begin(),
                this->host_centroids->begin()
                + this->num_features * this->num_clusters,
                device_old_centroids.begin(),
                this->queue).get_event();
        device_masses = decltype(device_masses)(
                this->num_clusters,
                this->queue.get_context()
                );

        assert(true ==
                this->scheduler.add_device(
                    this->context,
                    this->queue.get_device()
                    ));
        assert(true ==
                this->buffer_cache->add_device(
                    this->context,
                    this->queue.get_device(),
                    this->queue.get_device().global_memory_size()
                    - 64 * 1024 * 1024
                    ));
        auto points_handle = this->buffer_cache->add_object(
                (void*)this->host_points_partitioned.data(),
                this->host_points->size() * sizeof(PointT),
                ObjectMode::Immutable
                );
        auto labels_handle = this->buffer_cache->add_object(
                this->host_labels->data(),
                this->host_labels->size() * sizeof(LabelT),
                ObjectMode::Mutable
                );

        // If centroids initializer function is callable, then call
        if (this->centroids_initializer) {
            this->centroids_initializer(
                    device_old_centroids, // TODO: should be points vector
                    device_old_centroids
                    );
        }

        // Wait for all preprocessing steps to finish before
        // starting timer
        this->queue.finish();

        Timer::Timer total_timer;
        total_timer.start();

        uint32_t iterations = 0;
        while (iterations < this->max_iterations) {

            boost::compute::event fill_masses_event =
                boost::compute::fill_async(
                        device_masses.begin(),
                        device_masses.end(),
                        0,
                        this->queue
                        )
                .get_event();
            boost::compute::event fill_centroids_event =
                boost::compute::fill_async(
                        device_new_centroids.begin(),
                        device_new_centroids.end(),
                        0,
                        this->queue
                        )
                .get_event();

            auto labeling_lambda = [
                f_labeling = this->f_labeling,
                num_features = this->num_features,
                num_clusters = this->num_clusters,
                &device_old_centroids = this->device_old_centroids,
                &measurement = this->measurement,
                iterations
            ]
            (
             boost::compute::command_queue queue,
             size_t /* cl_offset */,
             size_t point_bytes,
             size_t label_bytes,
             boost::compute::buffer points,
             boost::compute::buffer labels
            )
            {

                boost::compute::wait_list wait_list;
                auto num_buffer_points = label_bytes / sizeof(LabelT);

                boost::compute::buffer_iterator<PointT>
                    points_begin(
                            points,
                            0
                            ),
                    points_end(
                            points,
                            point_bytes / sizeof(PointT)
                            );

                boost::compute::buffer_iterator<LabelT>
                    labels_begin(
                            labels,
                            0
                            ),
                    labels_end(
                            labels,
                            label_bytes / sizeof(LabelT)
                            );

                return f_labeling(
                        queue,
                        num_features,
                        num_buffer_points,
                        num_clusters,
                        points_begin,
                        points_end,
                        device_old_centroids.begin(),
                        device_old_centroids.end(),
                        labels_begin,
                        labels_end,
                        measurement->add_datapoint(iterations),
                        wait_list
                        );
            };

            std::future<std::deque<boost::compute::event>> ll_future;
            assert(true ==
                    scheduler.enqueue(
                        labeling_lambda,
                        points_handle,
                        labels_handle,
                        buffer_size,
                        buffer_size / this->num_features,
                        ll_future
                        ));

            auto mass_update_lambda = [
                f_mass_update = this->f_mass_update,
                num_clusters = this->num_clusters,
                &device_masses = this->device_masses,
                &measurement = this->measurement,
                iterations
            ]
            (
             boost::compute::command_queue queue,
             size_t /* cl_offset */,
             size_t label_bytes,
             boost::compute::buffer labels
            )
            {
                boost::compute::wait_list wait_list;
                auto num_buffer_labels = label_bytes / sizeof(LabelT);

                boost::compute::buffer_iterator<LabelT>
                    labels_begin(
                            labels,
                            0
                            ),
                    labels_end(
                            labels,
                            label_bytes / sizeof(LabelT)
                            );

                return f_mass_update(
                        queue,
                        num_buffer_labels,
                        num_clusters,
                        labels_begin,
                        labels_end,
                        device_masses.begin(),
                        device_masses.end(),
                        measurement->add_datapoint(iterations),
                        wait_list);
            };

            std::future<std::deque<boost::compute::event>> mu_future;
            assert(true ==
                    scheduler.enqueue(
                        mass_update_lambda,
                        labels_handle,
                        buffer_size / this->num_features,
                        mu_future
                        ));

            auto centroid_update_lambda = [
                f_centroid_update = this->f_centroid_update,
                num_features = this->num_features,
                num_clusters = this->num_clusters,
                &device_new_centroids = this->device_new_centroids,
                &device_masses = this->device_masses,
                &measurement = this->measurement,
                iterations
            ]
            (
             boost::compute::command_queue queue,
             size_t /* cl_offset */,
             size_t point_bytes,
             size_t label_bytes,
             boost::compute::buffer points,
             boost::compute::buffer labels
            )
            {
                boost::compute::wait_list wait_list;
                auto num_buffer_points = label_bytes / sizeof(LabelT);

                boost::compute::buffer_iterator<PointT>
                    points_begin(
                            points,
                            0
                            ),
                    points_end(
                            points,
                            point_bytes / sizeof(PointT)
                            );

                boost::compute::buffer_iterator<LabelT>
                    labels_begin(
                            labels,
                            0
                            ),
                    labels_end(
                            labels,
                            label_bytes / sizeof(LabelT)
                            );

                return f_centroid_update(
                        queue,
                        num_features,
                        num_buffer_points,
                        num_clusters,
                        points_begin,
                        points_end,
                        device_new_centroids.begin(),
                        device_new_centroids.end(),
                        labels_begin,
                        labels_end,
                        device_masses.begin(),
                        device_masses.end(),
                        measurement->add_datapoint(iterations),
                        wait_list
                        );
            };

            std::future<std::deque<boost::compute::event>> cu_future;
            assert(true ==
                    scheduler.enqueue(
                        centroid_update_lambda,
                        points_handle,
                        labels_handle,
                        buffer_size,
                        buffer_size / this->num_features,
                        cu_future
                        ));

            assert(true == scheduler.run());

            boost::compute::wait_list division_wait_list;
            matrix_divide.row(
                this->queue,
                this->num_features,
                this->num_clusters,
                device_new_centroids.begin(),
                device_new_centroids.end(),
                device_masses.begin(),
                device_masses.end(),
                this->measurement->add_datapoint(iterations),
                division_wait_list
                );

            std::swap(device_old_centroids, device_new_centroids);
            ++iterations;
        }

        // Wait for last queue to finish processing
        this->queue.finish();

        uint64_t total_time = total_timer
            .stop<std::chrono::nanoseconds>();
        this->measurement->add_datapoint()
            .set_name("TotalTime")
            .add_value() = total_time;

        boost::compute::event centroids_copy_event = boost::compute::copy_async(
                this->device_old_centroids.begin(),
                this->device_old_centroids.begin() + this->num_features * this->num_clusters,
                this->host_centroids->begin(),
                this->queue
                ).get_event();
        centroids_copy_event.wait();

        boost::compute::event masses_copy_event = boost::compute::copy_async(
                this->device_masses.begin(),
                this->device_masses.begin() + this->num_clusters,
                this->host_masses->begin(),
                this->queue
                ).get_event();
        masses_copy_event.wait();

        {
            char *begin, *iter, *end;
            size_t labels_content_size = buffer_size / this->num_features;
            for (
                    begin = (char*) this->host_labels->data(),
                    end = begin + this->host_labels->size() * sizeof(LabelT),
                    iter = begin;
                    iter < end;
                    iter += labels_content_size
                )
            {
                boost::compute::event labels_read_event;
                auto iter_step = (iter + labels_content_size > end)
                    ? end
                    : iter + labels_content_size
                    ;

                assert(true ==
                        buffer_cache->read(
                            this->queue,
                            labels_handle,
                            iter,
                            iter_step,
                            labels_read_event
                            ));
            }
        }

    }

    void set_labeler(LabelingConfiguration config) {
        LabelingFactory<PointT, LabelT, ColMajor> factory;
        f_labeling = factory.create(
                this->context,
                config,
                *this->measurement);
    }

    void set_mass_updater(MassUpdateConfiguration config) {
        MassUpdateFactory<LabelT, MassT> factory;
        f_mass_update = factory.create(
                this->context,
                config,
                *this->measurement);
    }

    void set_centroid_updater(CentroidUpdateConfiguration config) {
        CentroidUpdateFactory<PointT, LabelT, MassT, ColMajor> factory;
        f_centroid_update = factory.create(
                this->context,
                config,
                *this->measurement);
    }

    void set_labeling_context(boost::compute::context c) {
        if (context == boost::compute::context()) {
            context = c;
        }
        else {
            assert(context == c);
        }
    }

    void set_mass_update_context(boost::compute::context c) {
        if (context == boost::compute::context()) {
            context = c;
        }
        else {
            assert(context == c);
        }
    }

    void set_centroid_update_context(boost::compute::context c) {
        if (context == boost::compute::context()) {
            context = c;
        }
        else {
            assert(context == c);
        }
    }

    void set_labeling_queue(boost::compute::command_queue q) {

        if (this->queue == boost::compute::command_queue()) {
            this->queue = q;
        }
        else {
            assert(this->queue == q);
        }

        auto device = q.get_device();
        this->measurement->set_parameter(
                "LabelingPlatform",
                device.platform().name()
                );
        this->measurement->set_parameter(
                "LabelingDevice",
                device.name()
                );
    }

    void set_mass_update_queue(boost::compute::command_queue q) {
        if (this->queue == boost::compute::command_queue()) {
            this->queue = q;
        }
        else {
            assert(this->queue == q);
        }

        auto device = q.get_device();
        this->measurement->set_parameter(
                "MassUpdatePlatform",
                device.platform().name()
                );
        this->measurement->set_parameter(
                "MassUpdateDevice",
                device.name()
                );
    }

    void set_centroid_update_queue(boost::compute::command_queue q) {
        if (this->queue == boost::compute::command_queue()) {
            this->queue = q;
        }
        else {
            assert(this->queue == q);
        }

        auto device = q.get_device();
        this->measurement->set_parameter(
                "CentroidUpdatePlatform",
                device.platform().name()
                );
        this->measurement->set_parameter(
                "CentroidUpdateDevice",
                device.name()
                );
    }

private:
    static constexpr size_t buffer_size = 16ul * 1024ul * 1024ul;

    LabelingFunction f_labeling;
    MassUpdateFunction f_mass_update;
    CentroidUpdateFunction f_centroid_update;

    boost::compute::context context;
    boost::compute::command_queue queue;

    typename AbstractKmeans<PointT, LabelT, MassT, ColMajor>::template HostVector<PointT> host_points_partitioned;
    std::shared_ptr<SimpleBufferCache> buffer_cache;
    SingleDeviceScheduler scheduler;
    MatrixBinaryOp<PointT, MassT> matrix_divide;

    boost::compute::vector<PointT> device_old_centroids;
    boost::compute::vector<PointT> device_new_centroids;
    boost::compute::vector<MassT> device_masses;
};
} // namespace Clustering

#endif /* KMEANS_THREE_STAGE_BUFFERED_HPP */
