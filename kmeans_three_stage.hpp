/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef KMEANS_THREE_STAGE_HPP
#define KMEANS_THREE_STAGE_HPP

#include "abstract_kmeans.hpp"
#include "labeling_factory.hpp"
#include "mass_update_factory.hpp"
#include "centroid_update_factory.hpp"
#include "cl_kernels/matrix_binary_op.hpp"

#include "measurement/measurement.hpp"
#include "timer.hpp"

#include "container/vector_map_hack.hpp"

#include <functional>
#include <algorithm>
#include <vector>
#include <memory>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/fill.hpp>
#include <boost/compute/async/wait.hpp>
#include <boost/compute/allocator/pinned_allocator.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, typename MassT, bool ColMajor = true>
class KmeansThreeStage :
    public AbstractKmeans<PointT, LabelT, MassT, ColMajor>
{
public:
    template <typename T>
    using PinnedAllocator = boost::compute::pinned_allocator<T>;
    template <typename T>
    using Vector = boost::compute::vector<T>;
    template <typename T>
    using VectorPtr = std::shared_ptr<Vector<T>>;
    template <typename T>
    using PinnedVector = boost::compute::vector<T, PinnedAllocator<T>>;
    template <typename T>
    using PinnedVectorPtr = std::shared_ptr<PinnedVector<T>>;
    template <typename T>
    using HostVectorPtr = std::shared_ptr<std::vector<T>>;
    using Event = boost::compute::event;
    using Future = boost::compute::future<void>;

    using LabelingFunction = typename LabelingFactory<PointT, LabelT, ColMajor>::LabelingFunction;
    using MassUpdateFunction = typename MassUpdateFactory<LabelT, MassT>::MassUpdateFunction;
    using CentroidUpdateFunction = typename CentroidUpdateFactory<PointT, LabelT, MassT, ColMajor>::CentroidUpdateFunction;

    KmeansThreeStage() :
        AbstractKmeans<PointT, LabelT, MassT, ColMajor>()
    {}

    void run() {

        boost::compute::wait_list ll_wait_list, mu_wait_list, cu_wait_list;
        boost::compute::wait_list sync_labels_wait_list, sync_centroids_wait_list, sync_masses_wait_list;
        Event ll_event, mu_event, cu_event;
        Event sync_labels_event, sync_centroids_event, sync_masses_event;

        buffer_map.set_queues(
                this->q_labeling,
                this->q_mass_update,
                this->q_centroid_update);
        buffer_map.set_contexts(
                this->context_labeling,
                this->context_mass_update,
                this->context_centroid_update
                );
        buffer_map.set_parameters(
                this->num_features,
                this->num_points,
                this->num_clusters);
        buffer_map.set_points_buffer(
                this->host_points,
                this->measurement->add_datapoint());
        buffer_map.set_centroids_buffer(
                this->host_centroids,
                this->measurement->add_datapoint());
        buffer_map.set_labels_buffer();
        buffer_map.set_masses_buffer();

        this->matrix_divide.prepare(
                this->q_centroid_update.get_context(),
                matrix_divide.Divide
                );

        // If centroids initializer function is callable, then call
        if (this->centroids_initializer) {
            this->centroids_initializer(
                    buffer_map.get_points(BufferMap::ll),
                    buffer_map.get_centroids(BufferMap::ll));
        }

        // Wait for all preprocessing steps to finish before
        // starting timer
        this->q_labeling.finish();
        this->q_mass_update.finish();
        this->q_centroid_update.finish();

        Timer::Timer total_timer;
        total_timer.start();

        uint32_t iterations = 0;
        while (iterations < this->max_iterations) {

            // execute labeling
            sync_centroids_event = buffer_map.sync_centroids(
                    this->measurement->add_datapoint(iterations),
                    sync_centroids_wait_list);
            // TODO
            // ll_wait_list.insert(
            //         sync_centroids_event);
            ll_event = this->f_labeling(
                    this->q_labeling,
                    this->num_features,
                    this->num_points,
                    this->num_clusters,
                    buffer_map.get_points(BufferMap::ll).begin(),
                    buffer_map.get_points(BufferMap::ll).end(),
                    buffer_map.get_centroids(BufferMap::ll).begin(),
                    buffer_map.get_centroids(BufferMap::ll).end(),
                    buffer_map.get_labels(BufferMap::ll).begin(),
                    buffer_map.get_labels(BufferMap::ll).end(),
                    this->measurement->add_datapoint(iterations),
                    ll_wait_list);

            if (/* not converged */ true) {

                boost::compute::event fill_masses_event =
                    boost::compute::fill_async(
                            buffer_map.get_masses(BufferMap::mu).begin(),
                            buffer_map.get_masses(BufferMap::mu).end(),
                            0,
                            this->q_mass_update
                            )
                    .get_event();
                boost::compute::event fill_centroids_event =
                    boost::compute::fill_async(
                            buffer_map.get_centroids(BufferMap::cu).begin(),
                            buffer_map.get_centroids(BufferMap::cu).end(),
                            0,
                            this->q_centroid_update
                            )
                    .get_event();

                // execute mass update
                sync_labels_event = buffer_map.sync_labels(
                        this->measurement->add_datapoint(iterations),
                        sync_labels_wait_list);
                // TODO
                // mu_wait_list.insert(
                //         sync_labels_event);
                // cu_wait_list.insert(
                //         sync_labels_event);
                mu_event = this->f_mass_update(
                        this->q_mass_update,
                        this->num_points,
                        this->num_clusters,
                        buffer_map.get_labels(BufferMap::mu).begin(),
                        buffer_map.get_labels(BufferMap::mu).end(),
                        buffer_map.get_masses(BufferMap::mu).begin(),
                        buffer_map.get_masses(BufferMap::mu).end(),
                        this->measurement->add_datapoint(iterations),
                        mu_wait_list);
                // TODO
                // sync_masses_wait_list.insert(
                //         mu_event);

                // execute centroid update
                sync_masses_event = buffer_map.sync_masses(
                        sync_masses_wait_list);
                // TODO
                // cu_wait_list.insert(
                //         sync_masses_event);
                cu_event = this->f_centroid_update(
                        this->q_centroid_update,
                        this->num_features,
                        this->num_points,
                        this->num_clusters,
                        buffer_map.get_points(BufferMap::cu).begin(),
                        buffer_map.get_points(BufferMap::cu).end(),
                        buffer_map.get_centroids(BufferMap::cu).begin(),
                        buffer_map.get_centroids(BufferMap::cu).end(),
                        buffer_map.get_labels(BufferMap::cu).begin(),
                        buffer_map.get_labels(BufferMap::cu).end(),
                        buffer_map.get_masses(BufferMap::cu).begin(),
                        buffer_map.get_masses(BufferMap::cu).end(),
                        this->measurement->add_datapoint(iterations),
                        cu_wait_list);
                // TODO
                // sync_centroids_wait_list.insert(
                //         cu_event);

                boost::compute::wait_list division_wait_list;
                matrix_divide.row(
                        this->q_centroid_update,
                        this->num_features,
                        this->num_clusters,
                        buffer_map.get_centroids(BufferMap::cu).begin(),
                        buffer_map.get_centroids(BufferMap::cu).end(),
                        buffer_map.get_masses(BufferMap::cu).begin(),
                        buffer_map.get_masses(BufferMap::cu).end(),
                        this->measurement->add_datapoint(iterations),
                        division_wait_list
                        );
            }

            ++iterations;
        }

        // Wait for last queue to finish processing
        this->q_centroid_update.finish();

        uint64_t total_time = total_timer
            .stop<std::chrono::nanoseconds>();
        this->measurement->add_datapoint()
            .set_name("TotalTime")
            .add_value() = total_time;

        // copy centroids and labels to host
        buffer_map.get_centroids(
                this->host_centroids,
                this->measurement->add_datapoint());
        buffer_map.get_labels(
                this->host_labels,
                this->measurement->add_datapoint());
        buffer_map.get_masses(
                this->host_masses,
                this->measurement->add_datapoint());
    }

    void set_labeler(LabelingConfiguration config) {
        LabelingFactory<PointT, LabelT, ColMajor> factory;
        f_labeling = factory.create(
                this->context_labeling,
                config,
                *this->measurement);
    }

    void set_mass_updater(MassUpdateConfiguration config) {
        MassUpdateFactory<LabelT, MassT> factory;
        f_mass_update = factory.create(
                this->context_mass_update,
                config,
                *this->measurement);
    }

    void set_centroid_updater(CentroidUpdateConfiguration config) {
        CentroidUpdateFactory<PointT, LabelT, MassT, ColMajor> factory;
        f_centroid_update = factory.create(
                this->context_centroid_update,
                config,
                *this->measurement);
    }

    void set_labeling_context(boost::compute::context c) {
        context_labeling = c;
    }

    void set_mass_update_context(boost::compute::context c) {
        context_mass_update = c;
    }

    void set_centroid_update_context(boost::compute::context c) {
        context_centroid_update = c;
    }

    void set_labeling_queue(boost::compute::command_queue q) {
        q_labeling = q;

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
        q_mass_update = q;

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
        q_centroid_update = q;

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
    LabelingFunction f_labeling;
    MassUpdateFunction f_mass_update;
    CentroidUpdateFunction f_centroid_update;
    MatrixBinaryOp<PointT, MassT> matrix_divide;

    boost::compute::context context_labeling;
    boost::compute::context context_mass_update;
    boost::compute::context context_centroid_update;

    boost::compute::command_queue q_labeling;
    boost::compute::command_queue q_mass_update;
    boost::compute::command_queue q_centroid_update;

    struct BufferMap {
        enum Phase {ll = 0, mu, cu};

        void set_queues(boost::compute::command_queue q_ll, boost::compute::command_queue q_mu, boost::compute::command_queue q_cu) {
            queue.resize(3);
            queue[ll] = q_ll;
            queue[mu] = q_mu;
            queue[cu] = q_cu;

            device_map.resize(3);
            for (auto& v : device_map) {
                v.resize(3);
            }

            device_map[ll][ll] = true;
            device_map[ll][mu] = queue[ll] == queue[mu];
            device_map[ll][cu] = queue[ll] == queue[cu];
            device_map[mu][ll] = queue[mu] == queue[ll];
            device_map[mu][mu] = true;
            device_map[mu][cu] = queue[mu] == queue[cu];
            device_map[cu][ll] = queue[cu] == queue[ll];
            device_map[cu][mu] = queue[cu] == queue[mu];
            device_map[cu][cu] = true;
        }

        void set_contexts(
                boost::compute::context c_ll,
                boost::compute::context c_mu,
                boost::compute::context c_cu)
        {
            context.resize(3);
            context[ll] = c_ll;
            context[mu] = c_mu;
            context[cu] = c_cu;
        }

        void set_parameters(size_t num_features, size_t num_points, size_t num_clusters) {

            this->num_features = num_features;
            this->num_points = num_points;
            this->num_clusters = num_clusters;
        }

        void set_points_buffer(
                std::shared_ptr<const std::vector<PointT>> buf,
                Measurement::DataPoint& dp
                )
        {
            dp.set_name("PointsH2D");

            VectorPtr<PointT> dev_buf =
                std::make_shared<Vector<PointT>>(
                        buf->size(),
                        context[ll]);

            points.resize(3);

            // Clear buffers before allocating to avoid temporary
            // double space allocation
            points[ll].reset();
            points[mu].reset();
            points[cu].reset();

            points[ll] = dev_buf;
            points[mu] = nullptr;
            points[cu] = device_map[ll][cu] ? points[ll] :
                std::make_shared<Vector<PointT>>(
                        buf->size(),
                        context[cu]);

            Future ll_future =
                boost::compute::copy_async(
                        buf->begin(),
                        buf->end(),
                        points[ll]->begin(),
                        queue[ll]);

            if (not device_map[ll][cu]) {
                Future cu_future =
                    boost::compute::copy_async(
                            buf->begin(),
                            buf->end(),
                            points[cu]->begin(),
                            queue[cu]);

                dp.add_event() = cu_future.get_event();
                cu_future.wait();
            }

            dp.add_event() = ll_future.get_event();
            ll_future.wait();
        }

        void set_centroids_buffer(
                HostVectorPtr<PointT> buf,
                Measurement::DataPoint& dp
                )
        {
            dp.set_name("CentroidsH2D");

            VectorPtr<PointT> dev_buf =
                std::make_shared<Vector<PointT>>(
                        buf->size(),
                        context[ll]);

            centroids.resize(3);
            centroids[ll] = dev_buf;
            centroids[mu] = nullptr;
            centroids[cu] = device_map[cu][ll] ? centroids[ll] :
                std::make_shared<Vector<PointT>>(
                        buf->size(),
                        context[cu]);

            Future ll_future = boost::compute::copy_async(
                    buf->begin(),
                    buf->end(),
                    centroids[ll]->begin(),
                    queue[ll]);

            if (not device_map[ll][cu]) {
                Future cu_future = boost::compute::copy_async(
                        buf->begin(),
                        buf->end(),
                        centroids[cu]->begin(),
                        queue[cu]);

                dp.add_event() = cu_future.get_event();
                cu_future.wait();
            }

            dp.add_event() = ll_future.get_event();
            ll_future.wait();
        }

        void set_labels_buffer()
        {
            labels.resize(3);

            // Clear buffers before allocating to avoid temporary
            // double space allocation
            labels[ll].reset();
            labels[mu].reset();
            labels[cu].reset();

            labels[ll] = std::make_shared<PinnedVector<LabelT>>(
                    num_points,
                    0,
                    queue[ll]);
            labels[mu] = device_map[mu][ll] ? labels[ll] :
                std::make_shared<PinnedVector<LabelT>>(
                        num_points,
                        0,
                        queue[mu]);
            labels[cu] = device_map[cu][ll] ? labels[ll] :
                device_map[cu][mu] ? labels[mu] :
                std::make_shared<PinnedVector<LabelT>>(
                        num_points,
                        0,
                        queue[cu]);
        }

        void set_masses_buffer()
        {
            masses.resize(3);
            masses[ll] = nullptr;
            masses[mu] = std::make_shared<Vector<MassT>>(
                    num_clusters,
                    0,
                    queue[mu]);
            masses[cu] = device_map[cu][mu] ? masses[mu] :
                std::make_shared<Vector<MassT>>(
                        num_clusters,
                        0,
                        queue[cu]);
        }

        void get_centroids(
                HostVectorPtr<PointT> buf,
                Measurement::DataPoint& dp
                )
        {
            assert(buf->size() >= num_clusters * num_features);

            dp.set_name("CentroidsD2H");

            Future future = boost::compute::copy_async(
                    centroids[cu]->begin(),
                    centroids[cu]->begin()
                    + num_clusters * num_features,
                    buf->begin(),
                    queue[cu]);

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
                        labels[ll]->begin(),
                        labels[ll]->begin() + num_points,
                        buf->begin(),
                        queue[ll]);

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
                    masses[mu]->begin(),
                    masses[mu]->begin() + num_clusters,
                    buf->begin(),
                    queue[mu]);

            dp.add_event() = future.get_event();
            future.wait();
        }

        Event sync_centroids(
                Measurement::DataPoint& datapoint,
                boost::compute::wait_list const& /* wait_list */
                )
        {
            using Device = boost::compute::device;

            datapoint.set_name("SyncCentroids");

            if (not device_map[cu][ll]) {
                size_t num_elements = num_clusters * num_features;
                Future copy_future;
                if (queue[cu].get_device().type() == Device::cpu) {
                    auto& buf = centroids[cu]->get_buffer();

                    Event map_event;
                    PointT *buf_ptr = (PointT*) queue[cu]
                        .enqueue_map_buffer(
                                buf,
                                CL_MAP_READ | CL_MAP_WRITE,
                                0,
                                num_elements * sizeof(PointT));

                    copy_future = boost::compute::copy_async(
                            buf_ptr,
                            buf_ptr + num_elements,
                            centroids[ll]->begin(),
                            queue[ll]);

                    copy_future.wait();

                    queue[cu].enqueue_unmap_buffer(
                            buf,
                            buf_ptr);
                }
                else {
                    copy_future = boost::compute::copy_async(
                            centroids[cu]->begin(),
                            centroids[cu]->begin() + num_elements,
                            centroids[ll]->begin(),
                            queue[ll]);

                    copy_future.wait();
                }

                datapoint.add_event() = copy_future.get_event();
            }

            Event e;
            return e;
        }

        Event sync_labels(
                Measurement::DataPoint& datapoint,
                boost::compute::wait_list const& /* wait_list */
                )
        {
            using Device = boost::compute::device;

            datapoint.set_name("SyncLabels");

            if (not device_map[ll][mu]) {
                Event sync_event;
                auto dev_type = queue[mu].get_device().type();

                if (dev_type == Device::cpu) {

                    auto& dev_buf = labels[ll]->get_buffer();

                    LabelT *dev_buf_ptr = (LabelT*) queue[ll]
                        .enqueue_map_buffer(
                                dev_buf,
                                CL_MAP_READ,
                                0,
                                dev_buf.size(),
                                sync_event
                                );

                    // Warning: Potential use-after-free violation
                    // Probably works because we're using pinned memory
                    // and (hopefully) not exceeding the object's lifetime
                    boost::compute::buffer host_buf(
                            context[mu],
                            dev_buf.size(),
                            boost::compute::buffer::read_write |
                            boost::compute::buffer::use_host_ptr,
                            dev_buf_ptr
                            );

                    // Explicitly specialized template function
                    // Warning: This violates sanity
                    labels[mu]->assign(host_buf, host_buf);

                    queue[ll].enqueue_unmap_buffer(
                            labels[ll]->get_buffer(),
                            dev_buf_ptr);

                }
                else {
                    Future copy_future = boost::compute::copy_async(
                            labels[ll]->begin(),
                            labels[ll]->end(),
                            labels[mu]->begin(),
                            queue[mu]);

                    copy_future.wait();
                    sync_event = copy_future.get_event();
                }

                datapoint.add_event() = sync_event;
            }

            if (not device_map[ll][cu] && not device_map[mu][cu]) {
                Future copy_future;
                auto dev_type = queue[cu].get_device().type();

                if (dev_type == Device::cpu) {
                    auto& buf = labels[cu]->get_buffer();

                    LabelT *buf_ptr = (LabelT*) queue[cu]
                        .enqueue_map_buffer(
                                buf,
                                CL_MAP_READ | CL_MAP_WRITE,
                                0,
                                buf.size());

                    copy_future = boost::compute::copy_async(
                            labels[ll]->begin(),
                            labels[ll]->end(),
                            buf_ptr,
                            queue[ll]);

                    copy_future.wait();

                    queue[cu].enqueue_unmap_buffer(
                            buf,
                            buf_ptr);
                }
                else {
                    boost::compute::copy(
                            labels[ll]->begin(),
                            labels[ll]->end(),
                            labels[cu]->begin(),
                            queue[cu]);

                    copy_future.wait();
                }

                datapoint.add_event() = copy_future.get_event();
            }

            Event e;
            return e;
        }

        Event sync_masses(boost::compute::wait_list const& /* wait_list */)
        {
            if (not device_map[mu][cu]) {
                boost::compute::copy(
                        masses[mu]->begin(),
                        masses[mu]->begin() + num_clusters,
                        masses[cu]->begin(),
                        queue[cu]);
            }

            Event e;
            return e;
        }

        void shrink_centroids() {
            for (auto& buf : centroids) {
                if (buf) {
                    buf->resize(num_clusters * num_features);
                }
            }
        }

        void shrink_masses() {
            for (auto& buf : masses) {
                if (buf) {
                    buf->resize(num_clusters);
                }
            }
        }

        Vector<PointT>& get_points(BufferMap::Phase p) {
            return *points[p];
        }

        Vector<PointT>& get_centroids(BufferMap::Phase p) {
            return *centroids[p];
        }

        PinnedVector<LabelT>& get_labels(BufferMap::Phase p) {
            return *labels[p];
        }

        Vector<MassT>& get_masses(BufferMap::Phase p) {
            return *masses[p];
        }

        size_t num_features;
        size_t num_points;
        size_t num_clusters;
        std::vector<std::vector<int>> device_map;
        std::vector<boost::compute::command_queue> queue;
        std::vector<boost::compute::context> context;
        std::vector<VectorPtr<PointT>> points;
        std::vector<VectorPtr<PointT>> centroids;
        std::vector<PinnedVectorPtr<LabelT>> labels;
        std::vector<VectorPtr<MassT>> masses;
    } buffer_map;
};

}

#endif /* KMEANS_THREE_STAGE_HPP */
