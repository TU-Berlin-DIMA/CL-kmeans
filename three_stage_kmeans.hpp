#ifndef THREE_STAGE_KMEANS_HPP
#define THREE_STAGE_KMEANS_HPP

#include "abstract_kmeans.hpp"
#include "labeling_factory.hpp"
#include "mass_update_factory.hpp"
#include "centroid_update_factory.hpp"

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
class ThreeStageKmeans :
    public AbstractKmeans<PointT, LabelT, MassT, ColMajor>
{
public:
    template <typename T>
    using Vector = boost::compute::vector<T>;
    template <typename T>
    using VectorPtr = std::shared_ptr<Vector<T>>;
    using Event = boost::compute::event;

    using LabelingFunction = typename LabelingFactory<PointT, LabelT, ColMajor>::LabelingFunction;
    using MassUpdateFunction = typename MassUpdateFactory<LabelT, MassT>::MassUpdateFunction;
    using CentroidUpdateFunction = typename CentroidUpdateFactory<PointT, LabelT, MassT, ColMajor>::CentroidUpdateFunction;

    ThreeStageKmeans(boost::compute::context const& context = boost::compute::system::default_context()) :
        AbstractKmeans<PointT, LabelT, MassT, ColMajor>(context)
    {}

    void run() {

        boost::compute::wait_list ll_wait_list, mu_wait_list, cu_wait_list;
        boost::compute::wait_list sync_labels_wait_list, sync_centroids_wait_list, sync_masses_wait_list;
        Event ll_event, mu_event, cu_event;
        Event sync_labels_event, sync_centroids_event, sync_masses_event;

        // If don't have points device vector but do have host map,
        // create device vector and measure copy time
        if ((not this->points) and (not this->points_view.empty())) {
            this->points = std::make_shared<Vector<const PointT>>(
                    this->points_view.size(),
                    this->context);

            boost::compute::future<void> copy_future;
            copy_future = boost::compute::copy_async(
                    this->points_view.begin(),
                    this->points_view.end(),
                    this->points->begin(),
                    this->q_labeling);
            copy_future.wait();
        }

        std::vector<int> host_did_changes(1);
        VectorPtr<int> ll_did_changes = std::make_shared<Vector<int>>(
                host_did_changes.size(),
                this->context);

        // If centroids initializer function is callable, then call
        if (this->centroids_initializer) {
            this->centroids_initializer(
                    *this->points,
                    *this->centroids);
        }

        this->buffer_map.set_queues(
                this->q_labeling,
                this->q_mass_update,
                this->q_centroid_update);

        this->buffer_map.set_parameters(
                this->num_features,
                this->num_points,
                this->num_clusters);

        this->buffer_map.set_buffers(
                this->points,
                this->centroids,
                this->labels,
                this->masses);

        uint32_t iterations = 0;
        bool did_changes = true;
        while (did_changes == true && iterations < this->max_iterations) {

            // set did_changes to false on device
            boost::compute::fill(
                    ll_did_changes->begin(),
                    ll_did_changes->end(),
                    0);

            // execute labeling
            sync_centroids_event = buffer_map.sync_centroids(
                    sync_centroids_wait_list);
            ll_wait_list.insert(
                    sync_centroids_event);
            ll_event = this->f_labeling(
                    this->q_labeling,
                    this->num_features,
                    this->num_points,
                    this->num_clusters,
                    *ll_did_changes,
                    buffer_map.get_points(BufferMap::ll),
                    buffer_map.get_centroids(BufferMap::ll),
                    buffer_map.get_labels(BufferMap::ll),
                    this->logger,
                    ll_wait_list);

            // copy did_changes from device to host
            boost::compute::copy(
                    host_did_changes.begin(),
                    host_did_changes.end(),
                    ll_did_changes->begin(),
                    q_labeling);

            // inspect did_changes
            did_changes = std::any_of(
                    host_did_changes.cbegin(),
                    host_did_changes.cend(),
                    [](int i){ return i == 1; }
                    );

            if (did_changes == true) {
                // execute mass update
                sync_labels_event = buffer_map.sync_labels(
                        sync_labels_wait_list);
                mu_wait_list.insert(
                        sync_labels_event);
                cu_wait_list.insert(
                        sync_labels_event);
                mu_event = this->f_mass_update(
                        this->q_mass_update,
                        this->num_points,
                        this->num_clusters,
                        buffer_map.get_labels(BufferMap::mu),
                        buffer_map.get_masses(BufferMap::mu),
                        this->logger,
                        mu_wait_list);
                sync_masses_wait_list.insert(
                        mu_event);

                // execute centroid update
                sync_masses_event = buffer_map.sync_masses(
                        sync_masses_wait_list);
                cu_wait_list.insert(
                        sync_masses_event);
                cu_event = this->f_centroid_update(
                        this->q_centroid_update,
                        this->num_features,
                        this->num_points,
                        this->num_clusters,
                        buffer_map.get_points(BufferMap::cu),
                        buffer_map.get_centroids(BufferMap::cu),
                        buffer_map.get_labels(BufferMap::cu),
                        buffer_map.get_masses(BufferMap::cu),
                        this->logger,
                        cu_wait_list);
                sync_centroids_wait_list.insert(
                        cu_event);
            }

            ++iterations;
        }


        // copy centroids to host
        sync_centroids_event = buffer_map.sync_centroids(
                sync_centroids_wait_list);
        boost::compute::wait_for_all(
                sync_centroids_event);
    }

    void set_labeler(std::string flavor, LabelingConfiguration config) {
        LabelingFactory<PointT, LabelT, ColMajor> factory;
        f_labeling = factory.create(flavor, this->context, config);
    }

    void set_mass_updater(std::string flavor, MassUpdateConfiguration config) {
        MassUpdateFactory<LabelT, MassT> factory;
        f_mass_update = factory.create(flavor, this->context, config);
    }

    void set_centroid_updater(std::string flavor, CentroidUpdateConfiguration config) {
        CentroidUpdateFactory<PointT, LabelT, MassT, ColMajor> factory;
        f_centroid_update = factory.create(flavor, this->context, config);
    }

    void set_labeling_queue(boost::compute::command_queue q) {
        q_labeling = q;
    }

    void set_mass_update_queue(boost::compute::command_queue q) {
        q_mass_update = q;
    }

    void set_centroid_update_queue(boost::compute::command_queue q) {
        q_centroid_update = q;
    }

private:
    LabelingFunction f_labeling;
    MassUpdateFunction f_mass_update;
    CentroidUpdateFunction f_centroid_update;

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
            for (auto v : device_map) {
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

        void set_parameters(size_t num_features, size_t num_points, size_t num_clusters) {

            this->num_features = num_features;
            this->num_points = num_points;
            this->num_clusters = num_clusters;
        }

        void set_buffers(VectorPtr<const PointT> p_buf, VectorPtr<PointT> c_buf, VectorPtr<LabelT> l_buf, VectorPtr<MassT> m_buf) {

            points.resize(3);
            points[ll] = p_buf;
            points[mu] = nullptr;
            points[cu] = device_map[ll][cu] ? points[ll] :
                std::make_shared<Vector<const PointT>>(*points[ll], queue[cu]);

            c_buf->resize(num_clusters * num_features);
            centroids.resize(3);
            centroids[ll] = c_buf;
            centroids[mu] = nullptr;
            centroids[cu] = device_map[cu][ll] ? centroids[ll] :
                std::make_shared<Vector<PointT>>(*centroids[ll], queue[cu]);

            l_buf->resize(num_points);
            labels.resize(3);
            labels[ll] = l_buf;
            labels[mu] = device_map[mu][ll] ? labels[ll] :
                std::make_shared<Vector<LabelT>>(*labels[ll], queue[mu]);
            labels[cu] = device_map[cu][ll] ? labels[ll] :
                device_map[cu][mu] ? labels[mu] :
                std::make_shared<Vector<LabelT>>(*labels[ll], queue[cu]);

            m_buf->resize(num_clusters);
            masses[mu] = m_buf;
            masses[cu] = device_map[cu][mu] ? masses[mu] :
                std::make_shared<Vector<MassT>>(*masses[mu], queue[cu]);
        }

        Event sync_centroids(boost::compute::wait_list const& wait_list) {
            // TODO: defensive copy; lengths may not be the same
            if (not device_map[cu][ll]) {
                boost::compute::copy(
                        centroids[cu]->begin(),
                        centroids[cu]->end(),
                        centroids[ll]->begin(),
                        queue[ll]);
            }

            Event e;
            return e;
        }

        Event sync_labels(boost::compute::wait_list const& wait_list) {
            // TODO: defensive copy; lengths may not be the same
            if (not device_map[ll][mu]) {
                boost::compute::copy(
                        labels[ll]->begin(),
                        labels[ll]->end(),
                        labels[mu]->begin(),
                        queue[mu]);
            }

            // TODO: defensive copy; lengths may not be the same
            if (not device_map[ll][cu]) {
                boost::compute::copy(
                        labels[ll]->begin(),
                        labels[ll]->end(),
                        labels[cu]->begin(),
                        queue[cu]);
            }

            Event e;
            return e;
        }

        Event sync_masses(boost::compute::wait_list const& wait_list) {
            // TODO: defensive copy; lengths may not be the same
            if (not device_map[mu][cu]) {
                boost::compute::copy(
                        masses[mu]->begin(),
                        masses[mu]->end(),
                        masses[cu]->begin(),
                        queue[cu]);
            }

            Event e;
            return e;
        }

        Vector<const PointT>& get_points(BufferMap::Phase p) {
            return *points[p];
        }

        Vector<PointT>& get_centroids(BufferMap::Phase p) {
            return *centroids[p];
        }

        Vector<LabelT>& get_labels(BufferMap::Phase p) {
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
        std::vector<VectorPtr<const PointT>> points;
        std::vector<VectorPtr<PointT>> centroids;
        std::vector<VectorPtr<LabelT>> labels;
        std::vector<VectorPtr<MassT>> masses;
    } buffer_map;
};

}

#endif /* THREE_STAGE_KMEANS_HPP */
