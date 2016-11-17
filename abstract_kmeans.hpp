#ifndef ABSTRACT_KMEANS_HPP
#define ABSTRACT_KMEANS_HPP

#include "temp.hpp"

#include <functional>
#include <cstdint>
#include <memory>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, typename MassT, bool ColMajor = true>
class AbstractKmeans {
public:
    template <typename T>
    using Vector = boost::compute::vector<T>;
    template <typename T>
    using VectorPtr = std::shared_ptr<Vector<T>>;

    using InitCentroidsFunction = std::function<
        void(
                Vector<PointT>& points,
                Vector<PointT> & centroids
                )
        >;

    AbstractKmeans(boost::compute::context const& context = boost::compute::system::default_context()) :
        context(context),
        num_features(0),
        num_points(0),
        num_clusters(0),
        points(nullptr),
        host_points(nullptr),
        centroids(new Vector<PointT>(context)),
        masses(new Vector<MassT>(context)),
        labels(new Vector<LabelT>(context))
    {}

    virtual ~AbstractKmeans() {}

    virtual void set_max_iterations(size_t i) {
        this->max_iterations = i;
    }

    virtual void set_points(VectorPtr<PointT> p) {
        this->points = p;

        if (this->num_features != 0) {
            this->num_points = p->size() / this->num_features;
        }
    }

    virtual void set_points(std::shared_ptr<const std::vector<PointT>> p) {
        this->host_points = p;

        if (this->num_features != 0) {
            this->num_points = p->size() / this->num_features;
        }
    }

    virtual void set_features(size_t f) {
        this->num_features = f;

        if (this->points) {
            this->num_points = this->points->size() / f;
        }
        else if (this->host_points) {
            this->num_points = this->host_points->size() / f;
        }
    }

    virtual void set_clusters(size_t c) {
        this->num_clusters = c;
    }

    virtual void set_initializer(InitCentroidsFunction f) {
        this->centroids_initializer = f;
    }

    virtual Vector<PointT> const& get_centroids() const {
        return *this->centroids;
    }

    virtual Vector<LabelT> const& get_labels() const {
        return *this->labels;
    }

    virtual Vector<MassT> const& get_cluster_masses() const {
        return *this->masses;
    }

    virtual MeasurementLogger const& get_measurement_logger() const {
        return this->logger;
    }

    virtual void run() = 0;

    virtual void operator() (
            size_t max_iterations,
            size_t num_features,
            std::shared_ptr<const std::vector<PointT>> points,
            VectorPtr<PointT> centroids,
            VectorPtr<MassT> masses,
            VectorPtr<LabelT> labels) {

        this->max_iterations = max_iterations;
        this->num_features = num_features;
        this->num_points = points->size() / num_features;
        this->num_clusters = centroids->size() / num_features;
        this->host_points = points;
        this->centroids = centroids;
        this->masses = masses;
        this->labels = labels;

        this->run();
    }

protected:
    boost::compute::context context;
    size_t max_iterations;
    size_t num_features;
    size_t num_points;
    size_t num_clusters;
    VectorPtr<PointT> points;
    std::shared_ptr<const std::vector<PointT>> host_points;
    VectorPtr<PointT> centroids;
    VectorPtr<MassT> masses;
    VectorPtr<LabelT> labels;

    InitCentroidsFunction centroids_initializer;
    MeasurementLogger logger;
};

} // Clustering

#endif /* ABSTRACT_KMEANS_HPP */
