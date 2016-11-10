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
                Vector<const PointT>& points,
                Vector<PointT> & centroids
                )
        >;

    AbstractKmeans(boost::compute::context const& context = boost::compute::system::default_context()) :
        context(context),
        points(nullptr),
        centroids(new Vector<PointT>(context)),
        labels(new Vector<LabelT>(context)),
        masses(new Vector<MassT>(context))
    {}

    virtual ~AbstractKmeans() {}

    virtual void set_max_iterations(size_t i) {
        this->max_iterations = i;
    }

    virtual void set_points(VectorPtr<const PointT> p, size_t num_points) {
        this->points = p;
        this->num_points = num_points;
    }

    virtual void set_features(size_t f) {
        this->num_features = f;
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

protected:
    boost::compute::context context;
    size_t max_iterations;
    size_t num_features;
    size_t num_points;
    size_t num_clusters;
    VectorPtr<const PointT> points;
    VectorPtr<PointT> centroids;
    VectorPtr<LabelT> labels;
    VectorPtr<MassT> masses;

    InitCentroidsFunction centroids_initializer;
    MeasurementLogger logger;
};

} // Clustering

#endif /* ABSTRACT_KMEANS_HPP */
