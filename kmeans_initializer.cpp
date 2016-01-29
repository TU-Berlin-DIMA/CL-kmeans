#include "kmeans.hpp"

#include <random>

template <typename FP>
void cle::KmeansInitializer<FP>::forgy(
        std::vector<FP> const& points_x,
        std::vector<FP> const& points_y,
        std::vector<FP>& centroids_x,
        std::vector<FP>& centroids_y) {

    std::random_device rand;
    const size_t num_points = points_x.size();
    const size_t num_clusters = centroids_x.size();


    for (size_t c = 0; c != num_clusters; ++c) {
        size_t random_point = rand() % num_points;
        centroids_x[c] = points_x[random_point];
        centroids_y[c] = points_y[random_point];
    }
}

template <typename FP>
void cle::KmeansInitializer<FP>::first_x(
        std::vector<FP> const &points_x,
        std::vector<FP> const& points_y,
        std::vector<FP>& centroids_x,
        std::vector<FP>& centroids_y) {

    const size_t num_clusters = centroids_x.size();

    for (size_t c = 0; c != num_clusters; ++c) {
        centroids_x[c] = points_x[c];
        centroids_y[c] = points_y[c];
    }
}

template class cle::KmeansInitializer<float>;
template class cle::KmeansInitializer<double>;
