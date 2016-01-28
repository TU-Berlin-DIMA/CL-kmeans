#include "kmeans.hpp"

#include <random>

void cle::KmeansInitializer::random(
        std::vector<double> const& points_x,
        std::vector<double> const& points_y,
        std::vector<double>& centroids_x,
        std::vector<double>& centroids_y) {

    std::random_device rand;
    const size_t num_points = points_x.size();
    const size_t num_clusters = centroids_x.size();


    for (size_t c = 0; c != num_clusters; ++c) {
        size_t random_point = rand() % num_points;
        centroids_x[c] = points_x[random_point];
        centroids_y[c] = points_y[random_point];
    }
}

void cle::KmeansInitializer::first_x(
        std::vector<double> const &points_x,
        std::vector<double> const& points_y,
        std::vector<double>& centroids_x,
        std::vector<double>& centroids_y) {

    const size_t num_clusters = centroids_x.size();

    for (size_t c = 0; c != num_clusters; ++c) {
        centroids_x[c] = points_x[c];
        centroids_y[c] = points_y[c];
    }
}
