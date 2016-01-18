#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>

#include "csv.hpp"
#include "utils.hpp"

using centroid_distance = std::pair<size_t, double>;

double two_norm(double a_x, double a_y, double b_x, double b_y) {
    double t_x = b_x - a_x;
    double t_y = b_y - a_y;

    return t_x * t_x + t_y * t_y;
}

bool min_centroid_distance(centroid_distance const& a, centroid_distance const& b) {
    return std::get<1>(a) < std::get<1>(b);
}

void kmeans_naive(double const epsilon,
        std::vector<double> const& points_x, std::vector<double> const& points_y,
        std::vector<double>& centroids_x, std::vector<double>& centroids_y,
        std::vector<centroid_distance>& cluster_assignment) {

    assert(points_x.size() == points_y.size());
    assert(centroids_x.size() == centroids_y.size());

    if (cluster_assignment.size() != points_x.size()) {
        cluster_assignment.resize(points_x.size());
    }

    double old_sum_distances = 0;
    double sum_distances = 2 * epsilon;
    double distance = 0;
    centroid_distance min_centroid;
    centroid_distance cur_centroid;
    std::vector<size_t> cluster_size(centroids_x.size());

    while (std::abs(sum_distances - old_sum_distances) > epsilon) {

        // Phase 1: assign points to clusters
        old_sum_distances = sum_distances;
        sum_distances = 0;
        for (size_t p = 0; p != points_x.size(); ++p) {
            min_centroid = std::make_pair(
                std::numeric_limits<size_t>::max(),
                std::numeric_limits<double>::max());
 
            for (size_t c = 0; c != centroids_x.size(); ++c) {
                distance = two_norm(points_x[p], points_y[p], centroids_x[c], centroids_y[c]);
                cur_centroid = std::make_pair(c, distance);

                min_centroid = std::min(min_centroid, cur_centroid, min_centroid_distance);
            }

            cluster_assignment[p] = min_centroid;
            sum_distances += distance;
        }

        // Phase 2: calculate new clusters
        // Arithmetic mean of all points assigned to cluster
        std::fill(cluster_size.begin(), cluster_size.end(), 0);
        std::fill(centroids_x.begin(), centroids_x.end(), 0);
        std::fill(centroids_y.begin(), centroids_y.end(), 0);

        for (size_t p = 0; p != points_x.size(); ++p) {
            size_t c = std::get<0>(cluster_assignment[p]);

            cluster_size[c] += 1;
            centroids_x[c] += points_x[p];
            centroids_y[c] += points_y[p];
        }

        for (size_t c = 0; c != centroids_x.size(); ++c) {
            centroids_x[c] = centroids_x[c] / cluster_size[c];
            centroids_y[c] = centroids_y[c] / cluster_size[c];
        }
    }

}


int main(int argc, char **argv) {

    if (argc != 2) {
        std::cerr << "Usage: test_kmeans [input file]" << std::endl;
        return 1;
    }

    char *input_path = argv[1];

    cle::CSV csv;
    std::vector<double> points_x, points_y;

    csv.read_csv(input_path, points_x, points_y);
    std::cout << "Read CSV" << std::endl;

    cle::Utils::print_vector(points_x);

    constexpr size_t num_clusters = 9;
    double epsilon = 0.05;
    std::vector<centroid_distance> cluster_assignment;

    std::random_device rand;
    std::vector<double> centroids_x(num_clusters), centroids_y(num_clusters);

    for (size_t c = 0; c != centroids_x.size(); ++c) {
        size_t random_point = rand() % points_x.size();
        centroids_x[c] = points_x[random_point];
        centroids_y[c] = points_y[random_point];
    }

    kmeans_naive(epsilon, points_x, points_y, centroids_x, centroids_y, cluster_assignment);

    std::cout << "Clusters:" << std::endl;
    for (auto x : cluster_assignment) {
        std::cout << "Centroid: " << x.first << " Distance to centroid: " << x.second << std::endl;
    }
}
