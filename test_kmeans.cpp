#include "csv.hpp"
#include "timer.hpp"
#include "common.hpp"
#include "cle/common.hpp"

#include "kmeans.hpp"

#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <cstdint>


using centroid_distance = struct {
    size_t cluster;
    double distance;
};

using kmeans_stats = struct {
    uint64_t iterations;
};

double two_norm(double a_x, double a_y, double b_x, double b_y) {
    double t_x = b_x - a_x;
    double t_y = b_y - a_y;

    return t_x * t_x + t_y * t_y;
}

bool min_centroid_distance(centroid_distance const &a,
        centroid_distance const &b) {
    return a.distance < b.distance;
}

void kmeans_naive(uint32_t const max_iterations, std::vector<double> const &points_x,
        std::vector<double> const &points_y,
        std::vector<double> &centroids_x,
        std::vector<double> &centroids_y,
        std::vector<centroid_distance> &cluster_assignment,
        kmeans_stats &stats) {

    assert(points_x.size() == points_y.size());
    assert(centroids_x.size() == centroids_y.size());

    if (cluster_assignment.size() != points_x.size()) {
        cluster_assignment.resize(points_x.size());
    }

    bool did_changes;
    double distance = 0;
    centroid_distance min_centroid;
    centroid_distance cur_centroid;
    std::vector<size_t> cluster_size(centroids_x.size());
    uint64_t iterations;

    iterations = 0;
    did_changes = true;
    while (did_changes == true && iterations < max_iterations) {
        did_changes = false;

        // Phase 1: assign points to clusters
        for (size_t p = 0; p != points_x.size(); ++p) {
            min_centroid = {std::numeric_limits<size_t>::max(),
                std::numeric_limits<double>::max()};

            for (size_t c = 0; c != centroids_x.size(); ++c) {
                distance =
                    two_norm(points_x[p], points_y[p], centroids_x[c], centroids_y[c]);
                cur_centroid = {c, distance};

                min_centroid =
                    std::min(min_centroid, cur_centroid, min_centroid_distance);
            }

            if (min_centroid.cluster != cluster_assignment[p].cluster) {
                cluster_assignment[p] = min_centroid;
                did_changes = true;
            }
        }

        // Phase 2: calculate new clusters
        // Arithmetic mean of all points assigned to cluster
        std::fill(cluster_size.begin(), cluster_size.end(), 0);
        std::fill(centroids_x.begin(), centroids_x.end(), 0);
        std::fill(centroids_y.begin(), centroids_y.end(), 0);

        for (size_t p = 0; p != points_x.size(); ++p) {
            size_t c = cluster_assignment[p].cluster;

            cluster_size[c] += 1;
            centroids_x[c] += points_x[p];
            centroids_y[c] += points_y[p];
        }

        for (size_t c = 0; c != centroids_x.size(); ++c) {
            centroids_x[c] = centroids_x[c] / cluster_size[c];
            centroids_y[c] = centroids_y[c] / cluster_size[c];
        }

        ++iterations;
    }

    stats.iterations = iterations;
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
    std::cout << "Read " << input_path << std::endl;

    // cle::Utils::print_vector(points_x);

    constexpr size_t num_clusters = 9;
    constexpr uint32_t max_iterations = 100;
    size_t num_points = points_x.size();
    std::vector<centroid_distance> cluster_assignment_and_distance;
    std::vector<uint64_t> cluster_assignment(num_points);

    std::random_device rand;
    std::vector<double> centroids_x(num_clusters);
    std::vector<double> centroids_y(num_clusters);
    std::vector<uint64_t> cluster_size(num_clusters);

    kmeans_stats statistics = {};

    for (size_t c = 0; c != centroids_x.size(); ++c) {
        size_t random_point = rand() % points_x.size();
        centroids_x[c] = points_x[random_point];
        centroids_y[c] = points_y[random_point];
    }

    uint64_t nanos = 0;
    cle::Timer timer;
    timer.start();
    kmeans_naive(max_iterations, points_x, points_y, centroids_x, centroids_y,
            cluster_assignment_and_distance, statistics);
    nanos = timer.stop<std::chrono::microseconds>();

    std::cout << "Runtime: " << nanos << " µs" << std::endl;
    std::cout << "# iterations: " << statistics.iterations << std::endl;

    std::cout << "Clusters:" << std::endl;
    for (auto x : cluster_assignment_and_distance) {
        std::cout << "Centroid: " << x.cluster << std::endl;
    }

    cle::CLInitializer clinit;
    clinit.choose_platform_interactive();
    clinit.choose_device_interactive();

    cle::Kmeans_GPU kmeans_gpu(
            clinit.get_context(),
            clinit.get_commandqueue());
    kmeans_gpu.initialize();
    kmeans_gpu(
            max_iterations,
            points_x,
            points_y,
            centroids_x,
            centroids_y,
            cluster_size,
            cluster_assignment
            );
    kmeans_gpu.finalize();

    std::cout << "Clusters:" << std::endl;
    for (uint32_t p = 0; p < points_x.size(); ++p) {
        std::cout << "Point (" << points_x[p] << ", " << points_y[p] <<
            ") Centroid: " << cluster_assignment[p] << std::endl;
    }
}
