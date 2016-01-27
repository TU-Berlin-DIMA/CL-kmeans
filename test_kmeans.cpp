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

using kmeans_stats = struct { uint64_t iterations; };

double gaussian_distance(double a_x, double a_y, double b_x, double b_y) {
  double t_x = b_x - a_x;
  double t_y = b_y - a_y;

  return t_x * t_x + t_y * t_y;
}

void kmeans_naive(uint32_t const max_iterations,
                  std::vector<double> const &points_x,
                  std::vector<double> const &points_y,
                  std::vector<double> &centroids_x,
                  std::vector<double> &centroids_y,
                  std::vector<uint64_t>& cluster_size,
                  std::vector<uint64_t>& memberships,
                  kmeans_stats &stats) {

  assert(points_x.size() == points_y.size());
  assert(centroids_x.size() == centroids_y.size());
  assert(memberships.size() == points_x.size());
  assert(cluster_size.size() == centroids_x.size());

  bool did_changes;
  uint64_t iterations;

  iterations = 0;
  did_changes = true;
  while (did_changes == true && iterations < max_iterations) {
    did_changes = false;

    // Phase 1: assign points to clusters
    for (size_t p = 0; p != points_x.size(); ++p) {
        double min_distance = std::numeric_limits<double>::max();
        size_t min_centroid;

      for (size_t c = 0; c != centroids_x.size(); ++c) {
          double distance =
              gaussian_distance(points_x[p], points_y[p], centroids_x[c], centroids_y[c]);
          if (distance < min_distance) {
              min_distance = distance;
              min_centroid = c;
          }
      }

      if (min_centroid != memberships[p]) {
        memberships[p] = min_centroid;
        did_changes = true;
      }
    }

    // Phase 2: calculate new clusters
    // Arithmetic mean of all points assigned to cluster
    std::fill(cluster_size.begin(), cluster_size.end(), 0);
    std::fill(centroids_x.begin(), centroids_x.end(), 0);
    std::fill(centroids_y.begin(), centroids_y.end(), 0);

    for (size_t p = 0; p != points_x.size(); ++p) {
      size_t c = memberships[p];

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

void kmeans_initialize_random(std::vector<double> const &points_x,
                              std::vector<double> const &points_y,
                              std::vector<double> &centroids_x,
                              std::vector<double> &centroids_y) {

  std::random_device rand;
  const size_t num_points = points_x.size();
  const size_t num_clusters = centroids_x.size();

  for (size_t c = 0; c != num_clusters; ++c) {
    size_t random_point = rand() % num_points;
    centroids_x[c] = points_x[random_point];
    centroids_y[c] = points_y[random_point];
  }
}

void kmeans_initialize_first_x(std::vector<double> const &points_x,
                               std::vector<double> const &points_y,
                               std::vector<double> &centroids_x,
                               std::vector<double> &centroids_y) {

  const size_t num_clusters = centroids_x.size();

  for (size_t c = 0; c != num_clusters; ++c) {
    centroids_x[c] = points_x[c];
    centroids_y[c] = points_y[c];
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
  std::cout << "Read " << input_path << std::endl;

  // cle::Utils::print_vector(points_x);

  constexpr size_t num_clusters = 9;
  constexpr uint32_t max_iterations = 1;
  size_t num_points = points_x.size();
  std::vector<uint64_t> memberships_naive(num_points);
  std::vector<uint64_t> memberships_gpu(num_points);

  std::vector<double> centroids_x(num_clusters);
  std::vector<double> centroids_y(num_clusters);
  std::vector<uint64_t> cluster_size(num_clusters);

  kmeans_stats statistics = {};
  kmeans_initialize_first_x(points_x, points_y, centroids_x, centroids_y);

  uint64_t micros = 0;
  cle::Timer timer;
  timer.start();
  kmeans_naive(max_iterations, points_x, points_y, centroids_x, centroids_y,
               cluster_size, memberships_naive, statistics);
  micros = timer.stop<std::chrono::microseconds>();

  std::cout << "Runtime: " << micros << " µs" << std::endl;
  std::cout << "# iterations: " << statistics.iterations << std::endl;

  kmeans_initialize_first_x(points_x, points_y, centroids_x, centroids_y);

  cle::CLInitializer clinit;
  clinit.choose_platform_interactive();
  clinit.choose_device_interactive();

  cle::Kmeans_GPU kmeans_gpu(clinit.get_context(), clinit.get_commandqueue());
  kmeans_gpu.initialize();

  timer.start();
  kmeans_gpu(max_iterations, points_x, points_y, centroids_x, centroids_y,
             cluster_size, memberships_gpu);
  micros = timer.stop<std::chrono::microseconds>();
  std::cout << "Runtime: " << micros << " µs" << std::endl;

  kmeans_gpu.finalize();

  bool are_equal = std::equal(memberships_naive.begin(), memberships_naive.end(),
          memberships_gpu.begin());

  if (are_equal == true) {
      std::cout << "Naive and GPU cluster memberships are identical" << std::endl;
  }
  else {
      std::cout << "Naive and GPU cluster memberships are different; something is wrong!!!" << std::endl;
  }

  std::cout << "Point: Naive | GPU" << std::endl;
  uint32_t num_diff = 0;
  for (uint32_t p = 0; p < points_x.size(); ++p) {
      if (memberships_naive[p] != memberships_gpu[p]) {
          ++num_diff;
          std::cout << p << ": " << memberships_naive[p] << " | " << memberships_gpu[p] << std::endl;
      }
  }
  std::cout << "[" << num_diff << " / " << points_x.size() << "]" << std::endl;
}
