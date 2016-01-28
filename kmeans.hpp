#ifndef KMEANS_HPP
#define KMEANS_HPP

#include "kmeans_cl_api.hpp"

#include <vector>
#include <random>
#include <cstdint>
#include <cstddef> // size_t

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace cle {

using KmeansStats = struct { uint32_t iterations; };

class KmeansInitializer {
public:
    static void random(
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

    static void first_x(
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
};

class KmeansGPU {
public:
    KmeansGPU(
            cl::Context const& context,
            cl::CommandQueue const& queue
            );

    int initialize();
    int finalize();

    int operator() (
            uint32_t const max_iterations,
            std::vector<double> const& points_x,
            std::vector<double> const& points_y,
            std::vector<double>& centroids_x,
            std::vector<double>& centroids_y,
            std::vector<uint64_t>& cluster_size,
            std::vector<uint64_t>& memberships,
            KmeansStats& stats
            );

private:
    cle::Kmeans_With_Host_Kernel kmeans_kernel_;
    cl::Context context_;
    cl::CommandQueue queue_;

    size_t max_work_group_size_;
    std::vector<size_t> max_work_item_sizes_;
};

template <typename T>
class KmeansNaive {
public:
    int initialize() { return 1; }
    int finalize() { return 1; }

    void operator() (uint32_t const max_iterations,
                      std::vector<double> const& points_x,
                      std::vector<double> const& points_y,
                      std::vector<double>& centroids_x,
                      std::vector<double>& centroids_y,
                      std::vector<uint64_t>& cluster_size,
                      std::vector<uint64_t>& memberships,
                      KmeansStats& stats) {

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

private:
    double gaussian_distance(double a_x, double a_y, double b_x, double b_y) {
      double t_x = b_x - a_x;
      double t_y = b_y - a_y;

      return t_x * t_x + t_y * t_y;
    }

};

}
#endif /* KMEANS_HPP */
