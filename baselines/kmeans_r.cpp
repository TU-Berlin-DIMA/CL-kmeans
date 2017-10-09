/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2017, Lutz, Clemens <lutzcle@cml.li>
 */

#include "../matrix.hpp"
#include "../binary_format.hpp"
#include "../timer.hpp"

extern "C"
{
#include "r_stats_kmeans.h"
}

#include <iostream>
#include <string>
#include <chrono>
#include <cstdlib> // atoi

#ifndef FLOAT_T
#define FLOAT_T double
#endif

template <typename T>
class RKmeans {
public:

  void load(cle::Matrix<T, std::allocator<T>, size_t, true>& points)
  {
    points_ = std::move(points);
  }

  uint64_t operator() (
      uint32_t max_iterations,
      uint32_t num_clusters
      )
  {
    // No data copy, reference existing centroids
    cle::Matrix<T, std::allocator<T>, size_t, true> centroids;
    centroids.resize(num_clusters, points_.cols());

    for (size_t j = 0; j < centroids.cols(); ++j) {
      for (size_t i = 0; i < centroids.rows(); ++i) {
        centroids(i, j) = points_(i, j);
      }
    }

    std::vector<int> labels(points_.rows());
    std::vector<int> masses(num_clusters);
    std::vector<T> within_sum_squares(num_clusters);

    int num_points = (int) points_.rows();
    int k = (int) num_clusters;
    int num_features = (int) points_.cols();
    int itermax = (int) max_iterations;

    Timer::Timer cpu_timer;
    cpu_timer.start();

    run_lloyd(
        points_.data(),
        &num_points,
        &num_features,
        centroids.data(),
        &k,
        labels.data(),
        &itermax,
        masses.data(),
        within_sum_squares.data()
        );

    uint64_t total_time =
        cpu_timer.stop<std::chrono::milliseconds>();

    return total_time;
  }

private:
  // template <typename X>
  // void run_lloyd(X*, int*, int*, X*, int*, int*, int*, int*, X*) {}

  void run_lloyd(
    double *x,
    int *pn,
    int *pp,
    double *cen,
    int *pk,
    int *cl,
    int *pmaxiter,
    int *nc,
    double *wss
    )
  {
    kmeans_Lloyd(x, pn, pp, cen, pk, cl, pmaxiter, nc, wss);
  }

  void run_lloyd(
    float *x,
    int *pn,
    int *pp,
    float *cen,
    int *pk,
    int *cl,
    int *pmaxiter,
    int *nc,
    float *wss
    )
  {
    kmeans_Lloyd_float(x, pn, pp, cen, pk, cl, pmaxiter, nc, wss);
  }

  cle::Matrix<T, std::allocator<T>, size_t, true> points_;
};

int main(int argc, char **argv) {

  uint32_t k = 0;
  uint32_t max_iterations = 0;
  uint32_t repititions = 5;
  uint64_t time = 0;
  std::string file_path;
  RKmeans<FLOAT_T> rkmeans;

  if (argc != 4 || (argc == 2 && (std::string(argv[1]) == "--help"))) {
    std::cout
      << "Usage: kmeans_r [file] [max iterations] [k]"
      << std::endl;
    std::exit(1);
  }

  file_path = argv[1];
  max_iterations = std::atoi(argv[2]);
  k = std::atoi(argv[3]);

  {
    cle::Matrix<FLOAT_T, std::allocator<FLOAT_T>, size_t> matrix;
    Clustering::BinaryFormat binformat;

    binformat.read(file_path.c_str(), matrix);
    rkmeans.load(matrix);
  }

  std::cout << "Time (ms)" << std::endl;
  for (uint32_t r = 0; r < repititions; ++r) {
    time = rkmeans(max_iterations, k);
    std::cout << time << std::endl;
  }
}
