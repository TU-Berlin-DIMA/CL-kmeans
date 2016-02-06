/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "csv.hpp"
#include "common.hpp"
#include "cle/common.hpp"
#include "clustering_benchmark.hpp"

#include "kmeans.hpp"
#include "matrix.hpp"

#include <iostream>
#include <vector>
#include <algorithm>
#include <memory>

#include <boost/align/aligned_allocator.hpp>

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cerr << "Usage: test_kmeans [input file]" << std::endl;
    return 1;
  }

  char *input_path = argv[1];

  cle::CSV csv;
  cle::Matrix<float, cle::AlignedAllocatorFP32, uint32_t, true> points_32;
  cle::Matrix<double, cle::AlignedAllocatorFP64, uint64_t, true> points_64;

  cle::CLInitializer clinit;
  if (clinit.choose_platform_interactive() < 0) {
      return 0;
  }
  if (clinit.choose_device_interactive() < 0) {
      return 0;
  }

  csv.read_csv(input_path, points_32);
  csv.read_csv(input_path, points_64);

  std::cout << "Read " << input_path << std::endl;
  uint64_t bytes_read = points_64.size() * sizeof(double);
  std::cout << "Read " << bytes_read / 1024 / 1024 << " MiB" << std::endl;

  constexpr uint32_t num_runs = 5;
  constexpr uint32_t max_iterations = 100;
  constexpr uint64_t num_clusters = 10;
  uint64_t num_points = points_32.rows();

  cle::ClusteringBenchmark32Aligned bm32(
          num_runs, num_points, max_iterations,
          std::move(points_32));
  bm32.initialize(num_clusters, points_32.cols(), cle::KmeansInitializer32Aligned::first_x);

  cle::ClusteringBenchmark64Aligned bm64(
          num_runs, num_points, max_iterations,
          std::move(points_64));
  bm64.initialize(num_clusters, points_64.cols(), cle::KmeansInitializer64Aligned::first_x);

  cle::KmeansNaive32Aligned kmeans_naive_32;
  kmeans_naive_32.initialize();

  cle::KmeansNaive64Aligned kmeans_naive_64;
  kmeans_naive_64.initialize();

  cle::KmeansSIMD32 kmeans_simd_32;
  kmeans_simd_32.initialize();

  cle::KmeansGPUAssisted32Aligned kmeans_gpu_32(
          clinit.get_context(), clinit.get_commandqueue()
          );
  kmeans_gpu_32.initialize();

  cle::KmeansGPUAssisted64Aligned kmeans_gpu_64(
          clinit.get_context(), clinit.get_commandqueue()
          );
  kmeans_gpu_64.initialize();

  bm32.setVerificationReference(kmeans_naive_32);
  bm64.setVerificationReference(kmeans_naive_64);

  // int is_kmeans_simd_32_correct = bm32.verify(kmeans_simd_32);
  // if (is_kmeans_simd_32_correct) {
  //     std::cout << "SIMD 32 is correct" << std::endl;
  // }
  // else {
  //     std::cout << "SIMD 32 is wrong!!!" << std::endl;
  // }
  //
  int is_kmeans_gpu_32_correct = bm32.verify(kmeans_gpu_32);
  if (is_kmeans_gpu_32_correct) {
      std::cout << "GPU assisted 32 is correct" << std::endl;
  }
  else {
      std::cout << "GPU assisted 32 is wrong!!!" << std::endl;
  }

  int is_kmeans_gpu_64_correct = bm64.verify(kmeans_gpu_64);
  if (is_kmeans_gpu_64_correct) {
      std::cout << "GPU assisted 64 is correct" << std::endl;
  }
  else {
      std::cout << "GPU assisted 64 is wrong!!!" << std::endl;
  }

  cle::ClusteringBenchmarkStats bs_naive_32 = bm32.run(kmeans_naive_32);
  // bm32.print_labels();
  cle::ClusteringBenchmarkStats bs_gpu_32 = bm32.run(kmeans_gpu_32);

  // cle::ClusteringBenchmarkStats bs_simd_32 = bm32.run(kmeans_simd_32);
  cle::ClusteringBenchmarkStats bs_naive_64 = bm64.run(kmeans_naive_64);
  cle::ClusteringBenchmarkStats bs_gpu_64 = bm64.run(kmeans_gpu_64);

  kmeans_naive_32.finalize();
  kmeans_naive_64.finalize();
  kmeans_simd_32.finalize();
  kmeans_gpu_32.finalize();
  kmeans_gpu_64.finalize();

  bm32.finalize();
  bm64.finalize();

  bs_naive_32.print_times();
  // bs_simd_32.print_times();
  bs_naive_32.print_times();
  bs_naive_64.print_times();
  bs_gpu_64.print_times();

  // std::cout << "Point: Naive | GPU" << std::endl;
  // uint32_t num_diff = 0;
  // for (uint32_t p = 0; p < points_x.size(); ++p) {
  //     if (memberships_naive[p] != memberships_gpu[p]) {
  //         ++num_diff;
          // std::cout << p << ": " << memberships_naive[p] << " | " << memberships_gpu[p] << std::endl;
  //     }
  // }
  // std::cout << "[" << num_diff << " / " << points_x.size() << "]" << std::endl;
}
