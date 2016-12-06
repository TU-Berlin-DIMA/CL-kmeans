/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */
#include "binary_format.hpp"

#include "clustering_benchmark.hpp"
#include "configuration_parser.hpp"
#include "kmeans.hpp"
#include "matrix.hpp"

#include "three_stage_kmeans.hpp"

#include "SystemConfig.h"

#include <boost/program_options.hpp>
#include <boost/compute/core.hpp>

#include <iostream>
#include <cstdint>
#include <string>
#include <set>
#include <memory>
#include <stdexcept>

#ifdef CUDA_FOUND
#include <cuda_runtime.h>
#endif

// Suppress editor errors about BENCH_NAME not defined
#ifndef BENCH_NAME
#define BENCH_NAME ""
#endif

namespace po = boost::program_options;
namespace bc = boost::compute;

class CmdOptions {
public:
    int parse(int argc, char **argv) {
        char help_msg[] =
            "Usage: " BENCH_NAME " [OPTION] [K] [FILE]\n"
            "Options"
            ;

        po::options_description cmdline(help_msg);
        cmdline.add_options()
            ("help", "Produce help message")
            ("verbose", "Show additional information")
            ("runs", po::value<uint32_t>(&runs_),
             "Number of runs")
            ("k", po::value<uint32_t>(),
             "Number of Clusters")
            ("iterations",
             po::value<uint32_t>(&iterations_),
             "Number of iterations")
            ("verify", "Do verification pass")
            ("csv",
             po::value<std::string>(),
             "Output measurements to CSV file")
            ("config",
             po::value<std::string>(),
             "Configuration file")
            ;

        po::options_description hidden("Hidden options");
        hidden.add_options()
            ("input-file", po::value<std::string>(), "Input file")
            ;

        po::options_description visible;
        visible.add(cmdline).add(hidden);

        po::positional_options_description pos;
        pos.add("input-file", 1);

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(visible)
                .positional(pos).run(), vm);
        po::notify(vm);

        if (vm.count("k")) {
            k_ = vm["k"].as<uint32_t>();
        }

        if (vm.count("runs")) {
            runs_ = vm["runs"].as<uint32_t>();
        }

        if (vm.count("iterations")) {
            iterations_ = vm["iterations"].as<uint32_t>();
        }

        if (vm.count("input-file")) {
            input_file_ = vm["input-file"].as<std::string>();
        }

        if (vm.count("help")) {
            std::cout << cmdline << std::endl;
            return -1;
        }

        if (vm.count("verbose")) {
            verbose_ = true;
        }

        if (vm.count("verify")) {
            verify_ = true;
        }

        if (vm.count("csv")) {
            csv_ = true;
            csv_file_ = vm["csv"].as<std::string>();
        }

        if (vm.count("config")) {
            config_ = true;
            config_file_ = vm["config"].as<std::string>();
        }

        // Ensure we have required options
        if (input_file_.empty()) {
            std::cout << "No input file specified." << std::endl;
            return -1;
        }

        // Ensure we have a config file
        if (config_file_.empty()) {
            std::cout << "No config file specified." << std::endl;
            return -1;
        }

        return 1;
    }

    bool verbose() const {
        return verbose_;
    }

    uint32_t k() const {
        return k_;
    }

    uint32_t runs() const {
        return runs_;
    }

    uint32_t iterations() const {
        return iterations_;
    }

    std::string input_file() const {
        return input_file_;
    }

    bool verify() const {
        return verify_;
    }

    bool csv() const {
        return csv_;
    }

    std::string csv_file() const {
        return csv_file_;
    }

    bool config() const {
        return config_;
    }

    std::string config_file() const {
        return config_file_;
    }

private:
    bool verbose_ = false;
    uint32_t k_ = 0;
    uint32_t runs_ = 0;
    uint32_t iterations_ = 0;
    std::string input_file_;
    bool verify_ = false;
    bool csv_ = false;
    std::string csv_file_;
    bool config_ = false;
    std::string config_file_;
};

template <typename PointT, typename LabelT, typename MassT, bool ColMajor = true>
class Bench {
public:
    int run(CmdOptions options, Clustering::ConfigurationParser config) {
        auto bm_config = config.get_benchmark_configuration();
        auto km_config = config.get_kmeans_configuration();

        cle::Matrix<PointT, std::allocator<PointT>, size_t, true> points;

        Clustering::BinaryFormat binformat;
        binformat.read(options.input_file().c_str(), points);

        Clustering::ClusteringBenchmark<PointT, LabelT, MassT, ColMajor> bm(
                bm_config.runs,
                points.rows(),
                km_config.iterations,
                std::move(points));

        bm.initialize(
                km_config.clusters,
                points.cols(),
                Clustering::KmeansInitializer<PointT>::first_x);

        Clustering::KmeansNaive<PointT, LabelT, MassT> kmeans_naive;
        kmeans_naive.initialize();

        if (options.verify() || bm_config.verify) {
            bm.setVerificationReference(kmeans_naive);
        }

        Clustering::ClusteringBenchmarkStats bs(bm_config.runs);
        uint64_t verify_res = 0;

        if (km_config.pipeline == "three_stage") {
            auto ll_config =
                config.get_labeling_configuration();
            auto mu_config =
                config.get_mass_update_configuration();
            auto cu_config =
                config.get_centroid_update_configuration();

            bc::command_queue ll_queue, mu_queue, cu_queue;
            bc::context ll_context, mu_context, cu_context;

            {
                bc::device ll_dev =
                    bc::system::platforms()[ll_config.platform]
                    .devices()[ll_config.device];
                ll_context = bc::context(ll_dev);

                ll_queue = bc::command_queue(
                        ll_context,
                        ll_dev,
                        bc::command_queue::enable_profiling
                        );
            }

            if (
                    mu_config.platform == ll_config.platform
                    && mu_config.device == ll_config.device
                    ) {
                mu_queue = ll_queue;
                mu_context = ll_context;
            }
            else {
                bc::device mu_dev =
                    bc::system::platforms()[mu_config.platform]
                    .devices()[mu_config.device];
                mu_context = bc::context(mu_dev);

                mu_queue = bc::command_queue(
                        mu_context,
                        mu_dev,
                        bc::command_queue::enable_profiling
                        );
            }

            if (
                    cu_config.platform == ll_config.platform
                    && cu_config.device == ll_config.device
                    ) {
                cu_queue = ll_queue;
                cu_context = ll_context;
            }
            else if (
                    cu_config.platform == mu_config.platform
                    && cu_config.device == mu_config.device
                    ) {
                cu_queue = mu_queue;
                cu_context = mu_context;
            }
            else {
                bc::device cu_dev =
                    bc::system::platforms()[cu_config.platform]
                    .devices()[cu_config.device];
                cu_context = bc::context(cu_dev);

                cu_queue = bc::command_queue(
                        cu_context,
                        cu_dev,
                        bc::command_queue::enable_profiling
                        );
            }

            if (options.verbose()) {
                std::cout
                    << "Labeling device: "
                    << ll_queue.get_device().name()
                    << std::endl
                    << "Mass update device: "
                    << mu_queue.get_device().name()
                    << std::endl
                    << "Centroid update device: "
                    << cu_queue.get_device().name()
                    << std::endl;
            }

            Clustering::ThreeStageKmeans<
                PointT,
                LabelT,
                MassT,
                ColMajor
                    > kmeans;

            kmeans.set_labeling_queue(ll_queue);
            kmeans.set_mass_update_queue(mu_queue);
            kmeans.set_centroid_update_queue(cu_queue);
            kmeans.set_labeling_context(ll_context);
            kmeans.set_mass_update_context(mu_context);
            kmeans.set_centroid_update_context(cu_context);
            kmeans.set_labeler(ll_config);
            kmeans.set_mass_updater(mu_config);
            kmeans.set_centroid_updater(cu_config);

            if (options.verify() || bm_config.verify) {
                verify_res = bm.verify(kmeans);
            }
            else {
                bs = bm.run(kmeans);
            }
        }

        if (options.verbose()) {
            std::cout << "Pipeline: " << km_config.pipeline << " ";
            std::cout << "Types: "
                << km_config.point_type << " "
                << km_config.label_type << " "
                << km_config.mass_type << std::endl;
            if (options.verify() || bm_config.verify) {
                std::cout << "Centroid MSE: ";
                std::cout << bm.mse();
                std::cout << std::endl;
                if (verify_res == 0) {
                    std::cout << "Correct labels";
                    std::cout << std::endl;
                }
                else {
                    std::cout << verify_res << " incorrect labels";
                    std::cout << std::endl;
                    bm.print_result();
                }
            }
            else {
                bs.print_times();

            }
        }

        if (options.csv() && not (options.verify() || bm_config.verify)) {
            bs.to_csv(
                    options.csv_file().c_str(),
                    options.input_file().c_str()
                    );
        }

        kmeans_naive.finalize();
        bm.finalize();

        return 1;
    }
};

int main(int argc, char **argv) {
    int ret = 0;

    CmdOptions options;

    ret = options.parse(argc, argv);
    if (ret < 0) {
        return -1;
    }

    Clustering::ConfigurationParser config;
    config.parse_file(options.config_file());
    auto km_config = config.get_kmeans_configuration();

    if (
            km_config.point_type == "double" &&
            km_config.label_type == "uint64" &&
            km_config.mass_type == "uint64"
            ) {
        Bench<
            double,
            uint64_t,
            uint64_t,
            true
                > bench;
        ret = bench.run(options, config);
        if (ret < 0) {
            return ret;
        }
    }
    else if (
            km_config.point_type == "float" &&
            km_config.label_type == "uint32" &&
            km_config.mass_type == "uint32"
            ) {
        Bench<
            float,
            uint32_t,
            uint32_t,
            true
                > bench;
        ret = bench.run(options, config);
        if (ret < 0) {
            return ret;
        }
    }
    else {
        throw std::invalid_argument("Invalid type");
    }

#ifdef CUDA_FOUND
    cudaDeviceReset();
#endif

    return 0;
}
