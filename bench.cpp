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
#include "kmeans.hpp"
#include "matrix.hpp"

#include "three_stage_kmeans.hpp"

#include "SystemConfig.h"

#include <clext.hpp>
#include <boost/program_options.hpp>
#include <boost/compute/core.hpp>

#include <iostream>
#include <cstdint>
#include <string>
#include <set>
#include <memory>

#ifdef CUDA_FOUND
#include <cuda_runtime.h>
#endif

// Suppress editor errors about BENCH_NAME not defined
#ifndef BENCH_NAME
#define BENCH_NAME ""
#endif

namespace po = boost::program_options;

class CmdOptions {
public:
    enum class Algorithm {Naive, GPUAssisted, FeatureSum, ThreeStageKmeans};

    int parse(int argc, char **argv) {
        char help_msg[] =
            "Usage: " BENCH_NAME " [OPTION] [K] [FILE]\n"
            "Options"
            ;

        po::options_description cmdline(help_msg);
        cmdline.add_options()
            ("help", "Produce help message")
            ("quiet", "Suppress non-critical output")
            ("runs", po::value<uint32_t>(&runs_)->default_value(1),
             "Number of runs")
            ("max-iterations",
             po::value<uint32_t>(&max_iterations_)->default_value(100),
             "Maximum number of iterations")
            ("platform",
             po::value<uint32_t>(&platform_)->default_value(0),
             "OpenCL platform number")
            ("device",
             po::value<uint32_t>(&device_)->default_value(0),
             "OpenCL device number")
            ("64bit", "Run in 64-bit mode (doubles and unsigned long longs)")
            ("verify", "Do verification pass")
            ("verify-file",
             po::value<std::string>(),
             "Do verification pass with labels")
            ("csv",
             po::value<std::string>(),
             "Output measurements to CSV file")
            ("armadillo", "Run Armadillo K-means")
            ("naive", "Run Naive Lloyd's")
            ("gpu-assisted", "Run GPU assisted Lloyd's")
            ("feature-sum", "Run GPU feature sum Lloyd's")
            ("three-stage-kmeans", "Run three stage K-means")
            ;

        po::options_description hidden("Hidden options");
        hidden.add_options()
            ("k", po::value<uint32_t>(), "Number of clusters")
            ("input-file", po::value<std::string>(), "Input file")
            ;

        po::options_description visible;
        visible.add(cmdline).add(hidden);

        po::positional_options_description pos;
        pos.add("k", 1);
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

        if (vm.count("max-iterations")) {
            max_iterations_ = vm["max-iterations"].as<uint32_t>();
        }

        if (vm.count("platform")) {
            platform_ = vm["platform"].as<uint32_t>();
        }

        if (vm.count("device")) {
            device_ = vm["device"].as<uint32_t>();
        }

        if (vm.count("input-file")) {
            input_file_ = vm["input-file"].as<std::string>();
        }

        if (vm.count("help")) {
            std::cout << cmdline << std::endl;
            return -1;
        }

        if (vm.count("quiet")) {
            quiet_ = true;
        }

        if (vm.count("64bit")) {
            type64_ = true;
        }

        if (vm.count("verify")) {
            verify_ = true;
        }

        if (vm.count("verify-file")) {
            verify_ = true;
            verify_labels_ = true;
            labels_file_ = vm["verify-file"].as<std::string>();
        }

        if (vm.count("csv")) {
            csv_ = true;
            csv_file_ = vm["csv"].as<std::string>();
        }

        if (vm.count("naive")) {
            algorithms_.insert(Algorithm::Naive);
        }

        if (vm.count("gpu-assisted")) {
            algorithms_.insert(Algorithm::GPUAssisted);
        }

        if (vm.count("feature-sum")) {
            algorithms_.insert(Algorithm::FeatureSum);
        }

        if (vm.count("three-stage-kmeans")) {
            algorithms_.insert(Algorithm::ThreeStageKmeans);
        }

        // Ensure we have required options
        if (k_ == 0 || input_file_.empty()) {
            std::cout << "Enter k and a file or die" << std::endl;
            return -1;
        }

        // Warning about no algorithm selected
        if (algorithms_.size() == 0) {
            std::cout << "You might want to choose an algorithm" << std::endl;
            return -1;
        }

        return 1;
    }

    bool quiet() const {
        return quiet_;
    }

    uint32_t k() const {
        return k_;
    }

    uint32_t runs() const {
        return runs_;
    }

    uint32_t max_iterations() const {
        return max_iterations_;
    }

    uint32_t cl_platform() const {
        return platform_;
    }

    uint32_t cl_device() const {
        return device_;
    }

    std::string input_file() const {
        return input_file_;
    }

    bool type64() const {
        return type64_;
    }

    bool verify() const {
        return verify_;
    }

    bool verify_labels() const {
        return verify_labels_;
    }

    std::string labels_file() const {
        return labels_file_;
    }

    bool csv() const {
        return csv_;
    }

    std::string csv_file() const {
        return csv_file_;
    }

    std::set<Algorithm> algorithms() const {
        return algorithms_;
    }

private:
    bool quiet_ = false;
    uint32_t k_ = 0;
    uint32_t runs_ = 0;
    uint32_t max_iterations_ = 0;
    uint32_t platform_ = 0;
    uint32_t device_ = 0;
    std::string input_file_;
    bool type64_ = false;
    bool verify_ = false;
    bool verify_labels_ = false;
    std::string labels_file_;
    bool csv_ = false;
    std::string csv_file_;
    std::set<Algorithm> algorithms_;
};

template <typename FP, typename INT, typename AllocFP, typename AllocINT,
         bool COL_MAJOR>
class Bench {
public:
    int run(CmdOptions options) {
        int ret = 1;


        cle::Matrix<FP, AllocFP, INT, true> points;

        cle::CLInitializer clinit;
        if ((ret = clinit.init(options.cl_platform(), options.cl_device()))
                < 0) {
            return ret;
        }

        if (not options.quiet()) {
            cle_sanitize_done_return(
                    clinit.print_device_info());
        }

        {
            cle::BinaryFormat binformat;
            binformat.read(options.input_file().c_str(), points);
        }

        cle::ClusteringBenchmark<FP, INT, AllocFP, AllocINT, COL_MAJOR> bm(
                options.runs(), points.rows(), options.max_iterations(),
                std::move(points));
        bm.initialize(options.k(), points.cols(),
            cle::KmeansInitializer<FP, AllocFP, INT>::first_x);

        cle::KmeansNaive<FP, INT, AllocFP, AllocINT> kmeans_naive;
        kmeans_naive.initialize();

        if (options.verify_labels()) {
            std::cout << "Not implemented" << std::endl;
            std::vector<std::vector<INT, AllocINT>> reference_labels;
            bm.setVerificationReference(std::move(reference_labels[0]));
        }
        else if (options.verify()) {
            bm.setVerificationReference(kmeans_naive);
        }

        for (CmdOptions::Algorithm a : options.algorithms()) {
            cle::ClusteringBenchmarkStats bs(options.runs());
            uint64_t verify_res = 0;
            std::string name;

            switch (a) {
                case CmdOptions::Algorithm::Naive:
                    {
                        name = kmeans_naive.name();
                        if (options.verify()) {
                            verify_res = bm.verify(kmeans_naive);
                        }
                        else {
                            bs = bm.run(kmeans_naive);
                        }
                    }
                    break;
                case CmdOptions::Algorithm::GPUAssisted:
                    {
                        cle::KmeansGPUAssisted<FP, INT, AllocFP, AllocINT>
                            kmeans_gpu_assisted(
                                clinit.get_context(),
                                clinit.get_commandqueue()
                                    );
                        kmeans_gpu_assisted.initialize();
                        name = kmeans_gpu_assisted.name();
                        if (options.verify()) {
                            verify_res = bm.verify(kmeans_gpu_assisted);
                        }
                        else {
                            bs = bm.run(kmeans_gpu_assisted);
                        }
                        kmeans_gpu_assisted.finalize();
                    }
                    break;
                case CmdOptions::Algorithm::FeatureSum:
                    {
                        cle::LloydGPUFeatureSum<FP, INT, AllocFP, AllocINT>
                            lloyd_gpu_feature_sum(
                                    clinit.get_context(),
                                    clinit.get_commandqueue()
                                    );
                        lloyd_gpu_feature_sum.initialize();
                        name = lloyd_gpu_feature_sum.name();
                        if (options.verify()) {
                            verify_res = bm.verify(lloyd_gpu_feature_sum);
                        }
                        else {
                            bs = bm.run(lloyd_gpu_feature_sum);
                        }
                        lloyd_gpu_feature_sum.finalize();
                    }
                    break;
                case CmdOptions::Algorithm::ThreeStageKmeans:
                    {
                        Clustering::LabelingConfiguration labeling_config
                            = {
                                {1024, 1, 1},
                                {32, 1, 1},
                                1, 1, 1
                            };
                        Clustering::MassUpdateConfiguration mass_update_config
                            = {
                                {1024, 1, 1},
                                {32, 1, 1}
                            };
                        Clustering::CentroidUpdateConfiguration centroid_update_config
                            = {
                                {1024, 1, 1},
                                {32, 1, 1}
                            };
                        boost::compute::context context(clinit.get_context()());
                        boost::compute::command_queue queue(clinit.get_commandqueue()());
                        Clustering::ThreeStageKmeans<FP, INT, INT, true> kmeans(context);
                        kmeans.set_labeler("unroll_vector", labeling_config);
                        kmeans.set_mass_updater("global_atomic", mass_update_config);
                        kmeans.set_centroid_updater("feature_sum", centroid_update_config);
                        kmeans.set_labeling_queue(queue);
                        kmeans.set_mass_update_queue(queue);
                        kmeans.set_centroid_update_queue(queue);

                        if (options.verify()) {
                            verify_res = bm.verify(kmeans, queue);
                        }
                        else {
                            bs = bm.run(kmeans, queue);
                        }
                    }
                    break;
            }

            if (not options.quiet()) {
                std::cout << name << " ";
                std::cout << (options.type64() ? "64-bit" : "32-bit");
                std::cout << " ";
                if (options.verify()) {
                    if (verify_res == 0) {
                        std::cout << "correct";
                    }
                    else {
                        std::cout << verify_res << " incorrect labels";
                    }
                }
                else {
                    bs.print_times();

                }
                std::cout << std::endl;
            }

            if (options.csv() && not options.verify()) {
                bs.to_csv(
                        options.csv_file().c_str(),
                        options.input_file().c_str()
                        );
            }
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

    if (options.type64()) {
        Bench<
            double,
            uint64_t,
            std::allocator<double>,
            std::allocator<uint64_t>,
            true
                > bench;
        ret = bench.run(options);
        if (ret < 0) {
            return ret;
        }
    }
    else {
        Bench<
            float,
            uint32_t,
            std::allocator<float>,
            std::allocator<uint32_t>,
            true
                > bench;
        ret = bench.run(options);
        if (ret < 0) {
            return ret;
        }
    }

#ifdef CUDA_FOUND
    cudaDeviceReset();
#endif

    return 0;
}
