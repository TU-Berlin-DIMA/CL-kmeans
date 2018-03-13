/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "cluster_generator.hpp"

#include <cstdint>
#include <iostream>
#include <string>

#include <SystemConfig.h>

#include <boost/program_options.hpp>

// Suppress editor errors about GENERATOR_NAME not defined
#ifndef GENERATOR_NAME
#define GENERATOR_NAME ""
#endif

namespace po = boost::program_options;

class CmdOptions {
public:
    int parse(int argc, char **argv) {
        char help_msg[] =
            "Usage: " GENERATOR_NAME " [OPTION] [OUTPUT FILE]\n"
            "Options"
            ;

        po::options_description cmdline(help_msg);
        cmdline.add_options()
            ("help", "Produce help message")
            ("size", po::value<uint64_t>(&megabytes_)->default_value(100),
             "Target file size in MiB (as float-type data)")
            ("features", po::value<uint64_t>(&features_)->default_value(2),
             "Number of features (aka. dimensions)")
            ("clusters", po::value<uint64_t>(&clusters_)->default_value(10),
             "Number of clusters")
            ("radius", po::value<float>(&radius_)->default_value(10.0f),
             "Cluster radius (has Gaussian distribution)")
            ("domain_min", po::value<float>(&domain_min_)->default_value(-100.0f),
             "Domain space (minimum value for centroids)")
            ("domain_max", po::value<float>(&domain_max_)->default_value(100.0f),
             "Domain space (maximum value for centroids)")
            ("divisor", po::value<uint64_t>(&multiple_)->default_value(8),
             "Number of points are multiple of divisor")
            ;

        po::options_description hidden("Hidden options");
        hidden.add_options()
            ("output-file", po::value<std::string>(&output_file_),
             "output file")
            ;

        po::options_description visible;
        visible.add(cmdline).add(hidden);

        po::positional_options_description pos;
        pos.add("output-file", 1);

        po::variables_map vm;
        po::store(po::command_line_parser(argc, argv).options(visible)
                .positional(pos).run(),
                vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << cmdline << std::endl;
            return -1;
        }

        // Ensure we have required options
        if (output_file_.empty()) {
            std::cout << "Give me an output file!" << std::endl;
            return -1;
        }

        return 1;
    }

    uint64_t features() const {
        return features_;
    }

    uint64_t clusters() const {
        return clusters_;
    }

    uint64_t bytes() const {
        return megabytes_ * 1024 * 1024;
    }

    uint64_t multiple() const {
        return multiple_;
    }

    float radius() const {
        return radius_;
    }

    float domain_min() const {
        return domain_min_;
    }

    float domain_max() const {
        return domain_max_;
    }

    std::string output_file() const {
        return output_file_;
    }

private:
    std::string output_file_;
    uint64_t features_;
    uint64_t clusters_;
    uint64_t megabytes_;
    uint64_t multiple_;
    float radius_;
    float domain_min_;
    float domain_max_;
};

int main(int argc, char **argv) {

    CmdOptions options;
    if (options.parse(argc, argv) < 0) {
        return 1;
    }

    cle::ClusterGenerator generator;

    generator.total_size(options.bytes());
    generator.cluster_radius(options.radius());
    generator.domain(options.domain_min(), options.domain_max());
    generator.num_features(options.features());
    generator.num_clusters(options.clusters());
    generator.point_multiple(options.multiple());

    generator.generate_bin(options.output_file().c_str());
}
