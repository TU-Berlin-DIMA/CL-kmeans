/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "configuration_parser.hpp"

#include <string>
#include <fstream>
#include <vector>
#include <stdexcept>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace Clustering {

void ConfigurationParser::parse_file(std::string file) {

    // TODO: check if file exits and throw exception
    std::ifstream handle;
    handle.open(file);

    if (not handle.is_open()) {
        throw std::invalid_argument(
                "Could not open file \"" + file + "\"");
    }

    po::options_description desc;
    desc.add(benchmark_options());
    desc.add(kmeans_options());

    po::store(po::parse_config_file(handle, desc), this->vm);

    handle.close();
}

po::options_description ConfigurationParser::benchmark_options() {

    po::options_description desc;

    desc.add_options()

        ("benchmark.runs", po::value<size_t>())
        ("benchmark.verify", po::value<bool>())

        ;

    return desc;
}

po::options_description ConfigurationParser::kmeans_options() {

    po::options_description desc;

    desc.add_options()

        // K-means general options
        ("kmeans.clusters", po::value<size_t>())
        ("kmeans.pipeline", po::value<std::string>())
        ("kmeans.iterations", po::value<size_t>())
        ("kmeans.converge", po::value<bool>())
        ("kmeans.types.point", po::value<std::string>())
        ("kmeans.types.label", po::value<std::string>())
        ("kmeans.types.mass", po::value<std::string>())

        // Labeling specific
        ("kmeans.labeling.platform", po::value<size_t>())
        ("kmeans.labeling.device", po::value<size_t>())
        ("kmeans.labeling.strategy", po::value<std::string>())
        ("kmeans.labeling.global_size", po::value<std::vector<size_t>>())
        ("kmeans.labeling.local_size", po::value<std::vector<size_t>>())
        ("kmeans.labeling.vector_length", po::value<size_t>())
        ("kmeans.labeling.unroll_clusters_length", po::value<size_t>())
        ("kmeans.labeling.unroll_features_length", po::value<size_t>())

        // Mass sum specific
        ("kmeans.mass_update.platform", po::value<size_t>())
        ("kmeans.mass_update.device", po::value<size_t>())
        ("kmeans.mass_update.strategy", po::value<std::string>())
        ("kmeans.mass_update.global_size", po::value<std::vector<size_t>>())
        ("kmeans.mass_update.local_size", po::value<std::vector<size_t>>())
        ("kmeans.mass_update.vector_length", po::value<size_t>())

        // Centroid update specific
        ("kmeans.centroid_update.platform", po::value<size_t>())
        ("kmeans.centroid_update.device", po::value<size_t>())
        ("kmeans.centroid_update.strategy", po::value<std::string>())
        ("kmeans.centroid_update.global_size", po::value<std::vector<size_t>>())
        ("kmeans.centroid_update.local_size", po::value<std::vector<size_t>>())
        ("kmeans.centroid_update.local_features", po::value<size_t>())
        ("kmeans.centroid_update.thread_features", po::value<size_t>())

        // Fused specific
        ("kmeans.fused.platform", po::value<size_t>())
        ("kmeans.fused.device", po::value<size_t>())
        ("kmeans.fused.strategy", po::value<std::string>())
        ("kmeans.fused.global_size", po::value<std::vector<size_t>>())
        ("kmeans.fused.local_size", po::value<std::vector<size_t>>())

        ;

    return desc;
}

BenchmarkConfiguration ConfigurationParser::get_benchmark_configuration() {

    BenchmarkConfiguration conf;

    for (auto const& option : vm) {
        if (option.first == "benchmark.runs") {
            conf.runs = option.second.as<size_t>();
        }
        else if (option.first == "benchmark.verify") {
            conf.verify = option.second.as<bool>();
        }
    }

    return conf;
}

KmeansConfiguration ConfigurationParser::get_kmeans_configuration() {

    KmeansConfiguration conf;

    for (auto const& option : vm) {
        if (option.first == "kmeans.clusters") {
            conf.clusters = option.second.as<size_t>();
        }
        else if (option.first == "kmeans.pipeline") {
            conf.pipeline = option.second.as<std::string>();
        }
        else if (option.first == "kmeans.iterations") {
            conf.iterations = option.second.as<size_t>();
        }
        else if (option.first == "kmeans.converge") {
            conf.converge = option.second.as<bool>();
        }
        else if (option.first == "kmeans.types.point") {
            conf.point_type = option.second.as<std::string>();
        }
        else if (option.first == "kmeans.types.label") {
            conf.label_type = option.second.as<std::string>();
        }
        else if (option.first == "kmeans.types.mass") {
            conf.mass_type = option.second.as<std::string>();
        }
    }

    return conf;
}

LabelingConfiguration ConfigurationParser::get_labeling_configuration() {

    LabelingConfiguration conf;

    for (auto const& option : vm) {
        if (option.first == "kmeans.labeling.platform") {
            conf.platform = option.second.as<size_t>();
        }
        else if (option.first == "kmeans.labeling.device") {
            conf.device = option.second.as<size_t>();
        }
        else if (option.first == "kmeans.labeling.strategy") {
            conf.strategy = option.second.as<std::string>();
        }
        else if (option.first == "kmeans.labeling.global_size") {
            auto v = option.second.as<std::vector<size_t>>();
            if (v.size() > 3) {
                // throw
            }
            conf.global_size[0] = v[0];
            conf.global_size[1] = v.size() > 1 ? v[1] : 1;
            conf.global_size[2] = v.size() == 3 ? v[2] : 1;
        }
        else if (option.first == "kmeans.labeling.local_size") {
            auto v = option.second.as<std::vector<size_t>>();
            if (v.size() > 3) {
                // throw
            }
            conf.local_size[0] = v[0];
            conf.local_size[1] = v.size() > 1 ? v[1] : 1;
            conf.local_size[2] = v.size() == 3 ? v[2] : 1;
        }
        else if (option.first == "kmeans.labeling.vector_length") {
            conf.vector_length = option.second.as<size_t>();
        }
        else if (option.first == "kmeans.labeling.unroll_clusters_length") {
            conf.unroll_clusters_length = option.second.as<size_t>();
        }
        else if (option.first == "kmeans.labeling.unroll_features_length") {
            conf.unroll_features_length = option.second.as<size_t>();
        }

    }

    return conf;
}

MassUpdateConfiguration ConfigurationParser::get_mass_update_configuration() {
    MassUpdateConfiguration conf;

    for (auto const& option : vm) {
        if (option.first == "kmeans.mass_update.platform") {
            conf.platform = option.second.as<size_t>();
        }
        else if (option.first == "kmeans.mass_update.device") {
            conf.device = option.second.as<size_t>();
        }
        else if (option.first == "kmeans.mass_update.strategy") {
            conf.strategy = option.second.as<std::string>();
        }
        else if (option.first == "kmeans.mass_update.global_size") {
            auto v = option.second.as<std::vector<size_t>>();
            if (v.size() > 3) {
                // throw
            }
            conf.global_size[0] = v[0];
            conf.global_size[1] = v.size() > 1 ? v[1] : 1;
            conf.global_size[2] = v.size() == 3 ? v[2] : 1;
        }
        else if (option.first == "kmeans.mass_update.local_size") {
            auto v = option.second.as<std::vector<size_t>>();
            if (v.size() > 3) {
                // throw
            }
            conf.local_size[0] = v[0];
            conf.local_size[1] = v.size() > 1 ? v[1] : 1;
            conf.local_size[2] = v.size() == 3 ? v[2] : 1;
        }
        else if (option.first == "kmeans.mass_update.vector_length") {
            conf.vector_length = option.second.as<size_t>();
        }
    }

    return conf;
}

CentroidUpdateConfiguration ConfigurationParser::get_centroid_update_configuration() {
    CentroidUpdateConfiguration conf;

    for (auto const& option : vm) {
        if (option.first == "kmeans.centroid_update.platform") {
            conf.platform = option.second.as<size_t>();
        }
        else if (option.first == "kmeans.centroid_update.device") {
            conf.device = option.second.as<size_t>();
        }
        else if (option.first == "kmeans.centroid_update.strategy") {
            conf.strategy = option.second.as<std::string>();
        }
        else if (option.first == "kmeans.centroid_update.global_size") {
            auto v = option.second.as<std::vector<size_t>>();
            if (v.size() > 3) {
                // throw
            }
            conf.global_size[0] = v[0];
            conf.global_size[1] = v.size() > 1 ? v[1] : 1;
            conf.global_size[2] = v.size() == 3 ? v[2] : 1;
        }
        else if (option.first == "kmeans.centroid_update.local_size") {
            auto v = option.second.as<std::vector<size_t>>();
            if (v.size() > 3) {
                // throw
            }
            conf.local_size[0] = v[0];
            conf.local_size[1] = v.size() > 1 ? v[1] : 1;
            conf.local_size[2] = v.size() == 3 ? v[2] : 1;
        }
        else if (option.first == "kmeans.centroid_update.local_features") {
            conf.local_features = option.second.as<size_t>();
        }
        else if (option.first == "kmeans.centroid_update.thread_features") {
            conf.thread_features = option.second.as<size_t>();
        }
    }

    return conf;
}

FusedConfiguration ConfigurationParser::get_fused_configuration() {
    FusedConfiguration conf;

    for (auto const& option : vm) {
        if (option.first == "kmeans.fused.platform") {
            conf.platform = option.second.as<size_t>();
        }
        else if (option.first == "kmeans.fused.device") {
            conf.device = option.second.as<size_t>();
        }
        else if (option.first == "kmeans.fused.strategy") {
            conf.strategy = option.second.as<std::string>();
        }
        else if (option.first == "kmeans.fused.global_size") {
            auto v = option.second.as<std::vector<size_t>>();
            if (v.size() > 3) {
                // throw
            }
            conf.global_size[0] = v[0];
            conf.global_size[1] = v.size() > 1 ? v[1] : 1;
            conf.global_size[2] = v.size() == 3 ? v[2] : 1;
        }
        else if (option.first == "kmeans.fused.local_size") {
            auto v = option.second.as<std::vector<size_t>>();
            if (v.size() > 3) {
                // throw
            }
            conf.local_size[0] = v[0];
            conf.local_size[1] = v.size() > 1 ? v[1] : 1;
            conf.local_size[2] = v.size() == 3 ? v[2] : 1;
        }
    }

    return conf;
}

}
