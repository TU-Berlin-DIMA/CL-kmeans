/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifndef CONFIGURATION_PARSER_HPP
#define CONFIGURATION_PARSER_HPP

#include "benchmark_configuration.hpp"
#include "kmeans_configuration.hpp"
#include "labeling_configuration.hpp"
#include "mass_update_configuration.hpp"
#include "centroid_update_configuration.hpp"
#include "fused_configuration.hpp"

#include <cstddef>
#include <string>

#include <boost/program_options.hpp>

namespace Clustering {

class ConfigurationParser {

public:
    void parse_file(std::string file);
    BenchmarkConfiguration get_benchmark_configuration();
    KmeansConfiguration get_kmeans_configuration();
    LabelingConfiguration get_labeling_configuration();
    MassUpdateConfiguration get_mass_update_configuration();
    CentroidUpdateConfiguration get_centroid_update_configuration();
    FusedConfiguration get_fused_configuration();

private:
    boost::program_options::options_description benchmark_options();
    boost::program_options::options_description kmeans_options();

    boost::program_options::variables_map vm;

};
}

#endif /* CONFIGURATION_PARSER_HPP */
