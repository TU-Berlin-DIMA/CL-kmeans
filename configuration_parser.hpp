#ifndef CONFIGURATION_PARSER_HPP
#define CONFIGURATION_PARSER_HPP

#include "benchmark_configuration.hpp"
#include "kmeans_configuration.hpp"
#include "labeling_configuration.hpp"
#include "mass_update_configuration.hpp"
#include "centroid_update_configuration.hpp"

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

private:
    boost::program_options::options_description benchmark_options();
    boost::program_options::options_description kmeans_options();

    boost::program_options::variables_map vm;

};
}

#endif /* CONFIGURATION_PARSER_HPP */
