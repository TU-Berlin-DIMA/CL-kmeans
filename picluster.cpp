/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "kmeans.hpp"
#include "matrix.hpp"
#include "measurement/measurement.hpp"

#include "libs/jpeg_reader_writer/JPEGReader.h"
#include "libs/jpeg_reader_writer/JPEGWriter.h"

#include "SystemConfig.h"

#include <boost/program_options.hpp>

#include <iostream>
#include <string>
#include <cstdint>
#include <cassert>

// Suppress editor errors about PICLUSTER_NAME not defined
#ifndef PICLUSTER_NAME
#define PICLUSTER_NAME ""
#endif

namespace po = boost::program_options;

class CmdOptions {
public:
    int parse(int argc, char **argv) {
        char help_msg[] =
            "Usage: " PICLUSTER_NAME " [CLUSTERS] [IN FILE] [OUT FILE]";

        po::options_description hidden(help_msg);
        hidden.add_options()
            ("clusters",
             po::value<uint32_t>(&clusters_)->default_value(0),
             "Number of clusters")
            ("in_file",
             po::value<std::string>(&in_file_),
             "Input file")
            ("out_file",
             po::value<std::string>(&out_file_),
             "Output file")
            ;

        po::positional_options_description pos;
        pos.add("clusters", 1);
        pos.add("in_file", 1);
        pos.add("out_file", 1);

        po::variables_map vm;
        po::store(
                po::command_line_parser(argc, argv).options(hidden).positional(pos).run(),
                vm
                );
        po::notify(vm);

        // Ensure we have required options
        if (
                clusters_ == 0
                || in_file_.empty()
                || out_file_.empty()
           ) {
            std::cout
                << "Give me clusters, "
                << "an input and an output file"
                << std::endl;
            return -1;
        }

        return 1;
    }

    uint32_t clusters() const {
        return clusters_;
    }

    std::string input_file() const {
        return in_file_;
    }

    std::string output_file() const {
        return out_file_;
    }

private:
    uint32_t clusters_;
    std::string in_file_;
    std::string out_file_;
};

int picture_cluster(
        uint32_t const num_clusters,
        std::string const& input_file,
        std::string const& output_file
        ) {

    JPEGReader loader;
    loader.header(input_file);
    assert(loader.colorSpace() == JPEG::COLOR_RGB);
    // loader.setQuantization(256);
    // loader.setDither(JPEG::DITHER_NONE);
    std::cout << "Components: " << loader.components() << std::endl;
    std::cout << "ColorComponents: " << loader.colorComponents() << std::endl;
    std::cout << "Quantization: " << loader.quantization() << std::endl;

    uint64_t const length = loader.width() * loader.height();
    uint32_t const colors = loader.components();
    std::vector<uint8_t> buffer(length * colors);
    std::vector<uint8_t*> rows(loader.height());

    for (uint64_t i = 0; i < loader.height(); ++i) {
        rows[i] = &buffer[i * loader.width() * colors];
    }
    loader.load(rows.begin());

    cle::Matrix<double, std::allocator<double>, uint64_t> img_matrix;
    img_matrix.resize(length, 4);

    for (uint64_t i = 0; i < length; ++i) {
        uint8_t* pixel = &buffer[i*colors];
        img_matrix(i, 0) = pixel[0];
        img_matrix(i, 1) = pixel[1];
        img_matrix(i, 2) = pixel[2];
        img_matrix(i, 3) = 0;
    }

    cle::Matrix<double, std::allocator<double>, uint64_t> centroids;
    centroids.resize(num_clusters, 4);
    std::vector<uint64_t> cluster_mass(num_clusters);
    std::vector<uint64_t> labels(length);

    cle::KmeansNaive64 kmeans;
    cle::KmeansInitializer64 kmeans_init;
    Measurement::Measurement stats;

    kmeans_init.forgy(img_matrix, centroids);
    kmeans.initialize();
    kmeans(100, img_matrix, centroids, cluster_mass, labels, stats);
    kmeans.finalize();

    uint8_t mycolors[2 * 3];
    mycolors[0] = 18;
    mycolors[1] = 64;
    mycolors[2] = 98;
    mycolors[3] = 255;
    mycolors[4] = 121;
    mycolors[5] = 134;

    uint8_t replacement[3] = {255, 255, 255};

    for (uint64_t i = 0; i < length; ++i) {
        uint8_t* pixel = &buffer[i*colors];
        uint64_t label = labels[i];
        // if (label == 0) {
        //     pixel[0] = replacement[0];
        //     pixel[1] = replacement[1];
        //     pixel[2] = replacement[2];
        // }
        // else {#<{(| Leave original color |)}>#}
        // pixel[0] = centroids(label, 0);
        // pixel[1] = centroids(label, 1);
        // pixel[2] = centroids(label, 2);
        pixel[0] = mycolors[3*label];
        pixel[1] = mycolors[3*label + 1];
        pixel[2] = mycolors[3*label + 2];
    }

    std::cout << "Centroid colors (RGB):" << std::endl;
    for (uint32_t c = 0; c < num_clusters; ++c) {
        std::cout
            << (uint16_t) centroids(c, 0) << " "
            << (uint16_t) centroids(c, 1) << " "
            << (uint16_t) centroids(c, 2) << " "
            << std::endl;
    }

    JPEGWriter writer;
    writer.header(loader.width(), loader.height(), loader.colorComponents(), JPEG::COLOR_RGB);
    writer.setQuality(90);
    writer.write(output_file, rows.begin());

    return 1;
}

int main(int argc, char **argv) {

    CmdOptions options;
    if (options.parse(argc, argv) < 0) {
        return 1;
    }

    picture_cluster(
            options.clusters(),
            options.input_file(),
            options.output_file()
            );

    return 0;
}
