/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef LABELING_UNROLL_VECTOR_HPP
#define LABELING_UNROLL_VECTOR_HPP

#include "kernel_path.hpp"

#include "../labeling_configuration.hpp"
#include "../measurement/measurement.hpp"

#include <cassert>
#include <string>
#include <type_traits>
#include <vector>

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/memory/local_buffer.hpp>

namespace Clustering {

template <typename PointT, typename LabelT, bool ColMajor>
class LabelingUnrollVector {
public:
    using Event = boost::compute::event;
    using Context = boost::compute::context;
    using Kernel = boost::compute::kernel;
    using Program = boost::compute::program;
    template <typename T>
    using LocalBuffer = boost::compute::local_buffer<T>;

    LabelingUnrollVector() :
        kernel(num_unroll_variants)
    {}

    void prepare(Context context, LabelingConfiguration config) {
        this->config = config;

        std::string defines;
        if (std::is_same<cl_float, PointT>::value) {
            defines = "-DTYPE32";
        }
        else if (std::is_same<cl_double, PointT>::value) {
            defines = "-DTYPE64";
        }
        else {
            assert(false);
        }

        defines += " -DVEC_LEN="
            + std::to_string(this->config.vector_length);

        if (
                this->config.unroll_clusters_length == 0
                && this->config.unroll_features_length == 0
           ) {
            for (int i = 0; i < num_unroll_variants; ++i) {
                int unroll_features;
                int unroll_clusters;

                switch (i) {
                    case 0:
                        unroll_clusters = 2;
                        unroll_features = 2;
                        break;
                    case 1:
                        unroll_clusters = 2;
                        unroll_features = 4;
                        break;
                    case 2:
                        unroll_clusters = 2;
                        unroll_features = 8;
                        break;
                    case 3:
                        unroll_clusters = 2;
                        unroll_features = 16;
                        break;
                    case 4:
                        unroll_clusters = 4;
                        unroll_features = 2;
                        break;
                    case 5:
                        unroll_clusters = 4;
                        unroll_features = 4;
                        break;
                    case 6:
                        unroll_clusters = 4;
                        unroll_features = 8;
                        break;
                    case 7:
                        unroll_clusters = 8;
                        unroll_features = 2;
                        break;
                    case 8:
                        unroll_clusters = 8;
                        unroll_features = 4;
                        break;
                    case 9:
                        unroll_clusters = 16;
                        unroll_features = 2;
                        break;
                }

                std::string unroll =
                    " -DCLUSTERS_UNROLL="
                    + std::to_string(unroll_clusters)
                    + " -DFEATURES_UNROLL="
                    + std::to_string(unroll_features);


                Program program = Program::create_with_source_file(
                        PROGRAM_FILE,
                        context);

                program.build(defines + unroll);
                this->kernel[i] = program.create_kernel(KERNEL_NAME);
            }
        }
        else {
            std::string unroll =
                " -DCLUSTERS_UNROLL="
                + std::to_string(this->config.unroll_clusters_length)
                + " -DFEATURES_UNROLL="
                + std::to_string(this->config.unroll_features_length);

            Program program = Program::create_with_source_file(
                    PROGRAM_FILE,
                    context);

            program.build(defines + unroll);
            this->kernel[0] = program.create_kernel(KERNEL_NAME);
        }
    }

    Event operator() (
            boost::compute::command_queue queue,
            size_t num_features,
            size_t num_points,
            size_t num_clusters,
            boost::compute::vector<char>& did_changes,
            boost::compute::vector<PointT>& points,
            boost::compute::vector<PointT>& centroids,
            boost::compute::vector<LabelT>& labels,
            Measurement::DataPoint& datapoint,
            boost::compute::wait_list const& events) {

        datapoint.set_name("LabelingUnrollVector");

        LocalBuffer<PointT> local_centroids(num_clusters * num_features);

        int num = 0;
        if (
                this->config.unroll_clusters_length == 0
                && this->config.unroll_features_length == 0
           ) {
            int cluster_unroll = num_clusters;
            int feature_unroll = num_features;

            if (cluster_unroll > unroll_max) {
                cluster_unroll = unroll_max;
                feature_unroll = 2;
            }

            switch (cluster_unroll) {
                case 2:
                    switch (feature_unroll) {
                        case 2:
                            num = 0;
                            break;
                        case 4:
                            num = 1;
                            break;
                        case 8:
                            num = 2;
                            break;
                        case 16:
                        default:
                            num = 3;
                            break;
                    }
                    break;
                case 4:
                    switch (feature_unroll) {
                        case 2:
                            num = 4;
                            break;
                        case 4:
                            num = 5;
                            break;
                        case 8:
                        default:
                            num = 6;
                            break;
                    }
                    break;
                case 8:
                    switch (feature_unroll) {
                        case 2:
                            num = 7;
                            break;
                        case 4:
                        default:
                            num = 8;
                            break;
                    }
                    break;
                case 16:
                    switch (feature_unroll) {
                        case 2:
                        default:
                            num = 9;
                            break;
                    }
                    break;
                default:
                    num = -1;
            }

            assert(num != -1 /* unsupported num clusters or featurs */);
        }
        else {
            num = 0;
        }

        this->kernel[num].set_args(
                did_changes,
                points,
                centroids,
                labels,
                local_centroids,
                (LabelT) num_features,
                (LabelT) num_points,
                (LabelT) num_clusters);

        size_t work_offset[3] = {0, 0, 0};

        Event event;
        event = queue.enqueue_nd_range_kernel(
                this->kernel[num],
                1,
                work_offset,
                this->config.global_size,
                this->config.local_size,
                events);

        datapoint.add_event() = event;
        return event;
    }

private:
    static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("lloyd_labeling_vp_clc.cl");
    static constexpr const char* KERNEL_NAME = "lloyd_labeling_vp_clc";
    static constexpr const int num_unroll_variants = 10;
    static constexpr const int unroll_max = 16;

    std::vector<Kernel> kernel;
    LabelingConfiguration config;

};

}

#endif /* LABELING_UNROLL_VECTOR_HPP */
