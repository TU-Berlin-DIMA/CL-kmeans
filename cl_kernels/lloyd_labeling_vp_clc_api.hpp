/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef LLOYD_LABELING_VP_CLC_API_HPP
#define LLOYD_LABELING_VP_CLC_API_HPP

#include "kernel_path.hpp"

#include <clext.hpp>

#include <functional>
#include <cassert>
#include <type_traits>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace cle {

    template <typename CL_FP, typename CL_INT>
    class LloydLabelingVpClcAPI {
    public:
        cl_int initialize(cl::Context& context, int vector_length, unsigned int unroll_max = 16) {
            constexpr int num_tiles_variants = 10;
            vector_length_ = vector_length;
            unroll_max_ = unroll_max;

            cl_int error_code = CL_SUCCESS;

            std::string defines;
            if (std::is_same<cl_float, CL_FP>::value) {
                defines = "-DTYPE32";
            }
            else if (std::is_same<cl_double, CL_FP>::value) {
                defines = "-DTYPE64";
            }
            else {
                assert(false);
            }

            defines += " -DVEC_LEN=" + std::to_string(vector_length);

            labeling_vp_clc_kernel_.resize(num_tiles_variants);

            for (int i = 0; i < num_tiles_variants; ++i) {
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
                    " -DCLUSTERS_UNROLL=" + std::to_string(unroll_clusters)
                    + " -DFEATURES_UNROLL=" + std::to_string(unroll_features);

                defines += unroll;

                cl::Program program = make_program(context, PROGRAM_FILE, defines, error_code);
                if (error_code != CL_SUCCESS) {
                    return error_code;
                }

                labeling_vp_clc_kernel_[i].reset(new cl::Kernel(program, KERNEL_NAME, &error_code));
                sanitize_make_kernel(error_code, context, program);
            }

            return error_code;
        }

        /*
        * kernel prefix_sum
        * 
        * Performs exclusive prefix sum on input
        *
        * Input
        *       Buffer with cl_uint array
        *
        * Output
        *       Buffer with cl_uint array 
        *       cl_uint with total sum
        *
        */
        cl_int operator() (
                cl::EnqueueArgs const& args,
                TypedBuffer<cl_char>& did_changes,
                CL_INT num_features,
                CL_INT num_points,
                CL_INT num_clusters,
                TypedBuffer<CL_FP>& points,
                TypedBuffer<CL_FP>& centroids,
                TypedBuffer<CL_INT>& labels,
                cl::Event& event
                ) {

            // assert did_changes.size() == #global work items
            assert(points.size() == num_points * num_features);
            assert(labels.size() == num_points);
            assert(centroids.size() >= num_clusters * num_features);
            assert(num_points % vector_length_ == 0);
            assert(num_features <= unroll_max_);

            cl::LocalSpaceArg local_centroids =
                cl::Local(num_clusters * num_features * sizeof(CL_FP));

            unsigned int cluster_unroll = num_clusters;
            unsigned int feature_unroll = num_features;

            cluster_unroll = std::min(cluster_unroll, unroll_max_);

            int num = 0;
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
                            num = 3;
                            break;
                        default:
                            num = -1;
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
                            num = 6;
                            break;
                        default:
                            num = -1;
                    }
                    break;
                case 8:
                    switch (feature_unroll) {
                        case 2:
                            num = 7;
                            break;
                        case 4:
                            num = 8;
                            break;
                        default:
                            num = -1;
                    }
                    break;
                case 16:
                    switch (feature_unroll) {
                        case 2:
                            num = 9;
                            break;
                        default:
                            num = -1;
                    }
                    break;
                default:
                    num = -1;
            }

            assert(num != -1 /* unsupported num clusters or featurs */);

            cle_sanitize_val_return(
                    labeling_vp_clc_kernel_[num]->setArg(0, (cl::Buffer&)did_changes));

            cle_sanitize_val_return(
                    labeling_vp_clc_kernel_[num]->setArg(1, (cl::Buffer&)points));

            cle_sanitize_val_return(
                    labeling_vp_clc_kernel_[num]->setArg(2, (cl::Buffer&)centroids));

            cle_sanitize_val_return(
                    labeling_vp_clc_kernel_[num]->setArg(3, (cl::Buffer&)labels));

            cle_sanitize_val_return(
                    labeling_vp_clc_kernel_[num]->setArg(4, local_centroids));

            cle_sanitize_val_return(
                    labeling_vp_clc_kernel_[num]->setArg(5, num_features));

            cle_sanitize_val_return(
                    labeling_vp_clc_kernel_[num]->setArg(6, num_points));

            cle_sanitize_val_return(
                    labeling_vp_clc_kernel_[num]->setArg(7, num_clusters));

            cle_sanitize_val_return(
                    args.queue_.enqueueNDRangeKernel(
                    *(labeling_vp_clc_kernel_[num]),
                    args.offset_,
                    args.global_,
                    args.local_,
                    &args.events_,
                    &event
                    ));

            return CL_SUCCESS;
        }

    private:
        static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("lloyd_labeling_vp_clc.cl");
        static constexpr const char* KERNEL_NAME = "lloyd_labeling_vp_clc";

        std::vector<std::shared_ptr<cl::Kernel>> labeling_vp_clc_kernel_;
        int vector_length_;
        unsigned int unroll_max_;
    };
}

#endif /* LLOYD_LABELING_VP_CLC_API_HPP */
