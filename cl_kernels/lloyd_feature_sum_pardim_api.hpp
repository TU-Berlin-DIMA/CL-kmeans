/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef LLOYD_FEATURE_SUM_PARDIM_API_HPP
#define LLOYD_FEATURE_SUM_PARDIM_API_HPP

#include "kernel_path.hpp"

#include <clext.hpp>

#include <functional>
#include <cassert>
#include <type_traits>
#include <memory>
#include <algorithm>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace cle {

    template <typename CL_FP, typename CL_INT>
    class LloydFeatureSumPardimAPI {
    public:
        cl_int initialize(cl::Context& context) {
            cl_int error_code = CL_SUCCESS;

            std::string defines;
            if (std::is_same<cl_uint, CL_INT>::value) {
                defines = "-DTYPE32";
            }
            else if (std::is_same<cl_ulong, CL_INT>::value) {
                defines = "-DTYPE64";
            }
            else {
                assert(false);
            }
            defines += " -DNUM_THREAD_FEATURES="
                + std::to_string(num_features_per_thread_);

            cl::Program program = make_program(context, PROGRAM_FILE, defines, error_code);
            if (error_code != CL_SUCCESS) {
                return error_code;
            }

            lloyd_feature_sum_pardim_kernel_.reset(new cl::Kernel(program, KERNEL_NAME, &error_code));
            sanitize_make_kernel(error_code, context, program);

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
                CL_INT num_features,
                CL_INT num_points,
                CL_INT num_clusters,
                TypedBuffer<CL_FP>& points,
                TypedBuffer<CL_FP>& centroids,
                TypedBuffer<CL_INT>& mass,
                TypedBuffer<CL_INT>& labels,
                cl::Event& event
                ) {
            size_t const num_feature_tiles = num_features / num_features_per_thread_;
            size_t const num_local_features = 2; // essentially local_range[1]
            cl::NDRange global_range(args.global_[0] / num_feature_tiles, num_feature_tiles);
            cl::NDRange local_range;
            if (num_feature_tiles == 1) {
                local_range = cl::NDRange(args.local_[0], 1);
            }
            else {
                local_range = cl::NDRange(args.local_[0] / num_local_features, num_local_features);
            }

            assert(num_feature_tiles > 0);
            assert(global_range[1] == num_feature_tiles);
            assert(labels.size() == num_points);
            assert(centroids.size() >= global_range[0] * num_features * num_clusters);

            cl::LocalSpaceArg local_centroids =
                cl::Local(local_range[0] * local_range[1] * num_features_per_thread_ * num_clusters * sizeof(CL_FP));

            cle_sanitize_val_return(
                    lloyd_feature_sum_pardim_kernel_->setArg(0, (cl::Buffer&)points));

            cle_sanitize_val_return(
                    lloyd_feature_sum_pardim_kernel_->setArg(1, (cl::Buffer&)centroids));

            cle_sanitize_val_return(
                    lloyd_feature_sum_pardim_kernel_->setArg(2, (cl::Buffer&)mass));

            cle_sanitize_val_return(
                    lloyd_feature_sum_pardim_kernel_->setArg(3, (cl::Buffer&)labels));

            cle_sanitize_val_return(
                    lloyd_feature_sum_pardim_kernel_->setArg(4, local_centroids));

            cle_sanitize_val_return(
                    lloyd_feature_sum_pardim_kernel_->setArg(5, num_features));

            cle_sanitize_val_return(
                    lloyd_feature_sum_pardim_kernel_->setArg(6, num_points));

            cle_sanitize_val_return(
                    lloyd_feature_sum_pardim_kernel_->setArg(7, num_clusters));

            cle_sanitize_val_return(
                    args.queue_.enqueueNDRangeKernel(
                    *lloyd_feature_sum_pardim_kernel_,
                    args.offset_,
                    global_range,
                    local_range,
                    &args.events_,
                    &event
                    ));

            return CL_SUCCESS;
        }

        void set_num_features_per_thread(size_t n) {
            num_features_per_thread_ = n;
        }

        size_t get_num_global_blocks(
                size_t const global_size,
                size_t const /* local_size */,
                size_t const num_features,
                size_t const /* num_clusters */
                ) const {

            return global_size / (num_features / num_features_per_thread_);
        }

    private:
        static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("lloyd_feature_sum_pardim.cl");
        static constexpr const char* KERNEL_NAME = "lloyd_feature_sum_pardim";

        std::shared_ptr<cl::Kernel> lloyd_feature_sum_pardim_kernel_;
        size_t num_features_per_thread_ = 1;
    };
}

#endif /* LLOYD_FEATURE_SUM_PARDIM_API_HPP */
