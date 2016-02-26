/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef LLOYD_FEATURE_SUM_API_HPP
#define LLOYD_FEATURE_SUM_API_HPP

#include "kernel_path.hpp"

#include <clext.hpp>

#include <functional>
#include <cassert>
#include <type_traits>
#include <memory>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace cle {

    template <typename CL_FP, typename CL_INT>
    class LloydFeatureSumAPI {
    public:
        cl_int initialize(cl::Context& context) {
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

            cl::Program program = make_program(context, PROGRAM_FILE, defines, error_code);
            if (error_code != CL_SUCCESS) {
                return error_code;
            }

            feature_sum_kernel_.reset(
                    new cl::Kernel(program, KERNEL_NAME, &error_code));
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

            assert(points.size() == num_points * num_features);
            assert(labels.size() == num_points);
            assert(centroids.size() == num_clusters * num_features);
            assert(mass.size() >= num_clusters);

            const size_t local_size = args.local_[0];

            cl::LocalSpaceArg local_centroids =
                cl::Local(num_clusters * local_size * sizeof(CL_FP));
            cl::LocalSpaceArg local_points =
                cl::Local(local_size * local_size * sizeof(CL_FP));

            cle_sanitize_val_return(
                    feature_sum_kernel_->setArg(0, (cl::Buffer&)points));

            cle_sanitize_val_return(
                    feature_sum_kernel_->setArg(1, (cl::Buffer&)centroids));

            cle_sanitize_val_return(
                    feature_sum_kernel_->setArg(2, (cl::Buffer&)mass));

            cle_sanitize_val_return(
                    feature_sum_kernel_->setArg(3, (cl::Buffer&)labels));

            cle_sanitize_val_return(
                    feature_sum_kernel_->setArg(4, local_centroids));

            cle_sanitize_val_return(
                    feature_sum_kernel_->setArg(5, local_points));

            cle_sanitize_val_return(
                    feature_sum_kernel_->setArg(6, num_features));

            cle_sanitize_val_return(
                    feature_sum_kernel_->setArg(7, num_points));

            cle_sanitize_val_return(
                    feature_sum_kernel_->setArg(8, num_clusters));

            cle_sanitize_val_return(
                    args.queue_.enqueueNDRangeKernel(
                    *feature_sum_kernel_,
                    args.offset_,
                    args.global_,
                    args.local_,
                    &args.events_,
                    &event
                    ));

            return CL_SUCCESS;
        }

    private:
        static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("lloyd_feature_sum.cl");
        static constexpr const char* KERNEL_NAME = "lloyd_feature_sum";

        std::shared_ptr<cl::Kernel> feature_sum_kernel_;
    };
}

#endif /* LLOYD_FEATURE_SUM_API_HPP */
