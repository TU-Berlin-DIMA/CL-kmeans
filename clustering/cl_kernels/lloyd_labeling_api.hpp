/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef LLOYD_LABELING_API_HPP
#define LLOYD_LABELING_API_HPP

#include "../cle/common.hpp"

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
    class LloydLabelingAPI {
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

            labeling_kernel_.reset(new cl::Kernel(program, KERNEL_NAME, &error_code));
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
                TypedBuffer<cl_char>& did_changes,
                CL_INT num_features,
                CL_INT num_points,
                CL_INT num_clusters,
                TypedBuffer<CL_FP>& points,
                TypedBuffer<CL_FP>& centroids,
                TypedBuffer<CL_INT>& memberships,
                cl::Event& event
                ) {

            // assert did_changes.size() == #global work items
            assert(points.size() == num_points * num_features);
            assert(memberships.size() == num_points);
            assert(centroids.size() >= num_clusters * num_features);

            cl::LocalSpaceArg local_centroids = cl::Local(centroids.bytes());

            cle_sanitize_val_return(
                    labeling_kernel_->setArg(0, (cl::Buffer&)did_changes));

            cle_sanitize_val_return(
                    labeling_kernel_->setArg(1, (cl::Buffer&)points));

            cle_sanitize_val_return(
                    labeling_kernel_->setArg(2, (cl::Buffer&)centroids));

            cle_sanitize_val_return(
                    labeling_kernel_->setArg(3, (cl::Buffer&)memberships));

            cle_sanitize_val_return(
                    labeling_kernel_->setArg(4, local_centroids));

            cle_sanitize_val_return(
                    labeling_kernel_->setArg(5, num_features));

            cle_sanitize_val_return(
                    labeling_kernel_->setArg(6, num_points));

            cle_sanitize_val_return(
                    labeling_kernel_->setArg(7, num_clusters));

            cle_sanitize_val_return(
                    args.queue_.enqueueNDRangeKernel(
                    *labeling_kernel_,
                    args.offset_,
                    args.global_,
                    args.local_,
                    &args.events_,
                    &event
                    ));

            return CL_SUCCESS;
        }

    private:
        static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("lloyd_labeling.cl");
        static constexpr const char* KERNEL_NAME = "lloyd_labeling";

        std::shared_ptr<cl::Kernel> labeling_kernel_;
    };
}

#endif /* LLOYD_LABELING_API_HPP */
