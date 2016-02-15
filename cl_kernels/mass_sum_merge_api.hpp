/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef MASS_SUM_MERGE_API_HPP
#define MASS_SUM_MERGE_API_HPP

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

    template <typename CL_INT>
    class MassSumMergeAPI {
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

            cl::Program program = make_program(context, PROGRAM_FILE, defines, error_code);
            if (error_code != CL_SUCCESS) {
                return error_code;
            }

            mass_sum_kernel_.reset(new cl::Kernel(program, KERNEL_NAME, &error_code));
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
                CL_INT num_points,
                CL_INT num_clusters,
                TypedBuffer<CL_INT>& labels,
                TypedBuffer<CL_INT>& mass,
                cl::Event& event
                ) {

            assert(labels.size() == num_points);
            assert(mass.size() >=
                    num_clusters * (args.global_[0] / args.local_[0]));

            cl::LocalSpaceArg local_labels =
                cl::Local(num_clusters * sizeof(CL_INT));

            cle_sanitize_val_return(
                    mass_sum_kernel_->setArg(0, (cl::Buffer&)labels));

            cle_sanitize_val_return(
                    mass_sum_kernel_->setArg(1, (cl::Buffer&)mass));

            cle_sanitize_val_return(
                    mass_sum_kernel_->setArg(2, local_labels));

            cle_sanitize_val_return(
                    mass_sum_kernel_->setArg(3, num_points));

            cle_sanitize_val_return(
                    mass_sum_kernel_->setArg(4, num_clusters));

            cle_sanitize_val_return(
                    args.queue_.enqueueNDRangeKernel(
                    *mass_sum_kernel_,
                    args.offset_,
                    args.global_,
                    args.local_,
                    &args.events_,
                    &event
                    ));

            return CL_SUCCESS;
        }

    private:
        static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("mass_sum_merge.cl");
        static constexpr const char* KERNEL_NAME = "mass_sum_merge";

        std::shared_ptr<cl::Kernel> mass_sum_kernel_;
    };
}

#endif /* MASS_SUM_MERGE_API_HPP */
