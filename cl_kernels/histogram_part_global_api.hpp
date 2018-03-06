/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifndef HISTOGRAM_PART_GLOBAL_HPP
#define HISTOGRAM_PART_GLOBAL_HPP

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

    template <typename CL_INT>
    class HistogramPartGlobalAPI {
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
        * kernel HistogramPartLocal
        * 
        * Calculate histogram in partion per work group
        * using global memory
        *
        * Input
        *       Buffer with data item array
        *
        * Output
        *       Buffer with #(work groups) partial histograms array
        *
        */
        cl_int operator() (
                cl::EnqueueArgs const& args,
                CL_INT num_items,
                CL_INT num_bins,
                TypedBuffer<CL_INT>& in_items,
                TypedBuffer<CL_INT>& out_bins,
                cl::Event& event
                ) {

            assert(in_items.size() == num_items);
            assert(out_bins.size() >=
                    num_bins * (args.global_[0] / args.local_[0]));

            cle_sanitize_val_return(
                    mass_sum_kernel_->setArg(0, (cl::Buffer&)in_items));

            cle_sanitize_val_return(
                    mass_sum_kernel_->setArg(1, (cl::Buffer&)out_bins));

            cle_sanitize_val_return(
                    mass_sum_kernel_->setArg(2, num_items));

            cle_sanitize_val_return(
                    mass_sum_kernel_->setArg(3, num_bins));

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
        static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("histogram_part_global.cl");
        static constexpr const char* KERNEL_NAME = "histogram_part_global";

        std::shared_ptr<cl::Kernel> mass_sum_kernel_;
    };
}

#endif /* HISTOGRAM_PART_GLOBAL_HPP */
