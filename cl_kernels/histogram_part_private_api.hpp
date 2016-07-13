/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef HISTOGRAM_PART_PRIVATE_HPP
#define HISTOGRAM_PART_PRIVATE_HPP

#include "kernel_path.hpp"

#include <clext.hpp>

#include <functional>
#include <cassert>
#include <type_traits>
#include <memory>
#include <vector>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace cle {

    template <typename CL_INT>
    class HistogramPartPrivateAPI {
    public:
        cl_int initialize(cl::Context& context) {
            cl_int error_code = CL_SUCCESS;

            std::string define_type;
            if (std::is_same<cl_uint, CL_INT>::value) {
                define_type = "-DTYPE32";
            }
            else if (std::is_same<cl_ulong, CL_INT>::value) {
                define_type = "-DTYPE64";
            }
            else {
                assert(false);
            }

            histogram_kernel_.resize(NUM_KERNELS);

            for (uint32_t i = 0; i < NUM_KERNELS; ++i) {
                std::string define_bins = "-DNUM_BINS=" + std::to_string(2 << i);
                std::string defines = define_type + " " + define_bins;

                cl::Program program = make_program(context, PROGRAM_FILE, defines, error_code);
                if (error_code != CL_SUCCESS) {
                    return error_code;
                }

                histogram_kernel_[i].reset(new cl::Kernel(program, KERNEL_NAME, &error_code));
                sanitize_make_kernel(error_code, context, program);

                if (error_code != CL_SUCCESS) {
                    return error_code;
                }
            }

            return CL_SUCCESS;
        }

        /*
        * kernel HistogramPartLocal
        * 
        * Calculate histogram in partion per work group
        * using local memory
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

            assert(num_bins <= (2 << NUM_KERNELS));
            assert(in_items.size() == num_items);
            assert(out_bins.size() >=
                    num_bins * (args.global_[0] / args.local_[0]));

            CL_INT kid = 0;
            while (((CL_INT)2 << kid) < num_bins) ++kid;

            cle_sanitize_val_return(
                    histogram_kernel_[kid]->setArg(0, (cl::Buffer&)in_items));

            cle_sanitize_val_return(
                    histogram_kernel_[kid]->setArg(1, (cl::Buffer&)out_bins));

            cle_sanitize_val_return(
                    histogram_kernel_[kid]->setArg(2, num_items));

            cle_sanitize_val_return(
                    args.queue_.enqueueNDRangeKernel(
                    *histogram_kernel_[kid],
                    args.offset_,
                    args.global_,
                    args.local_,
                    &args.events_,
                    &event
                    ));

            return CL_SUCCESS;
        }

    private:
        static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("histogram_part_private.cl");
        static constexpr const char* KERNEL_NAME = "histogram_part_private";
        static constexpr const CL_INT NUM_KERNELS = 4;

        std::vector<std::shared_ptr<cl::Kernel>> histogram_kernel_;
    };
}

#endif /* HISTOGRAM_PART_PRIVATE_HPP */
