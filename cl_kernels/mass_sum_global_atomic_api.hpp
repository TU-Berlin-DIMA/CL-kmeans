/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef MASS_SUM_GLOBAL_ATOMIC_API_HPP
#define MASS_SUM_GLOBAL_ATOMIC_API_HPP

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
    class MassSumGlobalAtomicAPI {
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

            kernel_.reset(new cl::Kernel(program, KERNEL_NAME, &error_code));
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
            assert(mass.size() == num_clusters);

            cle_sanitize_val_return(
                    kernel_->setArg(0, (cl::Buffer&)labels));

            cle_sanitize_val_return(
                    kernel_->setArg(1, (cl::Buffer&)mass));

            cle_sanitize_val_return(
                    kernel_->setArg(2, num_points));

            cle_sanitize_val_return(
                    kernel_->setArg(3, num_clusters));

            cle_sanitize_val_return(
                    args.queue_.enqueueNDRangeKernel(
                    *kernel_,
                    args.offset_,
                    args.global_,
                    args.local_,
                    &args.events_,
                    &event
                    ));

            return CL_SUCCESS;
        }

    private:
        using Base_Kernel = cl::make_kernel<
            cl::Buffer&,
            cl::Buffer&,
            CL_INT,
            CL_INT
                >;

        using Kernel_Functor = std::function<typename Base_Kernel::type_>;

        static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("mass_sum_global_atomic.cl");
        static constexpr const char* KERNEL_NAME = "mass_sum_global_atomic";

        Kernel_Functor kernel_functor_;
        std::shared_ptr<cl::Kernel> kernel_;
    };
}

#endif /* MASS_SUM_GLOBAL_ATOMIC_API_HPP */
