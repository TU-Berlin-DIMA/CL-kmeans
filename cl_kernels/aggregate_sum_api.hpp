/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef AGGREGATE_SUM_API_HPP
#define AGGREGATE_SUM_API_HPP

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

    template <typename CL_TYPE, typename CL_INT>
    class AggregateSumAPI {
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
            defines += " -DCL_TYPE=";
            defines += cle::OpenCLType::to_str<CL_TYPE>();

            cl::Program program = make_program(context, PROGRAM_FILE, defines, error_code);
            if (error_code != CL_SUCCESS) {
                return error_code;
            }

            aggregate_sum_kernel_.reset(new cl::Kernel(program, KERNEL_NAME, &error_code));
            sanitize_make_kernel(error_code, context, program);

            return error_code;
        }

        /*
        * kernel aggregate sum
        */
        cl_int operator() (
                cl::EnqueueArgs const& args,
                CL_INT num_clusters,
                CL_INT num_blocks,
                TypedBuffer<CL_TYPE>& mass,
                cl::Event& event
                ) {

            assert(mass.size() >= num_clusters * num_blocks);

            cle_sanitize_val_return(
                    aggregate_sum_kernel_->setArg(0, (cl::Buffer&)mass));

            cle_sanitize_val_return(
                    aggregate_sum_kernel_->setArg(1, num_clusters));

            cle_sanitize_val_return(
                    aggregate_sum_kernel_->setArg(2, num_blocks));

            cle_sanitize_val_return(
                    args.queue_.enqueueNDRangeKernel(
                    *aggregate_sum_kernel_,
                    args.offset_,
                    args.global_,
                    args.local_,
                    &args.events_,
                    &event
                    ));

            return CL_SUCCESS;
        }

    private:
        static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("aggregate_sum_mass.cl");
        static constexpr const char* KERNEL_NAME = "aggregate_sum_mass";

        std::shared_ptr<cl::Kernel> aggregate_sum_kernel_;
    };
}

#endif /* AGGREGATE_SUM_API_HPP */
