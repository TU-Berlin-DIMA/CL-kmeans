/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef LLOYD_MERGE_SUM_API_HPP
#define LLOYD_MERGE_SUM_API_HPP

#include "../cle/common.hpp"

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
    class LloydMergeSumAPI {
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

            lloyd_merge_sum_kernel_.reset(new cl::Kernel(program, KERNEL_NAME, &error_code));
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

            assert(args.global_[0] >= num_features * num_clusters);
            assert(labels.size() == num_points);

            cl::LocalSpaceArg local_points =
                cl::Local(args.local_[0] * sizeof(CL_FP));

            cl::LocalSpaceArg local_mass =
                cl::Local(get_num_local_rows_points(
                            args.local_[0],
                            num_clusters)
                        * sizeof(CL_INT));

            cl::LocalSpaceArg local_labels =
                cl::Local(args.local_[0] * sizeof(CL_INT));

            cle_sanitize_val_return(
                    lloyd_merge_sum_kernel_->setArg(0, (cl::Buffer&)points));

            cle_sanitize_val_return(
                    lloyd_merge_sum_kernel_->setArg(1, (cl::Buffer&)centroids));

            cle_sanitize_val_return(
                    lloyd_merge_sum_kernel_->setArg(2, (cl::Buffer&)mass));

            cle_sanitize_val_return(
                    lloyd_merge_sum_kernel_->setArg(3, (cl::Buffer&)labels));

            cle_sanitize_val_return(
                    lloyd_merge_sum_kernel_->setArg(4, local_points));

            cle_sanitize_val_return(
                    lloyd_merge_sum_kernel_->setArg(5, local_mass));

            cle_sanitize_val_return(
                    lloyd_merge_sum_kernel_->setArg(6, local_labels));

            cle_sanitize_val_return(
                    lloyd_merge_sum_kernel_->setArg(7, num_features));

            cle_sanitize_val_return(
                    lloyd_merge_sum_kernel_->setArg(8, num_points));

            cle_sanitize_val_return(
                    lloyd_merge_sum_kernel_->setArg(9, num_clusters));

            cle_sanitize_val_return(
                    args.queue_.enqueueNDRangeKernel(
                    *lloyd_merge_sum_kernel_,
                    args.offset_,
                    args.global_,
                    args.local_,
                    &args.events_,
                    &event
                    ));

            return CL_SUCCESS;
        }

        size_t get_num_local_cols(
                size_t const local_size,
                size_t const num_clusters
                ) const {

            return std::max((size_t)1, local_size / num_clusters);
        }

        size_t get_num_local_rows_points(
                size_t const local_size,
                size_t const num_clusters
                ) const {

            return local_size / get_num_local_cols(local_size, num_clusters);
        }

        size_t get_num_local_rows_clusters(
                size_t const local_size,
                size_t const num_clusters
                ) const {

            return std::min(
                    num_clusters,
                    get_num_local_rows_points(local_size, num_clusters));
        }

        size_t get_num_global_blocks(
                size_t const global_size,
                size_t const local_size,
                size_t const num_features,
                size_t const num_clusters
                ) const {

            return global_size / (
                    get_num_local_rows_points(local_size, num_clusters)
                    * get_num_local_cols(local_size, num_clusters));
        }

    private:
        static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("lloyd_merge_sum.cl");
        static constexpr const char* KERNEL_NAME = "lloyd_merge_sum";

        std::shared_ptr<cl::Kernel> lloyd_merge_sum_kernel_;
    };
}

#endif /* LLOYD_MERGE_SUM_API_HPP */
