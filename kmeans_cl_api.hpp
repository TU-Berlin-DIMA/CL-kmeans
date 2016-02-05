/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef KMEANS_CL_API_HPP
#define KMEANS_CL_API_HPP

#include "cle/common.hpp"

#include <functional>
#include <cassert>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace cle {

    class Kmeans_With_Host_Kernel {
    public:
        cl_int initialize(cl::Context& context) {
            cl_int error_code = CL_SUCCESS;

            cl::Program program = make_program(context, PROGRAM_FILE, error_code);
            if (error_code != CL_SUCCESS) {
                return error_code;
            }

            kernel_functor_ = Base_Kernel(program,KERNEL_NAME, &error_code);
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
        void operator() (
                cl::EnqueueArgs const& args,
                TypedBuffer<cl_char>& did_changes,
                cl_ulong num_features,
                cl_ulong num_points,
                cl_ulong num_clusters,
                TypedBuffer<cl_double>& points,
                TypedBuffer<cl_double>& centroids,
                TypedBuffer<cl_ulong>& memberships
                ) {

            // assert did_changes.size() == #global work items
            assert(points.size() == num_points * num_features);
            assert(memberships.size() == num_points);
            assert(centroids.size() == num_clusters * num_features);

            const size_t centroid_bytes = centroids.bytes();
            cl::LocalSpaceArg local_centroids = cl::Local(centroid_bytes);
            cl::LocalSpaceArg local_old_centroids = cl::Local(centroid_bytes);

            kernel_functor_(
                    args,
                    did_changes,
                    points,
                    centroids,
                    memberships,
                    local_centroids,
                    local_old_centroids,
                    num_features,
                    num_points,
                    num_clusters
            );


        }

    private:
        typedef cl::make_kernel<
            cl::Buffer&,
            cl::Buffer&,
            cl::Buffer&,
            cl::Buffer&,
            cl::LocalSpaceArg,
            cl::LocalSpaceArg,
            cl_ulong,
            cl_ulong,
            cl_ulong
                > Base_Kernel;

        typedef std::function<Base_Kernel::type_> Kernel_Functor;

        static constexpr const char* PROGRAM_FILE = CL_KERNEL_FILE_PATH("kmeans.cl");
        static constexpr const char* KERNEL_NAME = "kmeans_with_host";

        Kernel_Functor kernel_functor_;
    };
}

#endif /* KMEANS_CL_API_HPP */
