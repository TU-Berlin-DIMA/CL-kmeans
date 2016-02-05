#include "kmeans.hpp"
/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */


#include "cle/common.hpp"

#include "kmeans_cl_api.hpp"

#include <cstdint>
#include <cstddef> // size_t
#include <vector>
#include <algorithm> // std::any_of

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

cle::KmeansGPUAssisted::KmeansGPUAssisted(
        cl::Context const& context, cl::CommandQueue const& queue)
    :
        context_(context), queue_(queue)
{}


int cle::KmeansGPUAssisted::initialize() {

    cl_int err = CL_SUCCESS;

    // Create kernel
    err = this->kmeans_kernel_.initialize(this->context_);
    if (err != CL_SUCCESS) {
        return err;
    }

    cl::Device device;
    cle_sanitize_val_return(
            this->queue_.getInfo(
                CL_QUEUE_DEVICE,
                &device
                ));

    cle_sanitize_val_return(
            device.getInfo(
                CL_DEVICE_MAX_WORK_ITEM_SIZES,
                &this->max_work_item_sizes_
            ));
    cle_sanitize_val_return(
            device.getInfo(
                CL_DEVICE_MAX_WORK_GROUP_SIZE,
                &this->max_work_group_size_
                ));

    return 1;
}

int cle::KmeansGPUAssisted::finalize() {
    return 1;
}

int cle::KmeansGPUAssisted::operator() (
        uint32_t const max_iterations,
        std::vector<double> const& points_x,
        std::vector<double> const& points_y,
        std::vector<double>& centroids_x,
        std::vector<double>& centroids_y,
        std::vector<uint64_t>& cluster_size,
        std::vector<uint64_t>& memberships,
        KmeansStats& stats
        ) {

    uint32_t iterations;
    bool did_changes;

    size_t global_size = this->max_work_group_size_ * this->max_work_item_sizes_[0];
    const size_t num_points = points_x.size();
    const size_t num_clusters = centroids_x.size();
    std::vector<cl_char> h_did_changes(global_size);

    cle::TypedBuffer<cl_char> d_did_changes(this->context_, CL_MEM_READ_WRITE, global_size);
    cle::TypedBuffer<cl_double> d_points_x(this->context_, CL_MEM_READ_ONLY, num_points);
    cle::TypedBuffer<cl_double> d_points_y(this->context_, CL_MEM_READ_ONLY, num_points);
    cle::TypedBuffer<cl_double> d_centroids_x(this->context_, CL_MEM_READ_ONLY, num_clusters);
    cle::TypedBuffer<cl_double> d_centroids_y(this->context_, CL_MEM_READ_ONLY, num_clusters);
    cle::TypedBuffer<cl_ulong> d_memberships(this->context_, CL_MEM_READ_WRITE, num_points);


    // copy points (x,y) host -> device
    cle_sanitize_val_return(
            this->queue_.enqueueWriteBuffer(
                d_points_x,
                CL_FALSE,
                0,
                d_points_x.bytes(),
                points_x.data()
                ));

    cle_sanitize_val_return(
            this->queue_.enqueueWriteBuffer(
                d_points_y,
                CL_FALSE,
                0,
                d_points_y.bytes(),
                points_y.data()
                ));

    cle_sanitize_val_return(
            this->queue_.enqueueWriteBuffer(
                d_memberships,
                CL_FALSE,
                0,
                d_memberships.bytes(),
                memberships.data()
                ));

    iterations = 0;
    did_changes = true;
    while (did_changes == true && iterations < max_iterations) {

        // set did_changes to false on device
        cle_sanitize_val(
                this->queue_.enqueueFillBuffer(
                    d_did_changes,
                    false,
                    0,
                    d_did_changes.bytes()
                    ));

        cle_sanitize_val_return(
                this->queue_.enqueueWriteBuffer(
                    d_centroids_x,
                    CL_FALSE,
                    0,
                    d_centroids_x.bytes(),
                    centroids_x.data()
                    ));

        cle_sanitize_val_return(
                this->queue_.enqueueWriteBuffer(
                    d_centroids_y,
                    CL_FALSE,
                    0,
                    d_centroids_y.bytes(),
                    centroids_y.data()
                    ));

        // execute kernel
        kmeans_kernel_(
                cl::EnqueueArgs(
                    this->queue_,
                    cl::NDRange(global_size),
                    cl::NDRange(this->max_work_group_size_)
                    ),
                d_did_changes,
                d_points_x,
                d_points_y,
                d_centroids_x,
                d_centroids_y,
                d_memberships
                );

        // copy did_changes device -> host
        cle_sanitize_val(
                this->queue_.enqueueReadBuffer(
                    d_did_changes,
                    CL_FALSE,
                    0,
                    d_did_changes.bytes(),
                    h_did_changes.data()
                    ));

        cle_sanitize_val(
                this->queue_.enqueueReadBuffer(
                    d_memberships,
                    CL_TRUE,
                    0,
                    d_memberships.bytes(),
                    memberships.data()
                    ));

        // inspect did_changes
        did_changes = std::any_of(
                h_did_changes.cbegin(),
                h_did_changes.cend(),
                [](cl_char i){ return i == 1; }
                );

        if (did_changes == true) {
            // Initialize to zero
            std::fill(centroids_x.begin(), centroids_x.end(), 0);
            std::fill(centroids_y.begin(), centroids_y.end(), 0);
            std::fill(cluster_size.begin(), cluster_size.end(), 0);

            // Calculate new centroids
            for (uint32_t p = 0; p < num_points; ++p) {
                uint32_t c = memberships[p];

                centroids_x[c] += points_x[p];
                centroids_y[c] += points_y[p];
                ++cluster_size[c];
            }

            for (uint32_t c = 0; c < num_clusters; ++c) {
                centroids_x[c] = centroids_x[c] / cluster_size[c];
                centroids_y[c] = centroids_y[c] / cluster_size[c];
            }
        }

        ++iterations;
    }
    std::cout << "iters: " << iterations << std::endl;

    return 1;
}
