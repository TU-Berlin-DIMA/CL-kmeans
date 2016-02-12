/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "lloyd_gpu_feature_sum.hpp"

#include "cle/common.hpp"
#include "cl_kernels/lloyd_labeling_api.hpp"
#include "cl_kernels/lloyd_feature_sum_api.hpp"
#include "cl_kernels/mass_sum_global_atomic_api.hpp"

#include <cstdint>
#include <cstddef> // size_t
#include <vector>
#include <algorithm> // std::any_of
#include <type_traits>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
cle::LloydGPUFeatureSum<FP, INT, AllocFP, AllocINT>::LloydGPUFeatureSum(
        cl::Context const& context, cl::CommandQueue const& queue)
    :
        context_(context), queue_(queue)
{}

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
char const* cle::LloydGPUFeatureSum<FP, INT, AllocFP, AllocINT>::name() const {

    return "Lloyd_GPU_Feature_Sum";
}

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
int cle::LloydGPUFeatureSum<FP, INT, AllocFP, AllocINT>::initialize() {

    cl_int err = CL_SUCCESS;

    // Create kernels
    err = labeling_kernel_.initialize(context_);
    if (err != CL_SUCCESS) {
        return err;
    }

    err = feature_sum_kernel_.initialize(context_);
    if (err != CL_SUCCESS) {
        return err;
    }

    err = mass_sum_kernel_.initialize(context_);
    if (err != CL_SUCCESS) {
        return err;
    }

    cl::Device device;
    cle_sanitize_val_return(
            queue_.getInfo(
                CL_QUEUE_DEVICE,
                &device
                ));

    cle_sanitize_val_return(
            device.getInfo(
                CL_DEVICE_MAX_WORK_ITEM_SIZES,
                &max_work_item_sizes_
            ));

    cle_sanitize_val_return(
            device.getInfo(
                CL_DEVICE_MAX_WORK_GROUP_SIZE,
                &max_work_group_size_
                ));

    return 1;
}

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
int cle::LloydGPUFeatureSum<FP, INT, AllocFP, AllocINT>::finalize() {
    return 1;
}

template <typename FP, typename INT, typename AllocFP, typename AllocINT>
int cle::LloydGPUFeatureSum<FP, INT, AllocFP, AllocINT>::operator() (
        uint32_t const max_iterations,
        cle::Matrix<FP, AllocFP, INT, true> const& points,
        cle::Matrix<FP, AllocFP, INT, true>& centroids,
        std::vector<INT, AllocINT>& mass,
        std::vector<INT, AllocINT>& labels,
        KmeansStats& stats
        ) {

    assert(points.cols() == centroids.cols());
    assert(points.rows() == labels.size());
    assert(centroids.rows() == mass.size());

    uint32_t iterations;
    bool did_changes;

    size_t global_size = max_work_group_size_ * max_work_item_sizes_[0];
    std::vector<cl_char> h_did_changes(global_size);

    cle::TypedBuffer<cl_char> d_did_changes(context_, CL_MEM_READ_WRITE, global_size);
    cle::TypedBuffer<CL_FP> d_points(context_, CL_MEM_READ_ONLY, points.size());
    cle::TypedBuffer<CL_FP> d_centroids(context_, CL_MEM_READ_WRITE, centroids.size());
    cle::TypedBuffer<CL_INT> d_mass(context_, CL_MEM_READ_WRITE, centroids.rows());
    cle::TypedBuffer<CL_INT> d_labels(context_, CL_MEM_READ_WRITE, points.rows());

    // copy points (x,y) host -> device
    cle_sanitize_val_return(
            queue_.enqueueWriteBuffer(
                d_points,
                CL_FALSE,
                0,
                d_points.bytes(),
                points.data()
                ));

    // copy labels host -> device
    cle_sanitize_val_return(
            queue_.enqueueFillBuffer(
                d_labels,
                0,
                0,
                d_labels.bytes()
                ));

    iterations = 0;
    did_changes = true;
    while (did_changes == true && iterations < max_iterations) {

        // set did_changes to false on device
        cle_sanitize_val(
                queue_.enqueueFillBuffer(
                    d_did_changes,
                    false,
                    0,
                    d_did_changes.bytes()
                    ));

        // copy centroids host -> device
        cle_sanitize_val_return(
                queue_.enqueueWriteBuffer(
                    d_centroids,
                    CL_FALSE,
                    0,
                    d_centroids.bytes(),
                    centroids.data()
                    ));

        // execute kernel
        cl::Event labeling_event;
        cle_sanitize_done_return(
                labeling_kernel_(
                cl::EnqueueArgs(
                    queue_,
                    cl::NDRange(global_size),
                    cl::NDRange(max_work_group_size_)
                    ),
                d_did_changes,
                points.cols(),
                points.rows(),
                centroids.rows(),
                d_points,
                d_centroids,
                d_labels,
                labeling_event
                ));

        // copy did_changes device -> host
        cle_sanitize_val(
                queue_.enqueueReadBuffer(
                    d_did_changes,
                    CL_TRUE,
                    0,
                    d_did_changes.bytes(),
                    h_did_changes.data()
                    ));

        // inspect did_changes
        did_changes = std::any_of(
                h_did_changes.cbegin(),
                h_did_changes.cend(),
                [](cl_char i){ return i == 1; }
                );

        if (did_changes == true) {
            // calculate sum of points per cluster
            cl::Event feature_sum_event;
            cle_sanitize_done_return(
                    feature_sum_kernel_(
                        cl::EnqueueArgs(
                            queue_,
                            cl::NDRange(global_size),
                            cl::NDRange(max_work_group_size_)
                            ),
                        points.cols(),
                        points.rows(),
                        centroids.rows(),
                        d_points,
                        d_centroids,
                        d_labels,
                        feature_sum_event
                        ));

            // calculate cluster mass
            cl::Event mass_sum_event;
            cle_sanitize_done_return(
                    mass_sum_kernel_(
                        cl::EnqueueArgs(
                            queue_,
                            cl::NDRange(global_size),
                            cl::NDRange(max_work_group_size_)
                            ),
                        points.rows(),
                        centroids.rows(),
                        d_labels,
                        d_mass,
                        mass_sum_event
                        ));

            // copy centroids device -> host
            cle_sanitize_val(
                    queue_.enqueueReadBuffer(
                        d_centroids,
                        CL_FALSE,
                        0,
                        d_centroids.bytes(),
                        centroids.data()
                        ));

            // copy cluster mass device -> host
            cle_sanitize_val(
                    queue_.enqueueReadBuffer(
                        d_mass,
                        CL_TRUE,
                        0,
                        d_mass.bytes(),
                        mass.data()
                        ));

            // divide point sums by cluster mass
            for (INT f = 0; f < centroids.cols(); ++f) {
                for (INT c = 0; c < centroids.rows(); ++c) {
                    centroids(c, f) = centroids(c, f) / mass[c];
                }
            }
        }

        ++iterations;
    }

    stats.iterations = iterations;

    return 1;
}

template class cle::LloydGPUFeatureSum<float, uint32_t, cle::AlignedAllocatorFP32, cle::AlignedAllocatorINT32>;
template class cle::LloydGPUFeatureSum<double, uint64_t, cle::AlignedAllocatorFP64, cle::AlignedAllocatorINT64>;
