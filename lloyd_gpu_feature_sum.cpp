/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "lloyd_gpu_feature_sum.hpp"

#include "cl_kernels/lloyd_labeling_api.hpp"
#include "cl_kernels/lloyd_feature_sum_api.hpp"
#include "cl_kernels/lloyd_merge_sum_api.hpp"
#include "cl_kernels/mass_sum_global_atomic_api.hpp"
#include "cl_kernels/mass_sum_merge_api.hpp"
#include "cl_kernels/aggregate_sum_api.hpp"

#include <clext.hpp>

#include <cstdint>
#include <cstddef> // size_t
#include <vector>
#include <algorithm> // std::any_of

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

    // Create kernels
    cle_sanitize_done_return(
            labeling_kernel_.initialize(context_));

    cle_sanitize_done_return(
            feature_sum_kernel_.initialize(context_));

    cle_sanitize_done_return(
            merge_sum_kernel_.initialize(context_));

    cle_sanitize_done_return(
            aggregate_centroid_kernel_.initialize(context_));

    cle_sanitize_done_return(
            mass_sum_kernel_.initialize(context_));

    cle_sanitize_done_return(
            mass_sum_merge_kernel_.initialize(context_));

    cle_sanitize_done_return(
            aggregate_mass_kernel_.initialize(context_));

    cle_sanitize_val_return(
            queue_.getInfo(
                CL_QUEUE_DEVICE,
                &device_
                ));

    cle_sanitize_done_return(
            cle::device_warp_size(device_, warp_size_));

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

    stats.start_experiment(device_);

    uint32_t iterations;
    bool did_changes;

    uint64_t const labeling_global_size =
        cle::optimize_global_size(points.rows(), warp_size_);
    uint64_t const mass_sum_global_atomic_global_size =
        cle::optimize_global_size(points.rows(), warp_size_);
    uint64_t const mass_sum_merge_global_size =
        cle::optimize_global_size(points.rows(), warp_size_);
    uint64_t const mass_sum_merge_num_work_groups =
        mass_sum_merge_global_size / warp_size_;
    uint64_t const aggregate_mass_num_work_items =
        mass_sum_merge_num_work_groups * centroids.rows();
    uint64_t const aggregate_mass_global_size =
        cle::optimize_global_size(aggregate_mass_num_work_items , warp_size_);
    uint64_t const feature_sum_global_size =
        cle::optimize_global_size(points.cols(), warp_size_);
    uint64_t const merge_sum_global_size = warp_size_ * 24 * 4;
    uint64_t const merge_sum_local_size = warp_size_ * 4;
    uint64_t const aggregate_centroid_global_size = warp_size_ * 24;
    uint64_t const centroid_merge_sum_num_blocks =
        merge_sum_kernel_.get_num_global_blocks(
                merge_sum_global_size,
                merge_sum_local_size,
                centroids.cols(),
                centroids.rows());

    std::vector<cl_char> h_did_changes(1);

    cle::TypedBuffer<cl_char> d_did_changes(context_, CL_MEM_READ_WRITE, 1);
    cle::TypedBuffer<CL_FP> d_points(context_, CL_MEM_READ_ONLY, points.size());
    cle::TypedBuffer<CL_FP> d_centroids(context_, CL_MEM_READ_WRITE, centroids.size() * centroid_merge_sum_num_blocks);
    cle::TypedBuffer<CL_INT> d_mass(context_, CL_MEM_READ_WRITE, aggregate_mass_num_work_items);
    cle::TypedBuffer<CL_INT> d_labels(context_, CL_MEM_READ_WRITE, points.rows());

    stats.buffer_info.emplace_back(
            cle::BufferInfo::Type::Changes,
            d_did_changes.bytes());
    stats.buffer_info.emplace_back(
            cle::BufferInfo::Type::Points,
            d_points.bytes());
    stats.buffer_info.emplace_back(
            cle::BufferInfo::Type::Centroids,
            d_centroids.bytes());
    stats.buffer_info.emplace_back(
            cle::BufferInfo::Type::Mass,
            d_mass.bytes());
    stats.buffer_info.emplace_back(
            cle::BufferInfo::Type::Labels,
            d_labels.bytes());

    // copy points (x,y) host -> device
    stats.data_points.emplace_back(
            cle::DataPoint::Type::H2DPoints,
            -1);
    cle_sanitize_val_return(
            queue_.enqueueWriteBuffer(
                d_points,
                CL_FALSE,
                0,
                d_points.bytes(),
                points.data(),
                NULL,
                &stats.data_points.back().get_event()
                ));

    // copy labels host -> device
    stats.data_points.emplace_back(
            cle::DataPoint::Type::FillLables,
            -1);
    cle_sanitize_val_return(
            queue_.enqueueFillBuffer(
                d_labels,
                std::numeric_limits<CL_INT>::max(),
                0,
                d_labels.bytes(),
                NULL,
                &stats.data_points.back().get_event()
                ));

    // copy centroids host -> device
    stats.data_points.emplace_back(
            cle::DataPoint::Type::H2DCentroids,
            -1);
    cle_sanitize_val_return(
            queue_.enqueueWriteBuffer(
                d_centroids,
                CL_FALSE,
                0,
                d_centroids.bytes(),
                centroids.data(),
                NULL,
                &stats.data_points.back().get_event()
                ));

    iterations = 0;
    did_changes = true;
    while (did_changes == true && iterations < max_iterations) {

        // set did_changes to false on device
        stats.data_points.emplace_back(
                cle::DataPoint::Type::FillChanges,
                iterations);
        cle_sanitize_val(
                queue_.enqueueFillBuffer(
                    d_did_changes,
                    false,
                    0,
                    d_did_changes.bytes(),
                    NULL,
                    &stats.data_points.back().get_event()
                    ));

        // execute kernel
        stats.data_points.emplace_back(
                cle::DataPoint::Type::LloydLabelingPlain,
                iterations);
        cle_sanitize_done_return(
                labeling_kernel_(
                cl::EnqueueArgs(
                    queue_,
                    cl::NDRange(labeling_global_size),
                    cl::NDRange(warp_size_)
                    ),
                d_did_changes,
                points.cols(),
                points.rows(),
                centroids.rows(),
                d_points,
                d_centroids,
                d_labels,
                stats.data_points.back().get_event()
                ));

        // copy did_changes device -> host
        stats.data_points.emplace_back(
                cle::DataPoint::Type::D2HChanges,
                iterations);
        cle_sanitize_val(
                queue_.enqueueReadBuffer(
                    d_did_changes,
                    CL_TRUE,
                    0,
                    d_did_changes.bytes(),
                    h_did_changes.data(),
                    NULL,
                    &stats.data_points.back().get_event()
                    ));

        // inspect did_changes
        did_changes = std::any_of(
                h_did_changes.cbegin(),
                h_did_changes.cend(),
                [](cl_char i){ return i == 1; }
                );

        if (did_changes == true) {
            // calculate cluster mass
            switch (mass_sum_strategy_) {
                case MassSumStrategy::GlobalAtomic:
                    stats.data_points.emplace_back(
                            cle::DataPoint::Type::LloydMassSumGlobalAtomic,
                            iterations);
                    cle_sanitize_done_return(
                            mass_sum_kernel_(
                                cl::EnqueueArgs(
                                    queue_,
                                    cl::NDRange(mass_sum_global_atomic_global_size),
                                    cl::NDRange(warp_size_)
                                    ),
                                points.rows(),
                                centroids.rows(),
                                d_labels,
                                d_mass,
                                stats.data_points.back().get_event()
                                ));
                    break;

                case MassSumStrategy::Merge:
                    stats.data_points.emplace_back(
                            cle::DataPoint::Type::LloydMassSumMerge,
                            iterations);
                    cle_sanitize_done_return(
                            mass_sum_merge_kernel_(
                                cl::EnqueueArgs(
                                    queue_,
                                    cl::NDRange(mass_sum_merge_global_size),
                                    cl::NDRange(warp_size_)
                                    ),
                                points.rows(),
                                centroids.rows(),
                                d_labels,
                                d_mass,
                                stats.data_points.back().get_event()
                                ));

                    // aggregate masses calculated by individual work groups
                    stats.data_points.emplace_back(
                            cle::DataPoint::Type::AggregateMass,
                            iterations);
                    cle_sanitize_done_return(
                            aggregate_mass_kernel_(
                                cl::EnqueueArgs(
                                    queue_,
                                    cl::NDRange(aggregate_mass_global_size),
                                    cl::NDRange(warp_size_)
                                    ),
                                centroids.rows(),
                                mass_sum_merge_num_work_groups,
                                d_mass,
                                stats.data_points.back().get_event()
                                ));
                    break;
            }

            // calculate sum of points per cluster
            switch (centroid_update_strategy_) {
                case CentroidUpdateStrategy::FeatureSum:
                    stats.data_points.emplace_back(
                            cle::DataPoint::Type::LloydCentroidsFeatureSum,
                            iterations);
                    cle_sanitize_done_return(
                            feature_sum_kernel_(
                                cl::EnqueueArgs(
                                    queue_,
                                    cl::NDRange(feature_sum_global_size),
                                    cl::NDRange(warp_size_)
                                    ),
                                points.cols(),
                                points.rows(),
                                centroids.rows(),
                                d_points,
                                d_centroids,
                                d_mass,
                                d_labels,
                                stats.data_points.back().get_event()
                                ));
                    break;
                case CentroidUpdateStrategy::MergeSum:
                    stats.data_points.emplace_back(
                            cle::DataPoint::Type::LloydCentroidsMergeSum,
                            iterations);
                    cle_sanitize_done_return(
                            merge_sum_kernel_(
                                cl::EnqueueArgs(
                                    queue_,
                                    cl::NDRange(merge_sum_global_size),
                                    cl::NDRange(merge_sum_local_size)
                                    ),
                                points.cols(),
                                points.rows(),
                                centroids.rows(),
                                d_points,
                                d_centroids,
                                d_mass,
                                d_labels,
                                stats.data_points.back().get_event()
                                ));

                    stats.data_points.emplace_back(
                            cle::DataPoint::Type::AggregateCentroids,
                            iterations);
                    cle_sanitize_done_return(
                            aggregate_centroid_kernel_(
                                cl::EnqueueArgs(
                                    queue_,
                                    cl::NDRange(aggregate_centroid_global_size),
                                    cl::NDRange(warp_size_)
                                    ),
                                centroids.size(),
                                centroid_merge_sum_num_blocks,
                                d_centroids,
                                stats.data_points.back().get_event()
                                ));
                    break;
            }
        }

        ++iterations;
    }

    stats.data_points.emplace_back(
            cle::DataPoint::Type::D2HLabels,
            -1);
    cle_sanitize_val(
            queue_.enqueueReadBuffer(
                d_labels,
                CL_TRUE,
                0,
                d_labels.bytes(),
                labels.data(),
                NULL,
                &stats.data_points.back().get_event()
                ));

    stats.iterations = iterations;

    return 1;
}

template class cle::LloydGPUFeatureSum<float, uint32_t, std::allocator<float>, std::allocator<uint32_t>>;
template class cle::LloydGPUFeatureSum<double, uint64_t, std::allocator<double>, std::allocator<uint64_t>>;
#ifdef USE_ALIGNED_ALLOCATOR
template class cle::LloydGPUFeatureSum<float, uint32_t, cle::AlignedAllocatorFP32, cle::AlignedAllocatorINT32>;
template class cle::LloydGPUFeatureSum<double, uint64_t, cle::AlignedAllocatorFP64, cle::AlignedAllocatorINT64>;
#endif
