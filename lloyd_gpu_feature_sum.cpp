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
#include "cl_kernels/lloyd_labeling_vp_clc_api.hpp"
#include "cl_kernels/lloyd_labeling_vp_clcp_api.hpp"
#include "cl_kernels/lloyd_feature_sum_api.hpp"
#include "cl_kernels/lloyd_merge_sum_api.hpp"
#include "cl_kernels/mass_sum_global_atomic_api.hpp"
#include "cl_kernels/mass_sum_merge_api.hpp"
#include "cl_kernels/aggregate_sum_api.hpp"
#include "timer.hpp"

#include <clext.hpp>

#include <cstdint>
#include <cstddef> // size_t
#include <vector>
#include <algorithm> // std::any_of
#include <string>

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
            labeling_vp_clc_kernel_.initialize(context_, 4, 8));

    cle_sanitize_done_return(
            labeling_vp_clcp_kernel_.initialize(context_, 4, 8));

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
            reduce_vector_parcol_kernel_.initialize(context_));

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
        Measurement::Measurement& stats
        ) {

    assert(points.cols() == centroids.cols());
    assert(points.rows() == labels.size());
    assert(centroids.rows() == mass.size());

    std::string device_name;
    cle_sanitize_val(
            device_.getInfo(CL_DEVICE_NAME, &device_name));
    stats.set_parameter(
            Measurement::ParameterType::Device,
            device_name.c_str() // remove trailing '\0'
            );

    uint32_t iterations;
    bool did_changes;
    cl::Event event_dummy;

    uint64_t const labeling_local_size = warp_size_ * 4;
    uint64_t const labeling_global_size = labeling_local_size * 90 * 32;
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

    stats
        .add_datapoint(Measurement::DataPointType::ChangesBuffer)
        .add_value()
        = d_did_changes.bytes();
    stats
        .add_datapoint(Measurement::DataPointType::PointsBuffer)
        .add_value()
        = d_points.bytes();
    stats
        .add_datapoint(Measurement::DataPointType::CentroidsBuffer)
        .add_value()
        = d_centroids.bytes();
    stats
        .add_datapoint(Measurement::DataPointType::MassBuffer)
        .add_value()
        = d_mass.bytes();
    stats
        .add_datapoint(Measurement::DataPointType::LabelsBuffer)
        .add_value()
        = d_labels.bytes();

    // copy points (x,y) host -> device
    cle_sanitize_val_return(
            queue_.enqueueWriteBuffer(
                d_points,
                CL_FALSE,
                0,
                d_points.bytes(),
                points.data(),
                NULL,
                &stats
                .add_datapoint(
                    Measurement::DataPointType::H2DPoints
                    )
                .add_opencl_event()
                ));

    // copy labels host -> device
    cle_sanitize_val_return(
            queue_.enqueueFillBuffer(
                d_labels,
                std::numeric_limits<CL_INT>::max(),
                0,
                d_labels.bytes(),
                NULL,
                &stats
                .add_datapoint(
                    Measurement::DataPointType::FillLables
                    )
                .add_opencl_event()
                ));

    // copy centroids host -> device
    cle_sanitize_val_return(
            queue_.enqueueWriteBuffer(
                d_centroids,
                CL_FALSE,
                0,
                d_centroids.bytes(),
                centroids.data(),
                NULL,
                &stats
                .add_datapoint(
                    Measurement::DataPointType::H2DCentroids
                    )
                .add_opencl_event()
                ));

    stats.start();
    Timer total_timer;
    total_timer.start();

    iterations = 0;
    did_changes = true;
    while (did_changes == true && iterations < max_iterations) {

        // set did_changes to false on device
        cle_sanitize_val(
                queue_.enqueueFillBuffer(
                    d_did_changes,
                    false,
                    0,
                    d_did_changes.bytes(),
                    NULL,
                    &stats
                    .add_datapoint(
                        Measurement::DataPointType::FillChanges,
                        iterations
                        )
                    .add_opencl_event()
                    ));

        // execute kernel
        switch (labeling_strategy_) {
            case LabelingStrategy::Plain:
                cle_sanitize_done_return(
                        labeling_kernel_(
                            cl::EnqueueArgs(
                                queue_,
                                cl::NDRange(labeling_global_size),
                                cl::NDRange(labeling_local_size)
                                ),
                            d_did_changes,
                            points.cols(),
                            points.rows(),
                            centroids.rows(),
                            d_points,
                            d_centroids,
                            d_labels,
                            stats
                            .add_datapoint(
                                Measurement::DataPointType::LloydLabelingPlain,
                                iterations
                                )
                            .add_opencl_event()
                            ));
                break;
            case LabelingStrategy::VpClc:
                cle_sanitize_done_return(
                        labeling_vp_clc_kernel_(
                            cl::EnqueueArgs(
                                queue_,
                                cl::NDRange(labeling_global_size),
                                cl::NDRange(labeling_local_size)
                                ),
                            d_did_changes,
                            points.cols(),
                            points.rows(),
                            centroids.rows(),
                            d_points,
                            d_centroids,
                            d_labels,
                            stats
                            .add_datapoint(
                                Measurement::DataPointType::LloydLabelingVpClc,
                                iterations
                                )
                            .add_opencl_event()
                            ));
                break;
            case LabelingStrategy::VpClcp:
                cle_sanitize_done_return(
                        labeling_vp_clcp_kernel_(
                            cl::EnqueueArgs(
                                queue_,
                                cl::NDRange(labeling_global_size),
                                cl::NDRange(labeling_local_size)
                                ),
                            d_did_changes,
                            points.cols(),
                            points.rows(),
                            centroids.rows(),
                            d_points,
                            d_centroids,
                            d_labels,
                            stats
                            .add_datapoint(
                                Measurement::DataPointType::LloydLabelingVpClcp,
                                iterations
                                )
                            .add_opencl_event()
                            ));
                break;
        }

        // copy did_changes device -> host
        cle_sanitize_val(
                queue_.enqueueReadBuffer(
                    d_did_changes,
                    CL_TRUE,
                    0,
                    d_did_changes.bytes(),
                    h_did_changes.data(),
                    NULL,
                    &stats
                    .add_datapoint(
                        Measurement::DataPointType::D2HChanges,
                        iterations
                        )
                    .add_opencl_event()
                    ));

        // inspect did_changes
        did_changes = std::any_of(
                h_did_changes.cbegin(),
                h_did_changes.cend(),
                [](cl_char i){ return i == 1; }
                );

        if (did_changes == true) {
            // calculate cluster mass
            Timer reduce_vector_parcol_timer;
            switch (mass_sum_strategy_) {
                case MassSumStrategy::GlobalAtomic:
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
                                stats
                                .add_datapoint(
                                    Measurement::DataPointType::LloydMassSumGlobalAtomic,
                                    iterations
                                    )
                                .add_opencl_event()
                                ));
                    break;

                case MassSumStrategy::Merge:
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
                                stats
                                .add_datapoint(
                                    Measurement::DataPointType::LloydMassSumMerge,
                                    iterations
                                    )
                                .add_opencl_event()
                                ));

                    // aggregate masses calculated by individual work groups
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
                                stats
                                .add_datapoint(
                                    Measurement::DataPointType::AggregateMass,
                                    iterations
                                    )
                                .add_opencl_event()
                                ));
                    break;

                case MassSumStrategy::MergeReduceVectorParcol:
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
                                stats
                                .add_datapoint(
                                    Measurement::DataPointType::LloydMassSumMerge,
                                    iterations
                                    )
                                .add_opencl_event()
                                ));

                    // aggregate masses calculated by individual work groups
                    reduce_vector_parcol_timer.start();
                    cle_sanitize_done_return(
                            reduce_vector_parcol_kernel_(
                                cl::EnqueueArgs(
                                    queue_,
                                    cl::NullRange,
                                    cl::NullRange
                                    ),
                                mass_sum_merge_num_work_groups,
                                centroids.rows(),
                                d_mass,
                                event_dummy
                                ));
                    stats
                        .add_datapoint(
                                Measurement::DataPointType::ReduceVectorParcol,
                                iterations
                                )
                        .add_value()
                        = reduce_vector_parcol_timer.stop<std::chrono::nanoseconds>();
                    break;
            }

            // calculate sum of points per cluster
            switch (centroid_update_strategy_) {
                case CentroidUpdateStrategy::FeatureSum:
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
                                stats
                                .add_datapoint(
                                    Measurement::DataPointType::LloydCentroidsFeatureSum,
                                    iterations
                                    )
                                .add_opencl_event()
                                ));
                    break;
                case CentroidUpdateStrategy::MergeSum:
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
                                stats
                                .add_datapoint(
                                    Measurement::DataPointType::LloydCentroidsMergeSum,
                                    iterations
                                    )
                                .add_opencl_event()
                                ));

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
                                stats
                                .add_datapoint(
                                    Measurement::DataPointType::AggregateCentroids,
                                    iterations
                                    )
                                .add_opencl_event()
                                ));
                    break;
            }
        }

        ++iterations;
    }

    stats.end();

    cle_sanitize_val(
            queue_.enqueueReadBuffer(
                d_labels,
                CL_TRUE,
                0,
                d_labels.bytes(),
                labels.data(),
                NULL,
                &stats
                .add_datapoint(
                    Measurement::DataPointType::D2HLabels
                    )
                .add_opencl_event()
                ));

    uint64_t total_time = total_timer.stop<std::chrono::nanoseconds>();
    stats
        .add_datapoint(
                Measurement::DataPointType::TotalTime
                )
        .add_value()
        = total_time;
    stats.set_parameter(
            Measurement::ParameterType::NumIterations,
            std::to_string(iterations)
            );

    return 1;
}

template class cle::LloydGPUFeatureSum<float, uint32_t, std::allocator<float>, std::allocator<uint32_t>>;
template class cle::LloydGPUFeatureSum<double, uint64_t, std::allocator<double>, std::allocator<uint64_t>>;
#ifdef USE_ALIGNED_ALLOCATOR
template class cle::LloydGPUFeatureSum<float, uint32_t, cle::AlignedAllocatorFP32, cle::AlignedAllocatorINT32>;
template class cle::LloydGPUFeatureSum<double, uint64_t, cle::AlignedAllocatorFP64, cle::AlignedAllocatorINT64>;
#endif
