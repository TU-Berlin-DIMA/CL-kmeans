/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef KMEANS_COMMON_HPP
#define KMEANS_COMMON_HPP

#include <cstdint>
#include <memory> // std::allocator
#include <vector>
#include <chrono>

#include <SystemConfig.h>

#ifdef USE_ALIGNED_ALLOCATOR
#include <boost/align/aligned_allocator.hpp>
#endif

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace cle {

class DataPoint {
public:
    // Don't forget to update cle_data_point_type_name strings in .cpp
    enum Type {
        H2DPoints = 0,
        H2DCentroids,
        D2HLabels,
        D2HChanges,
        FillChanges,
        FillLables,
        LloydLabelingNaive,
        LloydLabelingPlain,
        LloydLabelingVpClc,
        LloydLabelingVpClcp,
        LloydMassSumGlobalAtomic,
        LloydMassSumMerge,
        LloydCentroidsNaive,
        LloydCentroidsFeatureSum,
        LloydCentroidsMergeSum,
        AggregateMass,
        AggregateCentroids
    };

    DataPoint(Type type, int iteration, uint64_t nanoseconds);
    DataPoint(Type type, int iteration);

    Type get_type();
    char const* get_name();
    int get_iteration();
    uint64_t get_nanoseconds();
    cl::Event& get_event();

    static int get_num_types();
    static char const* type_to_name(Type type);

private:
    Type type_;
    int iteration_;
    uint64_t nanoseconds_;
    bool has_event_;
    cl::Event event_;
};

class BufferInfo {
public:
    enum Type {
        Changes,
        Points,
        Centroids,
        Labels,
        Mass
    };

    BufferInfo(Type type, size_t size);

    Type get_type();
    char const* get_name();
    size_t get_size();

private:
    Type type_;
    size_t size_;
};

class KmeansStats {
public:
    void start_experiment(cl::Device device);
    std::chrono::system_clock::time_point get_run_date() const;
    char const* get_device_name() const;

    inline cle::DataPoint& add_point(DataPoint::Type type, int iteration) {
        iter_points_.emplace_back(type, iteration);
        return iter_points_.back();
    }

    inline cle::DataPoint& add_point(DataPoint::Type type, int iteration, uint64_t nanoseconds) {
        iter_points_.emplace_back(type, iteration, nanoseconds);
        return iter_points_.back();
    }

    inline cle::DataPoint& add_point(DataPoint::Type type) {
        iter_points_.emplace_back(type, -1);
        return once_points_.back();
    }

    inline cle::DataPoint& add_point(DataPoint::Type type, uint64_t nanoseconds) {
        iter_points_.emplace_back(type, -1, nanoseconds);
        return once_points_.back();
    }

    std::vector<cle::DataPoint> data_points;
    std::vector<cle::BufferInfo> buffer_info;
    int iterations;

private:
    std::vector<cle::DataPoint> iter_points_;
    std::vector<cle::DataPoint> once_points_;

    static constexpr uint32_t max_device_name_length_ = 30;
    bool is_initialized_ = false;
    char device_name_[max_device_name_length_];
    std::chrono::system_clock::time_point run_date_;
};

#ifdef USE_ALIGNED_ALLOCATOR
using AlignedAllocatorFP32 =
    boost::alignment::aligned_allocator<float, 32>;
using AlignedAllocatorINT32 =
    boost::alignment::aligned_allocator<uint32_t, 32>;
using AlignedAllocatorFP64 =
    boost::alignment::aligned_allocator<double, 32>;
using AlignedAllocatorINT64 =
    boost::alignment::aligned_allocator<uint64_t, 32>;
#endif /* USE_ALIGNED_ALLOCATOR */

}

#endif /* KMEANS_COMMON_HPP */
