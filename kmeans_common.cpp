/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "kmeans_common.hpp"

#include <chrono>
#include <cassert>
#include <cstring>
#include <string>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <clext.hpp>

char const* cle_data_point_type_name[] = {
    "H2DPoints",
    "H2DCentroids",
    "D2HLabels",
    "D2HChanges",
    "FillChanges",
    "FillLables",
    "LloydLabelingNaive",
    "LloydLabelingPlain",
    "LloydLabelingVpClc",
    "LloydLabelingVpClcp",
    "LloydMassSumGlobalAtomic",
    "LloydMassSumMerge",
    "LloydCentroidsNaive",
    "LloydCentroidsFeatureSum",
    "LloydCentroidsMergeSum",
    "AggregateMass",
    "AggregateCentroids"
};

cle::DataPoint::DataPoint(Type type, int iteration, uint64_t nanoseconds)
    :
        type_(type),
        iteration_(iteration),
        nanoseconds_(nanoseconds),
        has_event_(false)
{}

cle::DataPoint::DataPoint(Type type, int iteration)
    :
        type_(type),
        iteration_(iteration),
        nanoseconds_(0),
        has_event_(true)
{}

cle::DataPoint::Type cle::DataPoint::get_type() {
    return type_;
}

char const* cle::DataPoint::get_name() {
    return cle_data_point_type_name[type_];
}

int cle::DataPoint::get_iteration() {
    return iteration_;
}

uint64_t cle::DataPoint::get_nanoseconds() {
    if (has_event_) {
        cl_ulong start, end;

        event_.wait();
        event_.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        event_.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
        return end - start;
    }
    else {
        return nanoseconds_;
    }
}

cl::Event& cle::DataPoint::get_event() {
    return event_;
}



char const* cle_buffer_info_type_name[] = {
    "Changes",
    "Points",
    "Centroids",
    "Labels",
    "Mass"
};

cle::BufferInfo::BufferInfo(Type type, size_t size)
    :
        type_(type),
        size_(size)
{}

cle::BufferInfo::Type cle::BufferInfo::get_type() {
    return type_;
}

char const* cle::BufferInfo::get_name() {
    return cle_buffer_info_type_name[type_];
}

size_t cle::BufferInfo::get_size() {
    return size_;
}

void cle::KmeansStats::start_experiment(cl::Device device) {
    assert(is_initialized_ == false);


    std::string device_name;
    cle_sanitize_val(
            device.getInfo(CL_DEVICE_NAME, &device_name));

    std::strncpy(
            device_name_,
            device_name.c_str(),
            max_device_name_length_
            );

    run_date_ = std::chrono::system_clock::now();
}

std::chrono::system_clock::time_point cle::KmeansStats::get_run_date() const {
    return run_date_;
}

char const* cle::KmeansStats::get_device_name() const {
    return device_name_;
}
