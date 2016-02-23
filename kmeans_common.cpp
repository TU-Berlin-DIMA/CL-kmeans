/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "kmeans_common.hpp"

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

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


