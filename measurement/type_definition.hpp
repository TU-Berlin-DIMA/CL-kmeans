/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef TYPE_DEFINITION_HPP
#define TYPE_DEFINITION_HPP

namespace Measurement {

struct DataPointType {
enum t {
    ChangesBuffer = 0,
    PointsBuffer,
    CentroidsBuffer,
    MassBuffer,
    LabelsBuffer,
    TotalTime,
    H2DPoints,
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
    AggregateCentroids,
    ReduceVectorParcol
};
};

struct ParameterType {
enum t {
    Version = 0,
    Filename,
    Hostname,
    Device,
    NumIterations,
    NumFeatures,
    NumPoints,
    NumClusters
};
};

} // namespace Measurement

#endif /* TYPE_DEFINITION_HPP */
