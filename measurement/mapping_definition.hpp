/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef MAPPING_DEFINITION_HPP
#define MAPPING_DEFINITION_HPP

#include "type_definition.hpp"

#include <map>
#include <string>

namespace Measurement {
namespace Mapping {

struct Unit {
enum u {
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
    Byte,
    Kilobyte,
    Megabyte,
    Gigabyte
};
};

std::map<DataPointType::t, std::string> type_name = {
    {DataPointType::ChangesBuffer, "ChangesBuffer"},
    {DataPointType::PointsBuffer, "PointsBuffer"},
    {DataPointType::CentroidsBuffer, "CentroidsBuffer"},
    {DataPointType::MassBuffer, "MassBuffer"},
    {DataPointType::LabelsBuffer, "LabelsBuffer"},
    {DataPointType::TotalTime, "TotalTime"},
    {DataPointType::H2DPoints, "H2DPoints"},
    {DataPointType::H2DCentroids, "H2DCentroids"},
    {DataPointType::D2HLabels, "D2HLabels"},
    {DataPointType::D2HChanges, "D2HChanges"},
    {DataPointType::FillChanges, "FillChanges"},
    {DataPointType::FillLables, "FillLables"},
    {DataPointType::LloydLabelingNaive, "LloydLabelingNaive"},
    {DataPointType::LloydLabelingPlain, "LloydLabelingPlain"},
    {DataPointType::LloydLabelingVpClc, "LloydLabelingVpClc"},
    {DataPointType::LloydLabelingVpClcp, "LloydLabelingVpClcp"},
    {DataPointType::LloydMassSumGlobalAtomic, "LloydMassSumGlobalAtomic"},
    {DataPointType::LloydMassSumMerge, "LloydMassSumMerge"},
    {DataPointType::LloydCentroidsNaive, "LloydCentroidsNaive"},
    {DataPointType::LloydCentroidsFeatureSum, "LloydCentroidsFeatureSum"},
    {DataPointType::LloydCentroidsMergeSum, "LloydCentroidsMergeSum"},
    {DataPointType::AggregateMass, "AggregateMass"},
    {DataPointType::AggregateCentroids, "AggregateCentroids"},
    {DataPointType::ReduceVectorParcol, "ReduceVectorParcol"},
    {DataPointType::HistogramPartGlobal, "HistogramPartGlobal"},
    {DataPointType::HistogramPartPrivate, "HistogramPartPrivate"},
    {DataPointType::LloydCentroidsFeatureSumPardim, "LloydCentroidsFeatureSumPardim"}
};

std::map<DataPointType::t, Unit::u> type_unit = {
    {DataPointType::ChangesBuffer, Unit::Byte},
    {DataPointType::PointsBuffer, Unit::Byte},
    {DataPointType::CentroidsBuffer, Unit::Byte},
    {DataPointType::MassBuffer, Unit::Byte},
    {DataPointType::LabelsBuffer, Unit::Byte},
    {DataPointType::TotalTime, Unit::Microsecond},
    {DataPointType::H2DPoints, Unit::Microsecond},
    {DataPointType::H2DCentroids, Unit::Microsecond},
    {DataPointType::D2HLabels, Unit::Microsecond},
    {DataPointType::D2HChanges, Unit::Microsecond},
    {DataPointType::FillChanges, Unit::Microsecond},
    {DataPointType::FillLables, Unit::Microsecond},
    {DataPointType::LloydLabelingNaive, Unit::Microsecond},
    {DataPointType::LloydLabelingPlain, Unit::Microsecond},
    {DataPointType::LloydLabelingVpClc, Unit::Microsecond},
    {DataPointType::LloydLabelingVpClcp, Unit::Microsecond},
    {DataPointType::LloydMassSumGlobalAtomic, Unit::Microsecond},
    {DataPointType::LloydMassSumMerge, Unit::Microsecond},
    {DataPointType::LloydCentroidsNaive, Unit::Microsecond},
    {DataPointType::LloydCentroidsFeatureSum, Unit::Microsecond},
    {DataPointType::LloydCentroidsMergeSum, Unit::Microsecond},
    {DataPointType::AggregateMass, Unit::Microsecond},
    {DataPointType::AggregateCentroids, Unit::Microsecond},
    {DataPointType::ReduceVectorParcol, Unit::Microsecond},
    {DataPointType::HistogramPartGlobal, Unit::Microsecond},
    {DataPointType::HistogramPartPrivate, Unit::Microsecond},
    {DataPointType::LloydCentroidsFeatureSumPardim, Unit::Microsecond}
};

std::map<Unit::u, std::string> unit_name = {
    {Unit::Second, "s"},
    {Unit::Millisecond, "ms"},
    {Unit::Microsecond, "us"},
    {Unit::Nanosecond, "ns"},
    {Unit::Byte, "B"},
    {Unit::Kilobyte, "KiB"},
    {Unit::Megabyte, "MiB"},
    {Unit::Gigabyte, "GiB"}
};

std::map<ParameterType::t, std::string> parameter_name = {
    {ParameterType::Version, "Version"},
    {ParameterType::Filename, "Filename"},
    {ParameterType::Hostname, "Hostname"},
    {ParameterType::Device, "Device"},
    {ParameterType::NumIterations, "NumIterations"},
    {ParameterType::NumFeatures, "NumFeatures"},
    {ParameterType::NumPoints, "NumPoints"},
    {ParameterType::NumClusters, "NumClusters"},
    {ParameterType::IntType, "IntType"},
    {ParameterType::FloatType, "FloatType"},
    {ParameterType::CLGlobalSize, "CLGlobalSize"},
    {ParameterType::CLLocalSize, "CLLocalSize"}
};

} // namespace Mapping
} // namespace Measurement

#endif /* MAPPING_DEFINITION_HPP */
