/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "measurement.hpp"

#include "mapping_definition.hpp"
#include "type_definition.hpp"

#include <boost/filesystem/path.hpp>
#include <cassert>
#include <chrono>
#include <fstream>
#include <unistd.h>
#include <random>
#include <sstream>

uint32_t const max_datetime_length = 30;
char const *const timestamp_format = "%F-%H-%M-%S";

char const *const experiment_file_suffix = "_expm";
char const *const measurements_file_suffix = "_mnts";

bool Measurement::DataPoint::is_iterative() { return iterative_; }

int Measurement::DataPoint::get_iteration() { return iteration_; }

uint64_t Measurement::DataPoint::get_value() {
  uint64_t value;

  if (has_event_) {
    cl_ulong start, end;

    event_.wait();
    event_.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    value = end - start;
  } else {
    value = value_;
  }

  uint64_t converted;
  Mapping::Unit::u unit = Mapping::type_unit[type_];
  switch (unit) {
      case Mapping::Unit::Second:
          converted = value / (1000 * 1000 * 1000);
          break;
      case Mapping::Unit::Millisecond:
          converted = value / (1000 * 1000);
          break;
      case Mapping::Unit::Microsecond:
          converted = value / 1000;
          break;
      case Mapping::Unit::Nanosecond:
          converted = value;
          break;
      case Mapping::Unit::Byte:
          converted = value;
          break;
      case Mapping::Unit::Kilobyte:
          converted = value / 1024;
          break;
      case Mapping::Unit::Megabyte:
          converted = value / (1024 * 1024);
          break;
      case Mapping::Unit::Gigabyte:
          converted = value / (1024 * 1024 * 1024);
          break;
      default:
          converted = value;
  }

  return converted;
}

Measurement::DataPointType::t Measurement::DataPoint::get_type() {  return type_;
}

Measurement::Measurement::Measurement() : is_started_(false) {}
Measurement::Measurement::~Measurement() {}

void Measurement::Measurement::start() {
  assert(is_started_ == false);

  is_started_ = true;
  run_date_ = std::chrono::system_clock::now();
}

void Measurement::Measurement::end() { assert(is_started_ == true); }

void Measurement::Measurement::set_parameter(
        ParameterType::t type,
        std::string value
        ) {
  parameters_[type] = value;
}

void Measurement::Measurement::write_csv(std::string filename) {
  assert(is_started_ == true);

  std::string experiment_id = get_unique_id();

  {
    std::string experiment_file =
        format_filename(filename, experiment_id, experiment_file_suffix);

    std::ofstream pf(experiment_file, std::ios_base::out | std::ios::trunc);

    pf << "ID";
    pf << ',';
    pf << "Timestamp";

    for (ParameterType::t type = (ParameterType::t)0;
            type < get_num_parameter_types();
            type = ParameterType::t(type+1)
        ) {
        std::string name = get_parameter_type_name(type);

        pf << ',';
        pf << name;
    }

    pf << '\n';

    pf << experiment_id;
    pf << ',';
    pf << get_datetime();

    for (ParameterType::t type = (ParameterType::t)0;
            type < get_num_parameter_types();
            type = ParameterType::t(type+1)
        ) {

        pf << ',';
        if (exists_parameter(type)) {
          std::string value = get_parameter_value(type);
          pf << value;
        }
    }

    pf << '\n';

    pf.close();
    pf.clear();
  }

  {
    std::string measurements_file =
        format_filename(filename, experiment_id, measurements_file_suffix);

    std::ofstream mf(measurements_file, std::ios_base::out | std::ios::trunc);

    mf << "ExperimentID";
    mf << ',';
    mf << "TypeID";
    mf << ',';
    mf << "TypeName";
    mf << ',';
    mf << "Iteration";
    mf << ',';
    mf << "Value";
    mf << ',';
    mf << "Unit";

    mf << '\n';

    for (DataPoint dp : data_points_) {
        mf << experiment_id;
        mf << ',';
        mf << get_datapoint_type_id(dp.get_type());
        mf << ',';
        mf << get_datapoint_type_name(dp.get_type());
        mf << ',';
        if (dp.is_iterative() == true) {
            mf << dp.get_iteration();
        }
        mf << ',';
        mf << dp.get_value();
        mf << ',';
        mf << get_datapoint_type_unit(dp.get_type());
        mf << '\n';
    }

    mf.close();
    mf.clear();
  }
}

int Measurement::Measurement::get_num_datapoint_types() {
  assert(Mapping::type_name.size() == Mapping::type_unit.size());
  return Mapping::type_name.size();
}

int Measurement::Measurement::get_num_parameter_types() {
  return Mapping::parameter_name.size();
}

int Measurement::Measurement::get_datapoint_type_id(DataPointType::t type) {
  return type;
}

int Measurement::Measurement::get_parameter_type_id(ParameterType::t type) {
  return type;
}

std::string Measurement::Measurement::get_datapoint_type_name(DataPointType::t type) {
  std::string type_name = Mapping::type_name[type];
  return type_name;
}

std::string Measurement::Measurement::get_parameter_type_name(ParameterType::t type) {
  std::string type_name = Mapping::parameter_name[type];
  return type_name;
}

std::string Measurement::Measurement::get_datapoint_type_unit(DataPointType::t type) {
  Mapping::Unit::u unit = Mapping::type_unit[type];
  std::string unit_name = Mapping::unit_name[unit];
  return unit_name;
}

bool Measurement::Measurement::exists_parameter(ParameterType::t type) {
  return (parameters_.count(type) == 1) ? true : false;
}

std::string Measurement::Measurement::get_parameter_value(ParameterType::t type) {
  std::string value = parameters_[type];
  return value;
}

std::string Measurement::Measurement::get_unique_id() {
  std::random_device rand;
  std::stringstream ss;

  ss << std::dec;
  ss << rand();

  return ss.str();
}

std::string Measurement::Measurement::get_datetime() {
  char datetime[max_datetime_length];
  std::time_t timet_date = std::chrono::system_clock::to_time_t(run_date_);
  std::tm *timeinfo_date = std::gmtime(&timet_date);
  std::strftime(datetime, max_datetime_length, timestamp_format, timeinfo_date);

  return datetime;
}

std::string Measurement::Measurement::format_filename(
        std::string base_file,
        std::string experiment_id,
        std::string data_suffix
        ) {
  boost::filesystem::path file_path(base_file);
  boost::filesystem::path file_parent = file_path.parent_path();
  boost::filesystem::path file_stem = file_path.stem();
  boost::filesystem::path file_suffix = file_path.extension();

  boost::filesystem::path filename;
  filename += file_parent;
  filename /= get_datetime();
  filename += '_';
  filename += experiment_id;
  filename += '_';
  filename += file_stem;
  filename += data_suffix;
  filename += file_suffix;

  return filename.string();
}

