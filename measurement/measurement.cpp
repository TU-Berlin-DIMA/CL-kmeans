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
#include <chrono>
#include <fstream>
#include <unistd.h>
#include <random>
#include <sstream>

uint32_t const max_datetime_length = 30;
char const *const timestamp_format = "%F-%H-%M-%S";

char const *const experiment_file_suffix = "_expm";
char const *const measurements_file_suffix = "_mnts";

std::string Measurement::DataPoint::get_name() { return name_; }

Measurement::Unit::u Measurement::DataPoint::get_unit() { return unit_; }

bool Measurement::DataPoint::is_iterative() { return iterative_; }

int Measurement::DataPoint::get_iteration() { return iteration_; }

uint64_t Measurement::DataPoint::get_value() {
  uint64_t value = 0;

  if (has_event_ && not events_.empty()) {
      for (Event const& e : events_) {
          e.wait();
          value += e.duration<std::chrono::nanoseconds>().count();
      }

  }
  else if (has_event_ && not cl_events_.empty()) {
      cl_ulong start, end;
      for (cl::Event const& e: cl_events_) {
          e.wait();
          e.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
          e.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
          value += end - start;
      }
  }
  else {
      for (uint64_t const& v : values_) {
          value += v;
      }
  }

  return value;
}

Measurement::Measurement::Measurement() {
    run_date_ = std::chrono::system_clock::now();
    set_parameter(ParameterType::TimeStamp, get_datetime());
}
Measurement::Measurement::~Measurement() {}

void Measurement::Measurement::set_parameter(
        ParameterType::t type,
        std::string value
        ) {
  parameters_[type] = value;
}

void Measurement::Measurement::write_csv(std::string filename) {
  std::string experiment_id = get_unique_id();

  {
    std::string experiment_file =
        format_filename(filename, experiment_id, experiment_file_suffix);

    std::ofstream pf(experiment_file, std::ios_base::out | std::ios::trunc);

    pf << "ExperimentID";
    pf << ',';
    pf << "ParameterName";
    pf << ',';
    pf << "Value";

    pf << "\n";

    for (auto const& p : parameters_) {
        pf << experiment_id;
        pf << ',';
        pf << get_parameter_type_name(p.first);
        pf << ',';
        pf << p.second;
        pf << '\n';
    }

    pf.close();
    pf.clear();
  }

  {
    std::string measurements_file =
        format_filename(filename, experiment_id, measurements_file_suffix);

    std::ofstream mf(measurements_file, std::ios_base::out | std::ios::trunc);

    mf << "ExperimentID";
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
        mf << dp.get_name();
        mf << ',';
        if (dp.is_iterative() == true) {
            mf << dp.get_iteration();
        }
        mf << ',';
        mf << dp.get_value();
        mf << ',';
        mf << get_unit_name(dp.get_unit());
        mf << '\n';
    }

    mf.close();
    mf.clear();
  }
}

int Measurement::Measurement::get_num_parameter_types() {
  return Mapping::parameter_name.size();
}

int Measurement::Measurement::get_parameter_type_id(ParameterType::t type) {
  return type;
}

std::string Measurement::Measurement::get_unit_name(Unit::u unit) {
    std::string unit_name = Mapping::unit_name[unit];
    return unit_name;
}

std::string Measurement::Measurement::get_parameter_type_name(ParameterType::t type) {
  std::string type_name = Mapping::parameter_name[type];
  return type_name;
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

