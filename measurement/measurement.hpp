/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#ifndef MEASUREMENT_HPP
#define MEASUREMENT_HPP

#include "type_definition.hpp"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <string>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace Measurement {

class Measurement;

class DataPoint {
friend class Measurement;

public:
  inline cl::Event &add_opencl_event() {
    has_event_ = true;
    return event_;
  }

  inline uint64_t &add_value() {
      return value_;
  }

private:
  inline DataPoint(DataPointType::t type)
      :
          type_(type),
          iterative_(false),
          has_event_(false)
    {}

  inline DataPoint(DataPointType::t type, int iteration)
      :
          type_(type),
          iterative_(true),
          iteration_(iteration),
          has_event_(false)
    {}

  bool is_iterative();
  int get_iteration();
  uint64_t get_value();
  DataPointType::t get_type();

  DataPointType::t type_;
  bool iterative_;
  int iteration_;
  bool has_event_;
  cl::Event event_;
  uint64_t value_;
};

class Measurement {
public:
  void start();
  void end();

  void set_parameter(ParameterType::t, std::string value);

  inline DataPoint &add_datapoint(DataPointType::t type) {
    data_points_.push_back(DataPoint(type));
    return data_points_.back();
  }

  inline DataPoint &add_datapoint(DataPointType::t type, int iteration) {
    data_points_.push_back(DataPoint(type, iteration));
    return data_points_.back();
  }

  void write_csv(std::string filename);

private:
  int get_num_datapoint_types();
  int get_num_parameter_types();
  int get_datapoint_type_id(DataPointType::t type);
  int get_parameter_type_id(ParameterType::t type);

  std::string get_datapoint_type_name(DataPointType::t type);
  std::string get_parameter_type_name(ParameterType::t type);
  std::string get_datapoint_type_unit(DataPointType::t type);
  bool exists_parameter(ParameterType::t type);
  std::string get_parameter_value(ParameterType::t type);

  std::string get_unique_id();
  std::string get_datetime();

  std::string format_filename(std::string basefile, std::string experiment_id, std::string suffix);

  bool is_started_;
  std::deque<DataPoint> data_points_;
  std::map<ParameterType::t, std::string> parameters_;
  std::chrono::system_clock::time_point run_date_;
};
}

#endif /* MEASUREMENT_HPP */
