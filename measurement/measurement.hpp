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
#include "units.hpp"

#include <cstdint>
#include <deque>
#include <map>
#include <string>

#include <boost/compute/event.hpp>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace Measurement {

class Measurement;

class DataPoint {
friend class Measurement;

using Event = boost::compute::event;

public:
    void set_name(std::string name) {
        name_ = name;
    }

    void set_unit(Unit::u unit) {
        unit_ = unit;
    }

    inline Event &add_event() {
        has_event_ = true;
        events_.push_back(Event());
        return events_.back();
    }

    inline cl::Event &add_opencl_event() {
        has_event_ = true;
        cl_events_.push_back(cl::Event());
        return cl_events_.back();
    }

    inline uint64_t &add_value() {
        values_.push_back(0);
        return values_.back();
    }

private:
    inline DataPoint()
        :
            iterative_(false),
            has_event_(false)
    {}

    inline DataPoint(int iteration)
        :
            iterative_(true),
            iteration_(iteration),
            has_event_(false)
    {}

    std::string get_name();
    Unit::u get_unit();
    bool is_iterative();
    int get_iteration();
    uint64_t get_value();

    std::string name_;
    Unit::u unit_;
    bool iterative_;
    int iteration_;
    bool has_event_;
    std::deque<Event> events_;
    std::deque<cl::Event> cl_events_;
    std::deque<uint64_t> values_;
};

class Measurement {
public:
  Measurement();
  ~Measurement();

  void set_parameter(ParameterType::t, std::string value);

  inline DataPoint &add_datapoint() {
    data_points_.push_back(DataPoint());
    return data_points_.back();
  }

  inline DataPoint &add_datapoint(int iteration) {
    data_points_.push_back(DataPoint(iteration));
    return data_points_.back();
  }

  void write_csv(std::string filename);

private:
  int get_num_parameter_types();
  int get_parameter_type_id(ParameterType::t type);

  std::string get_unit_name(Unit::u unit);
  std::string get_parameter_type_name(ParameterType::t type);
  bool exists_parameter(ParameterType::t type);
  std::string get_parameter_value(ParameterType::t type);

  std::string get_unique_id();
  std::string get_datetime();

  std::string format_filename(std::string basefile, std::string experiment_id, std::string suffix);

  std::deque<DataPoint> data_points_;
  std::map<ParameterType::t, std::string> parameters_;
  std::chrono::system_clock::time_point run_date_;
};
}

#endif /* MEASUREMENT_HPP */
