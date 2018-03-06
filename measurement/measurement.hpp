/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifndef MEASUREMENT_HPP
#define MEASUREMENT_HPP

#include <cstdint>
#include <deque>
#include <map>
#include <string>

#include <boost/compute/event.hpp>

namespace Measurement {

class Measurement;

class DataPoint {
friend class Measurement;

using Event = boost::compute::event;

public:
    DataPoint &set_name(std::string name) {
        name_ = name;

        return *this;
    }

    inline Event &add_event() {
        has_event_ = true;
        events_.push_back(Event());
        return events_.back();
    }

    inline uint64_t &add_value() {
        values_.push_back(0);
        return values_.back();
    }

    DataPoint& create_child() {
        children_.push_back(DataPoint());
        return children_.back();
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
    bool is_iterative();
    int get_iteration();
    uint64_t get_value();

    std::string name_;
    bool iterative_;
    int iteration_;
    bool has_event_;
    std::deque<Event> events_;
    std::deque<uint64_t> values_;
    std::deque<DataPoint> children_;
};

class Measurement {
public:
  Measurement();
  ~Measurement();

  void set_run(int run);
  void set_parameter(std::string name, std::string value);

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
  std::string get_unique_id();
  std::string get_datetime();

  std::string format_filename(std::string basefile, std::string experiment_id, std::string suffix);
  std::deque<DataPoint> get_flattened_datapoints();

  int run_;
  std::deque<DataPoint> data_points_;
  std::map<std::string, std::string> parameters_;
  std::chrono::system_clock::time_point run_date_;
};
}

#endif /* MEASUREMENT_HPP */
