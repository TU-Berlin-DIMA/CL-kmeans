/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 *
 *
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include "measurement.hpp"

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
  else {
      for (uint64_t const& v : values_) {
          value += v;
      }
  }

  if (not children_.empty()) {
      for (auto& child : children_) {
          value += child.get_value();
      }
  }

  return value;
}

Measurement::Measurement::Measurement() {
    run_date_ = std::chrono::system_clock::now();
    set_parameter("TimeStamp", get_datetime());
}
Measurement::Measurement::~Measurement() {}

void Measurement::Measurement::set_run(int run) { run_ = run; }

void Measurement::Measurement::set_parameter(
        std::string name,
        std::string value
        ) {
  parameters_[name] = value;
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
        pf << p.first;
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
    mf << "Run";
    mf << ',';
    mf << "TypeName";
    mf << ',';
    mf << "Iteration";
    mf << ',';
    mf << "Value";

    mf << '\n';

    for (DataPoint dp : get_flattened_datapoints()) {
        mf << experiment_id;
        mf << ',';
        mf << run_;
        mf << ',';
        mf << dp.get_name();
        mf << ',';
        if (dp.is_iterative() == true) {
            mf << dp.get_iteration();
        }
        mf << ',';
        mf << dp.get_value();
        mf << '\n';
    }

    mf.close();
    mf.clear();
  }
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

std::deque<Measurement::DataPoint>
Measurement::Measurement::get_flattened_datapoints() {
    std::deque<DataPoint> subpoints;

    for (auto& dp : data_points_) {
        if (not dp.children_.empty()) {

            DataPoint cp;
            if (dp.is_iterative()) {
                cp = DataPoint(dp.get_iteration());
            }
            cp.set_name(dp.get_name() + "#Sub");

            uint64_t child_values = 0;
            for (auto& child : dp.children_) {
                child_values += child.get_value();
            }

            cp.add_value() = child_values;
            subpoints.push_back(cp);
        }

        subpoints.push_back(dp);
    }

    return subpoints;
}
