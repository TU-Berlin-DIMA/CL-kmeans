/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016-2018, Lutz, Clemens <lutzcle@cml.li>"
 */

#ifndef OPENCL_SETUP_HPP
#define OPENCL_SETUP_HPP

#include <measurement/measurement.hpp>
#include <gtest/gtest.h>

#include <unistd.h>
#include <cstdint>
#include <string>

#include <boost/compute/core.hpp>

#include <Version.h>

uint32_t const max_hostname_length = 30;

class CLEnvironment : public ::testing::Environment {
public:
    virtual ~CLEnvironment() {}

    virtual void SetUp() {
        context = boost::compute::system::default_context();
        queue = boost::compute::system::default_queue();
        device = boost::compute::system::default_device();
    }

    boost::compute::context context;
    boost::compute::command_queue queue;
    boost::compute::device device;
};

CLEnvironment *clenv;

void measurement_setup(
        Measurement::Measurement& m,
        boost::compute::device device,
        int num_iterations) {

    char hostname[max_hostname_length];
    gethostname(hostname, max_hostname_length);

    m.set_parameter(
            "Device",
            device.name() // remove trailing '\0'
            );

    m.set_parameter(
            "Version",
            GIT_REVISION
            );
    m.set_parameter(
            "Hostname",
            hostname
            );

    m.set_parameter(
            "NumIterations",
            std::to_string(num_iterations)
            );
}

#endif /* OPENCL_SETUP_HPP */
