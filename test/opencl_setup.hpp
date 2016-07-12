#ifndef OPENCL_SETUP_HPP
#define OPENCL_SETUP_HPP

#include <clext.hpp>

#include <measurement/measurement.hpp>
#include <gtest/gtest.h>

#include <unistd.h>
#include <cstdint>
#include <string>

#include <Version.h>

uint32_t const max_hostname_length = 30;

class CLEnvironment : public ::testing::Environment {
public:
    virtual ~CLEnvironment() {}

    virtual void SetUp() {
        clinit.init(0, 0);
        context = clinit.get_context();
        queue = clinit.get_commandqueue();
        queue.getInfo(CL_QUEUE_DEVICE, &device);
    }

    cl::Context context;
    cl::CommandQueue queue;
    cl::Device device;

private:
    cle::CLInitializer clinit;

};

CLEnvironment *clenv;

void measurement_setup(
        Measurement::Measurement& m,
        cl::Device device,
        int num_iterations) {

    char hostname[max_hostname_length];
    gethostname(hostname, max_hostname_length);

    std::string device_name;
    device.getInfo(CL_DEVICE_NAME, &device_name);

    m.set_parameter(
            Measurement::ParameterType::Device,
            device_name.c_str() // remove trailing '\0'
            );

    m.set_parameter(
            Measurement::ParameterType::Version,
            GIT_REVISION
            );
    m.set_parameter(
            Measurement::ParameterType::Hostname,
            hostname
            );

    m.set_parameter(
            Measurement::ParameterType::NumIterations,
            std::to_string(num_iterations)
            );
}

#endif /* OPENCL_SETUP_HPP */
