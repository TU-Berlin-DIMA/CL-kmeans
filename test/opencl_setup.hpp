#ifndef OPENCL_SETUP_HPP
#define OPENCL_SETUP_HPP

#include <clext.hpp>

#include <measurement/measurement.hpp>

#include <unistd.h>
#include <cstdint>
#include <string>

#include <Version.h>

uint32_t const max_hostname_length = 30;

struct CLSetup : cle::CLInitializer {

    CLSetup() {
        this->init(0, 0);
        context = this->get_context();
        queue = this->get_commandqueue();
        queue.getInfo(CL_QUEUE_DEVICE, &device);
    }

    ~CLSetup() {
    }

    cl::Context context;
    cl::CommandQueue queue;
    cl::Device device;
};

BOOST_FIXTURE_TEST_SUITE(ClusteringTest, CLSetup)

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
