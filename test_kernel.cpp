/*
 * This Source Code Form is subject to the terms of the Mozilla Public License,
 * v. 2.0. If a copy of the MPL was not distributed with this file, You can
 * obtain one at http://mozilla.org/MPL/2.0/.
 * 
 * 
 * Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>
 */

#include <iostream>

#include "cl_kernels/reduce_vector_parcol_api.hpp"
#include "matrix.hpp"

#include <clext.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <memory> // allocator
#include <vector>
#include <algorithm> // equal
#include <cstdint>
#include <random>

#ifndef TEST_KERNEL_NAME
#define TEST_KERNEL_NAME ""
#endif

namespace po = boost::program_options;

class CmdOptions {
public:
    enum class Mode {GpuMemBandwidth, PciBandwidth};

    int parse(int argc, char **argv) {
        char help_msg[] =
            "Usage: " TEST_KERNEL_NAME " [OPTION]\n"
            "Options"
            ;

        po::options_description cmdline(help_msg);
        cmdline.add_options()
            ("help", "Produce help message")
            ("platform",
             po::value<unsigned int>(&platform_)->default_value(0),
             "OpenCL platform number")
            ("device",
             po::value<unsigned int>(&device_)->default_value(0),
             "OpenCL device number")
            ("gpumembw", "GPU Memory Bandwidth")
            ("pcibw", "PCI Bandwidth")
            ("buffersize",
             po::value<size_t>(&buffer_size_)->default_value(256),
             "Buffer size in MiB")
            ;

        po::variables_map vm;
        po::store(
                po::command_line_parser(argc, argv).options(cmdline).run(),
                vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << cmdline <<std::endl;
            return -1;
        }

        if (vm.count("platform")) {
            platform_ = vm["platform"].as<unsigned int>();
        }

        if (vm.count("device")) {
            device_ = vm["device"].as<unsigned int>();
        }

        if (vm.count("gpumembw")) {
            mode_ = Mode::GpuMemBandwidth;
        }

        if (vm.count("pcibw")) {
            mode_ = Mode::PciBandwidth;
        }

        if (vm.count("buffersize")) {
            buffer_size_ = vm["buffersize"].as<size_t>();
        }

        return 1;
    }

    Mode get_mode() const {
        return mode_;
    }

    unsigned int cl_platform() const {
        return platform_;
    }

    unsigned int cl_device() const {
        return device_;
    }

    size_t buffer_bytes() const {
        return buffer_size_ * 1024 * 1024;
    }

private:
    Mode mode_;
    unsigned int platform_;
    unsigned int device_;
    size_t buffer_size_;
};

class ReduceVectorParcolTest {
public:
    bool operator() (
            cle::Matrix<uint32_t, std::allocator<uint32_t>, uint32_t> const& data,
            std::vector<uint32_t> const& reduced
            ) {

        std::vector<uint32_t> reduced_verify(data.rows(), 0);
        for (uint32_t col = 0; col < data.cols(); ++col) {
            for (uint32_t row = 0; row < data.rows(); ++row) {
                reduced_verify[row] += data(row, col);
            }
        }

        return (reduced.size() == reduced_verify.size()) &&
            std::equal(
                reduced_verify.begin(), reduced_verify.end(),
                reduced.begin());
    }
};

int main(int argc, char **argv) {

    int ret = 0;

    CmdOptions options;

    ret = options.parse(argc, argv);
    if (ret < 0) {
        return 1;
    }

    cle::CLInitializer initializer;
    if (initializer.init(options.cl_platform(), options.cl_device()) < 0) {
        return 1;
    }
    cle_sanitize_done_return(
            initializer.print_device_info()
            );

    switch (options.get_mode()) {
        case CmdOptions::Mode::GpuMemBandwidth:

            break;
        case CmdOptions::Mode::PciBandwidth:

            break;
    }

    uint32_t num_rows = 64;
    uint32_t num_cols = options.buffer_bytes() / sizeof(uint32_t) / num_rows;

    cle::Matrix<uint32_t, std::allocator<uint32_t>, uint32_t> data;
    data.resize(num_rows, num_cols);

    std::default_random_engine rgen;
    std::uniform_int_distribution<uint32_t> uniform;
    std::generate(data.begin(), data.end(), [&](){return uniform(rgen);});
    // std::fill(data.begin(), data.end(), 1 << 31);
    // std::generate(data.begin(), data.end(), [](){static uint32_t x = 0; return x++;});

    cl::Context context_ = initializer.get_context();
    cl::CommandQueue queue_ = initializer.get_commandqueue();

    std::vector<uint32_t> res_buffer(num_rows);
    cle::TypedBuffer<cl_uint> d_buffer(context_, CL_MEM_READ_WRITE, data.size());

    cle::ReduceVectorParcolAPI<cl_uint, cl_uint> reducevector;
    cle_sanitize_done_return(
            reducevector.initialize(context_)
            );

    cle_sanitize_val_return(
            queue_.enqueueWriteBuffer(
                d_buffer,
                CL_FALSE,
                0,
                d_buffer.bytes(),
                data.data(),
                NULL,
                NULL));

    cl::Event event;
    cle_sanitize_done_return(
            reducevector(
                cl::EnqueueArgs(
                    queue_,
                    cl::NDRange(0),
                    cl::NDRange(0)
                    ),
                num_cols,
                num_rows,
                d_buffer,
                event));

    cle_sanitize_val_return(
            queue_.enqueueReadBuffer(
                d_buffer,
                CL_TRUE,
                0,
                res_buffer.size() * sizeof(uint32_t),
                res_buffer.data(),
                NULL,
                NULL));

    ReduceVectorParcolTest reducetest;
    if (reducetest(data, res_buffer)) {
        std::cout << "Passed" << std::endl;
    }
    else {
        std::cout << "Failed" << std::endl;
    }

    return 0;
}
