#include "cl_kernel.hpp"

#include "cl_sanitize.hpp"
#include "cl_string.hpp"

#include <string>
#include <fstream>
#include <utility>
#include <iostream>
#include <iterator>
#include <vector>
#include <climits>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif


cl::Program cle::make_program(cl::Context context, std::string file, cl_int& error_code) {
    cl::Program program;
    
    // Open OpenCL source file
    std::ifstream sourceFile(file);
    if (not sourceFile.good()) {
        std::cerr << "Failed to open program file "
            << file
            << "; ";

        // Further diagnosis
        if (sourceFile.eof()) {
            std::cerr << "at EOF";
        }
        else if (sourceFile.fail()) {
            std::cerr << "logical I/O error on read";
        }
        else if(sourceFile.bad()) {
            std::cerr << "read error on I/O";
        }
        std::cerr << std::endl;

        error_code = INT_MIN;
        return program;
    }
    
    // Read OpenCL source file to string buffer
    std::string sourceCode(
            std::istreambuf_iterator<char>(sourceFile),
            (std::istreambuf_iterator<char>())
            );

    cl::Program::Sources source(
            1,
            std::make_pair(sourceCode.c_str(), sourceCode.length()+1)
            );

    // Create program instance with source code
    cle_sanitize_ref(
            program = cl::Program(context, source, &error_code),
            error_code
            );
    if (error_code != CL_SUCCESS) {
        return program;
    }

    std::vector<cl::Device> context_devices;
    error_code = cle_sanitize_val(
            context.getInfo(CL_CONTEXT_DEVICES, &context_devices));

    error_code = cle_sanitize_val(
            program.build(context_devices));

    return program;
}

void cle::sanitize_make_kernel(cl_int error_code, cl::Context const& context, cl::Program const& program) {
    if (error_code != CL_SUCCESS) {
        std::string error_help_string;
        size_t error_help_size;
        std::vector<cl::Device> context_devices;
        
        std::cerr << "Failed to make kernel with: "
            << opencl_error_string(error_code);

        switch(error_code) {
            case CL_INVALID_PROGRAM:
            case CL_INVALID_PROGRAM_EXECUTABLE:
                // Print build logs
                std::cerr << "Printing program build log(s) for further diagnosis:" << std::endl;

                cle_sanitize_val(
                        context.getInfo(CL_CONTEXT_DEVICES, &context_devices));

                for (cl::Device device : context_devices) {
                    cle_sanitize_val(
                            program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &error_help_string));

                    std::cerr << error_help_string << std::endl;
                }

                break;

            case CL_INVALID_KERNEL_NAME:
                cle_sanitize_val(
                        program.getInfo(CL_PROGRAM_NUM_KERNELS, &error_help_size));
                cle_sanitize_val(
                        program.getInfo(CL_PROGRAM_KERNEL_NAMES, &error_help_string));

                std::cerr
                    << "  Hint: Kernel with this name wasn't found in program"
                    << std::endl
                    << "  Program knows of "
                    << error_help_size
                    << " kernels: "
                    << error_help_string
                    << std::endl;

                break;

            case CL_INVALID_KERNEL_DEFINITION:
                std::cerr << "  Hint: __kernel function signature doesn't match function call. Are parameters wrong?"
                    << std::endl;
                break;

            case CL_INVALID_VALUE:
                std::cerr << "  Hint: Kernel name is NULL"
                    << std::endl;
                break;

            case CL_OUT_OF_RESOURCES:
                std::cerr << "  Hint: Device resource allocation failed"
                    << std::endl;
                break;
        }
    }
}

cl_int cle::show_platforms(std::vector<cl::Platform> const& platforms) {
    cl_int err = CL_SUCCESS;
    unsigned int counter = 0;
    std::string platform_name, device_name, device_version;
    cl_device_type device_type;
    std::vector<cl::Device> devices_list;

    std::cout << "Platforms:" << std::endl;
    counter = 0;
    for (auto platform : platforms) {
        cle_sanitize_val_return(
                platform.getInfo(CL_PLATFORM_NAME, &platform_name));

        cle_sanitize_val_return(
                platform.getDevices(CL_DEVICE_TYPE_ALL, &devices_list));

        std::cout << "  (" << counter << ") " << platform_name << std::endl;
        for (auto device : devices_list) {
            cle_sanitize_val_return(
                    device.getInfo(CL_DEVICE_TYPE, &device_type));
            cle_sanitize_val_return(
                    device.getInfo(CL_DEVICE_NAME, &device_name));
            cle_sanitize_val_return(
                    device.getInfo(CL_DEVICE_VERSION, &device_version));

            std::cout << "    - "
                << cle::opencl_device_string(device_type)
                << " Device: "
                << " "
                << device_name
                << ", "
                << device_version
                << std::endl;
        }

        ++counter;
    }

    return err;
}
