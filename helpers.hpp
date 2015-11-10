
#ifndef HELPERS_HPP_
#define HELPERS_HPP_

#include <string>
#include <fstream>
#include <utility>
#include <iostream>
#include <vector>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

// Append path prefix defined in CMakeLists to OpenCL kernel file name
#define CL_KERNEL_FILE_PATH(FILE_NAME) CL_KERNELS_PATH "/" FILE_NAME

namespace cle {

    cl::Program make_program(cl::Context context, std::string file) {
        cl_int err = CL_SUCCESS;
        
        std::ifstream sourceFile(file);
        if (not sourceFile.good()) {
            std::cerr << "Failed to open program file " << file << std::endl;
        }
        
        std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));

        cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));

        cl::Program program = cl::Program(context, source, &err);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create program" << std::endl;
        }

        std::vector<cl::Device> context_devices;
        context.getInfo(CL_CONTEXT_DEVICES, &context_devices);

        err = program.build(context_devices);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to build program" << std::endl;
        }

        return program;
    }

    void sanitize_make_kernel(cl_int err, cl::Context const& context, cl::Program const& program) {
        if (err != CL_SUCCESS) {
            std::string error_help_string;
            size_t error_help_size;
            
            std::cerr << "Failed to make kernel with: ";
            switch(err) {
                case CL_INVALID_PROGRAM:
                    std::cerr << "invalid program" << std::endl;
                    break;
                case CL_INVALID_PROGRAM_EXECUTABLE:
                    std::cerr << "invalid program executable" << std::endl;
                    break;
                case CL_INVALID_KERNEL_NAME:
                    std::cerr << "invalid kernel name" << std::endl;
                    std::cerr << "  Hint: Kernel with this name wasn't found in program" << std::endl;

                    program.getInfo(CL_PROGRAM_NUM_KERNELS, &error_help_size);
                    program.getInfo(CL_PROGRAM_KERNEL_NAMES, &error_help_string);
                    std::cerr << "  Program knows of " << error_help_size << " kernels: " << error_help_string << std::endl;

                    break;
                case CL_INVALID_KERNEL_DEFINITION:
                    std::cerr << "invalid kernel definition" << std::endl;
                    std::cerr << "  Hint: __kernel function signature doesn't match function call. Are parameters wrong?" << std::endl;
                    break;
                case CL_INVALID_VALUE:
                    std::cerr << "invalid value" << std::endl;
                    std::cerr << "  Hint: Kernel name is NULL" << std::endl;
                    break;
                case CL_OUT_OF_RESOURCES:
                    std::cerr << "out of resources" << std::endl;
                    std::cerr << "  Hint: Device resource allocation failed" << std::endl;
                    break;
                case CL_OUT_OF_HOST_MEMORY:
                    std::cerr << "out of host memory" << std::endl;
                    break;
            }

            if (err == CL_INVALID_PROGRAM || err == CL_INVALID_PROGRAM_EXECUTABLE) {
                // Print build logs
                std::cout << "Printing program build log(s) for further diagnosis:" << std::endl;

                std::string build_log;
                std::vector<cl::Device> context_devices;

                context.getInfo(CL_CONTEXT_DEVICES, &context_devices);
                for (cl::Device device : context_devices) {
                    program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &build_log);
                    std::cout << build_log << std::endl;
                }
            }
        }
    }
}


#endif /* HELPERS_HPP_ */
