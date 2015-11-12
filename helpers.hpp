
#ifndef HELPERS_HPP_
#define HELPERS_HPP_

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

// Append path prefix defined in CMakeLists to OpenCL kernel file name
#define CL_KERNEL_FILE_PATH(FILE_NAME) CL_KERNELS_PATH "/" FILE_NAME

namespace cle {

    std::string opencl_error_string(cl_int error_code);

    cl::Program make_program(cl::Context context, std::string file, cl_int& error_code) {
        cl::Program program;
        
        // Open OpenCL source file
        std::ifstream sourceFile(file);
        if (not sourceFile.good()) {
            std::cerr << "Failed to open program file " << file << "; ";

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
        program = cl::Program(context, source, &error_code);
        if (error_code != CL_SUCCESS) {
            std::cerr << "Program constructor failed with "
                << opencl_error_string(error_code)
                << std::endl;

            return program;
        }

        std::vector<cl::Device> context_devices;
        context.getInfo(CL_CONTEXT_DEVICES, &context_devices);

        error_code = program.build(context_devices);
        if (error_code != CL_SUCCESS) {
            std::cerr << "Building program failed with "
                << opencl_error_string(error_code)
                << std::endl;
        }

        return program;
    }

    void sanitize_make_kernel(cl_int error_code, cl::Context const& context, cl::Program const& program) {
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

                    context.getInfo(CL_CONTEXT_DEVICES, &context_devices);
                    for (cl::Device device : context_devices) {
                        program.getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &error_help_string);
                        std::cerr << error_help_string << std::endl;
                    }

                    break;

                case CL_INVALID_KERNEL_NAME:
                    std::cerr << "  Hint: Kernel with this name wasn't found in program" << std::endl;

                    program.getInfo(CL_PROGRAM_NUM_KERNELS, &error_help_size);
                    program.getInfo(CL_PROGRAM_KERNEL_NAMES, &error_help_string);
                    std::cerr << "  Program knows of " << error_help_size << " kernels: " << error_help_string << std::endl;

                    break;

                case CL_INVALID_KERNEL_DEFINITION:
                    std::cerr << "  Hint: __kernel function signature doesn't match function call. Are parameters wrong?" << std::endl;
                    break;

                case CL_INVALID_VALUE:
                    std::cerr << "  Hint: Kernel name is NULL" << std::endl;
                    break;

                case CL_OUT_OF_RESOURCES:
                    std::cerr << "  Hint: Device resource allocation failed" << std::endl;
                    break;
            }
        }
    }

    std::string opencl_error_string(cl_int error_code) {
        std::string s;

        switch(error_code) {
            case CL_SUCCESS:
                s="CL_SUCCESS";
                break;
            case CL_DEVICE_NOT_FOUND:
                s="CL_DEVICE_NOT_FOUND";
                break;
            case CL_DEVICE_NOT_AVAILABLE:
                s="CL_DEVICE_NOT_AVAILABLE";
                break;
            case CL_COMPILER_NOT_AVAILABLE:
                s="CL_COMPILER_NOT_AVAILABLE";
                break;
            case CL_MEM_OBJECT_ALLOCATION_FAILURE:
                s="CL_MEM_OBJECT_ALLOCATION_FAILURE";
                break;
            case CL_OUT_OF_RESOURCES:
                s="CL_OUT_OF_RESOURCES";
                break;
            case CL_OUT_OF_HOST_MEMORY:
                s="CL_OUT_OF_HOST_MEMORY";
                break;
            case CL_PROFILING_INFO_NOT_AVAILABLE:
                s="CL_PROFILING_INFO_NOT_AVAILABLE";
                break;
            case CL_MEM_COPY_OVERLAP:
                s="CL_MEM_COPY_OVERLAP";
                break;
            case CL_IMAGE_FORMAT_MISMATCH:
                s="CL_IMAGE_FORMAT_MISMATCH";
                break;
            case CL_IMAGE_FORMAT_NOT_SUPPORTED:
                s="CL_IMAGE_FORMAT_NOT_SUPPORTED";
                break;
            case CL_BUILD_PROGRAM_FAILURE:
                s="CL_BUILD_PROGRAM_FAILURE";
                break;
            case CL_MAP_FAILURE:
                s="CL_MAP_FAILURE";
                break;
            case CL_MISALIGNED_SUB_BUFFER_OFFSET:
                s="CL_MISALIGNED_SUB_BUFFER_OFFSET";
                break;
            case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
                s="CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
                break;
            case CL_COMPILE_PROGRAM_FAILURE:
                s="CL_COMPILE_PROGRAM_FAILURE";
                break;
            case CL_LINKER_NOT_AVAILABLE:
                s="CL_LINKER_NOT_AVAILABLE";
                break;
            case CL_LINK_PROGRAM_FAILURE:
                s="CL_LINK_PROGRAM_FAILURE";
                break;
            case CL_DEVICE_PARTITION_FAILED:
                s="CL_DEVICE_PARTITION_FAILED";
                break;
            case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:
                s="CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
                break;
            case CL_INVALID_VALUE:
                s="CL_INVALID_VALUE";
                break;
            case CL_INVALID_DEVICE_TYPE:
                s="CL_INVALID_DEVICE_TYPE";
                break;
            case CL_INVALID_PLATFORM:
                s="CL_INVALID_PLATFORM";
                break;
            case CL_INVALID_DEVICE:
                s="CL_INVALID_DEVICE";
                break;
            case CL_INVALID_CONTEXT:
                s="CL_INVALID_CONTEXT";
                break;
            case CL_INVALID_QUEUE_PROPERTIES:
                s="CL_INVALID_QUEUE_PROPERTIES";
                break;
            case CL_INVALID_COMMAND_QUEUE:
                s="CL_INVALID_COMMAND_QUEUE";
                break;
            case CL_INVALID_HOST_PTR:
                s="CL_INVALID_HOST_PTR";
                break;
            case CL_INVALID_MEM_OBJECT:
                s="CL_INVALID_MEM_OBJECT";
                break;
            case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
                s="CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
                break;
            case CL_INVALID_IMAGE_SIZE:
                s="CL_INVALID_IMAGE_SIZE";
                break;
            case CL_INVALID_SAMPLER:
                s="CL_INVALID_SAMPLER";
                break;
            case CL_INVALID_BINARY:
                s="CL_INVALID_BINARY";
                break;
            case CL_INVALID_BUILD_OPTIONS:
                s="CL_INVALID_BUILD_OPTIONS";
                break;
            case CL_INVALID_PROGRAM:
                s="CL_INVALID_PROGRAM";
                break;
            case CL_INVALID_PROGRAM_EXECUTABLE:
                s="CL_INVALID_PROGRAM_EXECUTABLE";
                break;
            case CL_INVALID_KERNEL_NAME:
                s="CL_INVALID_KERNEL_NAME";
                break;
            case CL_INVALID_KERNEL_DEFINITION:
                s="CL_INVALID_KERNEL_DEFINITION";
                break;
            case CL_INVALID_KERNEL:
                s="CL_INVALID_KERNEL";
                break;
            case CL_INVALID_ARG_INDEX:
                s="CL_INVALID_ARG_INDEX";
                break;
            case CL_INVALID_ARG_VALUE:
                s="CL_INVALID_ARG_VALUE";
                break;
            case CL_INVALID_ARG_SIZE:
                s="CL_INVALID_ARG_SIZE";
                break;
            case CL_INVALID_KERNEL_ARGS:
                s="CL_INVALID_KERNEL_ARGS";
                break;
            case CL_INVALID_WORK_DIMENSION:
                s="CL_INVALID_WORK_DIMENSION";
                break;
            case CL_INVALID_WORK_GROUP_SIZE:
                s="CL_INVALID_WORK_GROUP_SIZE";
                break;
            case CL_INVALID_WORK_ITEM_SIZE:
                s="CL_INVALID_WORK_ITEM_SIZE";
                break;
            case CL_INVALID_GLOBAL_OFFSET:
                s="CL_INVALID_GLOBAL_OFFSET";
                break;
            case CL_INVALID_EVENT_WAIT_LIST:
                s="CL_INVALID_EVENT_WAIT_LIST";
                break;
            case CL_INVALID_EVENT:
                s="CL_INVALID_EVENT";
                break;
            case CL_INVALID_OPERATION:
                s="CL_INVALID_OPERATION";
                break;
            case CL_INVALID_GL_OBJECT:
                s="CL_INVALID_GL_OBJECT";
                break;
            case CL_INVALID_BUFFER_SIZE:
                s="CL_INVALID_BUFFER_SIZE";
                break;
            case CL_INVALID_MIP_LEVEL:
                s="CL_INVALID_MIP_LEVEL";
                break;
            case CL_INVALID_GLOBAL_WORK_SIZE:
                s="CL_INVALID_GLOBAL_WORK_SIZE";
                break;
            case CL_INVALID_PROPERTY:
                s="CL_INVALID_PROPERTY";
                break;
            case CL_INVALID_IMAGE_DESCRIPTOR:
                s="CL_INVALID_IMAGE_DESCRIPTOR";
                break;
            case CL_INVALID_COMPILER_OPTIONS:
                s="CL_INVALID_COMPILER_OPTIONS";
                break;
            case CL_INVALID_LINKER_OPTIONS:
                s="CL_INVALID_LINKER_OPTIONS";
                break;
            case CL_INVALID_DEVICE_PARTITION_COUNT:
                s="CL_INVALID_DEVICE_PARTITION_COUNT";
                break;
        }

        return s;
    }
}


#endif /* HELPERS_HPP_ */
