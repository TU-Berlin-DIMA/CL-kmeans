#include "cl_string.hpp"

#include <sstream>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif


std::string cle::opencl_device_string(cl_device_type device_type) {
    std::stringstream ss;

    while (device_type != 0) {
        switch(device_type) {
            case CL_DEVICE_TYPE_CPU:
                ss << "CPU";
                device_type ^= CL_DEVICE_TYPE_CPU;
                break;
            case CL_DEVICE_TYPE_GPU:
                ss << "GPU";
                device_type ^= CL_DEVICE_TYPE_GPU;
                break;
            case CL_DEVICE_TYPE_ACCELERATOR:
                ss << "ACCELERATOR";
                device_type ^= CL_DEVICE_TYPE_ACCELERATOR;
                break;
            case CL_DEVICE_TYPE_DEFAULT:
                ss << "DEFAULT";
                device_type ^= CL_DEVICE_TYPE_DEFAULT;
                break;
            case CL_DEVICE_TYPE_CUSTOM:
                ss << "CUSTOM";
                device_type ^= CL_DEVICE_TYPE_CUSTOM;
                break;
        }

        if (device_type != 0) {
            ss << '|';
        }
    }

    return ss.str();
}

std::string cle::opencl_error_string(cl_int error_code) {
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


