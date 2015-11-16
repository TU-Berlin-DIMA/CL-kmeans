
#ifndef HELPERS_HPP_
#define HELPERS_HPP_

#include <string>
#include <sstream>
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


// The syntax ({ x; }) is called a "statement expression"
// Statement expressions are supported by GCC and Clang
// More information:
// https://gcc.gnu.org/onlinedocs/gcc/Statement-Exprs.html
// http://stackoverflow.com/q/6440021

// Sanitize OpenCL errors returned by value
//
// Input
//      F: expression (without semicolon)
//
// Return value
//      the success or error value as cl_int
//
// Usage example
//      cl_int err; err = cle_sanitize_val( f() );
#define cle_sanitize_val(F)                             \
    ({                                                  \
        cl_int ERROR_CODE = F;                          \
        if (ERROR_CODE != CL_SUCCESS) {                 \
            std::cerr << "Failed "                      \
                << #F                                   \
                << " with: "                            \
                << cle::opencl_error_string(ERROR_CODE) \
                << std::endl;                           \
        }                                               \
        ERROR_CODE;                                     \
    })

// Sanitize OpenCL errors returned by value
// Calls "return" with error value as parameter if an error occurs
//
// Input
//      F: exprssion (without semicolon)
//
// Return value:
//      none    
//
// Usage example
//      cl_int g() { cle_sanitize_val_return( f() ); }
#define cle_sanitize_val_return(F)                      \
    do {                                                \
        cl_int error_code = CL_SUCCESS;                 \
        error_code = F;                                 \
        if (error_code != CL_SUCCESS) {                 \
            std::cerr << "Failed "                      \
                << #F                                   \
                << " with: "                            \
                << cle::opencl_error_string(error_code) \
                << std::endl;                           \
            return error_code;                          \
        }                                               \
    } while(0)

// Sanitize OpenCL errors returned by reference
//
// Input
//      F: expression (without semicolon)
//      ERROR_CODE: error variable passed by reference to F
//
// Return value:
//      the success or error value as cl_int
//
// Usage example
//      cl_int err; cle_sanitize_ref( f(&err), err);
#define cle_sanitize_ref(F, ERROR_CODE)                 \
    ({                                                  \
        ERROR_CODE = CL_SUCCESS;                        \
        F;                                              \
        if (ERROR_CODE != CL_SUCCESS) {                 \
            std::cerr << "Failed "                      \
                << #F                                   \
                << " with: "                            \
                << cle::opencl_error_string(ERROR_CODE) \
                << std::endl;                           \
        }                                               \
        ERROR_CODE;                                     \
    })

// Sanitize OpenCL errors returned by reference
// Calls "return" with error value as parameter if an error occurs
//
// Input
//      F: expression (without semicolon)
//      ERROR_CODE: error variable passed by reference to F
//
// Return value:
//      none
//
// Usage example
//      cl_int g() { cle_sanitize_ref_return( f(&err), err); }
#define cle_sanitize_ref_return(F, ERROR_CODE)          \
    do {                                                \
        ERROR_CODE = CL_SUCCESS;                        \
        F;                                              \
        if (ERROR_CODE != CL_SUCCESS) {                 \
            std::cerr << "Failed "                      \
                << #F                                   \
                << " with: "                            \
                << cle::opencl_error_string(ERROR_CODE) \
                << std::endl;                           \
            return ERROR_CODE;                          \
        }                                               \
    } while(0)


namespace cle {

    std::string opencl_device_string(cl_device_type device_type);
    std::string opencl_error_string(cl_int error_code);

    template <typename T>
    class TypedBuffer : public cl::Buffer {
    public:

        typedef T value_type;
        
        TypedBuffer(
                const cl::Context& context,
                cl_mem_flags flags,
                ::size_t size,
                T* host_ptr = NULL,
                cl_int* err = NULL
                )
            : cl::Buffer(context, flags, size * sizeof(T), host_ptr, err)
            , size_(size)
        {}

        TypedBuffer(
                cl_mem_flags flags,
                ::size_t size,
                T* host_ptr = NULL,
                cl_int* err = NULL
                )
            : cl::Buffer(flags, size * sizeof(T), host_ptr, err)
            , size_(size)
        {}

        // template<typename IteratorType>
        // TypedBuffer(
        //         IteratorType startIterator,
        //         IteratorType endIterator,
        //         bool readOnly,
        //         bool useHostPtr = false,
        //         cl_int* err = NULL)
        //     : cl::Buffer(startIterator, endIterator, readOnly, useHostPtr, err)
        // , size_((endIterator - startIterator) / sizeof(T))
        // {}

        // template<typename IteratorType>
        // TypedBuffer(
        //         const cl::Context& context,
        //         IteratorType startIterator,
        //         IteratorType endIterator,
        //         bool readOnly,
        //         bool useHostPtr = false,
        //         cl_int* err = NULL
        //         )
        //     : cl::Buffer(context, startIterator, endIterator, readOnly, useHostPtr, err)
        //     , size_((endIterator - startIterator) / sizeof(T))
        // {}

        // template<typename IteratorType>
        // TypedBuffer(
        //         cl::CommandQueue const& queue,
        //         IteratorType startIterator,
        //         IteratorType endIterator,
        //         bool readOnly,
        //         bool useHostPtr = false,
        //         cl_int* err = NULL
        //         )
        //     : cl::Buffer(queue, startIterator, endIterator, readOnly, useHostPtr, err)
        //     , size_((endIterator - startIterator) / sizeof(T))
        // {}

        TypedBuffer() : cl::Buffer(), size_(0) {}

        // __CL_EXPLICIT_CONSTRUCTORS TypedBuffer(cl_mem const& buffer)
        //     : cl::Buffer(buffer)
        //     , size_(0)
        // {}


        // Shallow copy
        // TypedBuffer& operator = (cl_mem const& rhs) {
        //     size_ = 0; // TODO: fetch size from cl::Buffer::getInfo()
        //     cl::Buffer::operator=(rhs);
        //     return *this;
        // }

        // TypedBuffer(cl::Buffer const& buf)
        //     : cl::Buffer(buf)
        //     , size_(0)
        // {}

        // TypedBuffer<T>& operator = (cl::Buffer const& rhs) {
        //     size_ = 0; // TODO: fetch size from cl::Buffer::getInfo()
        //     cl::Buffer::operator=(rhs);
        //     return *this;
        // }


        TypedBuffer<T>& operator = (TypedBuffer<T> const& rhs) {
            size_ = rhs.size();
            cl::Buffer::operator=(rhs);
            return *this;
        }

        TypedBuffer(TypedBuffer<T> const& buf)
            : cl::Buffer(buf)
            , size_(buf.size())
        {}

#if defined(CL_HPP_RVALUE_REFERENCE_SUPPORTED)
        // TypedBuffer(cl::Buffer&& buf) : cl::Buffer(std::move(buf)) {}
        //
        // TypedBuffer(TypedBuffer<T>&& buf) : cl::Buffer(std::move(buf)) {}
        //
        // TypedBuffer& operator = (cl::Buffer &&rhs) {
        //     size_ = 0; // TODO
        //     cl::Buffer::operator=(std::move(rhs));
        //     return *this;
        // }
        //
        // TypedBuffer& operator = (TypedBuffer &&rhs) {
        //     size_ = std::move(rhs.size_);
        //     cl::Buffer::operator=(std::move(rhs));
        //     return *this;
        // }
#endif

        ::size_t size() {
            return size_;
        }


    private:
        // Size in number of "T" elements 
        ::size_t size_;
    };

    cl::Program make_program(cl::Context context, std::string file, cl_int& error_code) {
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

    cl_int show_platforms(std::vector<cl::Platform> const& platforms) {
        cl_int err = CL_SUCCESS;
        std::string platform_name, device_name, device_version;
        cl_device_type device_type;
        std::vector<cl::Device> devices_list;

        std::cout << "Platforms:" << std::endl;
        for (auto platform : platforms) {
            cle_sanitize_val_return(
                    platform.getInfo(CL_PLATFORM_NAME, &platform_name));

            cle_sanitize_val_return(
                    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices_list));

            std::cout << "  - " << platform_name << std::endl;
            for (auto device : devices_list) {
                cle_sanitize_val_return(
                        device.getInfo(CL_DEVICE_TYPE, &device_type));
                cle_sanitize_val_return(
                        device.getInfo(CL_DEVICE_NAME, &device_name));
                cle_sanitize_val_return(
                        device.getInfo(CL_DEVICE_VERSION, &device_version));

                std::cout << "    + "
                    << cle::opencl_device_string(device_type)
                    << " Device: "
                    << " "
                    << device_name
                    << ", "
                    << device_version
                    << std::endl;
            }
        }

        return err;
    }

    std::string opencl_device_string(cl_device_type device_type) {
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
