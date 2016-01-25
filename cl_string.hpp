#ifndef CL_STRING_HPP
#define CL_STRING_HPP

#include <string>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace cle {

    std::string opencl_device_string(cl_device_type device_type);
    std::string opencl_error_string(cl_int error_code);

}

#endif /* CL_STRING_HPP */
