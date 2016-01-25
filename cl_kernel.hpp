#ifndef CL_KERNEL_HPP
#define CL_KERNEL_HPP

#include <vector>
#include <string>

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif


// Append path prefix defined in CMakeLists to OpenCL kernel file name
#define CL_KERNEL_FILE_PATH(FILE_NAME) CL_KERNELS_PATH "/" FILE_NAME

namespace cle {

    cl_int show_platforms(std::vector<cl::Platform> const& platforms);
    cl::Program make_program(cl::Context context, std::string file, cl_int& error_code);
    void sanitize_make_kernel(cl_int error_code, cl::Context const& context, cl::Program const& program);

}

#endif /* CL_KERNEL_HPP */
