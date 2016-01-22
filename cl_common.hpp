#ifndef CL_COMMON_HPP_
#define CL_COMMON_HPP_

#include <vector>
#include <string>

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
    cl_int show_platforms(std::vector<cl::Platform> const& platforms);
    cl::Program make_program(cl::Context context, std::string file, cl_int& error_code);
    void sanitize_make_kernel(cl_int error_code, cl::Context const& context, cl::Program const& program);


}


#endif /* CL_COMMON_HPP_ */
