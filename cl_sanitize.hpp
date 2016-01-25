#ifndef CL_SANITIZE_HPP
#define CL_SANITIZE_HPP

#include <iostream>

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


#endif /* CL_SANITIZE_HPP */
