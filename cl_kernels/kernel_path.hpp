#ifndef KERNEL_PATH_HPP
#define KERNEL_PATH_HPP

#include <SystemConfig.h>

// Append path prefix defined in CMakeLists to OpenCL kernel file name
#define CL_KERNEL_FILE_PATH(FILE_NAME) CL_KERNELS_PATH "/" FILE_NAME

#endif /* KERNEL_PATH_HPP */
