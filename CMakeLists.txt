CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
SET( CMAKE_EXPORT_COMPILE_COMMANDS 1 )
PROJECT(cljoin_operators)

SET(CMAKE_C_COMPILER clang)
SET(CMAKE_CXX_COMPILER clang++)
SET(CMAKE_C_FLAGS "-std=c99 -Wall -DUNIX")
SET(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG")
SET(CMAKE_CXX_FLAGS "-std=c++11 -Wall -DUNIX")
SET(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")

# Set OpenCL Include and Library paths manually; fix to use find_package() instead
FIND_PATH(OPENCL_INCLUDE_DIR CL/cl.h PATHS "/home/lutzcle/phd/opencl/include")
FIND_PATH(OPENCL_INCLUDE_DIR CL/cl.hpp PATHS "/home/lutzcle/phd/opencl/include")
FIND_LIBRARY(OPENCL_LIBRARY OpenCL PATHS "$HOME/phd/opencl/lib:$HOME/phd/opencl/opt/intel/opencl-1.2-5.0.0.57/lib64")
# FIND_PACKAGE(OpenCL REQUIRED)
INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIR})

ADD_EXECUTABLE(join join.c)
TARGET_LINK_LIBRARIES(join ${OPENCL_LIBRARY})

ADD_EXECUTABLE(test_prefix_sum test_prefix_sum.c)
TARGET_LINK_LIBRARIES(test_prefix_sum ${OPENCL_LIBRARY})

ADD_EXECUTABLE(test_prefix_sum_cpp test_prefix_sum.cpp)
TARGET_LINK_LIBRARIES(test_prefix_sum_cpp ${OPENCL_LIBRARY})
