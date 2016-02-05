# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.
# 
# 
# Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>

CMAKE_MINIMUM_REQUIRED(VERSION 3.2)
SET(CMAKE_EXPORT_COMPILE_COMMANDS 1) # For vim color_coded plugin
PROJECT(cljoin_operators)

SET(CL_KERNELS_SOURCE_PATH "${PROJECT_SOURCE_DIR}/cl_kernels")
SET(CL_KERNELS_INSTALL_PATH "${CMAKE_PREFIX_PATH}/opt/${PROJECT_NAME}")

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    SET(CL_KERNELS_PATH ${CL_KERNELS_SOURCE_PATH})
ELSEIF(CMAKE_BUILD_TYPE MATCHES Release OR NOT CMAKE_BUILD_TYPE)
    SET(CL_KERNELS_PATH ${CL_KERNELS_INSTALL_PATH})
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

SET(CMAKE_C_COMPILER clang)
SET(CMAKE_CXX_COMPILER clang++)
SET(CMAKE_C_FLAGS "-std=c99 -Wall -Wextra -DUNIX -DCL_KERNELS_PATH=\\\"${CL_KERNELS_PATH}\\\"")
SET(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG")
SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -march=haswell -Wall -Wextra -DUNIX -DCL_KERNELS_PATH=\\\"${CL_KERNELS_PATH}\\\"")
SET(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG -DMATRIX_BOUNDSCHECK")

FIND_PACKAGE(Boost REQUIRED)

FIND_PACKAGE(OpenCL REQUIRED)
INCLUDE_DIRECTORIES(${OpenCL_INCLUDE_DIR})

ADD_LIBRARY(CLExt SHARED cle/kernel.cpp cle/initializer.cpp cle/string.cpp)

ADD_EXECUTABLE(join join.c)
TARGET_LINK_LIBRARIES(join ${OpenCL_LIBRARY})

ADD_EXECUTABLE(test_prefix_sum test_prefix_sum.c)
TARGET_LINK_LIBRARIES(test_prefix_sum ${OpenCL_LIBRARY})

ADD_EXECUTABLE(test_prefix_sum_cpp test_prefix_sum.cpp)
TARGET_LINK_LIBRARIES(test_prefix_sum_cpp ${OpenCL_LIBRARY} CLExt)

ADD_EXECUTABLE(test_csv test_csv.cpp)
TARGET_LINK_LIBRARIES(test_csv)

ADD_EXECUTABLE(test_kmeans test_kmeans.cpp kmeans_initializer.cpp kmeans_naive.cpp kmeans_simd.cpp kmeans_gpu_assisted.cpp clustering_benchmark.cpp)
TARGET_LINK_LIBRARIES(test_kmeans ${OpenCL_LIBRARY} CLExt)

# Install OpenCL kernel source files
INSTALL(DIRECTORY ${CL_KERNELS_SOURCE_PATH} DESTINATION ${CL_KERNELS_INSTALL_PATH})

