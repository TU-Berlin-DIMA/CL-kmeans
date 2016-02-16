# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.
# 
# 
# Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET(CMAKE_EXPORT_COMPILE_COMMANDS 1) # For vim color_coded plugin
PROJECT(cljoin_operators)

SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${PROJECT_SOURCE_DIR}/cmake")

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
SET(CMAKE_CXX_FLAGS "-std=c++11 -O2 -ffp-contract=fast -march=haswell -fopenmp -Wall -Wextra -DUNIX")
SET(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG -DMATRIX_BOUNDSCHECK")

FIND_PACKAGE(Boost 1.49 REQUIRED COMPONENTS program_options)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

FIND_PACKAGE(OpenCL REQUIRED)
INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIRS})

FIND_PACKAGE(Armadillo)
IF(ARMADILLO_FOUND)
    INCLUDE_DIRECTORIES(${ARMADILLO_INCLUDE_DIRS})
ENDIF(ARMADILLO_FOUND)

ADD_LIBRARY(CLExt SHARED cle/kernel.cpp cle/initializer.cpp cle/optimization.cpp cle/string.cpp)

ADD_EXECUTABLE(join join.c)
TARGET_LINK_LIBRARIES(join ${OPENCL_LIBRARIES})

ADD_EXECUTABLE(test_prefix_sum test_prefix_sum.c)
TARGET_LINK_LIBRARIES(test_prefix_sum ${OPENCL_LIBRARIES})

ADD_EXECUTABLE(test_prefix_sum_cpp test_prefix_sum.cpp)
TARGET_LINK_LIBRARIES(test_prefix_sum_cpp ${OPENCL_LIBRARIES} CLExt)

ADD_EXECUTABLE(test_csv test_csv.cpp)
TARGET_LINK_LIBRARIES(test_csv)

SET(TEST_KMEANS_NAME "test_kmeans")
SET(TEST_KMEANS_SOURCES
    test_kmeans.cpp
    clustering_benchmark.cpp
    kmeans_initializer.cpp
    kmeans_naive.cpp
    kmeans_simd.cpp
    kmeans_gpu_assisted.cpp
    lloyd_gpu_feature_sum.cpp
    )
IF(ARMADILLO_FOUND)
    LIST(APPEND TEST_KMEANS_SOURCES kmeans_armadillo.cpp)
ENDIF(ARMADILLO_FOUND)
ADD_EXECUTABLE(test_kmeans ${TEST_KMEANS_SOURCES})
TARGET_LINK_LIBRARIES(test_kmeans ${OPENCL_LIBRARIES} CLExt)
IF(ARMADILLO_FOUND)
    TARGET_LINK_LIBRARIES(test_kmeans ${ARMADILLO_LIBRARIES})
ENDIF(ARMADILLO_FOUND)

SET(BENCH_NAME "bench")
SET(BENCH_SOURCES
    bench.cpp
    clustering_benchmark.cpp
    kmeans_initializer.cpp
    kmeans_naive.cpp
    kmeans_simd.cpp
    kmeans_gpu_assisted.cpp
    lloyd_gpu_feature_sum.cpp
    )
IF(ARMADILLO_FOUND)
    LIST(APPEND BENCH_SOURCES kmeans_armadillo.cpp)
ENDIF(ARMADILLO_FOUND)
ADD_EXECUTABLE(bench ${BENCH_SOURCES})
TARGET_LINK_LIBRARIES(bench ${OPENCL_LIBRARIES} ${Boost_LIBRARIES} CLExt)
IF(ARMADILLO_FOUND)
    TARGET_LINK_LIBRARIES(bench ${ARMADILLO_LIBRARIES})
ENDIF(ARMADILLO_FOUND)

# Install OpenCL kernel source files
INSTALL(DIRECTORY ${CL_KERNELS_SOURCE_PATH} DESTINATION ${CL_KERNELS_INSTALL_PATH})

CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/SystemConfig.h.in"
    "${PROJECT_BINARY_DIR}/SystemConfig.h")
INCLUDE_DIRECTORIES("${PROJECT_BINARY_DIR}")
