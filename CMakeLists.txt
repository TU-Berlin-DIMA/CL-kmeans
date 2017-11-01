# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.
# 
# 
# Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
SET(CMAKE_C_COMPILER clang)
SET(CMAKE_CXX_COMPILER clang++)
SET(CMAKE_EXPORT_COMPILE_COMMANDS 1) # For vim color_coded plugin
PROJECT(clustering)

SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${PROJECT_SOURCE_DIR}/cmake")

SET(CL_KERNELS_SOURCE_PATH "${PROJECT_SOURCE_DIR}/cl_kernels")
SET(CL_KERNELS_INSTALL_PATH "${CMAKE_PREFIX_PATH}/opt/${PROJECT_NAME}")

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    SET(CL_KERNELS_PATH ${CL_KERNELS_SOURCE_PATH})
ELSEIF(CMAKE_BUILD_TYPE MATCHES Release OR NOT CMAKE_BUILD_TYPE)
    SET(CL_KERNELS_PATH ${CL_KERNELS_INSTALL_PATH})
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

OPTION(MATRIX_BOUNDSCHECK "Enable matrix bounds checking" OFF)

SET(CMAKE_C_FLAGS "-std=c11 -O3 -ffp-contract=fast -march=native -Wall -Wextra -DUNIX")
SET(CMAKE_CXX_FLAGS "-std=c++11 -O3 -ffp-contract=fast -march=native -Wall -Wextra -DUNIX")
SET(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG")
SET(CMAKE_CXX_FLAGS_DEBUG "-g -DDEBUG")

ADD_SUBDIRECTORY(libs)
SET(GTEST_ROOT "${PROJECT_SOURCE_DIR}/libs/gtest/build")

FIND_PACKAGE(Boost 1.50 REQUIRED COMPONENTS program_options system filesystem)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/libs/boost_compute/include")

FIND_PACKAGE(OpenCL REQUIRED)
INCLUDE_DIRECTORIES(${OPENCL_INCLUDE_DIRS})

FIND_PACKAGE(CUDA)
IF(CUDA_FOUND)
    INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
ENDIF(CUDA_FOUND)

ADD_EXECUTABLE(test_csv test_csv.cpp)
TARGET_LINK_LIBRARIES(test_csv)

SET(BENCH_NAME "bench")
SET(BENCH_SOURCES
    bench.cpp
    binary_format.cpp
    clustering_benchmark.cpp
    configuration_parser.cpp
    kmeans_common.cpp
    kmeans_initializer.cpp
    kmeans_naive.cpp
    measurement/measurement.cpp
    )
ADD_EXECUTABLE(bench ${BENCH_SOURCES})
TARGET_LINK_LIBRARIES(bench ${OPENCL_LIBRARIES} ${Boost_LIBRARIES})
IF(CUDA_FOUND)
    TARGET_LINK_LIBRARIES(bench ${CUDA_LIBRARIES})
ENDIF(CUDA_FOUND)

SET(GENERATOR_NAME "generator")
SET(GENERATOR_SOURCES
    generator.cpp
    cluster_generator.cpp
    )
ADD_EXECUTABLE(generator ${GENERATOR_SOURCES})
TARGET_LINK_LIBRARIES(generator ${Boost_LIBRARIES})

FIND_PACKAGE(JPEG)
IF(JPEG_FOUND)
    SET(PICLUSTER_SOURCES
        picluster.cpp
        kmeans_common.cpp
        kmeans_initializer.cpp
        kmeans_naive.cpp
        measurement/measurement.cpp
        libs/jpeg_reader_writer/JPEGReader.cpp
        libs/jpeg_reader_writer/JPEGWriter.cpp
        )
    INCLUDE_DIRECTORIES(${JPEG_INCLUDE_DIR})
    ADD_EXECUTABLE(picluster ${PICLUSTER_SOURCES})
    TARGET_LINK_LIBRARIES(picluster ${OPENCL_LIBRARIES} ${Boost_LIBRARIES} ${JPEG_LIBRARIES})
ENDIF(JPEG_FOUND)

# Create Version.h file with current git revision string
FIND_PACKAGE(Git)
IF(GIT_EXECUTABLE)
    EXECUTE_PROCESS(
        COMMAND ${GIT_EXECUTABLE} log -1 --format="%H"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_REVISION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    MESSAGE(STATUS "Git revision is ${GIT_REVISION}")
ELSEIF (DEFINED GIT_REVISION)
    MESSAGE(STATUS "Git revision is ${GIT_REVISION}")
ELSE ()
    MESSAGE(FATAL_ERROR "GIT_REVISION not specified")
ENDIF(GIT_EXECUTABLE)

CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/Version.h.in"
    "${PROJECT_BINARY_DIR}/Version.h")

# Create SystemConfig.h file with configuration variables
CONFIGURE_FILE("${PROJECT_SOURCE_DIR}/SystemConfig.h.in"
    "${PROJECT_BINARY_DIR}/SystemConfig.h")
INCLUDE_DIRECTORIES("${PROJECT_BINARY_DIR}")

# Install OpenCL kernel source files
INSTALL(DIRECTORY ${CL_KERNELS_SOURCE_PATH} DESTINATION ${CL_KERNELS_INSTALL_PATH})

IF(ENABLE_BASELINES)
ADD_SUBDIRECTORY(baselines)
ENDIF(ENABLE_BASELINES)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
ADD_SUBDIRECTORY(test)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

