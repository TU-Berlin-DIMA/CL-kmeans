# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.
# 
# 
# Copyright (c) 2016, Lutz, Clemens <lutzcle@cml.li>

FIND_PACKAGE(Boost 1.50 REQUIRED COMPONENTS system filesystem)
FIND_PACKAGE(GTest 1.6 REQUIRED)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS})

FUNCTION(ADD_TEST_MODULE TEST_NAME TEST_SOURCE)
    GET_FILENAME_COMPONENT(TEST_TARGET ${TEST_SOURCE} NAME_WE)
    ADD_EXECUTABLE(${TEST_TARGET} ${TEST_SOURCE} ../measurement/measurement.cpp ../cluster_generator.cpp ${ARGN})
    TARGET_LINK_LIBRARIES(${TEST_TARGET}
        ${Boost_LIBRARIES}
        ${GTEST_LIBRARIES}
        ${OPENCL_LIBRARIES}
        ${CLEXT_LIBRARIES}
        Threads::Threads
        )
ENDFUNCTION()

ENABLE_TESTING()
# ADD_TEST_MODULE("histogram" histogram.cpp)
ADD_TEST_MODULE(
    "reduce_vector"
    reduce_vector.cpp
    )
# ADD_TEST_MODULE("feature_sum" feature_sum.cpp)
ADD_TEST_MODULE(
    "buffer_cache"
    buffer_cache.cpp
    ../simple_buffer_cache.cpp
    )
ADD_TEST_MODULE(
    "device_scheduler"
    device_scheduler.cpp
    ../single_device_scheduler.cpp
    ../simple_buffer_cache.cpp
    )
