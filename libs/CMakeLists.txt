IF (NOT EXISTS ${PROJECT_SOURCE_DIR}/libs/gtest/build)
    FILE(MAKE_DIRECTORY "${PROJECT_SOURCE_DIR}/libs/gtest/build")
    MESSAGE("-- Building gtest")
    EXECUTE_PROCESS(
        COMMAND ${CMAKE_COMMAND} "${PROJECT_SOURCE_DIR}/libs/gtest"
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/libs/gtest/build"
        RESULT_VARIABLE rv
        )
    EXECUTE_PROCESS(
        COMMAND make
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/libs/gtest/build"
        RESULT_VARIABLE rv
        )
ENDIF(NOT EXISTS ${PROJECT_SOURCE_DIR}/libs/gtest/build)
