
set(MLPACK_FOUND FALSE)

find_path(MLPACK_INCLUDE_DIR mlpack/core.hpp)
find_library(MLPACK_LIBRARIES libmlpack.so)

if (MLPACK_INCLUDE_DIR AND MLPACK_LIBRARIES)
    message(STATUS "Looking for libmlpack - found")
    set(TCMALLOC_FOUND TRUE)
else ()
    message(FATAL_ERROR "Looking for libmlpack - not found")
endif()
