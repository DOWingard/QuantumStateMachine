cmake_minimum_required(VERSION 3.20)
project(QuantumEngine VERSION 0.1.0 LANGUAGES CXX)

# Enforce Modern C++23
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ------------------------------------------------------------------------------
# Dependencies: Fetch Eigen automatically if not found locally
# ------------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
)

# Eigen's own CMakeLists.txt builds its unit tests, demos, and BLAS by
# default; disable all of that so FetchContent only brings in the headers.
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_BLAS OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_LAPACK OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(Eigen)

# ------------------------------------------------------------------------------
# Target 1: The Core Library
# ------------------------------------------------------------------------------
add_library(qstate
    src/QuantumState.cpp
)

# Specify Include Directories (Public headers)
target_include_directories(qstate PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link Eigen to our library
target_link_libraries(qstate PUBLIC Eigen3::Eigen)

# ------------------------------------------------------------------------------
# Clang-Specific Compiler Flags & Sanitizers
# ------------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(qstate PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Werror
    )

    # Enable AddressSanitizer and UndefinedBehaviorSanitizer for Debug builds
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(STATUS "Enforcing Clang Sanitizers (ASan + UBSan) for C++23 Debug build")
        target_compile_options(qstate PRIVATE 
            -fsanitize=address,undefined 
            -fno-omit-frame-pointer
        )
        target_link_options(qstate PRIVATE 
            -fsanitize=address,undefined
        )
    endif()
endif()

