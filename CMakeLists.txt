cmake_minimum_required(VERSION 3.25)
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
    SYSTEM         # consumers get -isystem, so our -Werror flags don't fire inside Eigen
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
# Library target
# ------------------------------------------------------------------------------
add_library(qputer_lib
    src/QuantumState.cpp
    src/QuantumGates.cpp
    src/ReadoutKernels.cpp
    src/StabilizerState.cpp
    src/QuantumStateMachine.cpp
)

# PIC so the static library can be linked into a shared Python extension module.
set_target_properties(qputer_lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Specify Include Directories (Public headers)
target_include_directories(qputer_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link Eigen to our library
target_link_libraries(qputer_lib PUBLIC Eigen3::Eigen)

# Gate kernels are parallelized over independent amplitude subspaces
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(qputer_lib PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "OpenMP not found: gate kernels will run single-threaded")
endif()

# PUBLIC so every TU sees the same Eigen vectorization/alignment settings
option(QPUTER_NATIVE "Optimize for the build host's CPU (-march=native)" ON)
if(QPUTER_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(qputer_lib PUBLIC -march=native)
endif()

# ------------------------------------------------------------------------------
# Shared warning + sanitizer flags, linked PRIVATE into every first-party target
# so the library, app, and tests are all held to the same bar.
# ------------------------------------------------------------------------------
add_library(qputer_build_flags INTERFACE)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(qputer_build_flags INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Werror
    )

    # ASan + UBSan for Debug builds; compile and link flags must both be present.
    set(QPUTER_SANITIZERS -fsanitize=address,undefined)
    target_compile_options(qputer_build_flags INTERFACE
        $<$<CONFIG:Debug>:${QPUTER_SANITIZERS} -fno-omit-frame-pointer>
    )
    target_link_options(qputer_build_flags INTERFACE
        $<$<CONFIG:Debug>:${QPUTER_SANITIZERS}>
    )
endif()

target_link_libraries(qputer_lib PRIVATE qputer_build_flags)

# ------------------------------------------------------------------------------
# Executable target
# ------------------------------------------------------------------------------
add_executable(qputer
    apps/main.cpp
)

target_link_libraries(qputer PRIVATE qputer_lib qputer_build_flags)

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------
# Project-scoped switch: BUILD_TESTING is a global shared with fetched
# dependencies, so it can't independently gate this project's tests.
option(QPUTER_BUILD_TESTS "Build the qputer unit test suite" ${PROJECT_IS_TOP_LEVEL})
if(QPUTER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ------------------------------------------------------------------------------
# Examples: worked algorithms to copy from (also run by ctest when tests are built)
# ------------------------------------------------------------------------------
option(QPUTER_BUILD_EXAMPLES "Build the example algorithms in examples/" ${PROJECT_IS_TOP_LEVEL})
if(QPUTER_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
