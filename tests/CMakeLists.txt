include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
    SYSTEM
    FIND_PACKAGE_ARGS NAMES GTest   # prefer an installed GoogleTest
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(qputer_tests
    test_state.cpp
    test_single_qubit_gates.cpp
    test_multi_qubit_gates.cpp
    test_dense_gates.cpp
    test_reference.cpp
    test_algorithms.cpp
    test_parallel.cpp
    test_state_machine.cpp
    test_readout.cpp
    test_measurement.cpp
    test_stabilizer.cpp
    test_backend_comparison.cpp
    test_channels.cpp
    test_entropy.cpp
)

target_link_libraries(qputer_tests PRIVATE qputer_lib qputer_build_flags GTest::gtest_main)

# The parallel-consistency suite pins thread counts through the OpenMP runtime.
if(OpenMP_CXX_FOUND)
    target_link_libraries(qputer_tests PRIVATE OpenMP::OpenMP_CXX)
endif()

include(GoogleTest)
gtest_discover_tests(qputer_tests DISCOVERY_TIMEOUT 30)
