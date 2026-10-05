# GoogleTest comes from the core test suite: fetched targets are global, but an installed GTest's
# imported targets are scoped to the directory that found them.
if(NOT TARGET GTest::gtest_main)
    find_package(GTest REQUIRED)
endif()

add_executable(noether_tests
    unit/test_lexer.cpp
    unit/test_parser.cpp
    unit/test_formatter.cpp
    unit/test_operators.cpp
    unit/test_golden.cpp
    unit/test_diagnostics.cpp
    unit/test_tools.cpp
    unit/test_research.cpp
)
target_link_libraries(noether_tests PRIVATE noether_lib qputer_build_flags GTest::gtest_main)
target_compile_definitions(noether_tests PRIVATE
    NOETHER_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    NOETHER_EXAMPLES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../examples"
    QSM_SKILLS_DIR="${QSM_SKILLS_DIR}")

include(GoogleTest)
gtest_discover_tests(noether_tests DISCOVERY_TIMEOUT 60 PROPERTIES LABELS noether)
