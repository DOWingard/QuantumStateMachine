# One executable per example. Each checks its result against theory and exits nonzero on a
# mismatch, so ctest also runs them as smoke tests.
function(qputer_example name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE qputer_lib qputer_build_flags)
    if(QPUTER_BUILD_TESTS)
        add_test(NAME example.${name} COMMAND ${name})
    endif()
endfunction()

qputer_example(phase_estimation)
qputer_example(variational)
qputer_example(repetition_code)
