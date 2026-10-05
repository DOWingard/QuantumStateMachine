# Noether: a scripting language for the Qputer quantum state machine.

set(NOETHER_STD_FILES qft.ntr grover.ntr arith.ntr gates.ntr)
list(TRANSFORM NOETHER_STD_FILES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/std/" OUTPUT_VARIABLE NOETHER_STD_DEPS)

string(JOIN "," NOETHER_STD_LIST ${NOETHER_STD_FILES})
set(NOETHER_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
add_custom_command(
    OUTPUT ${NOETHER_GEN_DIR}/EmbeddedStd.cpp
    COMMAND ${CMAKE_COMMAND} -DOUT=${NOETHER_GEN_DIR}/EmbeddedStd.cpp -DFUNC=embeddedStdLib
            -DROOT=${CMAKE_CURRENT_SOURCE_DIR}/std -DFILES=${NOETHER_STD_LIST}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
    DEPENDS ${NOETHER_STD_DEPS} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
    COMMENT "Embedding the Noether standard library")

# The agent skills live at the repository root, next to the projects they describe.
get_filename_component(QSM_SKILLS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../skills ABSOLUTE)
set(QSM_SKILL_FILES qsm/SKILL.md qsm-research/SKILL.md qsm-tune/SKILL.md qsm-tune/reference.md)
list(TRANSFORM QSM_SKILL_FILES PREPEND "${QSM_SKILLS_DIR}/" OUTPUT_VARIABLE QSM_SKILL_DEPS)
string(JOIN "," QSM_SKILL_LIST ${QSM_SKILL_FILES})
add_custom_command(
    OUTPUT ${NOETHER_GEN_DIR}/EmbeddedSkills.cpp
    COMMAND ${CMAKE_COMMAND} -DOUT=${NOETHER_GEN_DIR}/EmbeddedSkills.cpp -DFUNC=embeddedSkills
            -DROOT=${QSM_SKILLS_DIR} -DFILES=${QSM_SKILL_LIST}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
    DEPENDS ${QSM_SKILL_DEPS} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed.cmake
    COMMENT "Embedding the agent skills")

add_library(noether_lib STATIC
    src/Source.cpp src/Json.cpp src/Diagnostics.cpp src/Lexer.cpp src/Parser.cpp
    src/Values.cpp src/Compiler.cpp src/Eval.cpp src/Backend.cpp src/Lowering.cpp
    src/Formatter.cpp src/Sha256.cpp src/StdLib.cpp src/Estimate.cpp src/Executor.cpp src/Cli.cpp
    src/Draw.cpp src/Equiv.cpp src/Optimize.cpp src/Qasm.cpp src/Decompose.cpp src/Research.cpp
    src/Skills.cpp src/Repl.cpp
    ${NOETHER_GEN_DIR}/EmbeddedStd.cpp ${NOETHER_GEN_DIR}/EmbeddedSkills.cpp)
target_include_directories(noether_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(noether_lib PUBLIC qputer_lib PRIVATE qputer_build_flags)

add_executable(noether apps/main.cpp)
target_link_libraries(noether PRIVATE noether_lib qputer_build_flags)

if(QPUTER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
