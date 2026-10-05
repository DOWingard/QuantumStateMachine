#pragma once

#include <span>
#include <string_view>



namespace Noether
{

// A text file compiled into the binary: the standard library and the agent skills.
struct EmbeddedFile
{
    std::string_view path;
    std::string_view text;
};

std::span<const EmbeddedFile> embeddedStdLib(); // paths relative to std/, e.g. "qft.ntr"
std::span<const EmbeddedFile> embeddedSkills(); // paths relative to skills/, e.g. "qsm/SKILL.md"

} // namespace Noether
