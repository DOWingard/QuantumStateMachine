#pragma once

#include <optional>
#include <string>
#include <string_view>



namespace Noether
{

// Source of a bundled module by its import path ("std/qft.ntr"), or nullopt.
std::optional<std::string_view> stdSource(std::string_view importPath);

// Comma-separated import paths of every bundled module, for messages.
std::string stdModuleList();

} // namespace Noether
