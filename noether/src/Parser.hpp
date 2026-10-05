#pragma once

#include "Ast.hpp"
#include "Diagnostics.hpp"
#include "Lexer.hpp"

#include <string_view>



namespace Noether
{

inline constexpr std::string_view kLanguageVersion = "0.1";

// Parses one file. Errors are reported to `diags`; the parser resynchronises at the next line so
// that every independent error is reported. The result is always a (possibly partial) program.
Program parse(const SourceFile& file, std::uint32_t fileId, Diagnostics& diags);

// The EBNF grammar printed by `noether grammar`.
std::string_view grammarText();

} // namespace Noether
