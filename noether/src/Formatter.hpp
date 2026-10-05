#pragma once

#include "Ast.hpp"
#include "Source.hpp"

#include <string>



namespace Noether
{

// Canonical source text of an expression: Unicode by default, the ASCII-canonical spelling with
// `ascii`. Re-parsing the result gives the same tree.
std::string formatExpr(const Expr& e, bool ascii);

// Canonical text of a whole program, keeping comments, blank lines, statements that shared a line
// and one-line suites. Formatting is idempotent.
std::string formatProgram(const Program& prog, const SourceFile& file, bool ascii);

} // namespace Noether
