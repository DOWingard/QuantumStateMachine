#pragma once

#include <iosfwd>
#include <string>
#include <vector>



namespace Noether
{

// The `noether` command line. Returns the process exit code:
// 0 ok, 1 compile errors, 2 usage or I/O error, 3 assertion failed, 4 resource limit or timeout,
// 5 internal error, 6 research integrity failure.
int runCli(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

// Rewrites a source file to canonical Unicode Noether when it parses cleanly and the canonical text
// has the same parse tree. Returns true when the file changed; `error` is set if it could not be
// written.
bool canonicalizeFile(const std::string& path, std::string& error);

} // namespace Noether
