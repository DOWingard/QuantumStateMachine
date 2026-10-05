#pragma once

#include <iosfwd>
#include <string>
#include <vector>



namespace Noether::Interop
{

// The complete `noether` command line. `run`, `check` and `import` on OpenQASM (.qasm), Stim
// (.stim) and circuit JSON (.json) files, or with --format, are handled here; every other command
// goes to Noether::runCli. Exit codes are Noether's: 0 ok, 1 import or run errors, 2 usage or I/O
// error, 5 internal error.
int runCli(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

} // namespace Noether::Interop
