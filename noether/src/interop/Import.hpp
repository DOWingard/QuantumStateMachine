#pragma once

#include "Diagnostics.hpp"
#include "interop/Circuit.hpp"

#include <map>
#include <memory>
#include <optional>
#include <string>



namespace Noether::Interop
{

struct ImportOptions
{
    // qasm (version from the header), qasm2, qasm3, stim or circuit; empty picks it from the extension.
    std::string format;
    std::map<std::string, double> params; // OpenQASM 3 `input` values
};

struct ImportResult
{
    std::unique_ptr<SourceManager> sources = std::make_unique<SourceManager>();
    std::unique_ptr<Diagnostics> diags = std::make_unique<Diagnostics>(*sources);
    std::optional<ImportedCircuit> circuit; // set when there are no errors
    std::uint32_t mainFile = 0;
    std::string sha256;
    bool readFailed = false;
    std::string readError;
};

// The import format for a path: .qasm → qasm, .stim → stim, .json → circuit; empty otherwise.
std::string formatForPath(const std::string& path);

// Parses and lowers one file. Diagnostics carry spans into `sources`; W9003 lists declared
// qubits no operation touches.
ImportResult importFile(const std::string& path, const ImportOptions& options);
ImportResult importText(const std::string& path, std::string text, const ImportOptions& options);

} // namespace Noether::Interop
