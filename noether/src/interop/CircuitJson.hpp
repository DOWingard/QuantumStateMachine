#pragma once

#include "Diagnostics.hpp"
#include "interop/Circuit.hpp"

#include <cstdint>
#include <optional>



namespace Noether::Interop
{

inline constexpr std::string_view kCircuitSchema = "noether.circuit/1";

// Reads a `noether.circuit/1` document (written by the Python exporters and by
// `noether import --emit circuit`). Op names, qubit roles and params are exactly those of
// Qputer::Operation; the reader checks structure and ranges but holds no gate semantics. Unknown
// keys and schemas are E9007, located at the offending value; recorded notes become warnings.
std::optional<ImportedCircuit> readCircuitJson(const SourceFile& file, std::uint32_t fileId, Diagnostics& diags);

// The canonical document for `circuit`; warnings in `diags` are carried as notes.
Json circuitJson(const ImportedCircuit& circuit, const Diagnostics& diags);

} // namespace Noether::Interop
