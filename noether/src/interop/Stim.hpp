#pragma once

#include "Diagnostics.hpp"
#include "interop/Circuit.hpp"

#include <cstdint>
#include <optional>



namespace Noether::Interop
{

// Imports a Stim circuit. Qubit ids (often sparse coordinates) are compacted in ascending order and
// labelled q<id>; the k-th measurement result is clbit k. Cliffords lower to core Cliffords so the
// circuit stays on the stabilizer backend; Pauli noise becomes pauli_channel, correlated errors
// pauli_channel or kraus; REPEAT blocks are unrolled. DETECTOR and OBSERVABLE_INCLUDE become clbit
// masks with absolute record indices.
std::optional<ImportedCircuit> importStim(const SourceFile& file, std::uint32_t fileId, Diagnostics& diags);

// Every gate name the importer accepts, aliases included (fixture tests iterate over these).
std::vector<std::string_view> stimGateNames();

} // namespace Noether::Interop
