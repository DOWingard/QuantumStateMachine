#pragma once

#include "Diagnostics.hpp"
#include "interop/Circuit.hpp"

#include <Eigen/Core>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>



namespace Noether::Interop
{

struct QasmOptions
{
    int version = 0;                      // 2 or 3; 0 takes it from the header (3 when there is none)
    std::map<std::string, double> params; // values of OpenQASM 3 `input` declarations
};

// Imports OpenQASM 2 or 3. Qubits are numbered in declaration order with registers concatenated,
// clbits likewise. Gates are inlined with parameters constant-folded; `qelib1.inc` and
// `stdgates.inc` are embedded and parsed as ordinary definitions, with native core sequences as a
// fast path checked against them. Included files resolve relative to the including file.
std::optional<ImportedCircuit> importQasm(SourceManager& sources, std::uint32_t fileId, Diagnostics& diags,
                                          const QasmOptions& options);

// ---- The embedded include files, for tests ----

// Names of the gates `include` ("qelib1.inc" or "stdgates.inc") defines, in file order.
std::vector<std::string> qasmIncludeGates(std::string_view include);

struct QasmGateShape
{
    std::size_t params = 0;
    std::size_t qubits = 0;
    bool native = false; // has a fast path
};
std::optional<QasmGateShape> qasmGateShape(std::string_view include, std::string_view gate);

// Matrix of `gate` applied to qubits 0…k−1 in argument order, in core index order (index bit q is
// qubit q), from the fast path or from the embedded definition expanded down to U / CX / gphase.
Eigen::MatrixXcd qasmGateMatrix(std::string_view include, std::string_view gate, const std::vector<double>& params,
                                bool native);

} // namespace Noether::Interop
