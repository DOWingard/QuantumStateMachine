#pragma once

#include "Diagnostics.hpp"
#include "Ir.hpp"
#include "Json.hpp"

#include <cstdint>
#include <map>
#include <string>



namespace Noether
{

// Static resources of a compiled program, from its IR (noise-inserted channels excluded).
struct Resources
{
    std::size_t qubits = 0, clbits = 0;
    std::size_t ops = 0;          // operations, including measurements, resets and inline channels
    std::size_t depth = 0;        // ASAP layers over all qubit-touching ops
    std::size_t depth2q = 0;      // layers counting only ops on two or more qubits
    std::size_t count2q = 0;
    std::size_t tcount = 0;       // T, T† and Z-rotations by odd multiples of π/4
    std::size_t tdepth = 0;
    std::size_t measurements = 0;
    std::size_t nparams = 0;
    bool clifford = true;
    std::map<std::string, std::size_t> opsByKind;
    std::uint64_t peakBytes = 0;  // live register on the chosen backend
};

Resources resources(const Ir& ir);
Json resourcesJson(const Resources& r);

// The `noether.estimate/1` body (without the common header fields).
Json estimateJson(const Ir& ir, const SourceManager& sm);

// True when `op` is a T-type gate: T, T†, or Rz / P / Z-type Pauli rotation by an odd multiple of π/4.
bool isTGate(const IrOp& op);

} // namespace Noether
