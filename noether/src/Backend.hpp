#pragma once

#include "Diagnostics.hpp"
#include "Ir.hpp"

#include <cstdint>
#include <string>



namespace Noether
{

// True when the op is a Clifford operation (or a Pauli channel, measurement or reset) the stabilizer
// backend runs exactly. Angles must be exact multiples: θ ∈ (π/2)ℤ for rotations, λ ∈ πℤ for CP.
bool isClifford(const IrOp& op);

// Sets IrOp::clifford on every op event.
void tagClifford(Ir& ir);

// Chooses the backend for the whole program from the request (auto | statevector | stabilizer), the
// ops, the preparations and the readouts. Reports E6001 / E6002 / E6003 when nothing fits.
void selectBackend(Ir& ir, Diagnostics& d);

// Bytes the live register needs on a backend: 16·2^N for the state vector, ~N²/4 for the tableau.
std::uint64_t registerBytes(const std::string& backend, std::size_t nQubits);

} // namespace Noether
