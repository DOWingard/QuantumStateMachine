#pragma once

#include <QuantumGates.hpp>
#include <QuantumState.hpp>
#include "DenseGate.hpp"

#include <complex>
#include <cstddef>
#include <span>



namespace Qputer::detail
{

using Index = std::size_t;

struct QubitWeights
{
    double zero = 0.0; // sum |a_i|^2 over i with the qubit's bit clear
    double one = 0.0;  // ... with it set
};

// Every kernel is one O(2^N) pass unless noted, parallel above kParallelThreshold.
// Reductions use a block decomposition fixed by the problem size alone and combine
// partials in block order, so results are bitwise identical for any OpenMP thread count.

// out[i] = |a_i|^2 for every amplitude; out holds 2^N doubles.
void squaredMagnitudes(const QuantumStateVector& s, double* out);

// out[o] = sum of |a_i|^2 over all i whose bit qubits[j] equals bit j of o; out holds
// 2^k doubles. Qubits must be distinct and in range.
void marginalWeights(const QuantumStateVector& s, std::span<const Qubit> qubits, double* out);

QubitWeights qubitWeights(const QuantumStateVector& s, Qubit q);

// sum_i conj(a[i ^ xMask]) a[i] (-1)^popcount(i & zMask): <psi|P|psi> up to P's i^{#Y} phase.
std::complex<double> pauliSum(const QuantumStateVector& s, Index xMask, Index zMask);

// out[k] = ||K_k psi||^2 for the prepared operators of one Kraus channel: no controls, the
// same targets for every k. out holds ops.size() doubles. O(|ops| 2^(N+M)).
void krausWeights(const QuantumStateVector& s, std::span<const DenseGate> ops, double* out);

// Reduced density matrix rho_rc = sum_e a_(r,e) conj(a_(c,e)) of distinct, non-empty `qubits`
// over the assignments e of the other qubits, with qubits[0] the MSB of r and c. out receives
// the 2^k x 2^k matrix column-major (Eigen's default layout), exactly Hermitian with a real
// diagonal. O(2^(N+k)), and bitwise reproducible like the reductions.
void reducedDensityMatrix(const QuantumStateVector& s, std::span<const Qubit> qubits, std::complex<double>* out);

// a_i *= scale where (i & mask) == value, otherwise a_i = 0 (projection + renormalization).
void collapse(QuantumStateVector& s, Index mask, Index value, double scale);

// a_i *= scale for every amplitude.
void scaleAmplitudes(QuantumStateVector& s, double scale);

// Smallest i with sum_{j <= i} |a_j|^2 > target, for target in [0, ||psi||^2). Rounding
// past the end resolves to the last amplitude with nonzero weight, never a zero-weight one.
Index findCumulative(const QuantumStateVector& s, double target);

void setBasis(QuantumStateVector& s, Index index);

// Sizes must match.
void copyAmplitudes(const QuantumStateVector& from, QuantumStateVector& to);

// to[i] = from[i] for the to.size() amplitudes at `from`.
void copyAmplitudes(const std::complex<double>* from, QuantumStateVector& to);

} // namespace Qputer::detail
