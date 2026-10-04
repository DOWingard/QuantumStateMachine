#pragma once

#include <QuantumGates.hpp>
#include <QuantumState.hpp>

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

// Every kernel is one O(2^N) pass, parallel above kParallelThreshold. Reductions use a
// block decomposition fixed by the problem size alone and combine partials in block
// order, so results are bitwise identical for any OpenMP thread count.

// out[i] = |a_i|^2 for every amplitude; out holds 2^N doubles.
void squaredMagnitudes(const QuantumStateVector& s, double* out);

// out[o] = sum of |a_i|^2 over all i whose bit qubits[j] equals bit j of o; out holds
// 2^k doubles. Qubits must be distinct and in range.
void marginalWeights(const QuantumStateVector& s, std::span<const Qubit> qubits, double* out);

QubitWeights qubitWeights(const QuantumStateVector& s, Qubit q);

// sum_i conj(a[i ^ xMask]) a[i] (-1)^popcount(i & zMask): <psi|P|psi> up to P's i^{#Y} phase.
std::complex<double> pauliSum(const QuantumStateVector& s, Index xMask, Index zMask);

// a_i *= scale where (i & mask) == value, otherwise a_i = 0 (projection + renormalization).
void collapse(QuantumStateVector& s, Index mask, Index value, double scale);

// Smallest i with sum_{j <= i} |a_j|^2 > target, for target in [0, ||psi||^2). Rounding
// past the end resolves to the last amplitude with nonzero weight, never a zero-weight one.
Index findCumulative(const QuantumStateVector& s, double target);

void setBasis(QuantumStateVector& s, Index index);

// Sizes must match.
void copyAmplitudes(const QuantumStateVector& from, QuantumStateVector& to);

// to[i] = from[i] for the to.size() amplitudes at `from`.
void copyAmplitudes(const std::complex<double>* from, QuantumStateVector& to);

} // namespace Qputer::detail
