#pragma once

#include <QuantumGates.hpp>
#include <QuantumState.hpp>

#include <complex>
#include <cstddef>
#include <span>
#include <vector>
#include <Eigen/Core>



namespace Qputer::detail
{

// A dense (optionally controlled) gate reduced to exactly what the kernel reads: the
// amplitude offset of each local basis state and U in row-major order. Building it costs
// O(4^M); applying it costs only the O(2^(N+M)) kernel, so callers that apply one
// matrix repeatedly (circuit replay across shots) prepare it once.
struct DenseGate
{
    QubitList controls;
    QubitList targets;               // targets[0] = MSB of U's row/column index
    std::vector<std::size_t> offsets; // offsets[v] = amplitude offset of local basis state v
    std::vector<std::complex<double>> rowMajor;
};

// No validation: U must already be validated (unitary for a gate, one operator of a
// trace-preserving set for a Kraus channel) and 2^M x 2^M for M = targets.size(), with
// 1 <= M <= QuantumGate::kMaxDenseTargets and all qubits distinct.
DenseGate prepareDense(std::span<const Qubit> controls, std::span<const Qubit> targets, const Eigen::MatrixXcd& U);

// psi := U psi without checking unitarity, so it also applies a Kraus operator; qubit
// indices are still checked against the state, which is O(M) and allocation-free.
void applyDense(QuantumStateVector& state, const DenseGate& gate);

} // namespace Qputer::detail
