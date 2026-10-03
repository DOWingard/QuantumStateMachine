#pragma once

#include <QuantumState.hpp>

#include <cstddef>
#include <vector>
#include <Eigen/Core>



namespace Qputer
{

using Qubit = std::size_t;
using QubitList = std::vector<Qubit>;

// In-place unitary gates on a QuantumStateVector of N <= kMaxQubits qubits.
//
// Conventions:
//  - Qubit q is bit q of the amplitude index (qubit 0 = least significant).
//  - For a 2^M x 2^M matrix U applied to targets (t_0, ..., t_{M-1}), t_0 is the
//    MOST significant bit of U's row/column index, so U = A (x) B acts A on t_0, B on t_1.
//  - Controls are active-high: U acts only on the subspace where every control is |1>.
//
// Every gate is unitary (custom matrices are validated), so ||psi|| is preserved up
// to floating-point rounding. Invalid input throws before the state is touched.
class QuantumGate
{
    public:
    QuantumGate() = delete;

    // Dense matrices cost 16 * 4^M bytes and O(8^M) to validate; M = 10 -> 16 MiB.
    static constexpr std::size_t kMaxDenseTargets = 10;

    // Validation tolerance for max |(U^dagger U - I)_ij| of user-supplied matrices.
    static constexpr double kUnitaryTolerance = 1e-10;


    // ---- 1 qubit ----
    static void x(QuantumStateVector& state, Qubit target);
    static void y(QuantumStateVector& state, Qubit target);
    static void z(QuantumStateVector& state, Qubit target);
    static void h(QuantumStateVector& state, Qubit target);
    static void s(QuantumStateVector& state, Qubit target);
    static void sdg(QuantumStateVector& state, Qubit target);
    static void t(QuantumStateVector& state, Qubit target);
    static void tdg(QuantumStateVector& state, Qubit target);
    static void sx(QuantumStateVector& state, Qubit target);                         // sqrt(X)
    static void rx(QuantumStateVector& state, Qubit target, double theta);           // exp(-i theta X / 2)
    static void ry(QuantumStateVector& state, Qubit target, double theta);           // exp(-i theta Y / 2)
    static void rz(QuantumStateVector& state, Qubit target, double theta);           // exp(-i theta Z / 2)
    static void phase(QuantumStateVector& state, Qubit target, double lambda);       // diag(1, e^{i lambda})
    static void u3(QuantumStateVector& state, Qubit target, double theta, double phi, double lambda);

    // ---- 2 qubit ----
    static void cnot(QuantumStateVector& state, Qubit control, Qubit target);
    static void cz(QuantumStateVector& state, Qubit a, Qubit b);
    static void cphase(QuantumStateVector& state, Qubit a, Qubit b, double lambda);
    static void swap(QuantumStateVector& state, Qubit a, Qubit b);

    // ---- 3 qubit ----
    static void toffoli(QuantumStateVector& state, Qubit control0, Qubit control1, Qubit target);
    static void fredkin(QuantumStateVector& state, Qubit control, Qubit a, Qubit b);

    // ---- N qubit, O(2^N) regardless of how many qubits are involved ----
    static void mcx(QuantumStateVector& state, const QubitList& controls, Qubit target);
    static void mcz(QuantumStateVector& state, const QubitList& qubits);             // -1 on |1...1>
    static void mcphase(QuantumStateVector& state, const QubitList& qubits, double lambda);

    // ---- Arbitrary M-target unitary, optionally controlled, M <= kMaxDenseTargets ----
    static void apply(QuantumStateVector& state, const QubitList& targets, const Eigen::MatrixXcd& U);
    static void controlled(QuantumStateVector& state, const QubitList& controls,
                           const QubitList& targets, const Eigen::MatrixXcd& U);
};


}// namespace qstate
