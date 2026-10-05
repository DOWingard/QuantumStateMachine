#pragma once

#include "Values.hpp"

#include <QuantumState.hpp>
#include <QuantumStateMachine.hpp>

#include <vector>



namespace Noether
{

// Expands one IR op into core operations, with angles bound to `params`. On the stabilizer
// backend Clifford-angle rotations become S/H/X/Z/√X sequences (exact up to global phase), which
// the tableau runs; a Pauli measurement becomes the Clifford conjugation onto one qubit's Z, the
// Z measurement, and the conjugation undone. Global phases are dropped unless controlled.
std::vector<Qputer::Operation> lower(const IrOp& op, const std::vector<double>& params, bool stabilizer);

// Applies a unitary core operation to a raw state vector (dense helpers, equivalence checks).
void applyUnitary(Qputer::QuantumStateVector& s, const Qputer::Operation& op);

// Multiplies a dense matrix (targets[0] = most significant bit) into a raw state vector without
// requiring it to be unitary (observables, matrix elements).
void applyMatrix(Qputer::QuantumStateVector& s, const QubitList& targets, const Eigen::MatrixXcd& m);

// Dense matrix of a sequence of IR ops (unitary gates and observable matrices) on `support`
// (support[0] = most significant bit).
Eigen::MatrixXcd unitaryOf(const std::vector<IrOp>& ops, const QubitList& support, const std::vector<double>& params);

// Ops for preparing a stabilizer-product ket from |0…0⟩ (basis flips, then H / S per qubit).
std::vector<Qputer::Operation> productPreparation(const KetV& ket, std::uint64_t& basisLow);

} // namespace Noether
