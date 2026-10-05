#pragma once

// The one place that builds Qputer::Operation values for the importers: gate algebra over
// unitary sequences (control, inverse, power), exact dense fallbacks, Pauli-product measurement
// and rotation, channels, and validation of operations read from circuit JSON. A change to the
// core operation set touches this file only.

#include "interop/Circuit.hpp"

#include <Eigen/Core>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>



namespace Noether::Interop
{

using Qputer::Condition;

// A construct an importer cannot lower: an E9xxx diagnostic code and its message. Frontends attach
// the span of the statement being lowered.
class ImportError : public std::runtime_error
{
    public:
    ImportError(std::string code, const std::string& message) : std::runtime_error(message), diagCode(std::move(code)) {}
    const std::string& code() const { return diagCode; }

    private:
    std::string diagCode;
};

// One step of a unitary sequence: a core gate, or a global phase, under extra active-high
// controls. A global phase is unobservable until it is controlled, when it is a phase gate on
// the controls.
enum class Prim : std::uint8_t { X, Y, Z, H, S, Sdg, T, Tdg, SX, RX, RY, RZ, Phase, U3, Swap, Unitary, GPhase };

struct Gate
{
    Prim prim = Prim::X;
    QubitList controls;
    QubitList targets;          // none for GPhase
    std::vector<double> params; // RX RY RZ Phase GPhase {angle}; U3 {θ, φ, λ}
    Eigen::MatrixXcd matrix;    // Unitary only; targets[0] = most significant bit
};

using Seq = std::vector<Gate>; // time order: element 0 acts first

Gate makeGate(Prim prim, QubitList targets, std::vector<double> params = {}, QubitList controls = {});
Gate denseGate(QubitList targets, Eigen::MatrixXcd matrix);

// ---- Gate algebra; every result is exact, global phase included ----

// C(Uₙ⋯U₁) = C(Uₙ)⋯C(U₁). `active[k]` false makes controls[k] active on |0⟩ (X before and after).
Seq controlled(Seq body, const QubitList& controls, const std::vector<bool>& active = {});
Seq inverse(const Seq& body);
// Integers repeat; a lone rotation or diagonal phase gate scales its angle; anything else is the
// principal power Q·diag(λₖ^r)·Q† of its dense matrix (arg λ ∈ (−π, π]).
Seq power(const Seq& body, double exponent);

QubitList supportOf(const Seq& body); // in order of first use
Eigen::MatrixXcd baseMatrix(const Gate& g); // on g.targets, controls ignored; 1×1 for GPhase
Eigen::MatrixXcd matrixOf(const Seq& body, const QubitList& support); // support[0] = MSB

// ---- Circuit construction ----

struct PauliFactor
{
    char letter = 'Z'; // X, Y or Z
    Qubit qubit = 0;
};

class CircuitBuilder
{
    public:
    explicit CircuitBuilder(ImportedCircuit& circuit) : c(circuit) {}

    // Unitary steps as core operations, each carrying `condition`. Uncontrolled global phases are
    // dropped.
    void apply(const Seq& seq, const std::optional<Condition>& condition = std::nullopt);

    // Z measurement. `invert` reports the complement (X before and after); `flip` is the
    // probability of a reported-result flip, modelled as an X error just before the measurement.
    void measure(Qubit q, std::optional<std::size_t> clbit, bool invert = false, double flip = 0.0);

    // Measurement of the product P = ⊗ Pᵢ: a per-qubit basis change V onto Z, a CNOT chain onto the
    // last qubit, its Z measurement, and the chain and V undone. Exact, since V P V† = Z_last.
    void measurePauli(const std::vector<PauliFactor>& factors, std::optional<std::size_t> clbit, bool invert,
                      double flip = 0.0);

    // exp(∓iπ/4 P) up to global phase (Stim SPP / SPP_DAG): phases P's −1 eigenspace by i (−i).
    void rotatePauli(const std::vector<PauliFactor>& factors, bool dagger);

    void reset(Qubit q, const std::optional<Condition>& condition = std::nullopt);
    void pauliChannel(QubitList targets, std::vector<double> probabilities,
                      const std::optional<Condition>& condition = std::nullopt);
    void kraus(QubitList targets, std::vector<Eigen::MatrixXcd> operators,
               const std::optional<Condition>& condition = std::nullopt);

    std::size_t size() const { return c.ops.size(); }

    private:
    void push(Qputer::Operation op, const std::optional<Condition>& condition);
    void emit(const Gate& g, const std::optional<Condition>& condition);
    void basisChange(const std::vector<PauliFactor>& factors, bool undo);

    ImportedCircuit& c;
};

// ---- Operations read from circuit JSON ----

Qputer::Operation makeOperation(Qputer::OpKind kind, QubitList controls, QubitList targets, std::vector<double> params,
                                Eigen::MatrixXcd matrix, std::vector<Eigen::MatrixXcd> kraus,
                                std::optional<std::size_t> clbit, std::optional<Condition> condition);

// Empty when `op` is valid on nQubits qubits and nClbits clbits, otherwise the reason. Mirrors the
// core's checks so that a bad document fails before anything is allocated.
std::string validateOperation(const Qputer::Operation& op, std::size_t nQubits, std::size_t nClbits);

// ---- Runner support ----

// Clifford operations equal to `op` up to global phase when it is an uncontrolled rotation or phase
// by a multiple of π/2, or a cphase by a multiple of π; nullopt otherwise.
std::optional<std::vector<Qputer::Operation>> cliffordForm(const Qputer::Operation& op);

} // namespace Noether::Interop
