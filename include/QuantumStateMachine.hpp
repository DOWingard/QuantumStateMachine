#pragma once

#include <QuantumGates.hpp>
#include <QuantumState.hpp>
#include <StabilizerState.hpp>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include <Eigen/Core>



namespace Qputer
{

namespace detail { struct DenseGate; }

using Outcome = std::uint64_t;
using Counts = std::map<Outcome, std::size_t>; // classical register value -> occurrences


// xoshiro256** (Blackman & Vigna). 32 bytes of state and O(1) seeding let every shot own
// an independent stream derived from (seed, shot), so results do not depend on how shots
// are spread across threads. Satisfies UniformRandomBitGenerator.
class Rng
{
    public:
    using result_type = std::uint64_t;

    explicit Rng(std::uint64_t seed, std::uint64_t stream = 0) noexcept;

    static constexpr result_type min() noexcept { return 0; }
    static constexpr result_type max() noexcept { return ~result_type{0}; }

    result_type operator()() noexcept
    {
        const std::uint64_t result = std::rotl(s[1] * 5, 7) * 9;
        const std::uint64_t t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = std::rotl(s[3], 45);
        return result;
    }

    // Uniform on [0, 1) with 53 random bits.
    double uniform() noexcept { return static_cast<double>((*this)() >> 11) * 0x1.0p-53; }

    private:
    std::array<std::uint64_t, 4> s{};
};


enum class OpKind : std::uint8_t
{
    X, Y, Z, H, S, Sdg, T, Tdg, SX, RX, RY, RZ, Phase, U3,  // 1 qubit
    CNOT, CZ, CPhase, Swap,                                // 2 qubit
    Toffoli, Fredkin,                                      // 3 qubit
    MCX, MCZ, MCPhase,                                     // N qubit
    Unitary,                                               // dense, optionally controlled
    Measure, Reset,                                        // non-unitary
};

// Lowercase names matching the QuantumStateMachine method of the same operation.
std::string_view opName(OpKind kind);
std::optional<OpKind> opKindFromName(std::string_view name);

// True for the operations the stabilizer backend runs: the Clifford gates x y z h s sdg sx
// cnot cz swap, plus measure and reset.
bool stabilizerSupports(OpKind kind);


enum class Backend : std::uint8_t
{
    Auto,        // StateVector for N <= kMaxQubits, Stabilizer above
    StateVector, // 2^N amplitudes, every operation, N <= kMaxQubits
    Stabilizer,  // tableau, stabilizerSupports() operations only, N <= kMaxStabilizerQubits
};

// Classical feed-forward: the operation runs only while clbit holds `value`.
struct Condition
{
    std::size_t clbit = 0;
    bool value = true;
};

// One instruction of the recorded circuit. Qubit roles by kind:
//   controls: cnot/cz/cphase {c}, toffoli {c0, c1}, fredkin {c}, mcx {c...}, unitary {c...}
//   targets:  1-qubit gates, measure, reset {q}; cnot/cz/cphase/toffoli/mcx {t}; swap {a, b};
//             fredkin {a, b}; mcz/mcphase {all qubits}; unitary {t...}, targets[0] = MSB of U's index
// params: rx/ry/rz/phase {angle}, cphase/mcphase {lambda}, u3 {theta, phi, lambda}.
struct Operation
{
    OpKind kind = OpKind::X;
    QubitList controls;
    QubitList targets;
    std::vector<double> params;
    Eigen::MatrixXcd matrix;            // unitary only
    std::optional<std::size_t> clbit;   // measure only; nullopt discards the outcome
    std::optional<Condition> condition; // any kind except measure
};


// Stateful interface to one simulated quantum system: an N-qubit register, a classical
// register of up to 64 bits, a seeded RNG, and the circuit applied since the last
// preparation.
//
// The register is a state vector (any operation, 16 * 2^N bytes) or a stabilizer tableau
// (Clifford gates, measure and reset; O(N) per gate, O(N^2 / 64) per measurement, N^2 / 2
// bytes). Backend::Auto takes the state vector up to kMaxQubits and the tableau above it;
// either can be requested explicitly. Every readout and run() reports the same results on
// both, and for a Clifford circuit the same seed gives the same outcomes on both: random
// choices select an outcome by rank in index order from the same draws (barring
// floating-point ties at a state-vector CDF boundary).
//
// Execution is eager: every operation is validated, applied in place to the live state,
// then recorded. Readout never modifies the live state. run(shots) replays the recorded
// circuit from the recorded preparation on scratch registers, leaving the live system as is.
//
// Invariant: a live state vector is normalized to rounding error (preparation validates it,
// gates are unitary, measurement renormalizes), so readout needs no normalization pass.
//
// Conventions: qubit q is bit q of an amplitude index. A readout over (q_0, ..., q_{k-1})
// reports outcome index o with bit j = value of q_j, so readout over (0, ..., N-1) is the
// basis index itself. Classical bit c is bit c of classical_register() and of Counts keys.
//
// Invalid input throws before the live state, classical register or circuit change.
class QuantumStateMachine
{
    public:
    static constexpr std::size_t kMaxClbits = 64;

    // max |<psi|psi> - 1| accepted by prepare_state.
    static constexpr double kNormTolerance = 1e-10;

    // Starts in |0...0> with all clbits 0. Without a seed, one is drawn from std::random_device
    // (seed() reports it, so any run can be reproduced).
    explicit QuantumStateMachine(std::size_t numQubits, std::size_t numClbits = 0,
                                 std::optional<std::uint64_t> seed = std::nullopt,
                                 Backend backend = Backend::Auto);


    // ---- System ----
    std::size_t num_qubits() const { return nQubits; }
    std::size_t num_clbits() const { return nClbits; }
    std::uint64_t seed() const { return seedValue; }
    void reseed(std::uint64_t seed);
    Backend backend() const; // StateVector or Stabilizer, never Auto

    // The live register, for the backend that holds it; std::logic_error on the other one.
    const QuantumStateVector& state() const;
    const StabilizerState& stabilizer_state() const;

    // Amplitudes on either backend: a copy, or the tableau converted (N <= kMaxQubits, global
    // phase as in StabilizerState::to_state_vector).
    QuantumStateVector state_vector() const;

    Outcome classical_register() const { return creg; }
    bool clbit(std::size_t index) const;
    const std::vector<Operation>& circuit() const { return ops; }


    // ---- Preparation: each clears the circuit and classical register and becomes the
    //      starting point for run() ----
    void prepare();                                         // |0...0>
    void prepare_basis(Outcome index);                      // |index>; qubits 64 and above start in |0>
    void prepare_state(const Eigen::VectorXcd& amplitudes); // state vector only: size 2^N, normalized;
                                                            // keeps a copy for replay


    // ---- Gates ----
    QuantumStateMachine& x(Qubit target);
    QuantumStateMachine& y(Qubit target);
    QuantumStateMachine& z(Qubit target);
    QuantumStateMachine& h(Qubit target);
    QuantumStateMachine& s(Qubit target);
    QuantumStateMachine& sdg(Qubit target);
    QuantumStateMachine& t(Qubit target);
    QuantumStateMachine& tdg(Qubit target);
    QuantumStateMachine& sx(Qubit target);
    QuantumStateMachine& rx(Qubit target, double theta);
    QuantumStateMachine& ry(Qubit target, double theta);
    QuantumStateMachine& rz(Qubit target, double theta);
    QuantumStateMachine& phase(Qubit target, double lambda);
    QuantumStateMachine& u3(Qubit target, double theta, double phi, double lambda);

    QuantumStateMachine& cnot(Qubit control, Qubit target);
    QuantumStateMachine& cz(Qubit control, Qubit target);
    QuantumStateMachine& cphase(Qubit control, Qubit target, double lambda);
    QuantumStateMachine& swap(Qubit a, Qubit b);

    QuantumStateMachine& toffoli(Qubit control0, Qubit control1, Qubit target);
    QuantumStateMachine& fredkin(Qubit control, Qubit a, Qubit b);

    QuantumStateMachine& mcx(const QubitList& controls, Qubit target);
    QuantumStateMachine& mcz(const QubitList& qubits);
    QuantumStateMachine& mcphase(const QubitList& qubits, double lambda);

    QuantumStateMachine& unitary(const QubitList& targets, const Eigen::MatrixXcd& U);
    QuantumStateMachine& controlled_unitary(const QubitList& controls, const QubitList& targets,
                                            const Eigen::MatrixXcd& U);

    // Generic entry point (e.g. for circuits deserialized from JSON); same validation.
    QuantumStateMachine& append(Operation op);

    // Classical feed-forward for the NEXT appended gate or reset: it runs only if `clbit`
    // currently holds `value`, both now and on every replayed shot.
    QuantumStateMachine& when(std::size_t clbit, bool value = true);


    // ---- Measurement: collapses the live state and is recorded ----
    int measure(Qubit target, std::optional<std::size_t> clbit = std::nullopt);
    Outcome measure_all();            // qubit q -> clbit q; needs num_clbits() >= num_qubits()
    QuantumStateMachine& reset(Qubit target); // measure, then flip to |0> if needed


    // ---- Readout: non-destructive, not recorded ----
    // Vector results hold 2^k entries, so they need k <= kMaxQubits on either backend.
    Eigen::VectorXd probabilities() const;                              // |a_i|^2, size 2^N
    double probability(Outcome basisIndex) const;                       // qubits >= 64 read as |0>
    Eigen::VectorXd marginal_probabilities(const QubitList& qubits) const; // size 2^k

    // <psi|P|psi> for P = paulis[0] on qubits[0] (x) paulis[1] on qubits[1] ..., letters IXYZ.
    // Exactly -1, 0 or +1 on the stabilizer backend.
    double expectation(std::string_view paulis, const QubitList& qubits) const;

    // Born-rule draws over at most 64 `qubits` without collapse; advances the RNG.
    std::vector<Outcome> sample(const QubitList& qubits, std::size_t shots);
    Counts sample_counts(const QubitList& qubits, std::size_t shots);


    // ---- Circuit execution ----

    // Replays the circuit `shots` times from the recorded preparation and tallies the
    // classical register. When every measurement is terminal (no reset, no condition, no
    // later gate on a measured qubit) the state is simulated once and the shots are
    // sampled from it; otherwise each shot is an independent trajectory.
    Counts run(std::size_t shots);
    bool terminal_measurements_only() const;

    // Classical register value as text, clbit 0 rightmost.
    static std::string bitstring(Outcome value, std::size_t width);


    private:
    using Register = std::variant<QuantumStateVector, StabilizerState>;

    void validate(const Operation& op) const;
    int commit(Operation op);
    Register blankRegister() const; // |0...0> on the live backend
    void loadPreparation(QuantumStateVector& target) const;
    void loadPreparation(StabilizerState& target) const;
    void clearCircuit();
    void runSampled(std::uint64_t base, std::vector<Outcome>& outcomes) const;
    void runTrajectories(std::uint64_t base, std::vector<Outcome>& outcomes) const;

    std::size_t nQubits;
    std::size_t nClbits;
    Register live;
    Outcome creg = 0;
    std::vector<Operation> ops;
    // Aligned with ops: the replay-ready form of each validated unitary, null for every
    // other kind, so replays skip re-validation and re-layout. Immutable, hence shared
    // between copies of the machine.
    std::vector<std::shared_ptr<const detail::DenseGate>> dense;
    std::optional<Condition> pending;

    Outcome prepIndex = 0;                         // replay start when prepAmplitudes is empty
    std::optional<QuantumStateVector> prepAmplitudes; // state-vector backend only

    std::uint64_t seedValue;
    Rng rng;
};

} // namespace Qputer
