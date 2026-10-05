#include "TestSupport.hpp"

#include <QuantumStateMachine.hpp>

#include <stdexcept>

using namespace QputerTest;
using Qputer::Backend;
using Qputer::Counts;
using Qputer::OpKind;
using Qputer::Operation;
using Qputer::Outcome;
using Qputer::QuantumStateMachine;

namespace
{

constexpr std::size_t kAboveCutoff = Qputer::kMaxQubits + 1;

// The same machine on each backend.
struct Pair
{
    QuantumStateMachine sv, tab;

    Pair(std::size_t n, std::size_t clbits, std::uint64_t seed)
        : sv{n, clbits, seed, Backend::StateVector}, tab{n, clbits, seed, Backend::Stabilizer} {}

    template <class F>
    void both(F&& f)
    {
        f(sv);
        f(tab);
    }
};

QubitList range(std::size_t n)
{
    QubitList all(n);
    std::iota(all.begin(), all.end(), Qubit{0});
    return all;
}

} // namespace


// ---- Backend selection ----

TEST(Backends, AutoSwitchesToTheTableauAboveTheStateVectorCutoff)
{
    EXPECT_EQ(QuantumStateMachine{3}.backend(), Backend::StateVector);
    EXPECT_EQ(QuantumStateMachine{kAboveCutoff}.backend(), Backend::Stabilizer);
    EXPECT_EQ((QuantumStateMachine{3, 0, 1, Backend::Stabilizer}.backend()), Backend::Stabilizer);
    EXPECT_EQ((QuantumStateMachine{3, 0, 1, Backend::StateVector}.backend()), Backend::StateVector);
    EXPECT_THROW((QuantumStateMachine{kAboveCutoff, 0, 1, Backend::StateVector}), std::length_error);
    EXPECT_THROW((QuantumStateMachine{Qputer::kMaxStabilizerQubits + 1}), std::length_error);
    EXPECT_THROW((QuantumStateMachine{2, 0, 1, static_cast<Backend>(9)}), std::invalid_argument);
}

TEST(Backends, StabilizerSupportListsTheCliffordOperations)
{
    for (const OpKind k : {OpKind::X, OpKind::Y, OpKind::Z, OpKind::H, OpKind::S, OpKind::Sdg, OpKind::SX,
                           OpKind::CNOT, OpKind::CZ, OpKind::Swap, OpKind::Measure, OpKind::Reset,
                           OpKind::PauliChannel})
        EXPECT_TRUE(Qputer::stabilizerSupports(k)) << Qputer::opName(k);
    for (const OpKind k : {OpKind::T, OpKind::Tdg, OpKind::RX, OpKind::RY, OpKind::RZ, OpKind::Phase, OpKind::U3,
                           OpKind::CPhase, OpKind::Toffoli, OpKind::Fredkin, OpKind::MCX, OpKind::MCZ,
                           OpKind::MCPhase, OpKind::Unitary, OpKind::Kraus})
        EXPECT_FALSE(Qputer::stabilizerSupports(k)) << Qputer::opName(k);
}

TEST(Backends, AboveTheCutoffOnlyCliffordOperationsRun)
{
    QuantumStateMachine m{kAboveCutoff, 2, 1};
    m.h(0).cnot(0, 25).sx(3).cz(3, 4).swap(4, 5);
    const auto before = m.stabilizer_state().stabilizers();
    const std::size_t opsBefore = m.circuit().size();

    EXPECT_THROW(m.t(0), std::invalid_argument);
    EXPECT_THROW(m.rx(0, 0.1), std::invalid_argument);
    EXPECT_THROW(m.cphase(0, 1, 0.2), std::invalid_argument);
    EXPECT_THROW(m.toffoli(0, 1, 2), std::invalid_argument);
    EXPECT_THROW(m.mcz({0, 1}), std::invalid_argument);
    EXPECT_THROW(m.unitary({0}, matH()), std::invalid_argument);
    EXPECT_THROW(m.prepare_state(Eigen::VectorXcd::Zero(4)), std::invalid_argument);
    EXPECT_THROW(m.state(), std::logic_error);
    EXPECT_THROW(m.state_vector(), std::length_error);
    EXPECT_THROW(m.probabilities(), std::length_error);

    EXPECT_EQ(m.stabilizer_state().stabilizers(), before);
    EXPECT_EQ(m.circuit().size(), opsBefore);
    EXPECT_THROW(QuantumStateMachine{2}.stabilizer_state(), std::logic_error);
}

TEST(Backends, ThousandQubitGhzRunsOnTheTableau)
{
    constexpr std::size_t n = 1000, shots = 4000;
    QuantumStateMachine m{n, 64, 701};
    ASSERT_EQ(m.backend(), Backend::Stabilizer);
    m.h(0);
    for (Qubit q = 1; q < n; ++q) m.cnot(q - 1, q);

    EXPECT_EQ(m.expectation("ZZ", {0, 999}), 1.0);
    EXPECT_EQ(m.expectation(std::string(n, 'X'), range(n)), 1.0);
    EXPECT_EQ(m.expectation("Z", {500}), 0.0);
    EXPECT_EQ(m.probability(0), 0.5);
    EXPECT_EQ(m.probability(1), 0.0);
    EXPECT_EQ(m.marginal_probabilities({0, 999}), (Eigen::VectorXd(4) << 0.5, 0, 0, 0.5).finished());
    for (const Outcome o : m.sample({0, 500, 999}, 200)) EXPECT_TRUE(o == 0 || o == 7) << o;
    EXPECT_THROW(m.sample(range(65), 1), std::invalid_argument) << "outcomes are 64 bits wide";

    for (std::size_t c = 0; c < 64; ++c) m.measure(c * 15, c); // collapses the live state too
    const Outcome all = m.classical_register();
    EXPECT_TRUE(all == 0 || all == ~Outcome{0});

    const Counts counts = m.run(shots);
    ASSERT_LE(counts.size(), 2u);
    for (const auto& [reg, c] : counts) EXPECT_TRUE(reg == 0 || reg == ~Outcome{0}) << reg;
    EXPECT_TRUE(withinSigmas(frequency(counts, 0, shots), 0.5, shots));
}


// ---- The same circuit on both backends: equal to each other and to the expected answer ----

TEST(BackendComparison, RandomCliffordCircuitsAgreeWithEachOtherAndTheReference)
{
    std::mt19937_64 rng{702};
    for (std::size_t n = 1; n <= 6; ++n)
        for (int trial = 0; trial < 15; ++trial)
        {
            SCOPED_TRACE(std::format("n = {}, trial {}", n, trial));
            const auto ops = randomCliffordCircuit(n, 40, rng);
            Pair p{n, 0, 1};
            for (const auto& op : ops) p.both([&](auto& m) { m.append(op); });

            const Eigen::VectorXcd ref = referenceState(n, ops);
            EXPECT_TRUE(statesNearUpToPhase(p.tab.state_vector(), p.sv.state().vector()));
            EXPECT_TRUE(statesNearUpToPhase(p.tab.state_vector(), ref));
            EXPECT_TRUE(statesNear(p.sv.state(), ref));

            const Eigen::VectorXd probs = ref.cwiseAbs2();
            EXPECT_LE((p.tab.probabilities() - p.sv.probabilities()).cwiseAbs().maxCoeff(), kTol);
            EXPECT_LE((p.tab.probabilities() - probs).cwiseAbs().maxCoeff(), kTol);

            const QubitList sub = randomQubits(n, 1 + static_cast<std::size_t>(trial) % n, rng);
            EXPECT_LE((p.tab.marginal_probabilities(sub) - p.sv.marginal_probabilities(sub)).cwiseAbs().maxCoeff(),
                      kTol);
            for (Outcome o = 0; o < bit(n); ++o)
                EXPECT_NEAR(p.tab.probability(o), probs[ix(o)], kTol) << "index " << o;

            const QubitList all = range(n);
            for (const auto& pauli : allPauliStrings(std::min<std::size_t>(n, 3)))
            {
                const QubitList on(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(pauli.size()));
                const double expected = referenceExpectation(n, ref, pauli, on);
                EXPECT_NEAR(p.sv.expectation(pauli, on), expected, kTol) << pauli;
                EXPECT_EQ(p.tab.expectation(pauli, on), std::round(expected)) << pauli;
            }
        }
}

TEST(BackendComparison, GhzStateAgreesOnEveryReadout)
{
    constexpr std::size_t n = 5, shots = 20000;
    constexpr Outcome ones = 0b11111;
    Pair p{n, n, 703};
    p.both([](auto& m)
    {
        m.h(0);
        for (Qubit q = 1; q < n; ++q) m.cnot(q - 1, q);
    });

    Eigen::VectorXd expected = Eigen::VectorXd::Zero(ix(bit(n)));
    expected[0] = expected[ix(ones)] = 0.5;
    p.both([&](auto& m)
    {
        SCOPED_TRACE(m.backend() == Backend::Stabilizer ? "stabilizer" : "state vector");
        EXPECT_LE((m.probabilities() - expected).cwiseAbs().maxCoeff(), kTol);
        EXPECT_NEAR(m.expectation("ZZ", {1, 4}), 1.0, kTol);
        EXPECT_NEAR(m.expectation("XXXXX", range(n)), 1.0, kTol);
        EXPECT_NEAR(m.expectation("YYXXX", range(n)), -1.0, kTol);
        EXPECT_NEAR(m.expectation("Z", {2}), 0.0, kTol);
    });

    p.both([](auto& m) { m.measure_all(); });
    EXPECT_EQ(p.tab.classical_register(), p.sv.classical_register()) << "same seed, same collapse";
    EXPECT_TRUE(p.sv.classical_register() == 0 || p.sv.classical_register() == ones);

    const Counts a = p.sv.run(shots), b = p.tab.run(shots);
    EXPECT_EQ(b, a) << "same seed, same shots";
    ASSERT_LE(a.size(), 2u);
    for (const auto& [reg, c] : a) EXPECT_TRUE(reg == 0 || reg == ones) << reg;
    EXPECT_TRUE(withinSigmas(frequency(a, 0, shots), 0.5, shots));
}

TEST(BackendComparison, SuperdenseCodingDecodesEveryMessage)
{
    for (Outcome message = 0; message < 4; ++message)
    {
        SCOPED_TRACE(std::format("message {}", message));
        Pair p{2, 2, 704};
        p.both([&](auto& m)
        {
            m.h(0).cnot(0, 1);
            if (message & 2) m.x(0);
            if (message & 1) m.z(0);
            m.cnot(0, 1).h(0);
            m.measure(0, 0);
            m.measure(1, 1);
        });
        EXPECT_EQ(p.sv.classical_register(), message);
        EXPECT_EQ(p.tab.classical_register(), message);
        EXPECT_EQ(p.sv.run(500), (Counts{{message, 500}}));
        EXPECT_EQ(p.tab.run(500), (Counts{{message, 500}}));
    }
}

TEST(BackendComparison, TeleportationWithFeedForwardAgrees)
{
    // Teleport |+i> = S H |0> from qubit 0 to qubit 2, then undo the preparation there: clbit 2
    // reads 0 on every shot while the Bell-measurement bits are uniform.
    constexpr std::size_t shots = 8000;
    Pair p{3, 3, 705};
    p.both([](auto& m)
    {
        m.h(0).s(0);
        m.h(1).cnot(1, 2);
        m.cnot(0, 1).h(0);
        m.measure(0, 0);
        m.measure(1, 1);
        m.when(1).x(2);
        m.when(0).z(2);
        m.sdg(2).h(2);
        m.measure(2, 2);
    });
    ASSERT_FALSE(p.sv.terminal_measurements_only());
    EXPECT_EQ(p.tab.classical_register(), p.sv.classical_register());

    const Counts a = p.sv.run(shots), b = p.tab.run(shots);
    EXPECT_EQ(b, a) << "same seed, same trajectories";
    ASSERT_EQ(a.size(), 4u);
    for (const auto& [reg, c] : a)
    {
        EXPECT_EQ(reg & 0b100, 0u) << "teleported qubit read 1 with corrections " << (reg & 0b11);
        EXPECT_TRUE(withinSigmas(frequency(a, reg, shots), 0.25, shots));
    }
}

TEST(BackendComparison, ResetAndConditionedGatesAgree)
{
    // Bell pair; measure and reset qubit 0, then restore it from the classical bit: qubit 0
    // again equals qubit 1, so all three clbits match on every shot.
    constexpr std::size_t shots = 6000;
    Pair p{2, 3, 706};
    p.both([](auto& m)
    {
        m.h(0).cnot(0, 1);
        m.measure(0, 0);
        m.reset(0);
        m.when(0).x(0);
        m.measure(0, 1);
        m.measure(1, 2);
    });
    EXPECT_EQ(p.tab.classical_register(), p.sv.classical_register());

    const Counts a = p.sv.run(shots), b = p.tab.run(shots);
    EXPECT_EQ(b, a);
    ASSERT_LE(a.size(), 2u);
    for (const auto& [reg, c] : a) EXPECT_TRUE(reg == 0 || reg == 0b111) << reg;
    EXPECT_TRUE(withinSigmas(frequency(a, 0, shots), 0.5, shots));
}

TEST(BackendComparison, LiveMeasurementSequencesAgree)
{
    // Gates interleaved with mid-circuit measurements: both backends draw the same outcomes,
    // and each post-measurement state is the normalized projection of the reference.
    std::mt19937_64 rng{707};
    constexpr std::size_t n = 4;
    for (int trial = 0; trial < 20; ++trial)
    {
        SCOPED_TRACE(std::format("trial {}", trial));
        Pair p{n, 0, static_cast<std::uint64_t>(trial)};
        Eigen::VectorXcd ref = amplitudes(n, {{0, 1.0}});
        for (int round = 0; round < 6; ++round)
        {
            for (const auto& op : randomCliffordCircuit(n, 6, rng))
            {
                p.both([&](auto& m) { m.append(op); });
                ref = cliffordMatrix(n, op) * ref;
            }
            const Qubit q = randomQubits(n, 1, rng)[0];
            const int r = p.sv.measure(q);
            ASSERT_EQ(p.tab.measure(q), r) << "round " << round << ", qubit " << q;

            for (std::size_t i = 0; i < bit(n); ++i)
                if (static_cast<int>((i >> q) & 1U) != r) ref[ix(i)] = 0.0;
            ASSERT_GT(ref.norm(), 0.5) << "outcome " << r << " has probability 0";
            ref.normalize();
            EXPECT_TRUE(statesNearUpToPhase(p.sv.state(), ref));
            EXPECT_TRUE(statesNearUpToPhase(p.tab.state_vector(), ref));
        }
    }
}

TEST(BackendComparison, SamplingAgreesAndFollowsTheBornRule)
{
    std::mt19937_64 rng{708};
    constexpr std::size_t n = 5, shots = 20000;
    const auto ops = randomCliffordCircuit(n, 40, rng);
    const QubitList qubits{3, 0, 4};
    Pair p{n, 3, 708};
    for (const auto& op : ops) p.both([&](auto& m) { m.append(op); });

    const std::vector<Outcome> a = p.sv.sample(qubits, shots), b = p.tab.sample(qubits, shots);
    EXPECT_EQ(b, a) << "same seed, same draws";

    QuantumStateMachine probe{n};
    probe.prepare_state(referenceState(n, ops));
    const Eigen::VectorXd exact = probe.marginal_probabilities(qubits);
    Counts counts;
    for (const Outcome o : a) ++counts[o];
    for (Outcome o = 0; o < 8; ++o)
    {
        SCOPED_TRACE(std::format("outcome {}", o));
        EXPECT_TRUE(withinSigmas(frequency(counts, o, shots), exact[ix(o)], shots));
    }

    // Terminal measurements of the same state take the sampled run() path on both.
    for (std::size_t c = 0; c < qubits.size(); ++c) p.both([&](auto& m) { m.measure(qubits[c], c); });
    ASSERT_TRUE(p.tab.terminal_measurements_only());
    EXPECT_EQ(p.tab.run(shots), p.sv.run(shots));
}

TEST(BackendComparison, TableauRunsAreIndependentOfThreadCount)
{
    auto counts = [](bool midCircuit, int threads)
    {
        return withThreads(threads, [&]
        {
            QuantumStateMachine m{40, 3, 709};
            m.h(0).cnot(0, 39).h(20).s(20);
            if (midCircuit)
            {
                m.measure(0, 0);
                m.when(0).x(20);
            }
            m.cnot(20, 39);
            m.measure(20, 1);
            m.measure(39, 2);
            return m.run(20000);
        });
    };
    for (const bool mid : {false, true})
    {
        SCOPED_TRACE(std::format("mid-circuit = {}", mid));
        const Counts serial = counts(mid, 1);
        EXPECT_EQ(counts(mid, 4), serial);
        EXPECT_EQ(counts(mid, 3), serial);
    }
}
