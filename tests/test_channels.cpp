#include "TestSupport.hpp"

#include <QuantumStateMachine.hpp>

#include <array>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

using namespace QputerTest;
using Qputer::Backend;
using Qputer::Counts;
using Qputer::OpKind;
using Qputer::Operation;
using Qputer::Outcome;
using Qputer::QuantumStateMachine;

namespace
{

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

Operation op(OpKind kind, QubitList controls, QubitList targets, std::vector<double> params = {})
{
    Operation o;
    o.kind = kind;
    o.controls = std::move(controls);
    o.targets = std::move(targets);
    o.params = std::move(params);
    return o;
}

// |1> decays to |0> with probability gamma.
std::vector<Eigen::MatrixXcd> amplitudeDamping(double gamma)
{
    Eigen::MatrixXcd k0 = Eigen::MatrixXcd::Zero(2, 2), k1 = Eigen::MatrixXcd::Zero(2, 2);
    k0(0, 0) = 1.0;
    k0(1, 1) = std::sqrt(1.0 - gamma);
    k1(0, 1) = std::sqrt(gamma);
    return {k0, k1};
}

// `terms` probabilities, all zero except p on term k.
std::vector<double> onlyTerm(std::size_t terms, std::size_t k, double p = 1.0)
{
    std::vector<double> probs(terms, 0.0);
    probs[k] = p;
    return probs;
}

void applyLetter(QuantumStateVector& s, char letter, Qubit q)
{
    if (letter == 'X') QuantumGate::x(s, q);
    if (letter == 'Y') QuantumGate::y(s, q);
    if (letter == 'Z') QuantumGate::z(s, q);
}

void applyLetter(Qputer::StabilizerState& t, char letter, Qubit q)
{
    if (letter == 'X') t.x(q);
    if (letter == 'Y') t.y(q);
    if (letter == 'Z') t.z(q);
}

double bitFrequency(const Counts& counts, std::size_t clbit, std::size_t shots)
{
    std::size_t ones = 0;
    for (const auto& [reg, c] : counts)
        if ((reg >> clbit) & 1U) ones += c;
    return static_cast<double>(ones) / static_cast<double>(shots);
}

} // namespace


// ---- Multi-bit conditions ----

TEST(Conditions, WhenBitsGatesOnSeveralClbitsOnBothBackends)
{
    Pair p{5, 3, 801};
    p.both([](auto& m)
    {
        m.x(0).x(2);
        m.measure(0, 0);
        m.measure(1, 1);
        m.measure(2, 2);
        ASSERT_EQ(m.classical_register(), 0b101u);

        m.when_bits(0b101, 0b101).x(3); // clbits 0, 2 read 1, 1: runs
        m.when_bits(0b011, 0b001).x(1); // clbits 0, 1 read 1, 0: runs
        m.when_bits(0b111, 0b001).x(3); // register 101 != 001: skipped
        m.when_bits(0b110, 0b100).x(4); // clbits 1, 2 read 0, 1: runs
    });
    EXPECT_TRUE(statesNear(p.sv.state(), amplitudes(5, {{0b11111, 1.0}})));
    EXPECT_EQ(p.tab.probability(0b11111), 1.0);

    const auto& c = p.sv.circuit();
    ASSERT_EQ(c.size(), 9u);
    ASSERT_TRUE(c[5].condition.has_value() && c[7].condition.has_value());
    EXPECT_EQ(c[5].condition->mask, 0b101u);
    EXPECT_EQ(c[5].condition->value, 0b101u);
    EXPECT_EQ(c[7].condition->mask, 0b111u);
    EXPECT_EQ(c[7].condition->value, 0b001u);
    EXPECT_EQ(p.tab.circuit()[8].condition->mask, 0b110u);
}

TEST(Conditions, WhenBitsAgreesBetweenLivePassAndRun)
{
    // Uniform clbits 0..2; qubit 3 flips only when they read 101, qubit 4 when clbits 0 and 1
    // read 1 and 0, so clbits 3 and 4 are functions of clbits 0..2 on every shot.
    constexpr std::size_t shots = 8000;
    Pair p{5, 5, 802};
    p.both([](auto& m)
    {
        m.h(0).h(1).h(2);
        m.measure(0, 0);
        m.measure(1, 1);
        m.measure(2, 2);
        m.when_bits(0b111, 0b101).x(3);
        m.when_bits(0b011, 0b001).x(4);
        m.measure(3, 3);
        m.measure(4, 4);
    });
    auto consistent = [](Outcome reg)
    {
        const bool flip3 = (reg & 0b111) == 0b101, flip4 = (reg & 0b011) == 0b001;
        return (((reg >> 3) & 1U) != 0) == flip3 && (((reg >> 4) & 1U) != 0) == flip4;
    };
    EXPECT_TRUE(consistent(p.sv.classical_register())) << p.sv.classical_register();
    EXPECT_EQ(p.tab.classical_register(), p.sv.classical_register());
    ASSERT_FALSE(p.sv.terminal_measurements_only());

    const Counts a = p.sv.run(shots), b = p.tab.run(shots);
    EXPECT_EQ(b, a) << "same seed, same trajectories";
    EXPECT_EQ(a.size(), 8u);
    for (const auto& [reg, n] : a)
    {
        EXPECT_TRUE(consistent(reg)) << reg;
        EXPECT_TRUE(withinSigmas(frequency(a, reg, shots), 0.125, shots));
    }
}

TEST(Conditions, WhenIsTheSingleBitCase)
{
    QuantumStateMachine m{2, 3, 803};
    m.when(2).x(0);
    m.when(1, false).x(1);
    ASSERT_TRUE(m.circuit()[0].condition && m.circuit()[1].condition);
    EXPECT_EQ(m.circuit()[0].condition->mask, 0b100u);
    EXPECT_EQ(m.circuit()[0].condition->value, 0b100u);
    EXPECT_EQ(m.circuit()[1].condition->mask, 0b010u);
    EXPECT_EQ(m.circuit()[1].condition->value, 0u);
    EXPECT_TRUE(statesNear(m.state(), amplitudes(2, {{0b10, 1.0}}), 0.0)); // only the second ran
}

TEST(Conditions, InvalidConditionsThrowBeforeTheStateChanges)
{
    QuantumStateMachine m{2, 3, 804};
    m.h(0);
    const Eigen::VectorXcd before = m.state().vector();
    const std::size_t opsBefore = m.circuit().size();

    Operation conditioned = op(OpKind::X, {}, {1});
    EXPECT_THROW(m.when_bits(0, 0), std::invalid_argument);         // selects no clbit
    EXPECT_THROW(m.when_bits(0b001, 0b010), std::invalid_argument); // value outside the mask
    EXPECT_THROW(m.when_bits(0b1000, 0b1000), std::out_of_range);   // clbit 3 of 3
    EXPECT_THROW(m.when_bits(~Outcome{0}, 0), std::out_of_range);
    conditioned.condition = Qputer::Condition{0b1000, 0};
    EXPECT_THROW(m.append(conditioned), std::out_of_range);
    conditioned.condition = Qputer::Condition{};
    EXPECT_THROW(m.append(conditioned), std::invalid_argument);
    conditioned.condition = Qputer::Condition{0b011, 0b100};
    EXPECT_THROW(m.append(conditioned), std::invalid_argument);
    EXPECT_THROW((QuantumStateMachine{1, 0, 1}.when_bits(1, 1)), std::out_of_range);

    EXPECT_TRUE(statesNear(m.state(), before, 0.0));
    EXPECT_EQ(m.circuit().size(), opsBefore);

    // Nothing above left a pending condition: the next gate runs unconditioned.
    m.x(1);
    EXPECT_FALSE(m.circuit().back().condition.has_value());
    QuantumStateVector expected = fromAmplitudes(2, before);
    QuantumGate::x(expected, 1);
    EXPECT_TRUE(statesNear(m.state(), expected, 0.0));

    m.when_bits(0b11, 0b01);
    EXPECT_THROW(m.measure(0, 0), std::invalid_argument);

    // All 64 clbits are addressable; the register reads 0, so this condition fails.
    QuantumStateMachine wide{1, 64, 805};
    wide.when_bits(~Outcome{0}, Outcome{1} << 63).x(0);
    EXPECT_TRUE(statesNear(wide.state(), amplitudes(1, {{0, 1.0}}), 0.0));
}


// ---- Pauli channels ----

TEST(PauliChannel, CertainPaulisActLikeTheirGates)
{
    Pair p{3, 0, 811};
    p.both([](auto& m)
    {
        m.pauli_channel({0}, {1, 0, 0});            // X
        m.h(1).pauli_channel({1}, {0, 0, 1}).h(1); // H Z H = X
        m.pauli_channel({2}, {0, 1, 0});            // Y |0> = i |1>
    });
    EXPECT_TRUE(statesNear(p.sv.state(), amplitudes(3, {{0b111, kI}})));
    EXPECT_TRUE(statesNearUpToPhase(p.tab.state_vector(), amplitudes(3, {{0b111, 1.0}})));
    EXPECT_EQ(p.sv.circuit()[0].kind, OpKind::PauliChannel);
}

TEST(PauliChannel, TwoQubitTermsFollowTheDocumentedOrder)
{
    constexpr std::array<std::string_view, 15> terms{"IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI",
                                                     "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"};
    std::mt19937_64 rng{812};
    const auto prefix = randomCliffordCircuit(3, 20, rng);
    const Eigen::VectorXcd start = referenceState(3, prefix);

    for (std::size_t k = 0; k < terms.size(); ++k)
    {
        SCOPED_TRACE(std::format("term {} = {}", k, terms[k]));
        Pair p{3, 0, 812};
        for (const auto& o : prefix) p.both([&](auto& m) { m.append(o); });
        p.both([&](auto& m) { m.pauli_channel({2, 0}, onlyTerm(15, k)); }); // first letter on qubit 2

        QuantumStateVector ref = fromAmplitudes(3, start);
        applyLetter(ref, terms[k][0], 2);
        applyLetter(ref, terms[k][1], 0);
        EXPECT_TRUE(statesNear(p.sv.state(), ref));

        Qputer::StabilizerState t{3};
        for (const auto& o : prefix) applyClifford(t, o);
        applyLetter(t, terms[k][0], 2);
        applyLetter(t, terms[k][1], 0);
        EXPECT_EQ(p.tab.stabilizer_state().stabilizers(), t.stabilizers());
    }
}

TEST(PauliChannel, ZeroProbabilitiesLeaveTheStateUnchanged)
{
    std::mt19937_64 rng{813};
    const Eigen::VectorXcd psi = randomAmplitudes(3, rng);
    QuantumStateMachine m{3, 0, 813};
    m.prepare_state(psi);
    m.pauli_channel({1}, {0, 0, 0}).pauli_channel({0, 2}, std::vector<double>(15, 0.0));
    EXPECT_TRUE(statesNear(m.state(), psi, 0.0));
    EXPECT_EQ(m.circuit().size(), 2u);

    QuantumStateMachine t{3, 0, 813, Backend::Stabilizer};
    t.h(0).cnot(0, 2).s(1);
    const auto before = t.stabilizer_state().stabilizers();
    t.pauli_channel({1}, {0, 0, 0}).pauli_channel({2, 1}, std::vector<double>(15, 0.0));
    EXPECT_EQ(t.stabilizer_state().stabilizers(), before);
}

TEST(PauliChannel, FlipFrequenciesMatchTheirProbabilities)
{
    constexpr std::size_t shots = 20000;
    for (const double prob : {0.1, 0.37})
    {
        SCOPED_TRACE(std::format("p = {}", prob));
        Pair p{4, 4, 814};
        p.both([&](auto& m)
        {
            m.pauli_channel({0}, {prob, 0, 0});                    // X flips
            m.pauli_channel({1}, {0, prob, 0});                    // Y flips too
            m.pauli_channel({2}, {0, 0, prob});                    // Z never flips a Z outcome
            m.pauli_channel({3}, {prob / 3, prob / 3, prob / 3}); // depolarizing: 2p/3 flips
            for (std::size_t q = 0; q < 4; ++q) m.measure(q, q);
        });
        ASSERT_FALSE(p.sv.terminal_measurements_only());
        const Counts a = p.sv.run(shots);
        EXPECT_EQ(p.tab.run(shots), a) << "same seed, same trajectories";
        EXPECT_TRUE(withinSigmas(bitFrequency(a, 0, shots), prob, shots));
        EXPECT_TRUE(withinSigmas(bitFrequency(a, 1, shots), prob, shots));
        EXPECT_EQ(bitFrequency(a, 2, shots), 0.0);
        EXPECT_TRUE(withinSigmas(bitFrequency(a, 3, shots), 2.0 * prob / 3.0, shots));
    }

    // Two targets: XX with 0.3 flips both, ZI (Z on targets[0]) never shows.
    std::vector<double> probs(15, 0.0);
    probs[4] = 0.3;  // XX
    probs[11] = 0.2; // ZI
    Pair p{2, 2, 815};
    p.both([&](auto& m)
    {
        m.pauli_channel({1, 0}, probs);
        m.measure(0, 0);
        m.measure(1, 1);
    });
    const Counts a = p.sv.run(shots);
    EXPECT_EQ(p.tab.run(shots), a);
    ASSERT_LE(a.size(), 2u);
    for (const auto& [reg, n] : a) EXPECT_TRUE(reg == 0 || reg == 0b11) << reg;
    EXPECT_TRUE(withinSigmas(frequency(a, 0b11, shots), 0.3, shots));
}

TEST(PauliChannel, ChannelsForceIndependentTrajectories)
{
    QuantumStateMachine clean{2, 2, 816};
    clean.h(0).cnot(0, 1);
    clean.measure(0, 0);
    clean.measure(1, 1);
    EXPECT_TRUE(clean.terminal_measurements_only());

    QuantumStateMachine noisy{2, 2, 816};
    noisy.h(0).pauli_channel({1}, {0.2, 0, 0}).cnot(0, 1);
    noisy.measure(0, 0);
    noisy.measure(1, 1);
    EXPECT_FALSE(noisy.terminal_measurements_only());

    QuantumStateMachine damped{2, 2, 816};
    damped.h(0).kraus({1}, amplitudeDamping(0.2)).cnot(0, 1);
    damped.measure(0, 0);
    damped.measure(1, 1);
    EXPECT_FALSE(damped.terminal_measurements_only());
}

TEST(PauliChannel, SameSeedGivesTheSameOutcomesOnBothBackends)
{
    // Clifford layers interleaved with one- and two-qubit Pauli noise and mid-circuit
    // measurements. Both backends take one draw per channel and per measurement from the same
    // stream, and pick the same term from it, so live registers and run() counts coincide.
    std::mt19937_64 rng{817};
    constexpr std::size_t n = 5, shots = 4000;
    std::vector<double> two(15);
    for (std::size_t k = 0; k < two.size(); ++k) two[k] = 0.004 * static_cast<double>(k + 1);

    for (int trial = 0; trial < 6; ++trial)
    {
        SCOPED_TRACE(std::format("trial {}", trial));
        Pair p{n, n, 817 + static_cast<std::uint64_t>(trial)};
        for (int layer = 0; layer < 4; ++layer)
        {
            for (const auto& o : randomCliffordCircuit(n, 8, rng)) p.both([&](auto& m) { m.append(o); });
            const QubitList one = randomQubits(n, 1, rng), pair = randomQubits(n, 2, rng);
            const Qubit probe = randomQubits(n, 1, rng)[0];
            p.both([&](auto& m)
            {
                m.pauli_channel(one, {0.05, 0.1, 0.15});
                m.pauli_channel(pair, two);
                m.measure(probe, probe);
            });
            ASSERT_EQ(p.tab.classical_register(), p.sv.classical_register()) << "layer " << layer;
        }
        for (Qubit q = 0; q < n; ++q) p.both([&](auto& m) { m.measure(q, q); });
        EXPECT_EQ(p.tab.classical_register(), p.sv.classical_register());
        EXPECT_EQ(p.tab.run(shots), p.sv.run(shots));
    }
}

TEST(PauliChannel, ConditionedChannelsRunOnlyWhenTheConditionHolds)
{
    Pair p{2, 1, 818};
    p.both([](auto& m)
    {
        m.measure(0, 0);                                 // reads 0
        m.when(0).pauli_channel({1}, {1, 0, 0});         // skipped
        m.when(0, false).pauli_channel({0}, {1, 0, 0});  // runs
    });
    EXPECT_TRUE(statesNear(p.sv.state(), amplitudes(2, {{0b01, 1.0}})));
    EXPECT_EQ(p.tab.probability(0b01), 1.0);
}

TEST(PauliChannel, InvalidChannelsThrowBeforeTheStateChanges)
{
    QuantumStateMachine m{3, 1, 819};
    m.h(0).cnot(0, 2);
    const Eigen::VectorXcd before = m.state().vector();
    const std::size_t opsBefore = m.circuit().size();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    Operation withClbit = op(OpKind::PauliChannel, {}, {0}, {0.1, 0, 0});
    withClbit.clbit = 0;
    EXPECT_THROW(m.pauli_channel({0, 1, 2}, std::vector<double>(63, 0.0)), std::invalid_argument);
    EXPECT_THROW(m.pauli_channel({0}, {0.1, 0.1}), std::invalid_argument);
    EXPECT_THROW(m.pauli_channel({0, 1}, {0.1, 0.1, 0.1}), std::invalid_argument);
    EXPECT_THROW(m.pauli_channel({0}, {-0.1, 0, 0}), std::invalid_argument);
    EXPECT_THROW(m.pauli_channel({0}, {1.5, 0, 0}), std::invalid_argument);
    EXPECT_THROW(m.pauli_channel({0}, {0.5, 0.3, 0.3}), std::invalid_argument); // sums to 1.1
    EXPECT_THROW(m.pauli_channel({0}, {nan, 0, 0}), std::invalid_argument);
    EXPECT_THROW(m.pauli_channel({1, 1}, std::vector<double>(15, 0.0)), std::invalid_argument);
    EXPECT_THROW(m.pauli_channel({3}, {0.1, 0, 0}), std::out_of_range);
    EXPECT_THROW(m.pauli_channel({}, {}), std::invalid_argument);
    EXPECT_THROW(m.append(withClbit), std::invalid_argument);
    EXPECT_THROW(m.append(op(OpKind::PauliChannel, {1}, {0}, {0.1, 0, 0})), std::invalid_argument);

    EXPECT_TRUE(statesNear(m.state(), before, 0.0));
    EXPECT_EQ(m.circuit().size(), opsBefore);

    // A total within rounding of 1 is a valid channel.
    EXPECT_NO_THROW(m.pauli_channel({0}, {1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}));
    EXPECT_NO_THROW(m.pauli_channel({1}, {0.1, 0.2, 0.7}));
}


// ---- Kraus channels ----

TEST(Kraus, FullAmplitudeDampingSendsOneToZero)
{
    QuantumStateMachine m{2, 2, 821};
    m.x(0).x(1);
    m.kraus({0}, amplitudeDamping(1.0));
    EXPECT_TRUE(statesNear(m.state(), amplitudes(2, {{0b10, 1.0}}), 0.0));
    m.kraus({1}, amplitudeDamping(0.0)); // gamma = 0 is the identity
    EXPECT_TRUE(statesNear(m.state(), amplitudes(2, {{0b10, 1.0}}), 0.0));
    EXPECT_EQ(m.circuit().back().kind, OpKind::Kraus);
    EXPECT_EQ(m.circuit().back().kraus.size(), 2u);

    m.measure(0, 0);
    m.measure(1, 1);
    EXPECT_EQ(m.classical_register(), 0b10u);
    EXPECT_EQ(m.run(500), (Counts{{0b10, 500}}));
}

TEST(Kraus, BranchFrequenciesFollowTheBornRule)
{
    constexpr std::size_t shots = 20000;
    const double theta = 1.1, gamma = 0.3, dephase = 0.25;
    const double s2 = std::sin(theta / 2) * std::sin(theta / 2);

    Eigen::MatrixXcd z = Eigen::MatrixXcd::Identity(2, 2);
    z(1, 1) = -1.0;
    const std::vector<Eigen::MatrixXcd> dephasing{std::sqrt(1.0 - dephase) * Eigen::MatrixXcd::Identity(2, 2),
                                                   std::sqrt(dephase) * z};

    QuantumStateMachine m{3, 3, 822};
    m.x(0).kraus({0}, amplitudeDamping(gamma));           // |1>: stays with 1 - gamma
    m.ry(1, theta).kraus({1}, amplitudeDamping(gamma));   // state-dependent branch weights
    m.h(2).kraus({2}, dephasing).h(2);                    // |+> -> |-> with probability dephase
    for (std::size_t q = 0; q < 3; ++q) m.measure(q, q);

    const Counts c = m.run(shots);
    EXPECT_TRUE(withinSigmas(bitFrequency(c, 0, shots), 1.0 - gamma, shots));
    EXPECT_TRUE(withinSigmas(bitFrequency(c, 1, shots), (1.0 - gamma) * s2, shots));
    EXPECT_TRUE(withinSigmas(bitFrequency(c, 2, shots), dephase, shots));
}

TEST(Kraus, TargetsFollowTheUnitaryConvention)
{
    // A single unitary operator is that gate: targets[0] is the MSB of its index.
    std::mt19937_64 rng{823};
    const Eigen::MatrixXcd U = randomUnitary(4, rng);
    const Eigen::VectorXcd psi = randomAmplitudes(4, rng);
    QuantumStateMachine a{4, 0, 1}, b{4, 0, 1};
    a.prepare_state(psi);
    b.prepare_state(psi);
    a.kraus({3, 1}, {U});
    b.unitary({3, 1}, U);
    EXPECT_TRUE(statesNear(a.state(), b.state()));

    // X (x) I on targets {1, 0} flips qubit 1 with probability 1/2 and never qubit 0.
    constexpr std::size_t shots = 20000;
    const Eigen::MatrixXcd xi = kron(matX(), matI());
    QuantumStateMachine m{2, 2, 824};
    m.kraus({1, 0}, {std::sqrt(0.5) * xi, std::sqrt(0.5) * Eigen::MatrixXcd::Identity(4, 4)});
    m.measure(0, 0);
    m.measure(1, 1);
    const Counts c = m.run(shots);
    EXPECT_EQ(bitFrequency(c, 0, shots), 0.0);
    EXPECT_TRUE(withinSigmas(bitFrequency(c, 1, shots), 0.5, shots));
}

TEST(Kraus, PostBranchStateIsTheNormalizedImage)
{
    // Projective Kraus pair {|0><0|, |1><1|} on qubit 0 of a random state: the branch keeps
    // the matching half, renormalized.
    std::mt19937_64 rng{825};
    const Eigen::VectorXcd psi = randomAmplitudes(3, rng);
    Eigen::MatrixXcd p0 = Eigen::MatrixXcd::Zero(2, 2), p1 = Eigen::MatrixXcd::Zero(2, 2);
    p0(0, 0) = 1.0;
    p1(1, 1) = 1.0;
    QuantumStateMachine m{3, 0, 825};
    m.prepare_state(psi);
    m.kraus({0}, {p0, p1});

    const QuantumStateVector& out = m.state();
    const std::size_t branch = std::norm(out[1]) > 0.0 ? 1 : 0;
    Eigen::VectorXcd expected = psi;
    for (std::size_t i = 0; i < 8; ++i)
        if ((i & 1U) != branch) expected[ix(i)] = 0.0;
    expected.normalize();
    EXPECT_TRUE(statesNear(out, expected));

    // On targets {2, 0}, diag(1, 1, 0, 0) projects targets[0] = qubit 2 onto |0>.
    Eigen::MatrixXcd low = Eigen::MatrixXcd::Zero(4, 4), high = Eigen::MatrixXcd::Zero(4, 4);
    low(0, 0) = low(1, 1) = 1.0;
    high(2, 2) = high(3, 3) = 1.0;
    QuantumStateMachine two{3, 0, 826};
    two.prepare_state(psi);
    two.kraus({2, 0}, {low, high});
    const std::size_t top = std::norm(two.state()[4]) > 0.0 ? 1 : 0;
    Eigen::VectorXcd projected = psi;
    for (std::size_t i = 0; i < 8; ++i)
        if (((i >> 2) & 1U) != top) projected[ix(i)] = 0.0;
    projected.normalize();
    EXPECT_TRUE(statesNear(two.state(), projected));
}

TEST(Kraus, InvalidChannelsThrowBeforeTheStateChanges)
{
    QuantumStateMachine m{3, 1, 826};
    m.h(0);
    const Eigen::VectorXcd before = m.state().vector();
    const std::size_t opsBefore = m.circuit().size();
    const auto damping = amplitudeDamping(0.4);

    Eigen::MatrixXcd nan = damping[0];
    nan(0, 0) = std::numeric_limits<double>::quiet_NaN();
    Operation withClbit = op(OpKind::Kraus, {}, {0});
    withClbit.kraus = damping;
    withClbit.clbit = 0;
    Operation withControl = op(OpKind::Kraus, {1}, {0});
    withControl.kraus = damping;

    EXPECT_THROW(m.kraus({0}, {damping[0]}), std::invalid_argument); // sum K^dag K != I
    EXPECT_THROW(m.kraus({0}, {}), std::invalid_argument);
    EXPECT_THROW(m.kraus({0, 1}, damping), std::invalid_argument);   // 2x2 for two targets
    EXPECT_THROW(m.kraus({0}, {nan, damping[1]}), std::invalid_argument);
    EXPECT_THROW(m.kraus({0, 0}, {Eigen::MatrixXcd::Identity(4, 4)}), std::invalid_argument);
    EXPECT_THROW(m.kraus({5}, damping), std::out_of_range);
    EXPECT_THROW(m.kraus({}, damping), std::invalid_argument);
    EXPECT_THROW(m.append(withClbit), std::invalid_argument);
    EXPECT_THROW(m.append(withControl), std::invalid_argument);

    EXPECT_TRUE(statesNear(m.state(), before, 0.0));
    EXPECT_EQ(m.circuit().size(), opsBefore);

    QuantumStateMachine big{11, 0, 1};
    QubitList all(11);
    std::iota(all.begin(), all.end(), Qubit{0});
    EXPECT_THROW(big.kraus(all, {Eigen::MatrixXcd::Identity(2, 2)}), std::invalid_argument);

    QuantumStateMachine tab{2, 0, 1, Backend::Stabilizer};
    EXPECT_THROW(tab.kraus({0}, damping), std::invalid_argument);
    EXPECT_TRUE(tab.circuit().empty());
}

TEST(Kraus, ResultsAreIndependentOfThreadCount)
{
    // 14 qubits: the branch weights, the dense apply and the rescale all split across threads
    // on the live pass; run() splits shots. Each must give identical results.
    constexpr std::size_t n = 14;
    std::mt19937_64 rng{827};
    const std::vector<Eigen::MatrixXcd> mixed{std::sqrt(0.5) * Eigen::MatrixXcd::Identity(8, 8),
                                              std::sqrt(0.5) * randomUnitary(8, rng)};
    const std::vector<double> probs(15, 0.01);
    auto simulate = [&](int threads)
    {
        return withThreads(threads, [&]
        {
            QuantumStateMachine m{n, 3, 827};
            for (Qubit q = 0; q < n; q += 3) m.ry(q, 0.3 + 0.1 * static_cast<double>(q));
            m.cnot(0, 13).kraus({3}, amplitudeDamping(0.4)).kraus({13, 6}, {kron(matH(), matI())});
            m.kraus({2, 7, 11}, mixed);
            m.pauli_channel({9, 12}, probs);
            m.measure(3, 0);
            m.measure(13, 1);
            m.measure(6, 2);
            return std::pair{m.state().vector(), m.run(40)};
        });
    };
    const auto serial = simulate(1);
    for (const int threads : {3, 8})
    {
        SCOPED_TRACE(std::format("threads = {}", threads));
        const auto parallel = simulate(threads);
        ASSERT_EQ(parallel.first.size(), serial.first.size());
        EXPECT_EQ(std::memcmp(parallel.first.data(), serial.first.data(),
                              static_cast<std::size_t>(serial.first.size()) * sizeof(cd)), 0);
        EXPECT_EQ(parallel.second, serial.second);
    }
}
