#include "TestSupport.hpp"

#include <QuantumStateMachine.hpp>

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace QputerTest;
using Qputer::Counts;
using Qputer::Outcome;
using Qputer::QuantumStateMachine;

namespace
{

::testing::AssertionResult withinSigmas(double observed, double p, std::size_t shots, double sigmas = 5.0)
{
    const double sigma = std::sqrt(p * (1.0 - p) / static_cast<double>(shots));
    if (std::abs(observed - p) <= sigmas * sigma + 1e-12) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << std::format("frequency {:.5f} vs p = {:.5f} ({:.1f} sigma)",
                                                        observed, p, std::abs(observed - p) / sigma);
}

double frequency(const Counts& counts, Outcome o, std::size_t shots)
{
    return counts.contains(o) ? static_cast<double>(counts.at(o)) / static_cast<double>(shots) : 0.0;
}

// Runs `body` with the OpenMP team size pinned, restoring the previous setting.
template <class F>
auto withThreads(int threads, F&& body)
{
#ifdef _OPENMP
    const int saved = omp_get_max_threads();
    omp_set_num_threads(threads);
    auto result = body();
    omp_set_num_threads(saved);
    return result;
#else
    (void)threads;
    return body();
#endif
}

} // namespace


// ---- Projective measurement on the live state ----

TEST(Measurement, CollapseIsTheNormalizedProjection)
{
    std::mt19937_64 rng{301};
    for (int trial = 0; trial < 20; ++trial)
    {
        const std::size_t n = 5;
        const Qubit q = static_cast<Qubit>(trial) % n;
        const Eigen::VectorXcd psi = randomAmplitudes(n, rng);

        QuantumStateMachine m{n, 1, static_cast<std::uint64_t>(trial)};
        m.prepare_state(psi);
        const int r = m.measure(q, 0);

        Eigen::VectorXcd expected = psi;
        for (std::size_t i = 0; i < bit(n); ++i)
            if (static_cast<int>((i >> q) & 1U) != r) expected[ix(i)] = 0.0;
        expected.normalize();

        SCOPED_TRACE(std::format("trial {}, qubit {}, outcome {}", trial, q, r));
        EXPECT_TRUE(statesNear(m.state(), expected));
        EXPECT_EQ(m.clbit(0), r == 1);
    }
}

TEST(Measurement, OutcomeFrequenciesFollowBornRule)
{
    constexpr double theta = 1.0;
    constexpr std::size_t trials = 20000;
    const double p1 = std::pow(std::sin(theta / 2), 2);

    QuantumStateMachine m{2, 0, 302};
    std::size_t ones = 0;
    for (std::size_t k = 0; k < trials; ++k)
    {
        m.prepare();
        m.ry(1, theta);
        ones += static_cast<std::size_t>(m.measure(1));
    }
    EXPECT_TRUE(withinSigmas(static_cast<double>(ones) / trials, p1, trials));
}

TEST(Measurement, RepeatedMeasurementIsStable)
{
    QuantumStateMachine m{3, 0, 303};
    m.h(0).h(1).cnot(1, 2);
    const int first = m.measure(1);
    for (int k = 0; k < 10; ++k) EXPECT_EQ(m.measure(1), first);
    EXPECT_EQ(m.measure(2), first) << "qubit 2 is perfectly correlated with qubit 1";
}

TEST(Measurement, MeasureAllCollapsesToRecordedBasisState)
{
    std::mt19937_64 rng{304};
    for (const std::size_t n : {3u, 15u})
    {
        SCOPED_TRACE(std::format("n = {}", n));
        const Eigen::VectorXcd psi = randomAmplitudes(n, rng);
        QuantumStateMachine m{n, n, 304};
        m.prepare_state(psi);
        const Outcome index = m.measure_all();

        ASSERT_LT(index, bit(n));
        EXPECT_EQ(m.classical_register(), index);
        EXPECT_NEAR(m.probability(index), 1.0, kTol);
        EXPECT_NEAR(std::arg(m.state()[index]), std::arg(psi[ix(index)]), 1e-12) << "phase is preserved";
        EXPECT_EQ(m.circuit().size(), n);
        EXPECT_TRUE(m.terminal_measurements_only());
    }
    EXPECT_THROW(QuantumStateMachine(3, 2).measure_all(), std::invalid_argument);
}

TEST(Measurement, ResetReturnsQubitToZero)
{
    std::mt19937_64 rng{305};
    QuantumStateMachine m{4, 0, 305};
    m.prepare_state(randomAmplitudes(4, rng));
    m.reset(2);
    EXPECT_NEAR(m.marginal_probabilities({2})[0], 1.0, kTol);
    EXPECT_NEAR(m.state().norm(), 1.0, kTol);
}


// ---- Circuit execution ----

TEST(Execution, BellStateCountsAreCorrelated)
{
    constexpr std::size_t shots = 20000;
    QuantumStateMachine m{2, 2, 401};
    m.h(0).cnot(0, 1);
    m.measure(0, 0);
    m.measure(1, 1);
    ASSERT_TRUE(m.terminal_measurements_only());

    const Counts counts = m.run(shots);
    EXPECT_EQ(counts.size(), 2u);
    EXPECT_EQ(frequency(counts, 0b00, shots) + frequency(counts, 0b11, shots), 1.0);
    EXPECT_TRUE(withinSigmas(frequency(counts, 0b11, shots), 0.5, shots));
}

TEST(Execution, LargeGhzStateOnlyYieldsAllZerosOrAllOnes)
{
    constexpr std::size_t n = 18;
    QuantumStateMachine m{n, n, 402};
    m.h(0);
    for (Qubit q = 1; q < n; ++q) m.cnot(q - 1, q);
    m.measure_all();

    const Counts counts = m.run(10000);
    ASSERT_LE(counts.size(), 2u);
    for (const auto& [reg, c] : counts) EXPECT_TRUE(reg == 0 || reg == bit(n) - 1) << reg;
    EXPECT_TRUE(withinSigmas(frequency(counts, 0, 10000), 0.5, 10000));
}

TEST(Execution, TeleportationRecoversTheInputState)
{
    // Teleport R_z(phi) R_y(theta)|0> from qubit 0 to qubit 2 via feed-forward corrections,
    // then undo the preparation on qubit 2: it must read 0 on every shot.
    constexpr double theta = 1.1, phi = -0.7;
    QuantumStateMachine m{3, 3, 403};
    m.ry(0, theta).rz(0, phi);
    m.h(1).cnot(1, 2);
    m.cnot(0, 1).h(0);
    m.measure(0, 0);
    m.measure(1, 1);
    m.when(1).x(2);
    m.when(0).z(2);
    m.rz(2, -phi).ry(2, -theta);
    m.measure(2, 2);
    ASSERT_FALSE(m.terminal_measurements_only());

    const Counts counts = m.run(4000);
    std::size_t total = 0;
    for (const auto& [reg, c] : counts)
    {
        EXPECT_EQ(reg & 0b100, 0u) << "teleported qubit read 1 with corrections " << (reg & 0b11);
        total += c;
    }
    EXPECT_EQ(total, 4000u);
    EXPECT_EQ(counts.size(), 4u) << "all four Bell-measurement outcomes occur";
}

TEST(Execution, SampledAndTrajectoryStrategiesAgreeWithExactDistribution)
{
    std::mt19937_64 rng{404};
    constexpr std::size_t n = 6, shots = 40000;
    std::vector<GateOp> gates;
    for (int k = 0; k < 30; ++k) gates.push_back(randomGate(n, rng));

    QuantumStateVector psi{n + 1};
    for (const auto& g : gates) g(psi);

    // The trajectory variant resets the untouched ancilla n: same distribution, but the
    // reset rules out sampling from a single final state.
    auto build = [&](bool forceTrajectories)
    {
        QuantumStateMachine m{n + 1, 3, 404};
        m.prepare_state(psi.vector());
        if (forceTrajectories) m.reset(n);
        m.measure(4, 0);
        m.measure(1, 1);
        m.measure(5, 2);
        return m;
    };

    QuantumStateMachine sampled = build(false), trajectories = build(true);
    ASSERT_TRUE(sampled.terminal_measurements_only());
    ASSERT_FALSE(trajectories.terminal_measurements_only());

    QuantumStateMachine reference{n + 1};
    reference.prepare_state(psi.vector());
    const Eigen::VectorXd exact = reference.marginal_probabilities({4, 1, 5});
    const Counts a = sampled.run(shots), b = trajectories.run(shots);
    for (Outcome o = 0; o < 8; ++o)
    {
        SCOPED_TRACE(std::format("outcome {}", o));
        EXPECT_TRUE(withinSigmas(frequency(a, o, shots), exact[ix(o)], shots));
        EXPECT_TRUE(withinSigmas(frequency(b, o, shots), exact[ix(o)], shots));
    }
}

TEST(Execution, ReplaysFromThePreparedState)
{
    QuantumStateMachine m{3, 3, 405};
    m.prepare_basis(0b101);
    m.measure_all();
    EXPECT_EQ(m.run(100), (Counts{{0b101, 100}}));

    m.prepare_basis(0b001);
    m.x(1);
    m.measure_all();
    EXPECT_EQ(m.run(100), (Counts{{0b011, 100}}));
}

TEST(Execution, LaterMeasurementIntoSameClbitWins)
{
    QuantumStateMachine m{2, 1, 406};
    m.x(1);
    m.measure(0, 0);
    m.measure(1, 0);
    EXPECT_EQ(m.run(50), (Counts{{1, 50}}));
}

TEST(Execution, RunLeavesLiveSystemUntouched)
{
    QuantumStateMachine m{3, 3, 407};
    m.h(0).cnot(0, 1).ry(2, 0.4);
    m.measure(0, 0);
    m.reset(1);
    m.measure(2, 2);
    const Eigen::VectorXcd state = m.state().vector();
    const Outcome reg = m.classical_register();
    const std::size_t size = m.circuit().size();

    m.run(500);
    EXPECT_TRUE(statesNear(m.state(), state, 0.0));
    EXPECT_EQ(m.classical_register(), reg);
    EXPECT_EQ(m.circuit().size(), size);
}

TEST(Execution, RunIsReproducibleAndIndependentOfThreadCount)
{
    auto counts = [](std::size_t n, bool midCircuit, int threads)
    {
        return withThreads(threads, [&]
        {
            QuantumStateMachine m{n, 2, 408};
            m.h(0).ry(1, 0.9).cnot(0, 2);
            if (midCircuit)
            {
                m.measure(0, 0);
                m.when(0).x(1);
            }
            m.cnot(1, 2);
            m.measure(1, 1);
            m.measure(2, 0);
            return m.run(20000);
        });
    };

    // 3 qubits: shot-parallel trajectories / parallel sampling; 15 qubits: serial shots
    // with parallel gate and readout kernels.
    for (const std::size_t n : {3u, 15u})
        for (const bool mid : {false, true})
        {
            if (n == 15 && mid) continue; // 20000 trajectories at 2^15 amplitudes is slow under sanitizers
            SCOPED_TRACE(std::format("n = {}, mid-circuit = {}", n, mid));
            const Counts serial = counts(n, mid, 1);
            EXPECT_EQ(counts(n, mid, 1), serial);
            EXPECT_EQ(counts(n, mid, 4), serial);
        }
}

TEST(Execution, RepeatedRunsAdvanceTheRng)
{
    QuantumStateMachine m{4, 4, 409};
    for (Qubit q = 0; q < 4; ++q) m.h(q);
    m.measure_all();
    EXPECT_NE(m.run(64), m.run(64));
}

TEST(Execution, RejectsUnrunnableCircuits)
{
    QuantumStateMachine m{2, 1, 410};
    m.h(0);
    EXPECT_THROW(m.run(10), std::invalid_argument) << "no measurement into a clbit";
    m.measure(0);
    EXPECT_THROW(m.run(10), std::invalid_argument) << "discarded measurement only";
    m.measure(1, 0);
    EXPECT_THROW(m.run(0), std::invalid_argument);
    EXPECT_NO_THROW(m.run(1));
}

TEST(Execution, GroverFindsTheMarkedItem)
{
    constexpr std::size_t n = 6;
    constexpr Outcome marked = 0b101101;
    QuantumStateMachine m{n, n, 411};
    QubitList all;
    for (Qubit q = 0; q < n; ++q) all.push_back(q);

    auto flipZeros = [&](Outcome pattern)
    {
        for (Qubit q = 0; q < n; ++q)
            if (!((pattern >> q) & 1U)) m.x(q);
    };

    for (Qubit q = 0; q < n; ++q) m.h(q);
    const int iterations = static_cast<int>(std::floor(kPi / 4 * std::sqrt(static_cast<double>(bit(n)))));
    for (int k = 0; k < iterations; ++k)
    {
        flipZeros(marked);
        m.mcz(all);
        flipZeros(marked);
        for (Qubit q = 0; q < n; ++q) m.h(q);
        flipZeros(0);
        m.mcz(all);
        flipZeros(0);
        for (Qubit q = 0; q < n; ++q) m.h(q);
    }

    EXPECT_GT(m.probability(marked), 0.99);
    m.measure_all();
    const Counts counts = m.run(2000);
    EXPECT_GT(frequency(counts, marked, 2000), 0.98);
}

TEST(Execution, BitstringPutsClbitZeroRightmost)
{
    EXPECT_EQ(QuantumStateMachine::bitstring(0b0110, 4), "0110");
    EXPECT_EQ(QuantumStateMachine::bitstring(1, 3), "001");
    EXPECT_EQ(QuantumStateMachine::bitstring(0, 0), "");
    EXPECT_THROW(QuantumStateMachine::bitstring(0, 65), std::invalid_argument);
}

TEST(Execution, ReplayedUnitariesMatchReference)
{
    // Unitaries are validated and laid out once at append time; every replayed shot reuses
    // that prepared form. Cover 1-4 targets, with and without controls, on both strategies.
    std::mt19937_64 rng{412};
    constexpr std::size_t n = 6, ancilla = 5, shots = 40000;
    const std::vector<std::pair<QubitList, QubitList>> layout{
        {{}, {2}}, {{0}, {4, 1}}, {{}, {3, 0, 2}}, {{4}, {1, 3, 0, 2}}, {{1, 3}, {4}}, {{}, {0, 1, 2, 3}}};
    std::vector<Eigen::MatrixXcd> mats;
    for (const auto& [c, t] : layout) mats.push_back(randomUnitary(bit(t.size()), rng));

    Eigen::MatrixXcd reference = Eigen::MatrixXcd::Identity(ix(bit(n)), ix(bit(n)));
    for (std::size_t k = 0; k < layout.size(); ++k)
        reference = embed(n, layout[k].first, layout[k].second, mats[k]) * reference;
    const Eigen::VectorXcd psi0 = randomAmplitudes(n, rng);
    const Eigen::VectorXcd expected = reference * psi0;

    QuantumStateMachine probe{n};
    probe.prepare_state(expected);
    const Eigen::VectorXd exact = probe.marginal_probabilities({0, 2, 4});

    for (const bool forceTrajectories : {false, true})
    {
        SCOPED_TRACE(std::format("trajectories = {}", forceTrajectories));
        QuantumStateMachine m{n, 3, 412};
        m.prepare_state(psi0);
        // Measuring the entangled ancilla leaves the reduced state of qubits 0-4, and so
        // the averaged counts, unchanged; it only rules out single-state sampling.
        if (forceTrajectories) m.reset(ancilla);
        for (std::size_t k = 0; k < layout.size(); ++k)
            m.controlled_unitary(layout[k].first, layout[k].second, mats[k]);

        if (!forceTrajectories)
        {
            EXPECT_TRUE(statesNear(m.state(), expected, 1e-12)) << "live execution";
        }
        m.measure(0, 0);
        m.measure(2, 1);
        m.measure(4, 2);
        ASSERT_EQ(m.terminal_measurements_only(), !forceTrajectories);

        const QuantumStateMachine copy = m; // shares the prepared gates
        QuantumStateMachine copyRunner = copy;
        const Counts counts = m.run(shots);
        EXPECT_EQ(copyRunner.run(shots), counts) << "copies replay identically";
        for (Outcome o = 0; o < 8; ++o)
        {
            SCOPED_TRACE(std::format("outcome {}", o));
            EXPECT_TRUE(withinSigmas(frequency(counts, o, shots), exact[ix(o)], shots));
        }
    }
}
