#include "TestSupport.hpp"

#include <QuantumStateMachine.hpp>

#include <array>
#include <stdexcept>
#include <string>

using namespace QputerTest;
using Qputer::Outcome;
using Qputer::QuantumStateMachine;

namespace
{

QuantumStateMachine machineWith(const Eigen::VectorXcd& psi, std::size_t n, std::uint64_t seed = 1)
{
    QuantumStateMachine m{n, 0, seed};
    m.prepare_state(psi);
    return m;
}

// Marginal by definition: one pass over all basis states, gathering the selected bits.
Eigen::VectorXd bruteMarginal(const Eigen::VectorXcd& psi, const QubitList& qubits)
{
    Eigen::VectorXd p = Eigen::VectorXd::Zero(ix(bit(qubits.size())));
    for (Eigen::Index i = 0; i < psi.size(); ++i)
    {
        std::size_t o = 0;
        for (std::size_t j = 0; j < qubits.size(); ++j) o |= ((static_cast<std::size_t>(i) >> qubits[j]) & 1U) << j;
        p[ix(o)] += std::norm(psi[i]);
    }
    return p;
}

// <psi|P|psi> by applying P's factors with the gate kernels: shares no code with the
// readout kernel's bit-mask formulation.
double referencePauli(const Eigen::VectorXcd& psi, std::size_t n, const std::string& paulis, const QubitList& qubits)
{
    QuantumStateVector p = fromAmplitudes(n, psi);
    for (std::size_t k = 0; k < paulis.size(); ++k)
    {
        if (paulis[k] == 'X') QuantumGate::x(p, qubits[k]);
        if (paulis[k] == 'Y') QuantumGate::y(p, qubits[k]);
        if (paulis[k] == 'Z') QuantumGate::z(p, qubits[k]);
    }
    return psi.dot(p.vector()).real(); // Eigen's dot conjugates the left operand
}

} // namespace


TEST(Readout, ProbabilitiesAreSquaredMagnitudes)
{
    std::mt19937_64 rng{201};
    for (const std::size_t n : {1u, 4u, 15u})
    {
        SCOPED_TRACE(std::format("n = {}", n));
        const Eigen::VectorXcd psi = randomAmplitudes(n, rng);
        const QuantumStateMachine m = machineWith(psi, n);
        const Eigen::VectorXd p = m.probabilities();

        ASSERT_EQ(p.size(), psi.size());
        EXPECT_LE((p - psi.cwiseAbs2()).cwiseAbs().maxCoeff(), kTol);
        EXPECT_NEAR(p.sum(), 1.0, 1e-12);
        EXPECT_NEAR(m.probability(3 % psi.size()), std::norm(psi[ix(3 % bit(n))]), kTol);
    }
    EXPECT_THROW(QuantumStateMachine{2}.probability(4), std::out_of_range);
}

TEST(Readout, MarginalsMatchBruteForceForAnyQubitOrder)
{
    std::mt19937_64 rng{202};
    const std::size_t n = 6;
    const Eigen::VectorXcd psi = randomAmplitudes(n, rng);
    const QuantumStateMachine m = machineWith(psi, n);

    for (const QubitList& qs : {QubitList{0}, QubitList{5}, QubitList{3, 1}, QubitList{1, 3},
                                QubitList{5, 0, 2}, QubitList{0, 1, 2, 3, 4, 5}, QubitList{4, 2, 0, 5, 1, 3}})
    {
        SCOPED_TRACE(formatQubits(qs));
        const Eigen::VectorXd p = m.marginal_probabilities(qs);
        EXPECT_LE((p - bruteMarginal(psi, qs)).cwiseAbs().maxCoeff(), kTol);
        EXPECT_NEAR(p.sum(), 1.0, 1e-12);
    }

    // Full register in ascending order is the distribution over basis indices itself.
    EXPECT_LE((m.marginal_probabilities({0, 1, 2, 3, 4, 5}) - m.probabilities()).cwiseAbs().maxCoeff(), kTol);
}

TEST(Readout, MarginalsCoverHistogramAndOutcomeMajorPaths)
{
    // n = 16 crosses the parallel threshold; k = 3 uses per-block histograms, k = 13 sums
    // each outcome independently.
    std::mt19937_64 rng{203};
    const std::size_t n = 16;
    const Eigen::VectorXcd psi = randomAmplitudes(n, rng);
    const QuantumStateMachine m = machineWith(psi, n);

    for (const std::size_t k : {3u, 12u, 13u})
    {
        SCOPED_TRACE(std::format("k = {}", k));
        const QubitList qs = randomQubits(n, k, rng);
        const Eigen::VectorXd p = m.marginal_probabilities(qs);
        EXPECT_LE((p - bruteMarginal(psi, qs)).cwiseAbs().maxCoeff(), 1e-14);
    }
}

TEST(Readout, MarginalRejectsInvalidQubitLists)
{
    const QuantumStateMachine m{3};
    EXPECT_THROW(m.marginal_probabilities({}), std::invalid_argument);
    EXPECT_THROW(m.marginal_probabilities({0, 0}), std::invalid_argument);
    EXPECT_THROW(m.marginal_probabilities({3}), std::out_of_range);
}

TEST(Readout, PauliExpectationsOfKnownStates)
{
    QuantumStateMachine m{2, 0, 1};
    EXPECT_NEAR(m.expectation("Z", {0}), 1.0, kTol);
    EXPECT_NEAR(m.expectation("X", {0}), 0.0, kTol);
    EXPECT_NEAR(m.expectation("", {}), 1.0, kTol);

    m.h(0);  // |+>
    EXPECT_NEAR(m.expectation("X", {0}), 1.0, kTol);
    EXPECT_NEAR(m.expectation("Z", {0}), 0.0, kTol);
    m.s(0);  // |+i>
    EXPECT_NEAR(m.expectation("Y", {0}), 1.0, kTol);

    m.prepare();
    m.h(0).cnot(0, 1);  // (|00> + |11>)/sqrt2
    EXPECT_NEAR(m.expectation("ZZ", {0, 1}), 1.0, kTol);
    EXPECT_NEAR(m.expectation("XX", {0, 1}), 1.0, kTol);
    EXPECT_NEAR(m.expectation("YY", {0, 1}), -1.0, kTol);
    EXPECT_NEAR(m.expectation("ZI", {0, 1}), 0.0, kTol);
}

TEST(Readout, PauliExpectationsMatchGateReference)
{
    std::mt19937_64 rng{204};
    std::uniform_int_distribution<int> letter(0, 3);
    for (const std::size_t n : {3u, 7u, 16u})
    {
        const Eigen::VectorXcd psi = randomAmplitudes(n, rng);
        const QuantumStateMachine m = machineWith(psi, n);
        for (int trial = 0; trial < 12; ++trial)
        {
            const QubitList qs = randomQubits(n, 1 + static_cast<std::size_t>(trial) % n, rng);
            std::string paulis;
            for (std::size_t k = 0; k < qs.size(); ++k) paulis += "IXYZ"[letter(rng)];
            SCOPED_TRACE(std::format("n = {}, P = {} on {}", n, paulis, formatQubits(qs)));
            EXPECT_NEAR(m.expectation(paulis, qs), referencePauli(psi, n, paulis, qs), 1e-12);
        }
    }
}

TEST(Readout, PauliExpectationRejectsMalformedObservables)
{
    const QuantumStateMachine m{3};
    EXPECT_THROW(m.expectation("XZ", {0}), std::invalid_argument);
    EXPECT_THROW(m.expectation("XQ", {0, 1}), std::invalid_argument);
    EXPECT_THROW(m.expectation("xz", {0, 1}), std::invalid_argument);
    EXPECT_THROW(m.expectation("XX", {1, 1}), std::invalid_argument);
    EXPECT_THROW(m.expectation("X", {3}), std::out_of_range);
}

TEST(Readout, SamplingFollowsBornRuleWithoutCollapse)
{
    QuantumStateMachine m{3, 0, 205};
    m.ry(0, 1.2).ry(1, 2.0).cnot(1, 2);
    const Eigen::VectorXcd before = m.state().vector();

    constexpr std::size_t shots = 200000;
    const QubitList qs{0, 1};
    const Eigen::VectorXd p = m.marginal_probabilities(qs);
    const auto counts = m.sample_counts(qs, shots);

    for (Outcome o = 0; o < 4; ++o)
    {
        SCOPED_TRACE(std::format("outcome {}", o));
        const double freq = counts.contains(o) ? static_cast<double>(counts.at(o)) / shots : 0.0;
        EXPECT_TRUE(withinSigmas(freq, p[ix(o)], shots));
    }
    EXPECT_TRUE(statesNear(m.state(), before, 0.0)) << "sampling must not disturb the state";
    EXPECT_TRUE(m.circuit().size() == 3u) << "readout is not recorded";
}

TEST(Readout, SamplingNeverDrawsZeroProbabilityOutcomes)
{
    // 16 qubits -> 2^16 outcomes (per-shot binary search); 18 -> 2^18 (sorted sweep).
    for (const std::size_t n : {16u, 18u})
    {
        SCOPED_TRACE(std::format("n = {}", n));
        QuantumStateMachine m{n, 0, 206};
        m.h(0);
        for (Qubit q = 1; q < n; ++q) m.cnot(q - 1, q);

        QubitList all;
        for (Qubit q = 0; q < n; ++q) all.push_back(q);
        const auto counts = m.sample_counts(all, 20000);
        ASSERT_EQ(counts.size(), 2u);
        EXPECT_TRUE(counts.contains(0));
        EXPECT_TRUE(counts.contains(bit(n) - 1));
        EXPECT_TRUE(withinSigmas(static_cast<double>(counts.at(0)) / 20000, 0.5, 20000));
    }
}

TEST(Readout, SortedSweepSamplingFollowsBornRule)
{
    // 2^17 outcomes forces the sorted-sweep sampler; folding the draws onto two qubits
    // must reproduce that pair's marginal.
    std::mt19937_64 rng{207};
    constexpr std::size_t n = 17, shots = 200000;
    QuantumStateMachine m{n, 0, 207};
    for (Qubit q = 0; q < n; ++q) m.ry(q, 0.3 + 0.1 * static_cast<double>(q));
    for (int k = 0; k < 40; ++k)
    {
        const QubitList q = randomQubits(n, 2, rng);
        m.cnot(q[0], q[1]);
    }

    QubitList all;
    for (Qubit q = 0; q < n; ++q) all.push_back(q);
    const std::vector<Outcome> draws = m.sample(all, shots);

    const QubitList pair{3, 11};
    const Eigen::VectorXd exact = m.marginal_probabilities(pair);
    std::array<std::size_t, 4> folded{};
    for (const Outcome o : draws) ++folded[((o >> 3) & 1U) | (((o >> 11) & 1U) << 1)];
    for (std::size_t v = 0; v < 4; ++v)
    {
        SCOPED_TRACE(std::format("pair outcome {}", v));
        EXPECT_TRUE(withinSigmas(static_cast<double>(folded[v]) / shots, exact[ix(v)], shots));
    }
}
