#include "TestSupport.hpp"

#include <QuantumStateMachine.hpp>

#include <cstring>
#include <numeric>
#include <stdexcept>

using namespace QputerTest;
using Qputer::Backend;
using Qputer::QuantumStateMachine;

namespace
{

struct Pair
{
    QuantumStateMachine sv, tab;

    Pair(std::size_t n, std::uint64_t seed)
        : sv{n, 0, seed, Backend::StateVector}, tab{n, 0, seed, Backend::Stabilizer} {}

    template <class F>
    void both(F&& f)
    {
        f(sv);
        f(tab);
    }
};

QubitList range(std::size_t begin, std::size_t end)
{
    QubitList qs(end - begin);
    std::iota(qs.begin(), qs.end(), begin);
    return qs;
}

QubitList complement(std::size_t n, const QubitList& qubits)
{
    QubitList rest;
    for (Qubit q = 0; q < n; ++q)
        if (std::ranges::find(qubits, q) == qubits.end()) rest.push_back(q);
    return rest;
}

// rho_rc = sum over basis pairs (i, j) that agree off `qubits`, by brute force over all 4^n
// pairs; local index bit k-1-j is qubit qubits[j].
Eigen::MatrixXcd referenceDensityMatrix(std::size_t n, const Eigen::VectorXcd& psi, const QubitList& qubits)
{
    const std::size_t k = qubits.size();
    std::size_t mask = 0;
    for (const Qubit q : qubits) mask |= bit(q);
    auto local = [&](std::size_t i)
    {
        std::size_t r = 0;
        for (std::size_t j = 0; j < k; ++j) r |= ((i >> qubits[j]) & 1U) << (k - 1 - j);
        return r;
    };
    Eigen::MatrixXcd rho = Eigen::MatrixXcd::Zero(ix(bit(k)), ix(bit(k)));
    for (std::size_t i = 0; i < bit(n); ++i)
        for (std::size_t j = 0; j < bit(n); ++j)
            if ((i & ~mask) == (j & ~mask)) rho(ix(local(i)), ix(local(j))) += psi[ix(i)] * std::conj(psi[ix(j)]);
    return rho;
}

double referenceEntropy(const Eigen::MatrixXcd& rho)
{
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eig(rho, Eigen::EigenvaluesOnly);
    double s = 0.0;
    for (const double l : eig.eigenvalues())
        if (l > 1e-15) s -= l * std::log2(l);
    return s;
}

double binaryEntropy(double p) { return -p * std::log2(p) - (1.0 - p) * std::log2(1.0 - p); }

[[maybe_unused]] bool bitwiseEqual(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b) // OpenMP builds only
{
    return a.rows() == b.rows() && a.cols() == b.cols()
        && std::memcmp(a.data(), b.data(), static_cast<std::size_t>(a.size()) * sizeof(cd)) == 0;
}

} // namespace


// ---- Entropy ----

TEST(Entropy, BellPairCarriesOneBitOnBothBackends)
{
    Pair p{2, 901};
    p.both([](auto& m) { m.h(0).cnot(0, 1); });
    p.both([](auto& m)
    {
        SCOPED_TRACE(m.backend() == Backend::Stabilizer ? "stabilizer" : "state vector");
        EXPECT_NEAR(m.entropy({0}), 1.0, kTol);
        EXPECT_NEAR(m.entropy({1}), 1.0, kTol);
        EXPECT_EQ(m.entropy({}), 0.0);
        EXPECT_EQ(m.entropy({1, 0}), 0.0);
    });
    EXPECT_EQ(p.tab.entropy({0}), 1.0);
}

TEST(Entropy, ProductStatesCarryNone)
{
    Pair p{4, 902};
    p.both([](auto& m) { m.h(0).s(0).x(1).h(2).sx(3); });
    for (const QubitList& sub : {QubitList{0}, QubitList{1, 2}, QubitList{3, 0, 2}})
    {
        SCOPED_TRACE(formatQubits(sub));
        EXPECT_NEAR(p.sv.entropy(sub), 0.0, kTol);
        EXPECT_EQ(p.tab.entropy(sub), 0.0);
    }
}

TEST(Entropy, GhzCutsCarryOneBit)
{
    for (const std::size_t n : {3, 6, 9, 12})
    {
        SCOPED_TRACE(std::format("n = {}", n));
        Pair p{n, 903};
        p.both([&](auto& m)
        {
            m.h(0);
            for (Qubit q = 1; q < n; ++q) m.cnot(q - 1, q);
        });
        QubitList evens;
        for (Qubit q = 0; q < n; q += 2) evens.push_back(q);
        for (const QubitList& sub : {range(0, n / 2), range(n / 2, n), evens, QubitList{n - 1}})
        {
            EXPECT_NEAR(p.sv.entropy(sub), 1.0, 1e-12) << formatQubits(sub);
            EXPECT_EQ(p.tab.entropy(sub), 1.0) << formatQubits(sub);
        }
    }
}

TEST(Entropy, PartiallyEntangledPairMatchesTheBinaryEntropy)
{
    for (const double theta : {0.3, 1.0, 2.2})
    {
        QuantumStateMachine m{3, 0, 904};
        m.ry(0, theta).cnot(0, 2);
        const double c2 = std::cos(theta / 2) * std::cos(theta / 2);
        EXPECT_NEAR(m.entropy({0}), binaryEntropy(c2), 1e-12) << theta;
        EXPECT_NEAR(m.entropy({2, 1}), binaryEntropy(c2), 1e-12) << theta;
        EXPECT_NEAR(m.entropy({1}), 0.0, 1e-12) << theta;
    }
}

TEST(Entropy, RandomStatesMatchTheReferenceOnEitherSideOfTheCut)
{
    // |A| > N / 2 diagonalizes the complement's matrix instead; the spectra agree.
    std::mt19937_64 rng{905};
    constexpr std::size_t n = 7;
    const Eigen::VectorXcd psi = randomAmplitudes(n, rng);
    QuantumStateMachine m{n, 0, 905};
    m.prepare_state(psi);
    for (std::size_t k = 1; k < n; ++k)
    {
        const QubitList sub = randomQubits(n, k, rng);
        SCOPED_TRACE(formatQubits(sub));
        const double expected = referenceEntropy(referenceDensityMatrix(n, psi, sub));
        EXPECT_NEAR(m.entropy(sub), expected, 1e-10);
        EXPECT_NEAR(m.entropy(complement(n, sub)), expected, 1e-10);
    }
}

TEST(Entropy, MoreThanThirteenQubitsUseTheSmallerComplement)
{
    std::mt19937_64 rng{906};
    constexpr std::size_t n = 16;
    QuantumStateMachine m{n, 0, 906};
    m.prepare_state(randomAmplitudes(n, rng));
    const QubitList big = range(1, 15); // 14 qubits: past the explicit matrix limit
    const QubitList small = complement(n, big);
    EXPECT_THROW(m.reduced_density_matrix(big), std::length_error);
    EXPECT_NEAR(m.entropy(big), referenceEntropy(m.reduced_density_matrix(small)), 1e-12);
}

TEST(Entropy, StabilizerEntropyMatchesTheStateVectorOnRandomCliffordCircuits)
{
    std::mt19937_64 rng{907};
    for (std::size_t n = 8; n <= 12; ++n)
        for (int trial = 0; trial < 4; ++trial)
        {
            SCOPED_TRACE(std::format("n = {}, trial {}", n, trial));
            Pair p{n, 1};
            for (const auto& op : randomCliffordCircuit(n, 6 * n, rng)) p.both([&](auto& m) { m.append(op); });

            for (int s = 0; s < 8; ++s)
            {
                const QubitList sub = randomQubits(n, std::uniform_int_distribution<std::size_t>(0, n)(rng), rng);
                const double exact = p.tab.entropy(sub);
                EXPECT_EQ(exact, std::round(exact));
                EXPECT_EQ(static_cast<double>(p.tab.stabilizer_state().entanglement_entropy(sub)), exact);
                EXPECT_NEAR(p.sv.entropy(sub), exact, 1e-9) << formatQubits(sub);
            }
        }
}

TEST(Entropy, StabilizerEntropyIsSymmetricAndBoundedBeyondTheStateVector)
{
    std::mt19937_64 rng{908};
    constexpr std::size_t n = 150;
    QuantumStateMachine m{n, 0, 908};
    ASSERT_EQ(m.backend(), Backend::Stabilizer);
    for (const auto& op : randomCliffordCircuit(n, 3000, rng)) m.append(op);

    for (int s = 0; s < 20; ++s)
    {
        const QubitList sub = randomQubits(n, std::uniform_int_distribution<std::size_t>(0, n)(rng), rng);
        const double e = m.entropy(sub);
        EXPECT_EQ(e, m.entropy(complement(n, sub))) << sub.size();
        EXPECT_LE(e, static_cast<double>(std::min(sub.size(), n - sub.size())));
    }
    EXPECT_EQ(m.entropy(range(0, n)), 0.0);
}

TEST(Entropy, TwentyThousandQubitGhzHalfCutOnTheTableau)
{
    constexpr std::size_t n = 20000;
    QuantumStateMachine m{n, 0, 909};
    ASSERT_EQ(m.backend(), Backend::Stabilizer);
    m.h(0);
    for (Qubit q = 1; q < n; ++q) m.cnot(q - 1, q);

    EXPECT_EQ(m.entropy(range(0, n / 2)), 1.0);
    EXPECT_EQ(m.stabilizer_state().entanglement_entropy(range(n / 4, 3 * n / 4)), 1u);
}

TEST(Entropy, InvalidQubitsThrow)
{
    QuantumStateMachine sv{3, 0, 1};
    EXPECT_THROW(sv.entropy({1, 1}), std::invalid_argument);
    EXPECT_THROW(sv.entropy({3}), std::out_of_range);

    QuantumStateMachine tab{70, 0, 1};
    EXPECT_THROW(tab.entropy({0, 69, 0}), std::invalid_argument);
    EXPECT_THROW(tab.entropy({70}), std::out_of_range);

    Qputer::StabilizerState s{5};
    EXPECT_EQ(s.entanglement_entropy({}), 0u);
    const std::vector<Qubit> dup{2, 2}, outside{5};
    EXPECT_THROW(s.entanglement_entropy(dup), std::invalid_argument);
    EXPECT_THROW(s.entanglement_entropy(outside), std::out_of_range);
}


// ---- Reduced density matrix ----

TEST(ReducedDensityMatrix, BellPair)
{
    QuantumStateMachine m{2, 0, 911};
    m.h(0).cnot(0, 1);
    Eigen::MatrixXcd expected = Eigen::MatrixXcd::Zero(4, 4);
    expected(0, 0) = expected(0, 3) = expected(3, 0) = expected(3, 3) = 0.5;
    EXPECT_LE((m.reduced_density_matrix({0, 1}) - expected).cwiseAbs().maxCoeff(), kTol);
    EXPECT_LE((m.reduced_density_matrix({0}) - 0.5 * Eigen::MatrixXcd::Identity(2, 2)).cwiseAbs().maxCoeff(), kTol);
}

TEST(ReducedDensityMatrix, MatchesTheReferenceForRandomStates)
{
    // Up to 5 qubits the block-summed path runs, from 6 the column-split one.
    std::mt19937_64 rng{912};
    constexpr std::size_t n = 8;
    const Eigen::VectorXcd psi = randomAmplitudes(n, rng);
    QuantumStateMachine m{n, 0, 912};
    m.prepare_state(psi);
    for (const QubitList& sub : {QubitList{0}, QubitList{7}, QubitList{3, 0}, QubitList{1, 6, 2},
                                 QubitList{5, 0, 7, 2, 4}, QubitList{4, 1, 7, 0, 3, 6}, randomQubits(n, 7, rng),
                                 randomQubits(n, 8, rng)})
    {
        SCOPED_TRACE(formatQubits(sub));
        const Eigen::MatrixXcd rho = m.reduced_density_matrix(sub);
        EXPECT_LE((rho - referenceDensityMatrix(n, psi, sub)).cwiseAbs().maxCoeff(), kTol);
        EXPECT_TRUE(rho == Eigen::MatrixXcd(rho.adjoint())) << "exactly Hermitian";
        EXPECT_NEAR(rho.trace().real(), 1.0, kTol);
    }
}

TEST(ReducedDensityMatrix, IsIdenticalForEveryThreadCount)
{
#ifndef _OPENMP
    GTEST_SKIP() << "built without OpenMP; kernels are single-threaded";
#else
    // 15 qubits: past the parallel threshold, with both the block-summed (k <= 5) and the
    // column-split (k >= 6) paths, and entropy() diagonalizing either side.
    std::mt19937_64 rng{913};
    constexpr std::size_t n = 15;
    QuantumStateMachine m{n, 0, 913};
    m.prepare_state(randomAmplitudes(n, rng));
    for (const QubitList& sub : {QubitList{2}, QubitList{14, 3, 7}, range(9, 14), range(0, 6), randomQubits(n, 7, rng),
                                 randomQubits(n, 9, rng)})
    {
        SCOPED_TRACE(formatQubits(sub));
        const Eigen::MatrixXcd serial = withThreads(1, [&] { return m.reduced_density_matrix(sub); });
        const double serialEntropy = withThreads(1, [&] { return m.entropy(sub); });
        for (const int threads : {3, std::max(4, omp_get_num_procs())})
        {
            EXPECT_TRUE(bitwiseEqual(withThreads(threads, [&] { return m.reduced_density_matrix(sub); }), serial))
                << threads << " threads";
            EXPECT_EQ(withThreads(threads, [&] { return m.entropy(sub); }), serialEntropy) << threads << " threads";
        }
    }
#endif
}

TEST(ReducedDensityMatrix, InvalidRequestsThrow)
{
    QuantumStateMachine m{14, 0, 1};
    EXPECT_THROW(m.reduced_density_matrix({}), std::invalid_argument);
    EXPECT_THROW(m.reduced_density_matrix({0, 0}), std::invalid_argument);
    EXPECT_THROW(m.reduced_density_matrix({14}), std::out_of_range);
    EXPECT_THROW(m.reduced_density_matrix(range(0, 14)), std::length_error);

    QuantumStateMachine tab{3, 0, 1, Backend::Stabilizer};
    EXPECT_THROW(tab.reduced_density_matrix({0}), std::invalid_argument);
}
