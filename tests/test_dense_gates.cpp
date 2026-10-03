#include "TestSupport.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>

using namespace QputerTest;


TEST(DenseGates, FirstTargetIsMostSignificantBitOfMatrixIndex)
{
    std::mt19937_64 rng{11};
    const QuantumStateVector psi = randomState(4, rng);

    // CNOT matrix with the control on U's MSB: targets {c, t} must reproduce cnot(c, t).
    Eigen::Matrix4cd cnotMat = Eigen::Matrix4cd::Identity();
    cnotMat.bottomRightCorner<2, 2>() = matX();

    QuantumStateVector dense = psi, named = psi;
    QuantumGate::apply(dense, {3, 1}, cnotMat);
    QuantumGate::cnot(named, 3, 1);
    EXPECT_TRUE(statesNear(dense, named)) << "apply({c,t}, CNOT) = cnot(c,t)";

    // kron(A, B) acts A on targets[0] and B on targets[1].
    dense = psi;
    named = psi;
    QuantumGate::apply(dense, {2, 0}, kron(matH(), matX()));
    QuantumGate::h(named, 2);
    QuantumGate::x(named, 0);
    EXPECT_TRUE(statesNear(dense, named)) << "apply({2,0}, H (x) X) = H_2 X_0";
}

TEST(DenseGates, MatchNamedGatesForStandardMatrices)
{
    std::mt19937_64 rng{12};
    const QuantumStateVector psi = randomState(5, rng);
    auto both = [&](auto&& dense, auto&& named)
    {
        QuantumStateVector a = psi, b = psi;
        dense(a);
        named(b);
        return statesNear(a, b);
    };

    EXPECT_TRUE(both([](auto& s) { QuantumGate::apply(s, {3}, matH()); },
                     [](auto& s) { QuantumGate::h(s, 3); }))
        << "apply(H)";
    EXPECT_TRUE(both([](auto& s) { QuantumGate::apply(s, {1, 4}, matSwap()); },
                     [](auto& s) { QuantumGate::swap(s, 1, 4); }))
        << "apply(SWAP)";
    EXPECT_TRUE(both([](auto& s) { QuantumGate::controlled(s, {0, 4}, {2}, matX()); },
                     [](auto& s) { QuantumGate::toffoli(s, 0, 4, 2); }))
        << "controlled({c0,c1}, X) = Toffoli";
    EXPECT_TRUE(both([](auto& s) { QuantumGate::controlled(s, {3}, {0, 1}, matSwap()); },
                     [](auto& s) { QuantumGate::fredkin(s, 3, 0, 1); }))
        << "controlled({c}, SWAP) = Fredkin";
    EXPECT_TRUE(both([](auto& s) { QuantumGate::controlled(s, {1}, {2}, matZ()); },
                     [](auto& s) { QuantumGate::cz(s, 1, 2); }))
        << "controlled({c}, Z) = CZ";
}

// M = 1..5 targets covers the 2x2 kernel, the fixed-size 4x4 and 8x8 kernels, and the
// runtime-sized fallback; random qubit placement covers arbitrary bit orderings.
TEST(DenseGates, RandomUnitariesMatchReferenceOperator)
{
    constexpr std::size_t n = 7;
    std::mt19937_64 rng{13};

    for (std::size_t m = 1; m <= 5; ++m)
    {
        for (std::size_t nc = 0; nc + m <= n && nc <= 2; ++nc)
        {
            for (int trial = 0; trial < 3; ++trial)
            {
                const QubitList qs = randomQubits(n, m + nc, rng);
                const QubitList targets(qs.begin(), qs.begin() + static_cast<std::ptrdiff_t>(m));
                const QubitList controls(qs.begin() + static_cast<std::ptrdiff_t>(m), qs.end());
                const Eigen::MatrixXcd U = randomUnitary(bit(m), rng);
                SCOPED_TRACE(std::format("M = {}, controls = {}, targets = {}", m, formatQubits(controls), formatQubits(targets)));

                const Eigen::VectorXcd psi = randomAmplitudes(n, rng);
                QuantumStateVector s = fromAmplitudes(n, psi);
                if (controls.empty())
                    QuantumGate::apply(s, targets, U);
                else
                    QuantumGate::controlled(s, controls, targets, U);

                EXPECT_TRUE(statesNear(s, Eigen::VectorXcd(embed(n, controls, targets, U) * psi)));
                EXPECT_NEAR(s.norm(), 1.0, kTol);
            }
        }
    }
}

TEST(DenseGates, RejectInvalidInputWithoutTouchingState)
{
    std::mt19937_64 rng{14};
    QuantumStateVector s = randomState(4, rng);
    const Eigen::VectorXcd before = s.vector();

    const Eigen::MatrixXcd notUnitary = 2.0 * Eigen::MatrixXcd::Identity(2, 2);
    Eigen::MatrixXcd nanMat = Eigen::MatrixXcd::Identity(2, 2);
    nanMat(0, 0) = std::numeric_limits<double>::quiet_NaN();
    Eigen::MatrixXcd almostUnitary = Eigen::MatrixXcd::Identity(2, 2);
    almostUnitary(0, 1) = 1e-6;

    EXPECT_THROW(QuantumGate::apply(s, {0}, notUnitary), std::invalid_argument);
    EXPECT_THROW(QuantumGate::apply(s, {0}, nanMat), std::invalid_argument);
    EXPECT_THROW(QuantumGate::apply(s, {0}, almostUnitary), std::invalid_argument);
    EXPECT_THROW(QuantumGate::apply(s, {0, 1}, matH()), std::invalid_argument) << "dimension mismatch";
    EXPECT_THROW(QuantumGate::apply(s, {}, matH()), std::invalid_argument) << "no targets";
    EXPECT_THROW(QuantumGate::apply(s, {1, 1}, matSwap()), std::invalid_argument) << "duplicate target";
    EXPECT_THROW(QuantumGate::apply(s, {4}, matH()), std::out_of_range);
    EXPECT_THROW(QuantumGate::controlled(s, {2}, {2}, matX()), std::invalid_argument) << "control = target";
    EXPECT_THROW(QuantumGate::controlled(s, {7}, {0}, matX()), std::out_of_range);

    const QubitList tooMany(QuantumGate::kMaxDenseTargets + 1, 0);
    EXPECT_THROW(QuantumGate::apply(s, tooMany, matH()), std::invalid_argument);

    EXPECT_TRUE(statesNear(s, before, 0.0));
}
