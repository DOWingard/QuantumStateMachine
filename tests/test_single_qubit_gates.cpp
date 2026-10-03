#include "TestSupport.hpp"

#include <vector>

using namespace QputerTest;

namespace
{

struct SingleQubitCase
{
    std::string name;
    std::function<void(QuantumStateVector&, Qubit)> gate;
    Eigen::Matrix2cd expected;
};

const std::vector<double> kAngles{0.0, kPi / 3, kPi / 2, kPi, 3 * kPi / 2, -0.7, 2 * kPi};

std::vector<SingleQubitCase> textbookCases()
{
    std::vector<SingleQubitCase> cases{
        {"X", QuantumGate::x, matX()},
        {"Y", QuantumGate::y, matY()},
        {"Z", QuantumGate::z, matZ()},
        {"H", QuantumGate::h, matH()},
        {"S", QuantumGate::s, matS()},
        {"Sdg", QuantumGate::sdg, matSdg()},
        {"T", QuantumGate::t, matT()},
        {"Tdg", QuantumGate::tdg, matTdg()},
        {"SX", QuantumGate::sx, matSX()},
    };
    for (const double a : kAngles)
    {
        cases.push_back({std::format("RX({})", a), [a](auto& s, Qubit q) { QuantumGate::rx(s, q, a); }, matRX(a)});
        cases.push_back({std::format("RY({})", a), [a](auto& s, Qubit q) { QuantumGate::ry(s, q, a); }, matRY(a)});
        cases.push_back({std::format("RZ({})", a), [a](auto& s, Qubit q) { QuantumGate::rz(s, q, a); }, matRZ(a)});
        cases.push_back({std::format("P({})", a), [a](auto& s, Qubit q) { QuantumGate::phase(s, q, a); },
                         matPhase(a)});
        cases.push_back({std::format("U3({0},{0}/2,-{0}/3)", a),
                         [a](auto& s, Qubit q) { QuantumGate::u3(s, q, a, a / 2, -a / 3); },
                         matU3(a, a / 2, -a / 3)});
    }
    return cases;
}

// Applies `ops` in order to a copy of `start`.
QuantumStateVector run(QuantumStateVector start, std::initializer_list<GateOp> ops)
{
    for (const auto& op : ops) op(start);
    return start;
}

} // namespace


// Column c of a gate's matrix is its image of |c>: checks every gate against its textbook form.
TEST(SingleQubitGates, BasisImagesMatchTextbookMatrices)
{
    for (const auto& tc : textbookCases())
    {
        for (std::size_t c = 0; c < 2; ++c)
        {
            SCOPED_TRACE(std::format("{} |{}>", tc.name, c));
            QuantumStateVector s = basis(1, c);
            tc.gate(s, 0);
            EXPECT_TRUE(statesNear(s, Eigen::VectorXcd(tc.expected.col(ix(c)))));
        }
    }
}

TEST(SingleQubitGates, PreparesWellKnownStates)
{
    const auto plus = amplitudes(1, {{0, kInvSqrt2}, {1, kInvSqrt2}});
    const auto minus = amplitudes(1, {{0, kInvSqrt2}, {1, -kInvSqrt2}});
    const auto plusI = amplitudes(1, {{0, kInvSqrt2}, {1, kInvSqrt2 * kI}});

    QuantumStateVector s = basis(1, 0);
    QuantumGate::h(s, 0);
    EXPECT_TRUE(statesNear(s, plus)) << "H|0> = |+>";

    s = basis(1, 1);
    QuantumGate::h(s, 0);
    EXPECT_TRUE(statesNear(s, minus)) << "H|1> = |->";

    s = basis(1, 0);
    QuantumGate::h(s, 0);
    QuantumGate::s(s, 0);
    EXPECT_TRUE(statesNear(s, plusI)) << "SH|0> = |+i>";

    s = basis(1, 0);
    QuantumGate::ry(s, 0, kPi / 2);
    EXPECT_TRUE(statesNear(s, plus)) << "RY(pi/2)|0> = |+>";

    s = basis(1, 0);
    QuantumGate::rx(s, 0, kPi / 2);
    EXPECT_TRUE(statesNear(s, amplitudes(1, {{0, kInvSqrt2}, {1, -kInvSqrt2 * kI}}))) << "RX(pi/2)|0> = |-i>";

    s = basis(1, 0);
    QuantumGate::sx(s, 0);
    QuantumGate::sx(s, 0);
    EXPECT_TRUE(statesNear(s, amplitudes(1, {{1, 1.0}}))) << "SX^2 |0> = |1>";
}

// Algebraic identities checked on a random state, so they hold for every input, not just basis states.
TEST(SingleQubitGates, SatisfyStandardAlgebraicIdentities)
{
    std::mt19937_64 rng{0x5eed};
    const QuantumStateVector psi = randomState(1, rng);
    const double th = 0.913;
    auto g = [](auto f) { return GateOp{[f](QuantumStateVector& s) { f(s, Qubit{0}); }}; };

    const GateOp X = g(QuantumGate::x), Y = g(QuantumGate::y), Z = g(QuantumGate::z), H = g(QuantumGate::h);
    const GateOp S = g(QuantumGate::s), Sdg = g(QuantumGate::sdg), T = g(QuantumGate::t), Tdg = g(QuantumGate::tdg);
    const GateOp SX = g(QuantumGate::sx);

    const Eigen::VectorXcd v = psi.vector();
    EXPECT_TRUE(statesNear(run(psi, {H, H}), v)) << "H^2 = I";
    EXPECT_TRUE(statesNear(run(psi, {X, X}), v)) << "X^2 = I";
    EXPECT_TRUE(statesNear(run(psi, {S, Sdg}), v)) << "S Sdg = I";
    EXPECT_TRUE(statesNear(run(psi, {T, Tdg}), v)) << "T Tdg = I";
    EXPECT_TRUE(statesNear(run(psi, {H, Z, H}), run(psi, {X}))) << "HZH = X";
    EXPECT_TRUE(statesNear(run(psi, {H, X, H}), run(psi, {Z}))) << "HXH = Z";
    EXPECT_TRUE(statesNear(run(psi, {S, S}), run(psi, {Z}))) << "S^2 = Z";
    EXPECT_TRUE(statesNear(run(psi, {T, T}), run(psi, {S}))) << "T^2 = S";
    EXPECT_TRUE(statesNear(run(psi, {SX, SX}), run(psi, {X}))) << "SX^2 = X";
    EXPECT_TRUE(statesNear(run(psi, {Z, X}), (kI * -1.0) * run(psi, {Y}).vector())) << "XZ = -iY";

    // Pauli rotations at pi are the Paulis up to a global -i; at 2pi they are -I (spinor sign).
    const auto rx = [](double a) { return GateOp{[a](auto& s) { QuantumGate::rx(s, 0, a); }}; };
    const auto ry = [](double a) { return GateOp{[a](auto& s) { QuantumGate::ry(s, 0, a); }}; };
    const auto rz = [](double a) { return GateOp{[a](auto& s) { QuantumGate::rz(s, 0, a); }}; };
    const auto ph = [](double a) { return GateOp{[a](auto& s) { QuantumGate::phase(s, 0, a); }}; };
    const auto u3 = [](double t, double p, double l)
    { return GateOp{[=](auto& s) { QuantumGate::u3(s, 0, t, p, l); }}; };

    EXPECT_TRUE(statesNear(run(psi, {rx(kPi)}), -kI * run(psi, {X}).vector())) << "RX(pi) = -iX";
    EXPECT_TRUE(statesNear(run(psi, {ry(kPi)}), -kI * run(psi, {Y}).vector())) << "RY(pi) = -iY";
    EXPECT_TRUE(statesNear(run(psi, {rz(kPi)}), -kI * run(psi, {Z}).vector())) << "RZ(pi) = -iZ";
    EXPECT_TRUE(statesNear(run(psi, {rx(2 * kPi)}), -v)) << "RX(2pi) = -I";
    EXPECT_TRUE(statesNear(run(psi, {rz(th)}), expi(-th / 2) * run(psi, {ph(th)}).vector()))
        << "RZ(t) = e^{-it/2} P(t)";
    EXPECT_TRUE(statesNear(run(psi, {ph(kPi)}), run(psi, {Z}))) << "P(pi) = Z";
    EXPECT_TRUE(statesNear(run(psi, {ph(kPi / 4)}), run(psi, {T}))) << "P(pi/4) = T";
    EXPECT_TRUE(statesNear(run(psi, {u3(kPi / 2, 0, kPi)}), run(psi, {H}))) << "U3(pi/2, 0, pi) = H";
    EXPECT_TRUE(statesNear(run(psi, {u3(th, 0, 0)}), run(psi, {ry(th)}))) << "U3(t, 0, 0) = RY(t)";
    EXPECT_TRUE(statesNear(run(psi, {u3(th, -kPi / 2, kPi / 2)}), run(psi, {rx(th)})))
        << "U3(t, -pi/2, pi/2) = RX(t)";
    EXPECT_TRUE(statesNear(run(psi, {rx(th), rx(-th)}), v)) << "RX(t) RX(-t) = I";
}

// Qubit q is bit q of the amplitude index: X on qubit q of |0...0> must land on index 2^q.
TEST(SingleQubitGates, TargetQubitMapsToAmplitudeIndexBit)
{
    constexpr std::size_t n = 6;
    for (Qubit q = 0; q < n; ++q)
    {
        SCOPED_TRACE(std::format("target = {}", q));
        QuantumStateVector s{n};
        QuantumGate::x(s, q);
        EXPECT_TRUE(statesNear(s, amplitudes(n, {{bit(q), 1.0}}), 0.0));

        // H on qubit q of |0...0> splits weight between indices 0 and 2^q only.
        QuantumStateVector h{n};
        QuantumGate::h(h, q);
        EXPECT_TRUE(statesNear(h, amplitudes(n, {{0, kInvSqrt2}, {bit(q), kInvSqrt2}})));
    }
}

TEST(SingleQubitGates, RejectOutOfRangeTargetWithoutTouchingState)
{
    std::mt19937_64 rng{7};
    QuantumStateVector s = randomState(3, rng);
    const Eigen::VectorXcd before = s.vector();

    EXPECT_THROW(QuantumGate::x(s, 3), std::out_of_range);
    EXPECT_THROW(QuantumGate::h(s, 64), std::out_of_range);
    EXPECT_THROW(QuantumGate::rz(s, 99, 1.0), std::out_of_range);
    EXPECT_TRUE(statesNear(s, before, 0.0));
}
