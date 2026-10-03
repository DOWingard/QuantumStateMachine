#include "TestSupport.hpp"

#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using namespace QputerTest;

namespace
{

// Expected image of every basis state under a classical reversible gate (permutation of indices).
template <class Gate, class Map>
void expectPermutation(std::size_t n, Gate&& gate, Map&& expectedIndex, const std::string& label)
{
    for (std::size_t i = 0; i < bit(n); ++i)
    {
        SCOPED_TRACE(std::format("{} on |{}>", label, i));
        QuantumStateVector s = basis(n, i);
        gate(s);
        EXPECT_TRUE(statesNear(s, amplitudes(n, {{expectedIndex(i), 1.0}}), 0.0));
    }
}

// Expected image of every basis state under a diagonal gate.
template <class Gate, class Phase>
void expectDiagonal(std::size_t n, Gate&& gate, Phase&& expectedPhase, const std::string& label)
{
    for (std::size_t i = 0; i < bit(n); ++i)
    {
        SCOPED_TRACE(std::format("{} on |{}>", label, i));
        QuantumStateVector s = basis(n, i);
        gate(s);
        EXPECT_TRUE(statesNear(s, amplitudes(n, {{i, expectedPhase(i)}})));
    }
}

bool allSet(std::size_t i, const QubitList& qs)
{
    for (const Qubit q : qs)
        if (!((i >> q) & 1U)) return false;
    return true;
}

} // namespace


// ---- Truth tables ----

TEST(MultiQubitGates, CnotTruthTable)
{
    const std::vector<std::tuple<std::size_t, Qubit, Qubit>> layouts{{2, 0, 1}, {2, 1, 0}, {4, 0, 3}, {4, 3, 1}};
    for (const auto& [n, c, t] : layouts)
    {
        expectPermutation(
            n, [c, t](auto& s) { QuantumGate::cnot(s, c, t); },
            [c, t](std::size_t i) { return ((i >> c) & 1U) ? i ^ bit(t) : i; },
            std::format("CNOT({}->{}) n={}", c, t, n));
    }
}

TEST(MultiQubitGates, SwapTruthTable)
{
    for (const auto& [a, b] : std::vector<std::pair<Qubit, Qubit>>{{0, 1}, {0, 3}, {3, 1}})
    {
        expectPermutation(
            4, [a, b](auto& s) { QuantumGate::swap(s, a, b); },
            [a, b](std::size_t i)
            {
                const bool ba = (i >> a) & 1U, bb = (i >> b) & 1U;
                return ba == bb ? i : i ^ bit(a) ^ bit(b);
            },
            std::format("SWAP({},{})", a, b));
    }
}

TEST(MultiQubitGates, ToffoliTruthTable)
{
    for (const auto& [c0, c1, t] : std::vector<std::tuple<Qubit, Qubit, Qubit>>{{0, 1, 2}, {2, 0, 1}, {1, 2, 0}})
    {
        expectPermutation(
            3, [=](auto& s) { QuantumGate::toffoli(s, c0, c1, t); },
            [=](std::size_t i) { return allSet(i, {c0, c1}) ? i ^ bit(t) : i; },
            std::format("CCX({},{}->{})", c0, c1, t));
    }
}

TEST(MultiQubitGates, FredkinTruthTable)
{
    for (const auto& [c, a, b] : std::vector<std::tuple<Qubit, Qubit, Qubit>>{{0, 1, 2}, {2, 0, 1}, {1, 2, 0}})
    {
        expectPermutation(
            3, [=](auto& s) { QuantumGate::fredkin(s, c, a, b); },
            [=](std::size_t i)
            {
                if (!((i >> c) & 1U)) return i;
                const bool ba = (i >> a) & 1U, bb = (i >> b) & 1U;
                return ba == bb ? i : i ^ bit(a) ^ bit(b);
            },
            std::format("CSWAP({};{},{})", c, a, b));
    }
}

TEST(MultiQubitGates, CzAndCphaseOnlyPhaseTheOneOneComponent)
{
    const double lambda = 0.37;
    expectDiagonal(
        3, [](auto& s) { QuantumGate::cz(s, 0, 2); },
        [](std::size_t i) { return allSet(i, {0, 2}) ? cd{-1} : cd{1}; }, "CZ(0,2)");
    expectDiagonal(
        3, [=](auto& s) { QuantumGate::cphase(s, 2, 1, lambda); },
        [=](std::size_t i) { return allSet(i, {1, 2}) ? expi(lambda) : cd{1}; }, "CP(2,1)");
}

// n = 9 with 8 controls: X fires on exactly one pair of basis states out of 512.
TEST(MultiQubitGates, McxFlipsTargetOnlyWhenEveryControlIsSet)
{
    constexpr std::size_t n = 9;
    const QubitList controls{0, 1, 2, 3, 5, 6, 7, 8};
    const Qubit target = 4;
    expectPermutation(
        n, [&](auto& s) { QuantumGate::mcx(s, controls, target); },
        [&](std::size_t i) { return allSet(i, controls) ? i ^ bit(target) : i; }, "MCX(8 controls)");
}

TEST(MultiQubitGates, MczAndMcphasePhaseOnlyTheAllOnesSubspace)
{
    constexpr std::size_t n = 6;
    const QubitList qs{1, 3, 4};
    expectDiagonal(
        n, [&](auto& s) { QuantumGate::mcz(s, qs); },
        [&](std::size_t i) { return allSet(i, qs) ? cd{-1} : cd{1}; }, "MCZ{1,3,4}");
    expectDiagonal(
        n, [&](auto& s) { QuantumGate::mcphase(s, qs, 1.1); },
        [&](std::size_t i) { return allSet(i, qs) ? expi(1.1) : cd{1}; }, "MCP{1,3,4}");
}


// ---- Entangled states ----

TEST(MultiQubitGates, HadamardThenCnotPreparesAllFourBellStates)
{
    // |q1 q0> --H(q0), CNOT(q0->q1)--> Bell basis.
    const double r = kInvSqrt2;
    const std::vector<Eigen::VectorXcd> expected{
        amplitudes(2, {{0, r}, {3, r}}),   // |00> -> Phi+
        amplitudes(2, {{0, r}, {3, -r}}),  // |01> -> Phi-
        amplitudes(2, {{1, r}, {2, r}}),   // |10> -> Psi+
        amplitudes(2, {{1, -r}, {2, r}}),  // |11> -> Psi- (up to sign convention)
    };
    for (std::size_t i = 0; i < 4; ++i)
    {
        SCOPED_TRACE(std::format("input |{}>", i));
        QuantumStateVector s = basis(2, i);
        QuantumGate::h(s, 0);
        QuantumGate::cnot(s, 0, 1);
        EXPECT_TRUE(statesNear(s, expected[i]));
    }
}

TEST(MultiQubitGates, CnotFanOutPreparesGhzStates)
{
    for (std::size_t n = 2; n <= 9; ++n)
    {
        SCOPED_TRACE(std::format("n = {}", n));
        QuantumStateVector s{n};
        QuantumGate::h(s, 0);
        for (Qubit q = 1; q < n; ++q) QuantumGate::cnot(s, 0, q);
        EXPECT_TRUE(statesNear(s, amplitudes(n, {{0, kInvSqrt2}, {bit(n) - 1, kInvSqrt2}})));
    }
}


// ---- Decomposition identities (random 3-qubit state) ----

TEST(MultiQubitGates, SatisfyStandardDecompositionIdentities)
{
    std::mt19937_64 rng{42};
    const QuantumStateVector psi = randomState(4, rng);
    auto apply = [&](auto&& f)
    {
        QuantumStateVector s = psi;
        f(s);
        return s;
    };

    const auto cnot = apply([](auto& s) { QuantumGate::cnot(s, 1, 3); });
    EXPECT_TRUE(statesNear(apply([](auto& s) {
                               QuantumGate::h(s, 3);
                               QuantumGate::cz(s, 1, 3);
                               QuantumGate::h(s, 3);
                           }),
                           cnot))
        << "H_t CZ H_t = CNOT";

    EXPECT_TRUE(statesNear(apply([](auto& s) {
                               QuantumGate::cnot(s, 0, 2);
                               QuantumGate::cnot(s, 2, 0);
                               QuantumGate::cnot(s, 0, 2);
                           }),
                           apply([](auto& s) { QuantumGate::swap(s, 0, 2); })))
        << "three CNOTs = SWAP";

    EXPECT_TRUE(statesNear(apply([](auto& s) { QuantumGate::cz(s, 0, 3); }),
                           apply([](auto& s) { QuantumGate::cz(s, 3, 0); })))
        << "CZ is symmetric";

    EXPECT_TRUE(statesNear(apply([](auto& s) { QuantumGate::cphase(s, 0, 2, kPi); }),
                           apply([](auto& s) { QuantumGate::cz(s, 0, 2); })))
        << "CP(pi) = CZ";

    EXPECT_TRUE(statesNear(apply([](auto& s) { QuantumGate::mcx(s, {0, 2}, 1); }),
                           apply([](auto& s) { QuantumGate::toffoli(s, 0, 2, 1); })))
        << "MCX with 2 controls = Toffoli";

    EXPECT_TRUE(statesNear(apply([](auto& s) { QuantumGate::mcx(s, {2}, 0); }),
                           apply([](auto& s) { QuantumGate::cnot(s, 2, 0); })))
        << "MCX with 1 control = CNOT";

    EXPECT_TRUE(statesNear(apply([](auto& s) { QuantumGate::mcx(s, {}, 3); }),
                           apply([](auto& s) { QuantumGate::x(s, 3); })))
        << "MCX with 0 controls = X";

    EXPECT_TRUE(statesNear(apply([](auto& s) { QuantumGate::mcphase(s, {2}, 0.4); }),
                           apply([](auto& s) { QuantumGate::phase(s, 2, 0.4); })))
        << "MCP on 1 qubit = P";

    EXPECT_TRUE(statesNear(apply([](auto& s) { QuantumGate::mcz(s, {1, 0, 3}); }),
                           apply([](auto& s) { QuantumGate::mcphase(s, {0, 1, 3}, kPi); })))
        << "MCZ = MCP(pi), order-independent";

    // Toffoli = H_t . CCZ . H_t
    EXPECT_TRUE(statesNear(apply([](auto& s) {
                               QuantumGate::h(s, 2);
                               QuantumGate::mcz(s, {0, 1, 2});
                               QuantumGate::h(s, 2);
                           }),
                           apply([](auto& s) { QuantumGate::toffoli(s, 0, 1, 2); })))
        << "H_t CCZ H_t = Toffoli";
}


// ---- Input validation ----

TEST(MultiQubitGates, RejectInvalidQubitArgumentsWithoutTouchingState)
{
    std::mt19937_64 rng{9};
    QuantumStateVector s = randomState(4, rng);
    const Eigen::VectorXcd before = s.vector();

    EXPECT_THROW(QuantumGate::cnot(s, 1, 1), std::invalid_argument);
    EXPECT_THROW(QuantumGate::cnot(s, 0, 4), std::out_of_range);
    EXPECT_THROW(QuantumGate::swap(s, 2, 2), std::invalid_argument);
    EXPECT_THROW(QuantumGate::toffoli(s, 0, 0, 1), std::invalid_argument);
    EXPECT_THROW(QuantumGate::fredkin(s, 0, 1, 0), std::invalid_argument);
    EXPECT_THROW(QuantumGate::mcx(s, {0, 1, 2}, 2), std::invalid_argument);
    EXPECT_THROW(QuantumGate::mcz(s, {}), std::invalid_argument);
    EXPECT_THROW(QuantumGate::mcphase(s, {0, 5}, 1.0), std::out_of_range);
    EXPECT_TRUE(statesNear(s, before, 0.0));
}
