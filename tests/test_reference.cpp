#include "TestSupport.hpp"

#include <functional>
#include <string>
#include <vector>

using namespace QputerTest;

namespace
{

constexpr std::size_t kN = 5;

// Gate under test on qubits q[0..arity), paired with its matrix in the embed() convention.
struct ReferenceCase
{
    std::string name;
    std::size_t arity;
    std::function<void(QuantumStateVector&, const QubitList&)> gate;
    std::function<Eigen::MatrixXcd(std::size_t n, const QubitList&)> reference;
};

ReferenceCase single(std::string name, void (*g)(QuantumStateVector&, Qubit), Eigen::Matrix2cd m)
{
    return {std::move(name), 1, [g](auto& s, const QubitList& q) { g(s, q[0]); },
            [m](std::size_t n, const QubitList& q) { return embed(n, {}, {q[0]}, m); }};
}

std::vector<ReferenceCase> allNamedGates()
{
    const double a = 0.731, b = -1.29, c = 2.07;
    return {
        single("X", QuantumGate::x, matX()),
        single("Y", QuantumGate::y, matY()),
        single("Z", QuantumGate::z, matZ()),
        single("H", QuantumGate::h, matH()),
        single("S", QuantumGate::s, matS()),
        single("Sdg", QuantumGate::sdg, matSdg()),
        single("T", QuantumGate::t, matT()),
        single("Tdg", QuantumGate::tdg, matTdg()),
        single("SX", QuantumGate::sx, matSX()),
        {"RX", 1, [=](auto& s, const QubitList& q) { QuantumGate::rx(s, q[0], a); },
         [=](std::size_t n, const QubitList& q) { return embed(n, {}, {q[0]}, matRX(a)); }},
        {"RY", 1, [=](auto& s, const QubitList& q) { QuantumGate::ry(s, q[0], a); },
         [=](std::size_t n, const QubitList& q) { return embed(n, {}, {q[0]}, matRY(a)); }},
        {"RZ", 1, [=](auto& s, const QubitList& q) { QuantumGate::rz(s, q[0], a); },
         [=](std::size_t n, const QubitList& q) { return embed(n, {}, {q[0]}, matRZ(a)); }},
        {"P", 1, [=](auto& s, const QubitList& q) { QuantumGate::phase(s, q[0], a); },
         [=](std::size_t n, const QubitList& q) { return embed(n, {}, {q[0]}, matPhase(a)); }},
        {"U3", 1, [=](auto& s, const QubitList& q) { QuantumGate::u3(s, q[0], a, b, c); },
         [=](std::size_t n, const QubitList& q) { return embed(n, {}, {q[0]}, matU3(a, b, c)); }},
        {"CNOT", 2, [](auto& s, const QubitList& q) { QuantumGate::cnot(s, q[0], q[1]); },
         [](std::size_t n, const QubitList& q) { return embed(n, {q[0]}, {q[1]}, matX()); }},
        {"CZ", 2, [](auto& s, const QubitList& q) { QuantumGate::cz(s, q[0], q[1]); },
         [](std::size_t n, const QubitList& q) { return embed(n, {q[0]}, {q[1]}, matZ()); }},
        {"CP", 2, [=](auto& s, const QubitList& q) { QuantumGate::cphase(s, q[0], q[1], b); },
         [=](std::size_t n, const QubitList& q) { return embed(n, {q[0]}, {q[1]}, matPhase(b)); }},
        {"SWAP", 2, [](auto& s, const QubitList& q) { QuantumGate::swap(s, q[0], q[1]); },
         [](std::size_t n, const QubitList& q) { return embed(n, {}, {q[0], q[1]}, matSwap()); }},
        {"Toffoli", 3, [](auto& s, const QubitList& q) { QuantumGate::toffoli(s, q[0], q[1], q[2]); },
         [](std::size_t n, const QubitList& q) { return embed(n, {q[0], q[1]}, {q[2]}, matX()); }},
        {"Fredkin", 3, [](auto& s, const QubitList& q) { QuantumGate::fredkin(s, q[0], q[1], q[2]); },
         [](std::size_t n, const QubitList& q) { return embed(n, {q[0]}, {q[1], q[2]}, matSwap()); }},
        {"MCX(3)", 4, [](auto& s, const QubitList& q) { QuantumGate::mcx(s, {q[0], q[1], q[2]}, q[3]); },
         [](std::size_t n, const QubitList& q) { return embed(n, {q[0], q[1], q[2]}, {q[3]}, matX()); }},
        {"MCZ(4)", 4, [](auto& s, const QubitList& q) { QuantumGate::mcz(s, q); },
         [](std::size_t n, const QubitList& q) { return embed(n, {q[0], q[1], q[2]}, {q[3]}, matZ()); }},
        {"MCP(4)", 4, [=](auto& s, const QubitList& q) { QuantumGate::mcphase(s, q, c); },
         [=](std::size_t n, const QubitList& q) { return embed(n, {q[0], q[1], q[2]}, {q[3]}, matPhase(c)); }},
    };
}

// Every ordered tuple of `arity` distinct qubits from [0, n).
void forEachOrderedTuple(std::size_t n, std::size_t arity, QubitList& prefix,
                         const std::function<void(const QubitList&)>& visit)
{
    if (prefix.size() == arity)
    {
        visit(prefix);
        return;
    }
    for (Qubit q = 0; q < n; ++q)
    {
        if (std::find(prefix.begin(), prefix.end(), q) != prefix.end()) continue;
        prefix.push_back(q);
        forEachOrderedTuple(n, arity, prefix, visit);
        prefix.pop_back();
    }
}

} // namespace


// Exhaustive over qubit placement: every named gate, on every ordered choice of distinct
// qubits in a 5-qubit register, against the independently built full operator.
TEST(ReferenceOperator, EveryNamedGateMatchesOnEveryQubitPlacement)
{
    std::mt19937_64 rng{21};
    for (const auto& tc : allNamedGates())
    {
        QubitList prefix;
        forEachOrderedTuple(kN, tc.arity, prefix, [&](const QubitList& q)
        {
            SCOPED_TRACE(std::format("{} on {}", tc.name, formatQubits(q)));
            const Eigen::VectorXcd psi = randomAmplitudes(kN, rng);
            QuantumStateVector s = fromAmplitudes(kN, psi);
            tc.gate(s, q);
            EXPECT_TRUE(statesNear(s, Eigen::VectorXcd(tc.reference(kN, q) * psi)));
        });
    }
}

// Unitarity: ||psi|| stays 1 through a long random circuit mixing every gate family.
TEST(ReferenceOperator, RandomCircuitsPreserveNorm)
{
    constexpr std::size_t n = 8;
    constexpr int depth = 400;
    std::mt19937_64 rng{22};

    QuantumStateVector s = randomState(n, rng);
    for (int k = 0; k < depth; ++k)
    {
        randomGate(n, rng)(s);
        ASSERT_NEAR(s.norm(), 1.0, 1e-11) << "after gate " << k;
    }
}

// Inverse circuit: applying each gate's adjoint in reverse order returns the input state.
TEST(ReferenceOperator, CircuitFollowedByItsInverseIsIdentity)
{
    constexpr std::size_t n = 6;
    std::mt19937_64 rng{23};
    const QuantumStateVector psi = randomState(n, rng);
    QuantumStateVector s = psi;

    std::vector<GateOp> undo;
    for (int k = 0; k < 60; ++k)
    {
        const QubitList q = randomQubits(n, 3, rng);
        const Eigen::MatrixXcd U = randomUnitary(4, rng);
        const double th = std::uniform_real_distribution<double>(-kPi, kPi)(rng);

        QuantumGate::controlled(s, {q[2]}, {q[0], q[1]}, U);
        QuantumGate::u3(s, q[0], th, 0.3, -0.2);
        QuantumGate::cphase(s, q[1], q[2], th);
        QuantumGate::t(s, q[2]);
        undo.push_back([=](auto& st) {
            QuantumGate::tdg(st, q[2]);
            QuantumGate::cphase(st, q[1], q[2], -th);
            QuantumGate::u3(st, q[0], -th, 0.2, -0.3);  // U3(t,p,l)^dagger = U3(-t,-l,-p)
            QuantumGate::controlled(st, {q[2]}, {q[0], q[1]}, U.adjoint());
        });
    }
    for (auto it = undo.rbegin(); it != undo.rend(); ++it) (*it)(s);

    EXPECT_TRUE(statesNear(s, psi, 1e-11));
}
