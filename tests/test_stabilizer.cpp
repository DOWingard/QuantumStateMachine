#include "TestSupport.hpp"

#include <StabilizerState.hpp>

#include <stdexcept>

using namespace QputerTest;
using Qputer::Operation;
using Qputer::OutcomeSupport;
using Qputer::QuantumStateMachine;
using Qputer::StabilizerState;

namespace
{

StabilizerState tableauFor(std::size_t n, const std::vector<Operation>& ops)
{
    StabilizerState t{n};
    for (const auto& op : ops) applyClifford(t, op);
    return t;
}

// Exact marginal over `qubits` (bit j = qubits[j]) of a state vector.
Eigen::VectorXd marginalOf(std::size_t n, const Eigen::VectorXcd& v, const QubitList& qubits)
{
    QuantumStateMachine m{n};
    m.prepare_state(v);
    return m.marginal_probabilities(qubits);
}

// Every outcome of the support, by increasing rank (one word: width <= 64).
std::vector<std::uint64_t> enumerate(const OutcomeSupport& sup)
{
    const std::size_t d = sup.dimension();
    std::vector<std::uint64_t> out;
    for (std::uint64_t j = 0; j < (std::uint64_t{1} << d); ++j)
    {
        std::uint64_t o = sup.offset[0];
        for (std::size_t m = 0; m < d; ++m)
            if ((j >> (d - 1 - m)) & 1U) o ^= sup.row(m)[0];
        out.push_back(o);
    }
    return out;
}

} // namespace


TEST(Stabilizer, StartsInGroundState)
{
    const StabilizerState t{3};
    EXPECT_EQ(t.num_qubits(), 3u);
    EXPECT_EQ(t.stabilizers(), (std::vector<std::string>{"+ZII", "+IZI", "+IIZ"}));
    EXPECT_TRUE(statesNear(t.to_state_vector(), amplitudes(3, {{0, 1.0}}), 0.0));
}

TEST(Stabilizer, EveryBasisStateConvertsExactly)
{
    // Most of these are orthogonal to |0000>, the start state that would project to zero.
    StabilizerState t{4};
    for (std::size_t index = 0; index < 16; ++index)
    {
        SCOPED_TRACE(std::format("index {}", index));
        t.set_basis(index);
        EXPECT_TRUE(statesNear(t.to_state_vector(), amplitudes(4, {{index, 1.0}}), 0.0));
    }
}

TEST(Stabilizer, GatesConjugateGeneratorsByTheirTextbookRules)
{
    // Without measurement, generator i is U Z_i U^dagger, so each string below is fixed.
    auto after = [](std::size_t n, auto&& circuit)
    {
        StabilizerState t{n};
        circuit(t);
        return t.stabilizers();
    };
    using V = std::vector<std::string>;
    EXPECT_EQ(after(1, [](auto& t) { t.h(0); }), V{"+X"});
    EXPECT_EQ(after(1, [](auto& t) { t.x(0); }), V{"-Z"});
    EXPECT_EQ(after(1, [](auto& t) { t.y(0); }), V{"-Z"});
    EXPECT_EQ(after(1, [](auto& t) { t.z(0); }), V{"+Z"});
    EXPECT_EQ(after(1, [](auto& t) { t.h(0); t.y(0); }), V{"-X"});
    EXPECT_EQ(after(1, [](auto& t) { t.h(0); t.z(0); }), V{"-X"});
    EXPECT_EQ(after(1, [](auto& t) { t.h(0); t.s(0); }), V{"+Y"});
    EXPECT_EQ(after(1, [](auto& t) { t.h(0); t.s(0); t.s(0); }), V{"-X"});
    EXPECT_EQ(after(1, [](auto& t) { t.h(0); t.sdg(0); }), V{"-Y"});
    EXPECT_EQ(after(1, [](auto& t) { t.sx(0); }), V{"-Y"});
    EXPECT_EQ(after(1, [](auto& t) { t.sx(0); t.sx(0); }), V{"-Z"});
    EXPECT_EQ(after(2, [](auto& t) { t.h(0); t.cnot(0, 1); }), (V{"+XX", "+ZZ"}));
    EXPECT_EQ(after(2, [](auto& t) { t.h(0); t.h(1); t.cz(0, 1); }), (V{"+XZ", "+ZX"}));
    EXPECT_EQ(after(2, [](auto& t) { t.h(0); t.swap(0, 1); }), (V{"+IX", "+ZI"}));
}

TEST(Stabilizer, RandomCliffordCircuitsMatchTheReferenceOperator)
{
    std::mt19937_64 rng{601};
    for (std::size_t n = 1; n <= 5; ++n)
        for (int trial = 0; trial < 40; ++trial)
        {
            SCOPED_TRACE(std::format("n = {}, trial {}", n, trial));
            const auto ops = randomCliffordCircuit(n, 30, rng);
            EXPECT_TRUE(statesNearUpToPhase(tableauFor(n, ops).to_state_vector(), referenceState(n, ops)));
        }
}

TEST(Stabilizer, ConversionFixesTheGlobalPhase)
{
    StabilizerState y{1};
    y.y(0); // Y|0> = i|1>; the tableau has no phase to carry
    EXPECT_TRUE(statesNear(y.to_state_vector(), amplitudes(1, {{1, 1.0}}), 0.0));

    std::mt19937_64 rng{602};
    for (int trial = 0; trial < 30; ++trial)
    {
        const QuantumStateVector v = tableauFor(4, randomCliffordCircuit(4, 25, rng)).to_state_vector();
        std::size_t first = 0;
        while (std::abs(v[first]) < 1e-9) ++first;
        EXPECT_GT(v[first].real(), 0.0);
        EXPECT_EQ(v[first].imag(), 0.0);
        EXPECT_NEAR(v.norm(), 1.0, kTol);
    }
}

TEST(Stabilizer, ExpectationOfEveryPauliStringMatchesTheReference)
{
    std::mt19937_64 rng{603};
    constexpr std::size_t n = 3;
    const QubitList all{0, 1, 2};
    for (int trial = 0; trial < 20; ++trial)
    {
        const auto ops = randomCliffordCircuit(n, 25, rng);
        const StabilizerState t = tableauFor(n, ops);
        const Eigen::VectorXcd ref = referenceState(n, ops);
        for (const auto& p : allPauliStrings(n))
        {
            SCOPED_TRACE(std::format("trial {}, P = {}", trial, p));
            EXPECT_NEAR(t.expectation(p, all), referenceExpectation(n, ref, p, all), kTol);
        }
    }

    // Permuted and partial qubit lists address the same operators.
    StabilizerState bell{3};
    bell.h(2);
    bell.cnot(2, 0);
    EXPECT_EQ(bell.expectation("XX", QubitList{2, 0}), 1);
    EXPECT_EQ(bell.expectation("YY", QubitList{0, 2}), -1);
    EXPECT_EQ(bell.expectation("Z", QubitList{0}), 0);
    EXPECT_EQ(bell.expectation("Z", QubitList{1}), 1);
    EXPECT_EQ(bell.expectation("", QubitList{}), 1);
}

TEST(Stabilizer, ReportedGeneratorsStabilizeTheState)
{
    std::mt19937_64 rng{604};
    constexpr std::size_t n = 4;
    const QubitList all{0, 1, 2, 3};
    for (int trial = 0; trial < 20; ++trial)
    {
        const auto ops = randomCliffordCircuit(n, 30, rng);
        const Eigen::VectorXcd ref = referenceState(n, ops);
        for (const auto& g : tableauFor(n, ops).stabilizers())
        {
            SCOPED_TRACE(g);
            EXPECT_NEAR(referenceExpectation(n, ref, g.substr(1), all), g[0] == '+' ? 1.0 : -1.0, kTol);
        }
    }
}

TEST(Stabilizer, OutcomeSupportIsTheBornDistributionInRankOrder)
{
    std::mt19937_64 rng{605};
    constexpr std::size_t n = 5;
    for (int trial = 0; trial < 40; ++trial)
    {
        const auto ops = randomCliffordCircuit(n, 30, rng);
        const std::size_t k = 1 + static_cast<std::size_t>(trial) % n;
        const QubitList qubits = randomQubits(n, k, rng);
        SCOPED_TRACE(std::format("trial {}, qubits {}", trial, formatQubits(qubits)));

        const OutcomeSupport sup = tableauFor(n, ops).outcome_support(qubits);
        const Eigen::VectorXd exact = marginalOf(n, referenceState(n, ops), qubits);
        const double weight = std::ldexp(1.0, -static_cast<int>(sup.dimension()));

        std::vector<std::uint64_t> nonzero;
        for (std::uint64_t o = 0; o < bit(k); ++o)
            if (exact[ix(o)] > 1e-12)
            {
                nonzero.push_back(o);
                EXPECT_NEAR(exact[ix(o)], weight, kTol) << "outcome " << o;
            }
        EXPECT_EQ(enumerate(sup), nonzero) << "rank order is increasing index order";

        for (std::uint64_t o = 0; o < bit(k); ++o)
            EXPECT_EQ(sup.contains(std::vector<std::uint64_t>{o}), exact[ix(o)] > 1e-12) << "outcome " << o;
    }
}

TEST(Stabilizer, MeasurementCollapsesLikeTheReference)
{
    std::mt19937_64 rng{606};
    constexpr std::size_t n = 4;
    for (int trial = 0; trial < 60; ++trial)
    {
        const auto ops = randomCliffordCircuit(n, 25, rng);
        const Qubit q = static_cast<Qubit>(trial) % n;
        const int choice = trial % 2;
        Eigen::VectorXcd ref = referenceState(n, ops);
        const double p0 = marginalOf(n, ref, {q})[0];

        StabilizerState t = tableauFor(n, ops);
        const int r = t.measure(q, choice);
        SCOPED_TRACE(std::format("trial {}, qubit {}, p0 {}, outcome {}", trial, q, p0, r));
        if (std::abs(p0 - 0.5) < 1e-9) EXPECT_EQ(r, choice) << "random outcomes take the given choice";
        else EXPECT_EQ(r, p0 > 0.5 ? 0 : 1) << "certain outcomes ignore it";

        for (std::size_t i = 0; i < bit(n); ++i)
            if (static_cast<int>((i >> q) & 1U) != r) ref[ix(i)] = 0.0;
        EXPECT_TRUE(statesNearUpToPhase(t.to_state_vector(), ref.normalized()));
        EXPECT_EQ(t.measure(q, 1 - choice), r) << "repeated measurement is certain";
    }
}

TEST(Stabilizer, QubitsAcrossWordBoundariesBehaveLikeASmallRegister)
{
    // The same circuit on 5 qubits and on 5 qubits spread over 4 words of a 201-qubit
    // register: every Pauli expectation and every measurement must agree.
    std::mt19937_64 rng{607};
    constexpr std::size_t n = 5, big = 201;
    const QubitList place{0, 63, 64, 127, 200};
    const QubitList small{0, 1, 2, 3, 4};
    for (int trial = 0; trial < 10; ++trial)
    {
        SCOPED_TRACE(std::format("trial {}", trial));
        const auto ops = randomCliffordCircuit(n, 40, rng);
        StabilizerState a = tableauFor(n, ops);
        StabilizerState b{big};
        for (Operation op : ops)
        {
            for (auto& q : op.controls) q = place[q];
            for (auto& q : op.targets) q = place[q];
            applyClifford(b, op);
        }

        for (const auto& p : allPauliStrings(n)) ASSERT_EQ(a.expectation(p, small), b.expectation(p, place)) << p;
        const OutcomeSupport sa = a.outcome_support(small), sb = b.outcome_support(place);
        EXPECT_EQ(sa.offset, sb.offset);
        EXPECT_EQ(sa.basis, sb.basis);
        for (std::size_t k = 0; k < n; ++k)
        {
            const int choice = static_cast<int>(rng() & 1U);
            EXPECT_EQ(a.measure(small[k], choice), b.measure(place[k], choice));
        }
    }
}

TEST(Stabilizer, LargeGhzStateIsHandledExactly)
{
    constexpr std::size_t n = 1000;
    StabilizerState t{n};
    t.h(0);
    for (Qubit q = 1; q < n; ++q) t.cnot(q - 1, q);

    QubitList all(n);
    std::iota(all.begin(), all.end(), Qubit{0});
    std::string xs(n, 'X');
    EXPECT_EQ(t.expectation(xs, all), 1);
    xs[17] = xs[901] = 'Y';
    EXPECT_EQ(t.expectation(xs, all), -1);
    EXPECT_EQ(t.expectation("ZZ", QubitList{0, 999}), 1);
    EXPECT_EQ(t.expectation("Z", QubitList{500}), 0);

    const OutcomeSupport sup = t.outcome_support(all);
    EXPECT_EQ(sup.dimension(), 1u);
    EXPECT_TRUE(std::ranges::all_of(sup.offset, [](std::uint64_t w) { return w == 0; }));

    EXPECT_EQ(t.measure(500, 1), 1);
    for (const Qubit q : {Qubit{0}, Qubit{1}, Qubit{499}, Qubit{999}}) EXPECT_EQ(t.measure(q, 0), 1);
}

TEST(Stabilizer, InvalidInputThrowsWithoutChangingState)
{
    EXPECT_THROW(StabilizerState{0}, std::length_error);
    EXPECT_THROW(StabilizerState{Qputer::kMaxStabilizerQubits + 1}, std::length_error);
    EXPECT_THROW(StabilizerState{Qputer::kMaxQubits + 1}.to_state_vector(), std::length_error);

    StabilizerState t{3};
    t.h(0);
    t.cnot(0, 1);
    const auto before = t.stabilizers();

    EXPECT_THROW(t.h(3), std::out_of_range);
    EXPECT_THROW(t.cnot(0, 3), std::out_of_range);
    EXPECT_THROW(t.cz(1, 1), std::invalid_argument);
    EXPECT_THROW(t.swap(2, 2), std::invalid_argument);
    EXPECT_THROW(t.measure(3, 0), std::out_of_range);
    EXPECT_THROW(t.measure(0, 2), std::invalid_argument);
    EXPECT_THROW(t.outcome_support(QubitList{}), std::invalid_argument);
    EXPECT_THROW(t.outcome_support(QubitList{0, 0}), std::invalid_argument);
    EXPECT_THROW(t.outcome_support(QubitList{5}), std::out_of_range);
    EXPECT_THROW(t.expectation("XQ", QubitList{0, 1}), std::invalid_argument);
    EXPECT_THROW(t.expectation("XX", QubitList{0}), std::invalid_argument);
    EXPECT_THROW(t.expectation("XX", QubitList{1, 1}), std::invalid_argument);
    EXPECT_THROW(t.outcome_support(QubitList{0}).contains(std::vector<std::uint64_t>{}), std::invalid_argument);

    EXPECT_EQ(t.stabilizers(), before);
}
