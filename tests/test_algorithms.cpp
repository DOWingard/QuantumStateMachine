#include "TestSupport.hpp"

#include <cmath>
#include <tuple>
#include <vector>

using namespace QputerTest;

namespace
{

// Textbook QFT on qubit n-1 (MSB) .. 0, then bit-reversal swaps, so that
// QFT|j> = 2^{-n/2} sum_k exp(2 pi i j k / 2^n) |k> with the library's LSB-first indexing.
void qft(QuantumStateVector& s)
{
    const std::size_t n = s.num_qubits();
    for (std::size_t q = n; q-- > 0;)
    {
        QuantumGate::h(s, q);
        for (std::size_t c = q; c-- > 0;)
            QuantumGate::cphase(s, c, q, kPi / static_cast<double>(bit(q - c)));
    }
    for (std::size_t q = 0; q < n / 2; ++q) QuantumGate::swap(s, q, n - 1 - q);
}

void hadamardAll(QuantumStateVector& s, std::size_t count)
{
    for (Qubit q = 0; q < count; ++q) QuantumGate::h(s, q);
}

// Phase-flip oracle for the single marked basis state |marked> on qubits [0, n).
void markOracle(QuantumStateVector& s, std::size_t n, std::size_t marked)
{
    QubitList all;
    for (Qubit q = 0; q < n; ++q)
    {
        all.push_back(q);
        if (!((marked >> q) & 1U)) QuantumGate::x(s, q);
    }
    QuantumGate::mcz(s, all);
    for (Qubit q = 0; q < n; ++q)
        if (!((marked >> q) & 1U)) QuantumGate::x(s, q);
}

// Grover diffusion: H^n (I - 2|0><0|) H^n = -(2|s><s| - I), i.e. the reflection about the
// uniform state |s> up to a global phase that leaves probabilities unchanged.
void diffusion(QuantumStateVector& s, std::size_t n)
{
    hadamardAll(s, n);
    markOracle(s, n, 0);
    hadamardAll(s, n);
}

double probability(const QuantumStateVector& s, std::size_t index) { return std::norm(s[index]); }

} // namespace


TEST(Algorithms, QftOfBasisStateIsFourierVector)
{
    for (std::size_t n = 1; n <= 7; ++n)
    {
        const std::size_t N = bit(n);
        const double scale = 1.0 / std::sqrt(static_cast<double>(N));
        for (std::size_t j = 0; j < N; ++j)
        {
            SCOPED_TRACE(std::format("n = {}, j = {}", n, j));
            QuantumStateVector s = basis(n, j);
            qft(s);

            Eigen::VectorXcd expected(ix(N));
            for (std::size_t k = 0; k < N; ++k)
                expected[ix(k)] = scale * expi(2 * kPi * static_cast<double>((j * k) % N) / static_cast<double>(N));
            EXPECT_TRUE(statesNear(s, expected));
        }
    }
}

TEST(Algorithms, QftOfUniformSuperpositionIsZeroState)
{
    // H^n|0> = QFT|0>, and QFT^2 |j> = |-j mod N>, so QFT(H^n |0>) = |0>.
    for (std::size_t n = 1; n <= 8; ++n)
    {
        SCOPED_TRACE(std::format("n = {}", n));
        QuantumStateVector s{n};
        hadamardAll(s, n);
        qft(s);
        EXPECT_TRUE(statesNear(s, amplitudes(n, {{0, 1.0}})));
    }
}

// Two-qubit Grover finds the marked item with certainty after one iteration (theta = pi/6).
TEST(Algorithms, GroverTwoQubitsIsDeterministic)
{
    for (std::size_t marked = 0; marked < 4; ++marked)
    {
        SCOPED_TRACE(std::format("marked = {}", marked));
        QuantumStateVector s{2};
        hadamardAll(s, 2);
        markOracle(s, 2, marked);
        diffusion(s, 2);
        EXPECT_NEAR(probability(s, marked), 1.0, kTol);
    }
}

// After k iterations, P(marked) = sin^2((2k + 1) theta) with sin(theta) = 2^{-n/2};
// for n = 3, k = 2 this is the well-known 121/128.
TEST(Algorithms, GroverSuccessProbabilityMatchesClosedForm)
{
    for (std::size_t n = 3; n <= 9; ++n)
    {
        const double theta = std::asin(1.0 / std::sqrt(static_cast<double>(bit(n))));
        const auto iterations = static_cast<std::size_t>(std::floor(kPi / (4 * theta)));
        const std::size_t marked = (bit(n) - 1) / 3;  // an arbitrary non-trivial bit pattern
        SCOPED_TRACE(std::format("n = {}, k = {}, marked = {}", n, iterations, marked));

        QuantumStateVector s{n};
        hadamardAll(s, n);
        for (std::size_t k = 0; k < iterations; ++k)
        {
            markOracle(s, n, marked);
            diffusion(s, n);
        }

        const double expected = std::pow(std::sin(static_cast<double>(2 * iterations + 1) * theta), 2);
        EXPECT_NEAR(probability(s, marked), expected, 1e-11);
        if (n == 3)
        {
            EXPECT_NEAR(probability(s, marked), 121.0 / 128.0, 1e-11);
        }

        // All unmarked items share the remaining probability equally.
        const double rest = (1.0 - expected) / static_cast<double>(bit(n) - 1);
        for (std::size_t i = 0; i < bit(n); ++i)
        {
            if (i != marked)
            {
                EXPECT_NEAR(probability(s, i), rest, 1e-11) << "index " << i;
            }
        }
    }
}

// Bernstein-Vazirani: one query to f(x) = s.x (mod 2) leaves the data register in |s>.
TEST(Algorithms, BernsteinVaziraniRecoversSecretInOneQuery)
{
    for (const auto& [n, secrets] : std::vector<std::tuple<std::size_t, std::vector<std::size_t>>>{
             {4, {0b0000, 0b0001, 0b1010, 0b0111, 0b1111}},
             {8, {0b10110010, 0b01001101, 0b11111111}}})
    {
        for (const std::size_t secret : secrets)
        {
            SCOPED_TRACE(std::format("n = {}, s = {:#b}", n, secret));
            const Qubit ancilla = n;
            QuantumStateVector st{n + 1};
            QuantumGate::x(st, ancilla);
            hadamardAll(st, n + 1);
            for (Qubit q = 0; q < n; ++q)
                if ((secret >> q) & 1U) QuantumGate::cnot(st, q, ancilla);
            hadamardAll(st, n);

            // Data = |s>, ancilla = |-> = (|0> - |1>)/sqrt2.
            EXPECT_TRUE(statesNear(st, amplitudes(n + 1, {{secret, kInvSqrt2}, {secret | bit(ancilla), -kInvSqrt2}})));
        }
    }
}

// Deutsch-Jozsa: amplitude of |0...0> on the data register is +-1 for constant f, 0 for balanced f.
TEST(Algorithms, DeutschJozsaSeparatesConstantFromBalanced)
{
    constexpr std::size_t n = 4;
    constexpr Qubit ancilla = n;

    struct Oracle
    {
        const char* name;
        bool balanced;
        std::function<void(QuantumStateVector&)> apply;
    };
    const std::vector<Oracle> oracles{
        {"f = 0", false, [](auto&) {}},
        {"f = 1", false, [](auto& s) { QuantumGate::x(s, ancilla); }},
        {"f = x0", true, [](auto& s) { QuantumGate::cnot(s, 0, ancilla); }},
        {"f = x1 ^ x3", true, [](auto& s) { QuantumGate::cnot(s, 1, ancilla); QuantumGate::cnot(s, 3, ancilla); }},
        {"f = parity", true, [](auto& s) { for (Qubit q = 0; q < n; ++q) QuantumGate::cnot(s, q, ancilla); }},
    };

    for (const auto& o : oracles)
    {
        SCOPED_TRACE(o.name);
        QuantumStateVector s{n + 1};
        QuantumGate::x(s, ancilla);
        hadamardAll(s, n + 1);
        o.apply(s);
        hadamardAll(s, n);

        // Probability that the data register reads 0, marginalised over the ancilla.
        const double p0 = probability(s, 0) + probability(s, bit(ancilla));
        EXPECT_NEAR(p0, o.balanced ? 0.0 : 1.0, kTol);
    }
}

// Teleportation with deferred measurement: classically controlled corrections become
// CNOT / CZ, after which qubit 2 holds |psi> and qubits 0, 1 are left in |+>|+>.
TEST(Algorithms, TeleportationTransfersArbitraryState)
{
    const std::vector<std::tuple<double, double, double>> inputs{
        {0, 0, 0}, {kPi, 0, 0}, {kPi / 2, 0, kPi}, {1.234, -0.5, 2.1}, {2.9, 1.7, -0.3}};

    for (const auto& [th, ph, la] : inputs)
    {
        SCOPED_TRACE(std::format("psi = U3({}, {}, {})|0>", th, ph, la));
        const Eigen::Vector2cd psi = matU3(th, ph, la).col(0);

        QuantumStateVector s{3};
        QuantumGate::u3(s, 0, th, ph, la);
        QuantumGate::h(s, 1);
        QuantumGate::cnot(s, 1, 2);   // Bell pair shared by qubits 1, 2
        QuantumGate::cnot(s, 0, 1);
        QuantumGate::h(s, 0);         // Alice's Bell-basis rotation
        QuantumGate::cnot(s, 1, 2);   // X correction if m1
        QuantumGate::cz(s, 0, 2);     // Z correction if m0

        Eigen::VectorXcd expected(8);
        for (std::size_t i = 0; i < 8; ++i) expected[ix(i)] = 0.5 * psi[ix(i >> 2)];
        EXPECT_TRUE(statesNear(s, expected));
    }
}

// Superdense coding: Z^z X^x on one half of a Bell pair encodes two classical bits, and the
// Bell-basis decoder recovers them as basis state |x z> with probability 1.
TEST(Algorithms, SuperdenseCodingDecodesTwoBits)
{
    for (std::size_t msg = 0; msg < 4; ++msg)
    {
        const bool xBit = (msg >> 1) & 1U, zBit = msg & 1U;
        SCOPED_TRACE(std::format("message x={} z={}", xBit, zBit));

        QuantumStateVector s{2};
        QuantumGate::h(s, 0);
        QuantumGate::cnot(s, 0, 1);
        if (zBit) QuantumGate::z(s, 0);
        if (xBit) QuantumGate::x(s, 0);
        QuantumGate::cnot(s, 0, 1);
        QuantumGate::h(s, 0);

        EXPECT_NEAR(probability(s, msg), 1.0, kTol);
    }
}
