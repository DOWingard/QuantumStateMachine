// Bit-flip repetition code. A logical qubit a|0_L> + b|1_L> is stored as a|0...0> + b|1...1>
// on d data qubits. Ancillas between neighbouring data qubits measure the parities
// s_i = Z_i Z_{i+1} without measuring any data qubit, so superpositions survive. Since
// s_i = e_i XOR e_{i+1} for a bit-flip pattern e, the syndrome fixes e up to a global flip:
// e_0 = 0, e_{i+1} = e_i XOR s_i, or its complement. The decoder applies the lighter of the
// two, which succeeds while fewer than d/2 qubits flipped.
//
// The same code runs on both backends: d = 7 uses 13 qubits (state vector), d = 500 uses 999
// qubits, above the state-vector limit, so it runs on the stabilizer tableau.

#include <QuantumStateMachine.hpp>

#include <algorithm>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <vector>

using namespace Qputer;

namespace
{

enum class Logical { Zero, Plus };

struct MemoryResult
{
    bool inCodeSpace;  // every Z_0 Z_i = +1 after correction
    double logical;    // <Z_L> = <Z_0> for |0_L>, <X_L> = <X...X> for |+_L>
    Backend backend;
};

// Data qubit i is qubit 2i; the ancilla for the pair (i, i+1) is qubit 2i + 1.
MemoryResult runMemory(std::size_t d, Logical input, const std::vector<std::size_t>& flips)
{
    QuantumStateMachine m{2 * d - 1};
    QubitList data;
    for (std::size_t i = 0; i < d; ++i) data.push_back(2 * i);

    if (input == Logical::Plus)
    {
        m.h(data[0]);
        for (std::size_t i = 1; i < d; ++i) m.cnot(data[i - 1], data[i]);
    }
    for (const std::size_t i : flips) m.x(data[i]);

    // Syndrome extraction. Each outcome is read live: d - 1 bits can exceed the 64 clbits run() has.
    std::vector<int> syndrome(d - 1);
    for (std::size_t i = 0; i + 1 < d; ++i)
    {
        const Qubit ancilla = 2 * i + 1;
        m.cnot(data[i], ancilla).cnot(data[i + 1], ancilla);
        syndrome[i] = m.measure(ancilla);
    }

    std::vector<int> e(d, 0);
    for (std::size_t i = 0; i + 1 < d; ++i) e[i + 1] = e[i] ^ syndrome[i];
    const int complement = 2 * std::ranges::count(e, 1) > static_cast<std::ptrdiff_t>(d) ? 1 : 0;
    for (std::size_t i = 0; i < d; ++i)
        if (e[i] != complement) m.x(data[i]);

    bool inCodeSpace = true;
    for (std::size_t i = 1; i < d; ++i) inCodeSpace = inCodeSpace && m.expectation("ZZ", {data[0], data[i]}) > 0.5;
    const double logical = input == Logical::Zero ? m.expectation("Z", {data[0]})
                                                  : m.expectation(std::string(d, 'X'), data);
    return {inCodeSpace, logical, m.backend()};
}

} // namespace


int main()
{
    struct Case
    {
        std::string_view label;
        std::size_t d;
        Logical input;
        std::vector<std::size_t> flips;
        double expectedLogical; // -1: more than d/2 flips, so the decoder makes a logical error
    };
    const std::vector<Case> cases{
        {"|0_L>, 2 of 7 flipped", 7, Logical::Zero, {1, 5}, +1.0},
        {"|0_L>, 4 of 7 flipped", 7, Logical::Zero, {0, 2, 3, 6}, -1.0},
        {"|+_L>, 4 of 500 flipped", 500, Logical::Plus, {3, 4, 200, 431}, +1.0},
    };

    bool ok = true;
    for (const Case& c : cases)
    {
        const MemoryResult r = runMemory(c.d, c.input, c.flips);
        std::cout << std::format("{:24} {:4} qubits, {:12} backend: code space {}, logical {} = {:+.0f}\n", c.label,
                                 2 * c.d - 1, r.backend == Backend::Stabilizer ? "stabilizer" : "state-vector",
                                 r.inCodeSpace ? "yes" : "no", c.input == Logical::Zero ? "<Z>" : "<X>", r.logical);
        ok = ok && r.inCodeSpace && r.logical == c.expectedLogical;
    }

    if (!ok)
    {
        std::cerr << "repetition_code: a decoding result differs from the expected one\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
