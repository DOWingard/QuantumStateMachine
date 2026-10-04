#include <QuantumStateMachine.hpp>

#include <format>
#include <iostream>
#include <string_view>

using Qputer::Counts;
using Qputer::QuantumStateMachine;

namespace
{

void printProbabilities(std::string_view title, const QuantumStateMachine& m)
{
    std::cout << title << "\n";
    const Eigen::VectorXd p = m.probabilities();
    for (Eigen::Index i = 0; i < p.size(); ++i)
        if (p[i] > 1e-12)
            std::cout << std::format("  |{}>  p = {:.6f}\n",
                                     QuantumStateMachine::bitstring(static_cast<Qputer::Outcome>(i), m.num_qubits()),
                                     p[i]);
}

void printCounts(std::string_view title, const Counts& counts, std::size_t width)
{
    std::cout << title << "\n";
    for (const auto& [reg, c] : counts)
        std::cout << std::format("  {}  {}\n", QuantumStateMachine::bitstring(reg, width), c);
}

} // namespace


int main()
{
    constexpr std::uint64_t seed = 2026;
    constexpr std::size_t shots = 10000;

    // Bell pair: exact distribution off the state vector, then sampled shots.
    QuantumStateMachine bell{2, 2, seed};
    bell.h(0).cnot(0, 1);
    printProbabilities("Bell state probabilities:", bell);
    std::cout << std::format("  <ZZ> = {:+.6f}  <XX> = {:+.6f}  <YY> = {:+.6f}\n",
                             bell.expectation("ZZ", {0, 1}), bell.expectation("XX", {0, 1}),
                             bell.expectation("YY", {0, 1}));
    bell.measure_all();
    printCounts(std::format("Bell state, {} shots:", shots), bell.run(shots), 2);

    // GHZ on 5 qubits, marginal over a subset.
    QuantumStateMachine ghz{5, 5, seed};
    ghz.h(0);
    for (Qputer::Qubit q = 1; q < 5; ++q) ghz.cnot(q - 1, q);
    const Eigen::VectorXd marginal = ghz.marginal_probabilities({0, 4});
    std::cout << std::format("GHZ marginal over (q0, q4): [{:.3f}, {:.3f}, {:.3f}, {:.3f}]\n",
                             marginal[0], marginal[1], marginal[2], marginal[3]);

    // Teleportation: mid-circuit measurement and classically controlled corrections.
    // After undoing the input rotation on qubit 2, clbit 2 must always read 0.
    constexpr double theta = 1.1;
    QuantumStateMachine tele{3, 3, seed};
    tele.ry(0, theta);
    tele.h(1).cnot(1, 2);
    tele.cnot(0, 1).h(0);
    tele.measure(0, 0);
    tele.measure(1, 1);
    tele.when(1).x(2);
    tele.when(0).z(2);
    tele.ry(2, -theta);
    tele.measure(2, 2);
    printCounts(std::format("Teleportation, {} shots (bit 2 must be 0):", shots), tele.run(shots), 3);

    // 1000 qubits exceed the state-vector limit, so Auto runs this Clifford circuit on the
    // stabilizer tableau. Qubits 0, 500 and 999 always agree.
    constexpr std::size_t wide = 1000;
    QuantumStateMachine big{wide, 3, seed};
    big.h(0);
    for (Qputer::Qubit q = 1; q < wide; ++q) big.cnot(q - 1, q);
    std::cout << std::format("GHZ on {} qubits ({} backend): <Z0 Z999> = {:+.1f}  <Z500> = {:+.1f}\n", wide,
                             big.backend() == Qputer::Backend::Stabilizer ? "stabilizer" : "state-vector",
                             big.expectation("ZZ", {0, 999}), big.expectation("Z", {500}));
    big.measure(0, 0);
    big.measure(500, 1);
    big.measure(999, 2);
    printCounts(std::format("GHZ-{} qubits (0, 500, 999), {} shots:", wide, shots), big.run(shots), 3);

    std::cout << std::format("Seed {} reproduces every result above.\n", seed);
    return 0;
}
