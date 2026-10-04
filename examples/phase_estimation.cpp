// Quantum phase estimation. For a unitary U with eigenstate U|u> = e^{2 pi i phi} |u>, each of
// t counting qubits k controls U^(2^k), leaving the counting register in
//   sum_y e^{2 pi i phi y} |y> / 2^(t/2).
// The inverse QFT maps that to |phi 2^t> when phi 2^t is an integer; otherwise the readout
// peaks at the nearest integer y, with probability (sin(pi delta) / (2^t sin(pi delta / 2^t)))^2
// for delta = |phi 2^t - y|, which is at least 4 / pi^2.
//
// To estimate the phase of your own unitary, pass its matrix (2^M x 2^M, M <= 10) and a
// function that prepares one of its eigenstates on the target qubits.

#include <QuantumStateMachine.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <format>
#include <functional>
#include <iostream>
#include <numbers>
#include <utility>
#include <vector>

using namespace Qputer;

namespace
{

constexpr double kPi = std::numbers::pi;

using Preparation = std::function<void(QuantumStateMachine&, const QubitList& targets)>;

// q[0] is the least significant bit: the QFT's gates in reverse order with negated angles.
void inverseQft(QuantumStateMachine& m, const QubitList& q)
{
    const std::size_t t = q.size();
    for (std::size_t i = 0; i < t / 2; ++i) m.swap(q[i], q[t - 1 - i]);
    for (std::size_t j = 0; j < t; ++j)
    {
        for (std::size_t k = 0; k < j; ++k) m.cphase(q[k], q[j], -kPi / std::ldexp(1.0, static_cast<int>(j - k)));
        m.h(q[j]);
    }
}

// Counting qubits 0..t-1 read out y (clbit k = qubit k); U acts on the qubits after them.
Counts estimatePhase(const Eigen::MatrixXcd& U, const Preparation& prepareEigenstate, std::size_t t,
                     std::size_t shots, std::uint64_t seed)
{
    const auto numTargets = static_cast<std::size_t>(std::countr_zero(static_cast<std::size_t>(U.rows())));
    QuantumStateMachine m{t + numTargets, t, seed};
    QubitList counting, targets;
    for (Qubit k = 0; k < t; ++k) counting.push_back(k);
    for (Qubit k = 0; k < numTargets; ++k) targets.push_back(t + k);

    prepareEigenstate(m, targets);
    for (const Qubit k : counting) m.h(k);
    Eigen::MatrixXcd power = U; // U^(2^k) by repeated squaring
    for (const Qubit k : counting)
    {
        m.controlled_unitary({k}, targets, power);
        power = power * power;
    }
    inverseQft(m, counting);
    for (const Qubit k : counting) m.measure(k, k);
    return m.run(shots);
}

} // namespace


int main()
{
    // U = diag(1, e^{2 pi i phi}) with eigenstate |1>; phi 2^t = 76.8 falls between outcomes.
    constexpr double phi = 0.3;
    constexpr std::size_t t = 8, shots = 10000;
    Eigen::MatrixXcd U = Eigen::MatrixXcd::Identity(2, 2);
    U(1, 1) = std::polar(1.0, 2 * kPi * phi);

    const Counts counts = estimatePhase(U, [](QuantumStateMachine& m, const QubitList& q) { m.x(q[0]); }, t,
                                        shots, 2026);

    std::vector<std::pair<Outcome, std::size_t>> ranked(counts.begin(), counts.end());
    std::ranges::sort(ranked, std::greater{}, &std::pair<Outcome, std::size_t>::second);
    std::cout << std::format("Phase estimation of phi = {} with {} counting qubits, {} shots:\n", phi, t, shots);
    for (std::size_t r = 0; r < std::min<std::size_t>(3, ranked.size()); ++r)
        std::cout << std::format("  y = {:3}  phi ~ {:.5f}  {:5} shots\n", ranked[r].first,
                                 std::ldexp(static_cast<double>(ranked[r].first), -static_cast<int>(t)),
                                 ranked[r].second);

    const double scaled = std::ldexp(phi, static_cast<int>(t));
    const auto peak = static_cast<Outcome>(std::lround(scaled));
    const double delta = std::abs(scaled - static_cast<double>(peak));
    const double pPeak = std::pow(std::sin(kPi * delta) / (std::ldexp(1.0, static_cast<int>(t)) *
                                                           std::sin(kPi * std::ldexp(delta, -static_cast<int>(t)))), 2);
    const double observed = static_cast<double>(counts.contains(peak) ? counts.at(peak) : 0) / shots;
    std::cout << std::format("Peak y = {}: observed probability {:.4f}, theory {:.4f}\n", peak, observed, pPeak);

    if (ranked.front().first != peak || std::abs(observed - pPeak) > 0.02)
    {
        std::cerr << "phase_estimation: readout does not match the theoretical peak\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
