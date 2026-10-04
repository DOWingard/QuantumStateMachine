// Variational quantum eigensolver (VQE): minimize E(theta) = <psi(theta)|H|psi(theta)> over a
// parameterized circuit psi(theta). H is a weighted sum of Pauli strings, so E is the same
// weighted sum of Pauli expectation values. Every parameter enters through one gate
// exp(-i theta P / 2) (here RY), for which the parameter-shift rule gives the exact gradient
//   dE/dtheta_i = [E(theta + pi/2 e_i) - E(theta - pi/2 e_i)] / 2.
// On hardware each <P> is estimated from shots; the simulator returns it exactly. The
// gradient drives Adam (Kingma & Ba, 2015), which converges here in about a third of the
// steps plain gradient descent needs.
//
// To solve your own problem, replace the Hamiltonian terms and the ansatz.

#include <QuantumStateMachine.hpp>

#include <cmath>
#include <complex>
#include <cstdlib>
#include <format>
#include <iostream>
#include <numbers>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Dense>

using namespace Qputer;

namespace
{

constexpr double kPi = std::numbers::pi;

struct PauliTerm
{
    double coeff;
    std::string paulis; // letter k acts on qubits[k]
    QubitList qubits;
};

using Hamiltonian = std::vector<PauliTerm>;

// Open transverse-field Ising chain: H = -sum_i Z_i Z_{i+1} - h sum_i X_i.
Hamiltonian transverseFieldIsing(std::size_t n, double h)
{
    Hamiltonian H;
    for (Qubit i = 0; i + 1 < n; ++i) H.push_back({-1.0, "ZZ", {i, i + 1}});
    for (Qubit i = 0; i < n; ++i) H.push_back({-h, "X", {i}});
    return H;
}

// `layers` blocks of RY on every qubit followed by a CNOT chain, then a final RY layer:
// n (layers + 1) parameters. RY and CNOT are real, which suffices because this H is real.
std::size_t parameterCount(std::size_t n, std::size_t layers) { return n * (layers + 1); }

void ansatz(QuantumStateMachine& m, std::span<const double> theta, std::size_t layers)
{
    const std::size_t n = m.num_qubits();
    std::size_t p = 0;
    for (std::size_t l = 0; l <= layers; ++l)
    {
        for (Qubit q = 0; q < n; ++q) m.ry(q, theta[p++]);
        if (l < layers)
            for (Qubit q = 0; q + 1 < n; ++q) m.cnot(q, q + 1);
    }
}

double energy(const Hamiltonian& H, std::size_t n, std::span<const double> theta, std::size_t layers)
{
    QuantumStateMachine m{n, 0, 0}; // fixed seed: nothing here is random
    ansatz(m, theta, layers);
    double e = 0.0;
    for (const PauliTerm& term : H) e += term.coeff * m.expectation(term.paulis, term.qubits);
    return e;
}

// Lowest eigenvalue of H built as a dense 2^n x 2^n matrix: a check that is only feasible for
// small n. Qubit q is bit q of the basis index, so qubit n-1 is the leftmost Kronecker factor.
double exactGroundEnergy(const Hamiltonian& H, std::size_t n)
{
    const Eigen::Index dim = Eigen::Index{1} << n;
    const std::complex<double> i{0.0, 1.0};
    Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(dim, dim);
    for (const PauliTerm& term : H)
    {
        Eigen::MatrixXcd P = Eigen::MatrixXcd::Identity(1, 1);
        for (std::size_t q = n; q-- > 0;)
        {
            Eigen::Matrix2cd sigma = Eigen::Matrix2cd::Identity();
            for (std::size_t k = 0; k < term.qubits.size(); ++k)
                if (term.qubits[k] == q)
                {
                    if (term.paulis[k] == 'X') sigma << 0, 1, 1, 0;
                    if (term.paulis[k] == 'Y') sigma << 0, -i, i, 0;
                    if (term.paulis[k] == 'Z') sigma << 1, 0, 0, -1;
                }
            Eigen::MatrixXcd next(P.rows() * 2, P.cols() * 2);
            for (Eigen::Index r = 0; r < P.rows(); ++r)
                for (Eigen::Index c = 0; c < P.cols(); ++c) next.block<2, 2>(2 * r, 2 * c) = P(r, c) * sigma;
            P = std::move(next);
        }
        M += term.coeff * P;
    }
    return Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd>(M, Eigen::EigenvaluesOnly).eigenvalues()[0];
}

} // namespace


int main()
{
    // Depth limits what the ansatz can reach: 3 layers are still 1e-3 above E0 after 2000
    // steps, while 5 layers converge in 200.
    constexpr std::size_t n = 4, layers = 5, steps = 200;
    constexpr double h = 1.0, learningRate = 0.05, beta1 = 0.9, beta2 = 0.999;
    const Hamiltonian H = transverseFieldIsing(n, h);

    // Small random start: theta = 0 is a stationary point of this ansatz.
    std::mt19937_64 rng{2026};
    std::normal_distribution<double> start{0.0, 0.1};
    std::vector<double> theta(parameterCount(n, layers));
    for (double& t : theta) t = start(rng);

    std::vector<double> grad(theta.size()), moment1(theta.size()), moment2(theta.size());
    for (std::size_t step = 0; step <= steps; ++step)
    {
        if (step % 50 == 0)
            std::cout << std::format("step {:3}: E = {:+.8f}\n", step, energy(H, n, theta, layers));
        for (std::size_t p = 0; p < theta.size(); ++p)
        {
            std::vector<double> shifted = theta;
            shifted[p] = theta[p] + kPi / 2;
            const double plus = energy(H, n, shifted, layers);
            shifted[p] = theta[p] - kPi / 2;
            grad[p] = 0.5 * (plus - energy(H, n, shifted, layers));
        }

        // Adam: per-parameter step from bias-corrected running means of g and g^2.
        const double correction1 = 1.0 - std::pow(beta1, static_cast<double>(step + 1));
        const double correction2 = 1.0 - std::pow(beta2, static_cast<double>(step + 1));
        for (std::size_t p = 0; p < theta.size(); ++p)
        {
            moment1[p] = beta1 * moment1[p] + (1.0 - beta1) * grad[p];
            moment2[p] = beta2 * moment2[p] + (1.0 - beta2) * grad[p] * grad[p];
            theta[p] -= learningRate * (moment1[p] / correction1) / (std::sqrt(moment2[p] / correction2) + 1e-12);
        }
    }

    const double found = energy(H, n, theta, layers), exact = exactGroundEnergy(H, n);
    std::cout << std::format("VQE on a {}-qubit transverse-field Ising chain (h = {}):\n"
                             "  variational E = {:+.8f}\n  exact E0      = {:+.8f}\n", n, h, found, exact);

    if (!(found - exact < 1e-4))
    {
        std::cerr << "variational: did not reach the exact ground energy\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
