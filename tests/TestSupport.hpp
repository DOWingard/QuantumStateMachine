#pragma once

#include <QuantumGates.hpp>
#include <QuantumState.hpp>
#include <QuantumStateMachine.hpp>
#include <StabilizerState.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <format>
#include <initializer_list>
#include <functional>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <Eigen/Dense>
#include <gtest/gtest.h>

#ifdef _OPENMP
#include <omp.h>
#endif



namespace QputerTest
{

using cd = std::complex<double>;
using Qputer::QuantumGate;
using Qputer::QuantumStateVector;
using Qputer::Qubit;
using Qputer::QubitList;

// Tight enough to catch any wrong matrix entry, loose enough for O(depth * eps) rounding.
inline constexpr double kTol = 1e-12;
inline constexpr double kInvSqrt2 = std::numbers::sqrt2 / 2.0;
inline constexpr double kPi = std::numbers::pi;
inline const cd kI{0.0, 1.0};

inline Eigen::Index ix(std::size_t i) { return static_cast<Eigen::Index>(i); }
inline std::size_t bit(std::size_t q) { return std::size_t{1} << q; }
inline cd expi(double phi) { return std::polar(1.0, phi); }


// ---- State construction ----

// Computational basis state |index> on n qubits (qubit q = bit q of index).
inline QuantumStateVector basis(std::size_t n, std::size_t index)
{
    QuantumStateVector s{n};
    s[0] = 0.0;
    s[index] = 1.0;
    return s;
}

inline QuantumStateVector fromAmplitudes(std::size_t n, const Eigen::VectorXcd& amps)
{
    QuantumStateVector s{n};
    for (std::size_t i = 0; i < s.size(); ++i) s[i] = amps[ix(i)];
    return s;
}

inline Eigen::VectorXcd randomAmplitudes(std::size_t n, std::mt19937_64& rng)
{
    std::normal_distribution<double> g;
    Eigen::VectorXcd v(ix(bit(n)));
    for (auto& a : v) a = cd{g(rng), g(rng)};
    return v.normalized();
}

inline QuantumStateVector randomState(std::size_t n, std::mt19937_64& rng)
{
    return fromAmplitudes(n, randomAmplitudes(n, rng));
}

// Haar-distributed unitary: QR of a complex Ginibre matrix with R's diagonal phases removed.
inline Eigen::MatrixXcd randomUnitary(std::size_t dim, std::mt19937_64& rng)
{
    std::normal_distribution<double> g;
    Eigen::MatrixXcd z(ix(dim), ix(dim));
    for (auto& a : z.reshaped()) a = cd{g(rng), g(rng)};
    const Eigen::HouseholderQR<Eigen::MatrixXcd> qr(z);
    Eigen::MatrixXcd q = qr.householderQ();
    const Eigen::MatrixXcd r = qr.matrixQR().triangularView<Eigen::Upper>();
    for (Eigen::Index k = 0; k < ix(dim); ++k) q.col(k) *= r(k, k) / std::abs(r(k, k));
    return q;
}

// Distinct qubits drawn uniformly from [0, n).
inline QubitList randomQubits(std::size_t n, std::size_t count, std::mt19937_64& rng)
{
    QubitList all(n);
    for (std::size_t q = 0; q < n; ++q) all[q] = q;
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(count);
    return all;
}


// ---- Textbook single-qubit matrices ----

inline Eigen::Matrix2cd mat(cd m00, cd m01, cd m10, cd m11)
{
    Eigen::Matrix2cd m;
    m << m00, m01, m10, m11;
    return m;
}

inline Eigen::Matrix2cd matI() { return mat(1, 0, 0, 1); }
inline Eigen::Matrix2cd matX() { return mat(0, 1, 1, 0); }
inline Eigen::Matrix2cd matY() { return mat(0, -kI, kI, 0); }
inline Eigen::Matrix2cd matZ() { return mat(1, 0, 0, -1); }
inline Eigen::Matrix2cd matH() { return mat(kInvSqrt2, kInvSqrt2, kInvSqrt2, -kInvSqrt2); }
inline Eigen::Matrix2cd matPhase(double l) { return mat(1, 0, 0, expi(l)); }
inline Eigen::Matrix2cd matS() { return matPhase(kPi / 2); }
inline Eigen::Matrix2cd matSdg() { return matPhase(-kPi / 2); }
inline Eigen::Matrix2cd matT() { return matPhase(kPi / 4); }
inline Eigen::Matrix2cd matTdg() { return matPhase(-kPi / 4); }
inline Eigen::Matrix2cd matSX() { return 0.5 * mat(cd{1, 1}, cd{1, -1}, cd{1, -1}, cd{1, 1}); }

inline Eigen::Matrix2cd matRX(double t)
{
    return mat(std::cos(t / 2), -kI * std::sin(t / 2), -kI * std::sin(t / 2), std::cos(t / 2));
}
inline Eigen::Matrix2cd matRY(double t)
{
    return mat(std::cos(t / 2), -std::sin(t / 2), std::sin(t / 2), std::cos(t / 2));
}
inline Eigen::Matrix2cd matRZ(double t) { return mat(expi(-t / 2), 0, 0, expi(t / 2)); }
inline Eigen::Matrix2cd matU3(double t, double p, double l)
{
    const double c = std::cos(t / 2), s = std::sin(t / 2);
    return mat(c, -expi(l) * s, expi(p) * s, expi(p + l) * c);
}

inline Eigen::Matrix4cd matSwap()
{
    Eigen::Matrix4cd m = Eigen::Matrix4cd::Zero();
    m(0, 0) = m(1, 2) = m(2, 1) = m(3, 3) = 1;
    return m;
}

inline Eigen::MatrixXcd kron(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b)
{
    Eigen::MatrixXcd k(a.rows() * b.rows(), a.cols() * b.cols());
    for (Eigen::Index r = 0; r < a.rows(); ++r)
        for (Eigen::Index c = 0; c < a.cols(); ++c)
            k.block(r * b.rows(), c * b.cols(), b.rows(), b.cols()) = a(r, c) * b;
    return k;
}


// ---- Reference operator ----

// Full 2^n x 2^n operator for U on `targets` (targets[0] = MSB of U's index), active only
// where every control is |1>. Built column-by-column from the documented convention,
// sharing no code with the library kernels, so agreement is independent evidence.
inline Eigen::MatrixXcd embed(std::size_t n, const QubitList& controls, const QubitList& targets,
                              const Eigen::MatrixXcd& U)
{
    const std::size_t dim = bit(n);
    const std::size_t m = targets.size();
    Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(ix(dim), ix(dim));

    for (std::size_t col = 0; col < dim; ++col)
    {
        bool active = true;
        for (const Qubit c : controls) active = active && ((col >> c) & 1U);
        if (!active)
        {
            M(ix(col), ix(col)) = 1.0;
            continue;
        }

        std::size_t v = 0;
        std::size_t cleared = col;
        for (const Qubit t : targets)
        {
            v = (v << 1) | ((col >> t) & 1U);
            cleared &= ~bit(t);
        }

        for (std::size_t r = 0; r < bit(m); ++r)
        {
            std::size_t row = cleared;
            for (std::size_t k = 0; k < m; ++k)
                if ((r >> (m - 1 - k)) & 1U) row |= bit(targets[k]);
            M(ix(row), ix(col)) = U(ix(r), ix(v));
        }
    }
    return M;
}


// ---- Assertions ----

// Nonzero amplitudes, truncated so failures on large registers stay readable.
inline std::string formatState(const Eigen::VectorXcd& v, std::size_t maxEntries = 32)
{
    std::string out;
    std::size_t shown = 0;
    for (Eigen::Index i = 0; i < v.size(); ++i)
    {
        if (std::abs(v[i]) <= 1e-15) continue;
        if (shown++ == maxEntries) return out + "  ...\n";
        out += std::format("  [{}] {:+.6f}{:+.6f}i\n", i, v[i].real(), v[i].imag());
    }
    return out.empty() ? "  (all zero)\n" : out;
}

inline std::string formatQubits(const QubitList& qs)
{
    std::string out = "{";
    for (std::size_t k = 0; k < qs.size(); ++k) out += std::format("{}{}", k ? "," : "", qs[k]);
    return out + "}";
}

// Elementwise |actual - expected| <= tol; on failure reports the worst index and both states.
inline ::testing::AssertionResult statesNear(const Eigen::VectorXcd& actual, const Eigen::VectorXcd& expected,
                                             double tol = kTol)
{
    if (actual.size() != expected.size())
        return ::testing::AssertionFailure()
               << "size mismatch: actual " << actual.size() << " vs expected " << expected.size();

    Eigen::Index worst = 0;
    const double err = (actual - expected).cwiseAbs().maxCoeff(&worst);
    if (err <= tol) return ::testing::AssertionSuccess();

    return ::testing::AssertionFailure()
           << std::format("max |diff| = {:.3e} > {:.1e} at index {}\nactual:\n{}expected:\n{}",
                          err, tol, worst, formatState(actual), formatState(expected));
}

inline ::testing::AssertionResult statesNear(const QuantumStateVector& actual, const Eigen::VectorXcd& expected,
                                             double tol = kTol)
{
    return statesNear(actual.vector(), expected, tol);
}

inline ::testing::AssertionResult statesNear(const QuantumStateVector& actual, const QuantumStateVector& expected,
                                             double tol = kTol)
{
    return statesNear(actual.vector(), expected.vector(), tol);
}

// statesNear after rotating `actual` by the global phase that aligns it with `expected` at
// expected's largest amplitude: states that differ only by e^{i phi} compare equal.
inline ::testing::AssertionResult statesNearUpToPhase(const Eigen::VectorXcd& actual, const Eigen::VectorXcd& expected,
                                                      double tol = kTol)
{
    if (actual.size() != expected.size()) return statesNear(actual, expected, tol);
    Eigen::Index k = 0;
    expected.cwiseAbs().maxCoeff(&k);
    if (std::abs(actual[k]) == 0.0) return statesNear(actual, expected, tol);
    const cd ratio = expected[k] / actual[k];
    return statesNear(actual * (ratio / std::abs(ratio)), expected, tol);
}

inline ::testing::AssertionResult statesNearUpToPhase(const QuantumStateVector& actual, const Eigen::VectorXcd& expected,
                                                      double tol = kTol)
{
    return statesNearUpToPhase(actual.vector(), expected, tol);
}

// Amplitudes from an index -> value list; unspecified entries are zero.
inline Eigen::VectorXcd amplitudes(std::size_t n, std::initializer_list<std::pair<std::size_t, cd>> entries)
{
    Eigen::VectorXcd v = Eigen::VectorXcd::Zero(ix(bit(n)));
    for (const auto& [i, a] : entries) v[ix(i)] = a;
    return v;
}


// ---- Random circuits ----

using GateOp = std::function<void(QuantumStateVector&)>;

// One randomly chosen named or dense gate on random distinct qubits (requires n >= 4).
inline GateOp randomGate(std::size_t n, std::mt19937_64& rng)
{
    std::uniform_int_distribution<int> pick(0, 15);
    std::uniform_real_distribution<double> angle(-kPi, kPi);
    const QubitList q = randomQubits(n, 4, rng);
    const double a = angle(rng), b = angle(rng), c = angle(rng);

    switch (pick(rng))
    {
        case 0:  return [=](auto& s) { QuantumGate::h(s, q[0]); };
        case 1:  return [=](auto& s) { QuantumGate::x(s, q[0]); };
        case 2:  return [=](auto& s) { QuantumGate::y(s, q[0]); };
        case 3:  return [=](auto& s) { QuantumGate::t(s, q[0]); };
        case 4:  return [=](auto& s) { QuantumGate::sx(s, q[0]); };
        case 5:  return [=](auto& s) { QuantumGate::rx(s, q[0], a); };
        case 6:  return [=](auto& s) { QuantumGate::ry(s, q[0], a); };
        case 7:  return [=](auto& s) { QuantumGate::rz(s, q[0], a); };
        case 8:  return [=](auto& s) { QuantumGate::u3(s, q[0], a, b, c); };
        case 9:  return [=](auto& s) { QuantumGate::cnot(s, q[0], q[1]); };
        case 10: return [=](auto& s) { QuantumGate::cphase(s, q[0], q[1], a); };
        case 11: return [=](auto& s) { QuantumGate::swap(s, q[0], q[1]); };
        case 12: return [=](auto& s) { QuantumGate::toffoli(s, q[0], q[1], q[2]); };
        case 13: return [=](auto& s) { QuantumGate::fredkin(s, q[0], q[1], q[2]); };
        case 14:
        {
            const Eigen::MatrixXcd U = randomUnitary(4, rng);
            return [=](auto& s) { QuantumGate::apply(s, {q[0], q[1]}, U); };
        }
        default:
        {
            const Eigen::MatrixXcd U = randomUnitary(8, rng);
            return [=](auto& s) { QuantumGate::controlled(s, {q[3]}, {q[0], q[1], q[2]}, U); };
        }
    }
}


// ---- Clifford circuits (the gates both backends run) ----

// Random x y z h s sdg sx cnot cz swap on random distinct qubits; two-qubit gates need n >= 2.
inline std::vector<Qputer::Operation> randomCliffordCircuit(std::size_t n, std::size_t depth, std::mt19937_64& rng)
{
    using Qputer::OpKind;
    constexpr OpKind oneQubit[] = {OpKind::X, OpKind::Y, OpKind::Z, OpKind::H, OpKind::S, OpKind::Sdg, OpKind::SX};
    constexpr OpKind twoQubit[] = {OpKind::CNOT, OpKind::CZ, OpKind::Swap};
    std::uniform_int_distribution<std::size_t> pick(0, n >= 2 ? 9 : 6);

    std::vector<Qputer::Operation> ops(depth);
    for (auto& op : ops)
    {
        const std::size_t g = pick(rng);
        if (g < 7)
        {
            op.kind = oneQubit[g];
            op.targets = randomQubits(n, 1, rng);
            continue;
        }
        op.kind = twoQubit[g - 7];
        const QubitList q = randomQubits(n, 2, rng);
        if (op.kind == OpKind::Swap) op.targets = q;
        else
        {
            op.controls = {q[0]};
            op.targets = {q[1]};
        }
    }
    return ops;
}

// Full 2^n x 2^n operator of a Clifford op, from the textbook matrices via embed().
inline Eigen::MatrixXcd cliffordMatrix(std::size_t n, const Qputer::Operation& op)
{
    using Qputer::OpKind;
    switch (op.kind)
    {
        case OpKind::X:    return embed(n, {}, op.targets, matX());
        case OpKind::Y:    return embed(n, {}, op.targets, matY());
        case OpKind::Z:    return embed(n, {}, op.targets, matZ());
        case OpKind::H:    return embed(n, {}, op.targets, matH());
        case OpKind::S:    return embed(n, {}, op.targets, matS());
        case OpKind::Sdg:  return embed(n, {}, op.targets, matSdg());
        case OpKind::SX:   return embed(n, {}, op.targets, matSX());
        case OpKind::CNOT: return embed(n, op.controls, op.targets, matX());
        case OpKind::CZ:   return embed(n, op.controls, op.targets, matZ());
        case OpKind::Swap: return embed(n, {}, op.targets, matSwap());
        default: throw std::invalid_argument("cliffordMatrix: not a Clifford gate");
    }
}

// The circuit applied to |0...0> through cliffordMatrix: the independent oracle.
inline Eigen::VectorXcd referenceState(std::size_t n, const std::vector<Qputer::Operation>& ops)
{
    Eigen::VectorXcd v = amplitudes(n, {{0, 1.0}});
    for (const auto& op : ops) v = cliffordMatrix(n, op) * v;
    return v;
}

inline void applyClifford(Qputer::StabilizerState& t, const Qputer::Operation& op)
{
    using Qputer::OpKind;
    switch (op.kind)
    {
        case OpKind::X:    t.x(op.targets[0]); break;
        case OpKind::Y:    t.y(op.targets[0]); break;
        case OpKind::Z:    t.z(op.targets[0]); break;
        case OpKind::H:    t.h(op.targets[0]); break;
        case OpKind::S:    t.s(op.targets[0]); break;
        case OpKind::Sdg:  t.sdg(op.targets[0]); break;
        case OpKind::SX:   t.sx(op.targets[0]); break;
        case OpKind::CNOT: t.cnot(op.controls[0], op.targets[0]); break;
        case OpKind::CZ:   t.cz(op.controls[0], op.targets[0]); break;
        case OpKind::Swap: t.swap(op.targets[0], op.targets[1]); break;
        default: throw std::invalid_argument("applyClifford: not a Clifford gate");
    }
}

// <v|P|v> for P = paulis[k] on qubits[k], built from textbook matrices via embed().
inline double referenceExpectation(std::size_t n, const Eigen::VectorXcd& v, std::string_view paulis,
                                   const QubitList& qubits)
{
    Eigen::VectorXcd pv = v;
    for (std::size_t k = 0; k < paulis.size(); ++k)
    {
        const Eigen::Matrix2cd m = paulis[k] == 'X' ? matX() : paulis[k] == 'Y' ? matY()
                                 : paulis[k] == 'Z' ? matZ() : matI();
        pv = embed(n, {}, {qubits[k]}, m) * pv;
    }
    return v.dot(pv).real();
}

// Every Pauli string over n qubits, letter k acting on qubit k.
inline std::vector<std::string> allPauliStrings(std::size_t n)
{
    std::vector<std::string> out{""};
    for (std::size_t q = 0; q < n; ++q)
    {
        std::vector<std::string> next;
        for (const auto& s : out)
            for (const char c : {'I', 'X', 'Y', 'Z'}) next.push_back(s + c);
        out = std::move(next);
    }
    return out;
}


// ---- Statistics and threading ----

inline ::testing::AssertionResult withinSigmas(double observed, double p, std::size_t shots, double sigmas = 5.0)
{
    const double sigma = std::sqrt(p * (1.0 - p) / static_cast<double>(shots));
    if (std::abs(observed - p) <= sigmas * sigma + 1e-12) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << std::format("frequency {:.5f} vs p = {:.5f} ({:.1f} sigma)",
                                                        observed, p, std::abs(observed - p) / sigma);
}

inline double frequency(const Qputer::Counts& counts, Qputer::Outcome o, std::size_t shots)
{
    return counts.contains(o) ? static_cast<double>(counts.at(o)) / static_cast<double>(shots) : 0.0;
}

// Runs `body` with the OpenMP team size pinned, restoring the previous setting.
template <class F>
auto withThreads(int threads, F&& body)
{
#ifdef _OPENMP
    const int saved = omp_get_max_threads();
    omp_set_num_threads(threads);
    auto result = body();
    omp_set_num_threads(saved);
    return result;
#else
    (void)threads;
    return body();
#endif
}


} // namespace QputerTest
