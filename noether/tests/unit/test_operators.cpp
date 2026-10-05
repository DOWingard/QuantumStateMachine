// Per-operator tests. Every spelling in tokens.def (punctuation, gates, gate aliases, functions,
// channels, keywords) must transpile to its canonical form and act on the QuantumStateMachine as
// intended. Gates are checked against independently built matrices on a generic product state, and
// Clifford gates additionally against the stabilizer backend. Coverage tests at the end fail when a
// tokens.def entry has no test here.

#include "TestUtil.hpp"

#include "Lexer.hpp"

#include <Eigen/Dense>

#include <numbers>
#include <regex>
#include <set>

using namespace Noether;
using namespace NoetherTest;

namespace
{

using Mat = Eigen::MatrixXcd;
using Vec = Eigen::VectorXcd;
constexpr double kPi = std::numbers::pi;
const cd kI{0.0, 1.0};

// ---- Reference matrices, written from their textbook definitions ----

Mat m2(cd a, cd b, cd c, cd d)
{
    Mat m(2, 2);
    m << a, b, c, d;
    return m;
}
Mat diag(std::initializer_list<cd> d)
{
    Mat m = Mat::Zero(static_cast<Eigen::Index>(d.size()), static_cast<Eigen::Index>(d.size()));
    Eigen::Index k = 0;
    for (const cd x : d) m(k, k) = x, ++k;
    return m;
}
Mat rx(double t) { return m2(std::cos(t / 2), -kI * std::sin(t / 2), -kI * std::sin(t / 2), std::cos(t / 2)); }
Mat ry(double t) { return m2(std::cos(t / 2), -std::sin(t / 2), std::sin(t / 2), std::cos(t / 2)); }
Mat rz(double t) { return m2(std::exp(-kI * t / 2.0), 0, 0, std::exp(kI * t / 2.0)); }
Mat ph(double l) { return m2(1, 0, 0, std::exp(kI * l)); }
Mat u3(double t, double p, double l)
{
    return m2(std::cos(t / 2), -std::exp(kI * l) * std::sin(t / 2), std::exp(kI * p) * std::sin(t / 2), std::exp(kI * (p + l)) * std::cos(t / 2));
}
const Mat kX = m2(0, 1, 1, 0), kY = m2(0, -kI, kI, 0), kZ = m2(1, 0, 0, -1), kH = m2(1, 1, 1, -1) / std::sqrt(2.0);
const Mat kS = ph(kPi / 2), kT = ph(kPi / 4), kId = Mat::Identity(2, 2);
const Mat kSX = 0.5 * m2(1.0 + kI, 1.0 - kI, 1.0 - kI, 1.0 + kI);

Mat kron(const Mat& a, const Mat& b)
{
    Mat out(a.rows() * b.rows(), a.cols() * b.cols());
    for (Eigen::Index i = 0; i < a.rows(); ++i)
        for (Eigen::Index j = 0; j < a.cols(); ++j) out.block(i * b.rows(), j * b.cols(), b.rows(), b.cols()) = a(i, j) * b;
    return out;
}

// U acting on the targets after the controls, active when the controls read `pattern` (MSB first).
Mat controlled(const Mat& u, std::size_t nControls, std::size_t pattern)
{
    const Eigen::Index blocks = Eigen::Index{1} << nControls, d = u.rows();
    Mat m = Mat::Identity(blocks * d, blocks * d);
    m.block(static_cast<Eigen::Index>(pattern) * d, static_cast<Eigen::Index>(pattern) * d, d, d) = u;
    return m;
}
Mat ctrl(const Mat& u) { return controlled(u, 1, 1); }

Mat kSwap()
{
    Mat m = Mat::Zero(4, 4);
    m(0, 0) = m(1, 2) = m(2, 1) = m(3, 3) = 1;
    return m;
}
Mat kISwap()
{
    Mat m = Mat::Zero(4, 4);
    m(0, 0) = m(3, 3) = 1;
    m(1, 2) = m(2, 1) = kI;
    return m;
}

// exp(-i t A) for Hermitian A.
Mat expmi(const Mat& a, double t)
{
    Eigen::SelfAdjointEigenSolver<Mat> es(a);
    const Eigen::VectorXd ev = es.eigenvalues();
    Vec phases(ev.size());
    for (Eigen::Index k = 0; k < ev.size(); ++k) phases[k] = std::exp(-kI * t * ev[k]);
    return es.eigenvectors() * phases.asDiagonal() * es.eigenvectors().adjoint();
}

// ---- States: index bit q = qubit q; Noether labels put q0 leftmost ----

constexpr std::size_t kN = 3;
const double kA[kN] = {0.3, 0.5, 0.9}, kB[kN] = {0.7, -1.1, 0.4};
const std::string kGenericPrepare = "prepare (cos(0.3)|0⟩ + e^{0.7i} sin(0.3)|1⟩) ⊗ (cos(0.5)|0⟩ + e^{-1.1i} sin(0.5)|1⟩) ⊗ "
                                    "(cos(0.9)|0⟩ + e^{0.4i} sin(0.9)|1⟩)";

Vec genericState()
{
    Vec v(1 << kN);
    for (Eigen::Index i = 0; i < v.size(); ++i)
    {
        cd a = 1.0;
        for (std::size_t q = 0; q < kN; ++q) a *= ((i >> q) & 1) ? std::exp(kI * kB[q]) * std::sin(kA[q]) : cd(std::cos(kA[q]));
        v[i] = a;
    }
    return v;
}

// Applies m to `targets` (targets[0] = most significant bit of m's index).
Vec applyOn(const Vec& psi, const Mat& m, const std::vector<int>& targets)
{
    const std::size_t k = targets.size();
    Vec out = Vec::Zero(psi.size());
    for (Eigen::Index i = 0; i < psi.size(); ++i)
    {
        std::size_t row = 0;
        for (std::size_t j = 0; j < k; ++j) row |= static_cast<std::size_t>((i >> targets[j]) & 1) << (k - 1 - j);
        for (std::size_t col = 0; col < (std::size_t{1} << k); ++col)
        {
            Eigen::Index src = i;
            for (std::size_t j = 0; j < k; ++j)
            {
                const Eigen::Index bit = Eigen::Index{1} << targets[j];
                src = ((col >> (k - 1 - j)) & 1) ? (src | bit) : (src & ~bit);
            }
            out[i] += m(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) * psi[src];
        }
    }
    return out;
}

Vec stateOf(const Executed& run, std::string_view label)
{
    Vec v = Vec::Zero(1 << kN);
    const PrintRecord* p = run.print(label);
    if (!p) return v;
    for (const auto& [key, a] : amplitudes(*p))
    {
        Eigen::Index i = 0;
        for (std::size_t q = 0; q < key.size(); ++q)
            if (key[q] == '1') i |= Eigen::Index{1} << q;
        v[i] = a;
    }
    return v;
}

// |⟨a|b⟩|² / (|a|²|b|²): equality up to a global phase.
double overlap(const Vec& a, const Vec& b) { return std::norm(a.dot(b)) / (a.squaredNorm() * b.squaredNorm()); }

std::string program(const std::string& prelude, const std::string& body, const std::string& prepare)
{
    return "noether 0.1\nbackend statevector\nqubits q[3]\n" + prelude + prepare + "\n" + body + "\nprint |ψ⟩ as state\n";
}

std::vector<std::string> allPaulis()
{
    std::vector<std::string> out;
    for (int code = 1; code < 64; ++code)
    {
        std::string s;
        for (std::size_t q = 0; q < kN; ++q)
        {
            const int l = (code >> (2 * q)) & 3;
            if (l) s += std::format("{}{}_{}", s.empty() ? "" : " ", "XYZ"[l - 1], q);
        }
        out.push_back(s);
    }
    return out;
}

// ---- Gate cases ----

struct GateCase
{
    GateCase(std::vector<std::string> s, std::string c, std::vector<std::string> e, Mat matrix, std::vector<int> t,
             bool cliff = false, std::string pre = "")
        : spellings(std::move(s)), canonical(std::move(c)), equivalents(std::move(e)), m(std::move(matrix)), targets(std::move(t)),
          clifford(cliff), prelude(std::move(pre))
    {
    }
    std::vector<std::string> spellings;   // all format to one canonical text
    std::string canonical;                // expected canonical text ("" = only check agreement)
    std::vector<std::string> equivalents; // different constructs with the same action
    Mat m;
    std::vector<int> targets;
    bool clifford = false;
    std::string prelude;
};

const double th = 0.37, phi = 1.1, lam = -0.6;

std::vector<GateCase> gateCases()
{
    const Mat cnot20 = ctrl(kX);
    const Mat U = m2(0.6, 0.8, -0.8, 0.6);
    const std::string letU = "let U = [[0.6, 0.8], [-0.8, 0.6]]\n";
    const Mat xz = kron(kX, kZ);
    const Mat xxyy = kron(kX, kX) + kron(kY, kY);
    return {
        {{"I_1"}, "I_1", {}, kId, {1}, true},
        {{"X_1"}, "X_1", {"[[0, 1], [1, 0]]_1", "Y_1 Z_1"}, kX, {1}, true},
        {{"Y_1"}, "Y_1", {"X_1 Z_1"}, kY, {1}, true},
        {{"Z_1"}, "Z_1", {"S_1 S_1"}, kZ, {1}, true},
        {{"T^4_1"}, "T^4_1", {"T_1 T_1 T_1 T_1"}, kZ, {1}, false}, // T gates are not Clifford one by one
        {{"H_1"}, "H_1", {"H^3_1", "U3(π/2, 0, π)_1"}, kH, {1}, true},
        {{"S_1"}, "S_1", {"P(π/2)_1"}, kS, {1}, true},
        {{"T_1 T_1"}, "T_1 T_1", {"T^2_1"}, kS, {1}, false},
        {{"S†_1", "Sdg_1", "S\\dagger_1", "S^\\dagger_1"}, "S†_1", {"S^3_1"}, kS.adjoint(), {1}, true},
        {{"T_1"}, "T_1", {"P(π/4)_1"}, kT, {1}, false},
        {{"T†_1", "Tdg_1", "T\\dagger_1"}, "T†_1", {"T^7_1"}, kT.adjoint(), {1}, false},
        {{"√X_1", "SX_1", "sqrt X_1"}, "√X_1", {"H_1 S_1 H_1"}, kSX, {1}, true},
        {{"√X†_1", "SX\\dagger_1"}, "√X†_1", {"H_1 S†_1 H_1"}, kSX.adjoint(), {1}, true},
        {{"Rx(0.37)_1", "RX(0.37)_1"}, "Rx(0.37)_1", {"e^{-i 0.185 X_1}", "exp(-i·0.185·X_1)"}, rx(th), {1}, false},
        {{"Ry(0.37)_1", "RY(0.37)_1"}, "Ry(0.37)_1", {"e^{-i 0.185 Y_1}"}, ry(th), {1}, false},
        {{"Rz(0.37)_1", "RZ(0.37)_1"}, "Rz(0.37)_1", {"e^{-i 0.185 Z_1}"}, rz(th), {1}, false},
        {{"P(0.37)_1", "Phase(0.37)_1"}, "P(0.37)_1", {}, ph(th), {1}, false},
        {{"U3(0.37, 1.1, -0.6)_1"}, "U3(0.37, 1.1, -0.6)_1", {}, u3(th, phi, lam), {1}, false},
        {{"Rx(0.37)†_1", "Rx(0.37)\\dagger_1"}, "Rx(0.37)†_1", {"Rx(-0.37)_1"}, rx(-th), {1}, false},
        {{"U3(0.37, 1.1, -0.6)†_1"}, "U3(0.37, 1.1, -0.6)†_1", {}, u3(th, phi, lam).adjoint(), {1}, false},
        // Clifford angles go through the stabilizer's exact decompositions.
        {{"Rx(π/2)_1", "RX(pi/2)_1"}, "Rx(π/2)_1", {}, rx(kPi / 2), {1}, true},
        {{"Ry(-π/2)_1"}, "Ry(-π/2)_1", {}, ry(-kPi / 2), {1}, true},
        {{"Rz(π)_1"}, "Rz(π)_1", {}, rz(kPi), {1}, true},
        {{"P(3π/2)_1"}, "P(3π/2)_1", {}, ph(3 * kPi / 2), {1}, true},
        {{"U3(π/2, π/2, -π)_1"}, "U3(π/2, π/2, -π)_1", {}, u3(kPi / 2, kPi / 2, -kPi), {1}, true},
        // Two- and three-qubit gates, on reversed or interleaved qubits to catch ordering mistakes.
        {{"CNOT_{2→0}", "CX_{2->0}", "CNOT_{2 \\to 0}", "CNOT_{2 \\rightarrow 0}", "CNOT_{2 ⟶ 0}"}, "CNOT_{2→0}",
         {"C_{2}(X_0)", "C_2(X_0)"}, cnot20, {2, 0}, true},
        {{"CZ_{2, 0}", "CZ_{2,0}"}, "CZ_{2,0}", {"C_{2}(Z_0)", "C_{0}(Z_2)", "CP(π)_{2, 0}"}, ctrl(kZ), {2, 0}, true},
        {{"CP(-0.6)_{2, 0}", "CPhase(-0.6)_{2,0}"}, "CP(-0.6)_{2,0}", {"C_{2}(P(-0.6)_0)"}, diag({1, 1, 1, std::exp(kI * lam)}), {2, 0}},
        {{"SWAP_{2, 0}"}, "SWAP_{2,0}", {"CNOT_{2→0} CNOT_{0→2} CNOT_{2→0}"}, kSwap(), {2, 0}, true},
        {{"C_{2}(Y_0)"}, "C_{2}(Y_0)", {}, ctrl(kY), {2, 0}, true},
        {{"C_{2}(H_0)"}, "C_{2}(H_0)", {}, ctrl(kH), {2, 0}},
        {{"C_{2}(Rx(0.37)_0)"}, "C_{2}(Rx(0.37)_0)", {}, ctrl(rx(th)), {2, 0}},
        {{"C_{2}(U3(0.37, 1.1, -0.6)_0)"}, "C_{2}(U3(0.37, 1.1, -0.6)_0)", {}, ctrl(u3(th, phi, lam)), {2, 0}},
        {{"C_{2}(S†_0)", "C_{2}(Sdg_0)"}, "C_{2}(S†_0)", {}, ctrl(kS.adjoint()), {2, 0}},
        {{"C_{2}(√X_0)", "C_{2}(SX_0)"}, "C_{2}(√X_0)", {}, ctrl(kSX), {2, 0}},
        {{"C_{¬2}(H_0)", "C_{not 2}(H_0)", "C_{\\neg 2}(H_0)", "C_{\\lnot 2}(H_0)"}, "C_{¬2}(H_0)", {}, controlled(kH, 1, 0), {2, 0}},
        {{"Toffoli_{2, 0→1}", "CCX_{2,0->1}"}, "Toffoli_{2,0→1}", {"C_{2, 0}(X_1)", "C_{2}(CNOT_{0→1})"}, controlled(kX, 2, 3), {2, 0, 1}},
        {{"Fredkin_{1→2, 0}", "CSWAP_{1->2,0}"}, "Fredkin_{1→2,0}", {"C_{1}(SWAP_{2, 0})"}, ctrl(kSwap()), {1, 2, 0}},
        {{"C_{0, ¬2}(Z_1)"}, "C_{0,¬2}(Z_1)", {}, controlled(kZ, 2, 2), {0, 2, 1}},
        {{"C_{q[0..1]}(P(-0.6)_{q[2]})"}, "C_{q[0..1]}(P(-0.6)_{q[2]})", {}, controlled(ph(lam), 2, 3), {0, 1, 2}},
        // Matrices: literal, controlled, powers, adjoint, two-qubit.
        {{"U_1"}, "U_1", {}, U, {1}, false, letU},
        {{"C_{2}(U_0)"}, "C_{2}(U_0)", {}, ctrl(U), {2, 0}, false, letU},
        {{"U^3_1"}, "U^3_1", {"U_1 U_1 U_1"}, U * U * U, {1}, false, letU},
        {{"U†_1", "U\\dagger_1"}, "U†_1", {}, U.adjoint(), {1}, false, letU},
        {{"[[1, 0, 0, 0], [0, 0, i, 0], [0, i, 0, 0], [0, 0, 0, 1]]_{2, 0}"}, "[[1, 0, 0, 0], [0, 0, i, 0], [0, i, 0, 0], [0, 0, 0, 1]]_{2,0}", {}, kISwap(), {2, 0}},
        // Pauli exponentials and products.
        {{"e^{-i 0.37 X_0 Z_2}", "exp(-i 0.37 X_0 Z_2)"}, "e^{-i 0.37 X_0 Z_2}", {}, expmi(xz, th), {0, 2}},
        {{"e^{-i π/4 X_0 Z_2}"}, "e^{-i π/4 X_0 Z_2}", {}, expmi(xz, kPi / 4), {0, 2}, true},
        {{"e^{-i 0.37 (X_0 X_1 + Y_0 Y_1)}"}, "e^{-i 0.37 (X_0 X_1 + Y_0 Y_1)}", {}, expmi(xxyy, th), {0, 1}},
        {{"H_1 S_1"}, "H_1 S_1", {}, kH * kS, {1}, true},
        {{"S_1 H_1"}, "S_1 H_1", {}, kS * kH, {1}, true},
        {{"(H_1 S_1)^2"}, "(H_1 S_1)^2", {}, kH * kS * kH * kS, {1}, true},
        {{"X_0 ⊗ Z_2", "X_0 \\otimes Z_2"}, "X_0 ⊗ Z_2", {"X_0 Z_2"}, kron(kX, kZ), {0, 2}, true},
        // Registers: broadcast, slices, tensor powers, big products.
        {{"H_q"}, "H_q", {"H_0 H_1 H_2", "∏_{j=0}^{2} H_j"}, kron(kron(kH, kH), kH), {0, 1, 2}, true},
        {{"H^⊗3_q", "H^{\\otimes 3}_q"}, "H^⊗3_q", {}, kron(kron(kH, kH), kH), {0, 1, 2}, true},
        {{"X_{q[0..1]}"}, "X_{q[0..1]}", {}, kron(kX, kX), {0, 1}, true},
        {{"∏_{j=0}^{2} X_j", "\\prod_{j=0}^{2} X_j"}, "∏_{j=0}^{2} X_j", {}, kron(kron(kX, kX), kX), {0, 1, 2}, true},
        // Definitions: let-gates, defs with classical and register parameters, adjoints, std.
        {{"Bell_{2, 0}"}, "Bell_{2,0}", {}, cnot20 * kron(kH, kId), {2, 0}, true, "let Bell_{a,b} = CNOT_{a→b} H_a\n"},
        {{"R(0.37)_1"}, "R(0.37)_1", {}, rz(th) * rx(th), {1}, false, "def R(t)_{a}: Rz(t)_a Rx(t)_a\n"},
        {{"R(0.37)†_1"}, "R(0.37)†_1", {}, (rz(th) * rx(th)).adjoint(), {1}, false, "def R(t)_{a}: Rz(t)_a Rx(t)_a\n"},
        {{"Ladder_q"}, "Ladder_q", {}, kron(kId, ctrl(kX)) * kron(ctrl(kX), kId), {0, 1, 2}, true,
         "def Ladder_{r[n]}:\n    for k ∈ 0..n - 2:\n        CNOT_{r[k]→r[k + 1]}\n"},
        {{"ISWAP_{2, 0}"}, "ISWAP_{2,0}", {}, kISwap(), {2, 0}, true, "import \"std/gates.ntr\"\n"},
        {{"Rzz(0.37)_{0, 2}"}, "Rzz(0.37)_{0,2}", {}, expmi(kron(kZ, kZ), th / 2), {0, 2}, false, "import \"std/gates.ntr\"\n"},
    };
}

// ---- Token templates: '@' is replaced by each spelling of the token ----

struct TokenCase
{
    Tok tok;
    std::string tmpl;                  // program body with '@'
    std::vector<std::string> programs; // explicit programs, for tokens that need context
};

std::vector<TokenCase> tokenCases()
{
    return {
        {Tok::Underscore, "qubits q[1]\nX@0\nassert ⟨Z_0⟩ ≈ -1", {}},
        {Tok::Caret, "assert 2@3 == 8", {}},
        {Tok::TensorPow, "",
         {"qubits q[3]\nprepare |1⟩^⊗3\nassert ⟨Z_0 Z_1 Z_2⟩ ≈ -1", "qubits q[3]\nprepare |1>^{\\otimes 3}\nassert ⟨Z_0 Z_1 Z_2⟩ ≈ -1",
          "qubits q[3]\nprepare |1>^\\otimes 3\nassert ⟨Z_0 Z_1 Z_2⟩ ≈ -1"}},
        {Tok::Dagger, "qubits q[1]\nprepare |+⟩\nS@_0 S_0\nassert ⟨X_0⟩ ≈ 1", {}},
        {Tok::LParen, "assert @2 + 3)·2 == 10", {}},
        {Tok::RParen, "assert (2 + 3@·2 == 10", {}},
        {Tok::LBracket, "qubits q@2]\nX_{q[1]}\nassert ⟨Z_1⟩ ≈ -1", {}},
        {Tok::RBracket, "qubits q[2@\nX_{q[1]}\nassert ⟨Z_1⟩ ≈ -1", {}},
        {Tok::LBrace, "qubits q[12]\nX_@10}\nassert ⟨Z_10⟩ ≈ -1", {}},
        {Tok::RBrace, "qubits q[12]\nX_{10@\nassert ⟨Z_10⟩ ≈ -1", {}},
        {Tok::Comma, "qubits q[2]\nX_0\nCNOT_{0→1}\nCZ_{0@1}\nassert ⟨Z_1⟩ ≈ -1", {}},
        {Tok::Semicolon, "qubits q[1]@ bits c[1]\nX_0\nc ← measure_q\nassert c[0]", {}},
        {Tok::Colon, "qubits q[1]\nif true@ X_0\nassert ⟨Z_0⟩ ≈ -1", {}},
        {Tok::DotDot, "qubits q[3]\nfor k ∈ 0@2: X_k\nassert ⟨Z_0 Z_1 Z_2⟩ ≈ -1", {}},
        {Tok::Assign, "let x @ 3\nassert x == 3", {}},
        {Tok::LeftArrow, "qubits q[1]; bits c[1]\nX_0\nc[0] @ measure Z_0\nassert c[0]", {}},
        {Tok::Arrow, "qubits q[2]\nX_0\nCNOT_{0 @ 1}\nassert ⟨Z_1⟩ ≈ -1", {}},
        {Tok::Otimes, "qubits q[2]\nprepare |1⟩ @ |0⟩\nassert ⟨Z_0⟩ ≈ -1 and ⟨Z_1⟩ ≈ 1", {}},
        {Tok::Plus, "assert 2 @ 3 == 5", {}},
        {Tok::Minus, "assert 5 @ 3 == 2", {}},
        {Tok::Cdot, "assert 2 @ 3 == 6", {}},
        {Tok::Slash, "assert 6 @ 3 == 2", {}},
        {Tok::Sqrt, "assert @(4) == 2", {}},
        {Tok::Sum, "assert @_{j=1}^{4} j == 10", {}},
        {Tok::Prod, "assert @_{j=1}^{4} j == 24", {}},
        {Tok::Not, "assert @ false", {}},
        {Tok::EqEq, "assert 2 @ 2", {}},
        {Tok::NotEq, "assert 2 @ 3", {}},
        {Tok::Less, "assert 2 @ 3", {}},
        {Tok::LessEq, "assert 3 @ 3", {}},
        {Tok::Greater, "assert 3 @ 2", {}},
        {Tok::GreaterEq, "assert 3 @ 3", {}},
        {Tok::Approx, "assert 0.1 + 0.2 @ 0.3", {}},
        {Tok::PlusMinus, "assert 1 ≈ 1.05 @ 0.1", {}},
        {Tok::In, "qubits q[1]\nparam θ @ [0, 1] = 0.5\nRy(θ)_0\nassert ⟨Z_0⟩ ≈ cos(0.5)", {}},
        {Tok::LAngle, "",
         {"qubits q[1]\nX_0\nassert ⟨Z_0⟩ ≈ -1", "qubits q[1]\nX_0\nassert \\expval{Z_0} ≈ -1", "qubits q[1]\nX_0\nassert \\langle Z_0 \\rangle ≈ -1"}},
        {Tok::RAngle, "", {}}, // tested with LAngle
        {Tok::Abs, "assert @-3@ == 3", {}},
        {Tok::Dot, "", {}}, // only diagnoses Qiskit spellings; see QiskitDotIsDiagnosed
    };
}

std::map<Tok, std::vector<std::string>> spellingsByToken()
{
    std::map<Tok, std::vector<std::string>> m;
#define NTR_PUNCT(Name, canonical, ascii)                                                                                  \
    m[Tok::Name].push_back(canonical);                                                                                     \
    if (std::string(ascii) != std::string(canonical)) m[Tok::Name].push_back(ascii);
#define NTR_ASCII_OP(text, Name) m[Tok::Name].push_back(text);
#define NTR_UNICODE_ALT(text, Name) m[Tok::Name].push_back(text);
#define NTR_LATEX_TOK(command, Name) m[Tok::Name].push_back("\\" command);
#include "tokens.def"
    for (auto& [tok, v] : m)
    {
        std::ranges::sort(v);
        v.erase(std::ranges::unique(v).begin(), v.end());
    }
    return m;
}

std::string fill(const std::string& tmpl, const std::string& spelling)
{
    std::string out;
    for (const char c : tmpl) out += c == '@' ? spelling : std::string(1, c);
    return out;
}

} // namespace

// ============================================================================================
// Gates
// ============================================================================================

TEST(Operators, GatesTranspileToTheirCanonicalSpelling)
{
    for (const GateCase& g : gateCases())
    {
        const std::string pre = "noether 0.1\nqubits q[3]\n" + g.prelude;
        const std::string canonical = format(pre + g.spellings[0] + "\n");
        if (!g.canonical.empty())
        {
            EXPECT_EQ(canonical.substr(canonical.size() - g.canonical.size() - 1), g.canonical + "\n") << g.spellings[0];
        }
        for (const std::string& s : g.spellings) EXPECT_EQ(format(pre + s + "\n"), canonical) << "spelling `" << s << "`";
    }
}

TEST(Operators, GatesActAsTheirMatricesOnTheStateVector)
{
    const Vec psi0 = genericState();
    for (const GateCase& g : gateCases())
    {
        const Vec expected = applyOn(psi0, g.m, g.targets);
        std::vector<std::string> forms = g.spellings;
        forms.insert(forms.end(), g.equivalents.begin(), g.equivalents.end());
        for (const std::string& body : forms)
        {
            const Executed run = expectRuns(program(g.prelude, body, kGenericPrepare));
            EXPECT_NEAR(overlap(expected, stateOf(run, "state")), 1.0, 1e-12) << "`" << body << "`";
        }
    }
}

TEST(Operators, CliffordGatesAgreeOnBothBackends)
{
    const std::vector<std::string> paulis = allPaulis();
    std::string printLine = "print ";
    for (std::size_t k = 0; k < paulis.size(); ++k) printLine += (k ? ", ⟨" : "⟨") + paulis[k] + "⟩";
    for (const GateCase& g : gateCases())
    {
        if (!g.clifford) continue;
        std::vector<std::string> forms = g.spellings;
        forms.insert(forms.end(), g.equivalents.begin(), g.equivalents.end());
        for (const std::string& body : forms)
        {
            if (body.contains("[[")) continue; // a written matrix always runs on the state vector
            const std::string src = "noether 0.1\nqubits q[3]\n" + g.prelude + "prepare |+⟩ ⊗ |+i⟩ ⊗ |1⟩\n" + body + "\n" + printLine + "\n";
            CompileOptions sv, stab;
            sv.backend = "statevector";
            stab.backend = "stabilizer";
            const Executed a = expectRuns(src, sv);
            const Executed b = expectRuns(src, stab);
            ASSERT_EQ(b.c->compiler->ir().backend, "stabilizer") << body;
            ASSERT_EQ(a.r.prints.size(), paulis.size());
            ASSERT_EQ(b.r.prints.size(), paulis.size());
            for (std::size_t k = 0; k < paulis.size(); ++k)
                EXPECT_NEAR(*a.r.prints[k].number, *b.r.prints[k].number, 1e-12) << "`" << body << "` ⟨" << paulis[k] << "⟩";
        }
    }
}

TEST(Operators, GateTensorPowersStayGates)
{
    // G^⊗n of a one-qubit gate is n gates, not a 2^n × 2^n matrix: it scales past the dense limit and
    // stays Clifford when G is.
    const Executed run = expectRuns("noether 0.1\nqubits q[20]\nH^⊗20_q\nassert ⟨X_0 X_19⟩ ≈ 1\n");
    EXPECT_EQ(run.c->compiler->ir().backend, "stabilizer");
    const Executed wrong = runSource("noether 0.1\nqubits q[3]\nH^⊗2_q\n");
    EXPECT_EQ(wrong.codes(), std::vector<std::string>{"E5009"});
}

TEST(Operators, NonCliffordGatesAreRejectedByTheStabilizer)
{
    for (const std::string body : {"T_0", "Rx(0.37)_0", "Toffoli_{0, 1→2}", "C_{0}(H_1)"})
    {
        CompileOptions stab;
        stab.backend = "stabilizer";
        const Executed run = runSource("noether 0.1\nqubits q[3]\n" + std::string(body) + "\n", stab);
        EXPECT_EQ(run.codes(), std::vector<std::string>{"E6001"}) << body;
    }
}

// ============================================================================================
// Punctuation and operator tokens
// ============================================================================================

TEST(Operators, EveryTokenSpellingTranspilesAndRuns)
{
    const auto spellings = spellingsByToken();
    for (const TokenCase& t : tokenCases())
    {
        std::vector<std::string> programs = t.programs;
        if (!t.tmpl.empty())
            for (const std::string& s : spellings.at(t.tok)) programs.push_back(fill(t.tmpl, s));
        if (programs.empty()) continue;
        const std::string canonical = format("noether 0.1\n" + programs.front() + "\n");
        EXPECT_NE(canonical.find(std::string(canonicalText(t.tok))), std::string::npos) << tokName(t.tok) << ": " << canonical;
        for (const std::string& p : programs)
        {
            const std::string src = "noether 0.1\n" + p + "\n";
            EXPECT_EQ(format(src), canonical) << tokName(t.tok) << " spelled as in:\n" << p;
            const Executed run = expectRuns(src);
            EXPECT_FALSE(run.r.asserts.empty()) << p;
        }
    }
}

TEST(Operators, QiskitDotIsDiagnosed)
{
    const Executed run = runSource("noether 0.1\nqubits q[2]\nqc.cx(0, 1)\n");
    ASSERT_EQ(run.codes(), std::vector<std::string>{"E2009"});
    const Diagnostic d = run.c->diags->sorted().front();
    ASSERT_FALSE(d.fixes.empty());
    EXPECT_EQ(d.fixes[0].edits[0].text, "CNOT_{0→1}");
}

// ============================================================================================
// Functions
// ============================================================================================

namespace
{
const std::map<std::string, std::string>& functionPrograms()
{
    static const std::map<std::string, std::string> m{
        {"sin", "assert sin(π/6) ≈ 0.5 and \\sin(0) == 0"},
        {"cos", "assert cos(π/3) ≈ 0.5 and \\cos(0) == 1"},
        {"tan", "assert tan(π/4) ≈ 1 and \\tan(0) == 0"},
        {"exp", "assert exp(1) ≈ e and \\exp(0) ≈ 1 and |exp(i π) + 1| ≈ 0"},
        {"log", "assert log(e^2) ≈ 2 and \\ln(1) == 0 and \\log(e) ≈ 1"},
        {"sqrt", "assert sqrt(2)^2 ≈ 2 and √(9) == 3 and \\sqrt{16} == 4"},
        {"abs", "assert abs(-2.5) == 2.5 and |3 - 5| == 2 and |3 + 4i| ≈ 5"},
        {"floor", "assert floor(2.7) == 2 and floor(-0.5) == -1"},
        {"ceil", "assert ceil(2.1) == 3 and ceil(-0.5) == 0"},
        {"min", "assert min(3, 1, 2) == 1"},
        {"max", "assert max(3, 1, 2) == 3"},
        // Product formula for H = X + Z from |0⟩: ⟨Z⟩(t) = 1/2 + 1/2 cos(2√2 t).
        {"trotter", "qubits q[1]\nlet Ham = X_0 + Z_0, t = 0.7\ntrotter(Ham, t, steps=200, order=2)\n"
                    "assert ⟨Z_0⟩ ≈ 1/2 + 1/2 cos(2√2·t) ± 1e-4\n"
                    "prepare |0⟩\ntrotter(Ham, t, steps=400, order=1)\nassert ⟨Z_0⟩ ≈ 1/2 + 1/2 cos(2√2·t) ± 1e-2\n"
                    "prepare |0⟩\ntrotter(Ham, t, steps=20, order=4)\nassert ⟨Z_0⟩ ≈ 1/2 + 1/2 cos(2√2·t) ± 1e-5"},
        {"entropy", "qubits q[3]\nCNOT_{0→1} H_0\nassert entropy(ρ_{q[0]}) ≈ 1 and entropy(ρ_{q[0..1]}) ≈ 0 and entropy(ρ_{q[2]}) ≈ 0"},
        {"fidelity", "qubits q[2]\nCNOT_{0→1} H_0\nassert fidelity((|00⟩ + |11⟩)/√2) ≈ 1 and fidelity(|00⟩) ≈ 0.5 and fidelity(|01⟩) ≈ 0"},
        {"marginal", "qubits q[2]; bits a[1], b[1]; seed 4\nX_1\na[0] ← measure Z_0\nb[0] ← measure Z_1\ncounts ← run 50\n"
                     "let m = marginal(counts, b)\nassert m[\"1\"] == 50 and marginal(counts, a)[\"0\"] == 50"},
        {"count", "qubits q[2]\nH_0 H_1 CNOT_{0→1}\nassert count(H) == 2 and count(CNOT) == 1"},
        {"depth", "qubits q[2]\nH_0 H_1 CNOT_{0→1} X_0\nassert depth() == 3"},
    };
    return m;
}

const std::map<std::string, std::string>& channelPrograms()
{
    // Analytic decay laws; `trajectories` averages and the assert adds 2·stderr to the tolerance.
    static const std::map<std::string, std::string> m{
        {"flip", "qubits q[1]; trajectories 4000; seed 11\nflip(0.1)_0\nassert ⟨Z_0⟩ ≈ 1 - 2·0.1 ± 0.01"},
        {"dephase", "qubits q[1]; trajectories 4000; seed 12\nprepare |+⟩\ndephase(0.15)_0\nassert ⟨X_0⟩ ≈ 1 - 2·0.15 ± 0.01"},
        {"depolarize", "qubits q[1]; trajectories 4000; seed 13\ndepolarize(0.3)_0\nassert ⟨Z_0⟩ ≈ 1 - 4/3·0.3 ± 0.01"},
        {"depolarize2", "qubits q[2]; trajectories 4000; seed 14\ndepolarize2(0.3)_{0, 1}\nassert ⟨Z_0⟩ ≈ 1 - 16/15·0.3 ± 0.01"},
        {"pauli", "qubits q[2]; trajectories 4000; seed 15\nprepare |0⟩ ⊗ |+⟩\npauli(0.1, 0.05, 0.2)_0\npauli(0.1, 0.05, 0.2)_1\n"
                  "assert ⟨Z_0⟩ ≈ 1 - 2·(0.1 + 0.05) ± 0.01 and ⟨X_1⟩ ≈ 1 - 2·(0.05 + 0.2) ± 0.01"},
        {"ampdamp", "qubits q[1]; trajectories 4000; seed 16\nX_0\nampdamp(0.25)_0\nassert ⟨Z_0⟩ ≈ -1 + 2·0.25 ± 0.01"},
        {"kraus", "qubits q[1]; trajectories 4000; seed 17\nlet p = 0.2\nkraus([√(1 - p)·[[1, 0], [0, 1]], √p·[[0, 1], [1, 0]]])_0\n"
                  "assert ⟨Z_0⟩ ≈ 1 - 2p ± 0.01"},
    };
    return m;
}

// Each keyword with a program that depends on it for its result.
const std::map<std::string, std::string>& keywordPrograms()
{
    static const std::map<std::string, std::string> m{
        {"noether", "assert true"},
        {"qubits", "qubits a[2], b[1]\nX_{b[0]}\nassert ⟨Z_2⟩ ≈ -1"},
        {"bits", "qubits q[1]; bits c[2]\nX_0\nc[1] ← measure Z_0\nassert c[1] and ¬c[0]"},
        {"seed", "qubits q[1]; bits c[1]; seed 99\nH_0\nc ← measure_q\ncounts ← run 200\nassert counts[\"0\"] + counts[\"1\"] == 200"},
        {"backend", "backend stabilizer\nqubits q[100]\nH_0\nfor k ∈ 0..98: CNOT_{k→k + 1}\nassert ⟨Z_0 Z_{99}⟩ ≈ 1"},
        {"trajectories", "qubits q[1]; trajectories 2000; seed 3\nflip(0.25)_0\nassert ⟨Z_0⟩ ≈ 0.5 ± 0.02"},
        {"let", "let a = 2, b = a + 1\nassert b == 3"},
        {"param", "qubits q[1]\nparam θ ∈ [-π, π] = 0.4\nparam φ[2] ∈ [0, 1] = 0\nRy(θ)_0 Rz(φ[1])_0\nassert ⟨Z_0⟩ ≈ cos(0.4)"},
        {"def", "qubits q[2]\ndef Pair_{a,b}: CNOT_{a→b} H_a\nPair_{0, 1}\nassert ⟨Z_0 Z_1⟩ ≈ 1 and ⟨X_0 X_1⟩ ≈ 1"},
        {"proc", "qubits q[2]; bits c[1]\nproc Flip_{a,b}:\n    c[0] ← measure Z_a\n    if c[0]: X_b\nX_0\nFlip_{0, 1}\nassert ⟨Z_1⟩ ≈ -1"},
        {"prepare", "qubits q[2]\nprepare (|01⟩ - |10⟩)/√2\nassert ⟨Z_0 Z_1⟩ ≈ -1 and ⟨X_0 X_1⟩ ≈ -1"},
        {"measure", "qubits q[2]; bits c[3]\nCNOT_{0→1} H_0\nc[0] ← measure X_0 X_1\nc[1] ← measure Z_0 Z_1\nc[2] ← measure -Z_0 Z_1\n"
                    "assert ¬c[0] and ¬c[1] and c[2]"},
        {"reset", "qubits q[2]\nX_0 X_1\nreset_0\nreset_{q[1]}\nassert ⟨Z_0⟩ ≈ 1 and ⟨Z_1⟩ ≈ 1"},
        {"print", "qubits q[1]\nprint ⟨Z_0⟩\nassert true"},
        {"as", "qubits q[1]\nprint ⟨Z_0⟩ as z\nassert true"},
        {"assert", "assert 1 + 1 == 2"},
        {"run", "qubits q[2]; bits c[2]; seed 5\nX_1\nc ← measure_q\ncounts ← run 100\nassert counts[\"01\"] == 100"},
        {"import", "import \"std/qft.ntr\"\nqubits q[3]\nX_2\nQFT_q\nQFT†_q\nassert |⟨001|ψ⟩|^2 ≈ 1"},
        {"for", "qubits q[4]\nfor k ∈ [0, 3]: X_k\nassert ⟨Z_0⟩ ≈ -1 and ⟨Z_1⟩ ≈ 1 and ⟨Z_3⟩ ≈ -1"},
        {"by", "qubits q[5]\nfor k ∈ 0..4 by 2: X_k\nassert ⟨Z_0 Z_2 Z_4⟩ ≈ -1 and ⟨Z_1⟩ ≈ 1 and ⟨Z_3⟩ ≈ 1"},
        {"if", "qubits q[2]; bits c[1]\nX_0\nc[0] ← measure Z_0\nif c[0]: X_1\nassert ⟨Z_1⟩ ≈ -1"},
        {"else", "qubits q[2]; bits c[1]\nc[0] ← measure Z_0\nif c[0]: X_0\nelse: X_1\nassert ⟨Z_0⟩ ≈ 1 and ⟨Z_1⟩ ≈ -1"},
        {"and", "qubits q[3]; bits c[2]\nX_0 X_1\nc ← measure_{q[0..1]}\nif c[0] and c[1]: X_2\nassert ⟨Z_2⟩ ≈ -1"},
        {"or", "qubits q[3]; bits c[2]\nX_1\nc ← measure_{q[0..1]}\nif c[0] or c[1]: X_2\nassert ⟨Z_2⟩ ≈ -1"},
        {"noise", "qubits q[1]; trajectories 2000; seed 21\nnoise:\n    after gate1: flip(0.1)\nX_0\nassert ⟨Z_0⟩ ≈ -0.8 ± 0.02"},
        {"after", "qubits q[1]; trajectories 2000; seed 22\nnoise:\n    after gate1: dephase(0.2)\nH_0\nassert ⟨X_0⟩ ≈ 0.6 ± 0.03"},
        {"before", "qubits q[1]; bits c[1]; seed 23\nnoise:\n    before measure: flip(0.3)\nc ← measure_q\ncounts ← run 4000\n"
                   "assert counts[\"1\"]/4000 ≈ 0.3 ± 0.03"},
        {"true", "assert true"},
        {"false", "assert ¬false"},
    };
    return m;
}
} // namespace

TEST(Operators, FunctionsComputeTheirDefinitions)
{
    for (const auto& [name, body] : functionPrograms())
    {
        SCOPED_TRACE(name);
        expectRuns("noether 0.1\n" + body + "\n");
    }
}

TEST(Operators, ChannelsFollowTheirDecayLaws)
{
    for (const auto& [name, body] : channelPrograms())
    {
        SCOPED_TRACE(name);
        expectRuns("noether 0.1\n" + body + "\n");
    }
}

TEST(Operators, PauliChannelsRunOnTheStabilizer)
{
    // The same Pauli channels on the tableau, sampled by trajectories.
    for (const std::string name : {"flip", "dephase", "depolarize", "depolarize2", "pauli"})
    {
        SCOPED_TRACE(name);
        CompileOptions stab;
        stab.backend = "stabilizer";
        const Executed run = expectRuns("noether 0.1\n" + channelPrograms().at(name) + "\n", stab);
        EXPECT_EQ(run.c->compiler->ir().backend, "stabilizer");
    }
}

TEST(Operators, NonPauliChannelsNeedTheStateVector)
{
    for (const std::string name : {"ampdamp", "kraus"})
    {
        CompileOptions stab;
        stab.backend = "stabilizer";
        const Executed run = runSource("noether 0.1\n" + channelPrograms().at(name) + "\n", stab);
        EXPECT_EQ(run.codes(), std::vector<std::string>{"E6001"}) << name;
    }
}

TEST(Operators, KeywordsDriveTheStateMachine)
{
    for (const auto& [kw, body] : keywordPrograms())
    {
        SCOPED_TRACE(kw);
        expectRuns("noether 0.1\n" + body + "\n");
    }
}

TEST(Operators, SeedMakesRunsReproducible)
{
    auto counts = [](const std::string& seed)
    {
        const Executed run = expectRuns("noether 0.1\nqubits q[4]; bits c[4]; seed " + seed + "\nH_q\nc ← measure_q\ncounts ← run 500\n");
        return run.r.runs.at(0).counts;
    };
    EXPECT_EQ(counts("8"), counts("8"));
    EXPECT_NE(counts("8"), counts("9"));

    // An unseeded run reports the seed it drew, and passing it back reproduces the run.
    TempDir dir;
    const auto f = dir.path / "unseeded.ntr";
    writeText(f, "noether 0.1\nqubits q[4]; bits c[4]\nH_q\nc ← measure_q\ncounts ← run 500\n");
    const auto first = Json::parse(cli({"run", f.string(), "--json", "--no-timing"}).out);
    ASSERT_TRUE(first && first->find("seed")->isNumber());
    const std::int64_t seed = first->find("seed")->asInt();
    EXPECT_GE(seed, 0);
    const auto again = Json::parse(cli({"run", f.string(), "--json", "--no-timing", "--seed", std::to_string(seed)}).out);
    ASSERT_TRUE(again);
    EXPECT_EQ(again->find("runs")->dump(-1), first->find("runs")->dump(-1));
}

TEST(Operators, ParamsRebindWithoutRecompiling)
{
    CompileOptions o;
    o.sets = {{"theta", "1.2"}};
    const Executed run = expectRuns("noether 0.1\nqubits q[1]\nparam θ ∈ [-π, π] = 0\nRy(θ)_0\nprint ⟨Z_0⟩ as z\n", o);
    EXPECT_NEAR(run.number("z"), std::cos(1.2), 1e-12);
}

TEST(Operators, GreekAndLatexIdentifiersNameTheSameBinding)
{
    expectRuns("noether 0.1\nlet theta = 0.3\nassert \\theta == θ and θ == theta and \\vartheta == θ\n"
               "let \\varphi = 2\nassert phi == 2 and φ == 2\nlet \\mathcal{H} = 5\nassert 𝓗 == 5\n");
}

// ============================================================================================
// Readouts
// ============================================================================================

TEST(Operators, ReadoutsReturnTheExpectedValues)
{
    const Executed run = expectRuns(
        "noether 0.1\nqubits q[2]; bits c[2]; seed 2\n"
        "let φ = (|00⟩ + |11⟩)/√2, M = [[2, 1], [1, -1]]\n"
        "CNOT_{0→1} H_0\n"
        "print ⟨Z_0 Z_1⟩ as zz, ⟨X_0 X_1 + Z_0⟩ as sum, ⟨M_0⟩ as dense, |⟨11|ψ⟩|^2 as p11, ⟨φ|ψ⟩ as ov, ⟨00|X_0 X_1|11⟩ as mel\n"
        "print |ψ⟩ as state, ρ_{q[0]} as reduced\n"
        "c ← measure_q\ncounts ← run 400\nprint counts as counts\n");
    EXPECT_NEAR(run.number("zz"), 1.0, 1e-12);
    EXPECT_NEAR(run.number("sum"), 1.0, 1e-12);
    EXPECT_NEAR(run.number("dense"), 0.5, 1e-12); // ρ_0 = I/2, tr(M)/2
    EXPECT_NEAR(run.number("p11"), 0.5, 1e-12);
    EXPECT_NEAR(run.number("ov"), 1.0, 1e-12);
    EXPECT_NEAR(run.number("mel"), 1.0, 1e-12); // ⟨00|X⊗X|11⟩, independent of the state
    ASSERT_TRUE(run.print("state") && run.print("reduced") && run.print("counts")) << run.diagnostics;
    const auto amps = amplitudes(*run.print("state"));
    ASSERT_EQ(amps.size(), 2U);
    EXPECT_NEAR(amps.at("00").real(), 1 / std::sqrt(2.0), 1e-12);
    EXPECT_NEAR(amps.at("11").real(), 1 / std::sqrt(2.0), 1e-12);
    ASSERT_TRUE(run.print("reduced")->value.contains("matrix")) << run.print("reduced")->value.dump(-1);
    const Json& rho = *run.print("reduced")->value.find("matrix");
    EXPECT_NEAR(rho.asArray()[0].asArray()[0].asArray()[0].asDouble(), 0.5, 1e-12);
    EXPECT_NEAR(rho.asArray()[0].asArray()[1].asArray()[0].asDouble(), 0.0, 1e-12);
    const Json& counts = run.print("counts")->value;
    EXPECT_EQ(counts.find("00")->asInt() + counts.find("11")->asInt(), 400);
    EXPECT_FALSE(counts.contains("01"));
}

// ============================================================================================
// Coverage: every tokens.def entry has a test above
// ============================================================================================

TEST(OperatorCoverage, EveryPunctuationTokenHasACase)
{
    std::set<Tok> covered;
    for (const TokenCase& t : tokenCases()) covered.insert(t.tok);
#define NTR_PUNCT(Name, canonical, ascii) EXPECT_TRUE(covered.contains(Tok::Name)) << "no token case for " << canonical;
#include "tokens.def"
}

TEST(OperatorCoverage, EveryContextualSpellingIsExercised)
{
    // Tokens tested through explicit programs must still use every spelling tokens.def lists.
    const auto spellings = spellingsByToken();
    for (const TokenCase& t : tokenCases())
    {
        if (!t.tmpl.empty() || t.programs.empty()) continue;
        std::string all;
        for (const auto& p : t.programs) all += p + "\n";
        for (const std::string& s : spellings.at(t.tok))
        {
            // The ASCII form of a token pair (\expval{ … }) is written as a whole.
            const std::string probe = s == "^{\\otimes}" ? "^{\\otimes" : s;
            EXPECT_NE(all.find(probe), std::string::npos) << tokName(t.tok) << " spelling `" << s << "` is not exercised";
        }
    }
}

TEST(OperatorCoverage, EveryGateAndAliasHasACase)
{
    std::set<std::string> names;
    std::string text;
    for (const GateCase& g : gateCases())
    {
        std::vector<std::string> forms = g.spellings;
        forms.insert(forms.end(), g.equivalents.begin(), g.equivalents.end());
        for (const std::string& f : forms)
        {
            text += f + "\n";
            SourceManager sm;
            Diagnostics d(sm);
            const auto id = sm.add("t", f);
            for (const Token& t : lex(sm.file(id), id, d).tokens)
                if (t.kind == Tok::Ident) names.insert(t.text);
        }
    }
#define NTR_GATE(name, roles, params, clifford, meaning) EXPECT_TRUE(names.contains(name)) << "no gate case uses " << name;
#include "tokens.def"
#define NTR_GATE_ALIAS(alias, canonical, dagger)                                                                           \
    EXPECT_TRUE(std::regex_search(text, std::regex(std::string("(^|[^A-Za-z])") + alias + "([^A-Za-z]|$)")))               \
        << "no gate case spells the alias " << alias;
#include "tokens.def"
}

TEST(OperatorCoverage, EveryFunctionChannelAndKeywordHasACase)
{
#define NTR_FUNC(name, meaning) EXPECT_TRUE(functionPrograms().contains(name)) << "no function case for " << name;
#include "tokens.def"
#define NTR_CHANNEL(name, arity, params, pauli, meaning) EXPECT_TRUE(channelPrograms().contains(name)) << "no channel case for " << name;
#include "tokens.def"
#define NTR_KEYWORD(Name, text) EXPECT_TRUE(keywordPrograms().contains(text)) << "no keyword case for " << text;
#include "tokens.def"
}
