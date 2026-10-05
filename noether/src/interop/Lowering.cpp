#include "interop/Lowering.hpp"

#include <QuantumGates.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <numbers>
#include <Eigen/Eigenvalues>



namespace Noether::Interop
{

using Qputer::OpKind;
using Qputer::Operation;
using cd = std::complex<double>;

namespace
{
    constexpr double kPi = std::numbers::pi;
    constexpr std::size_t kMaxDense = Qputer::QuantumGate::kMaxDenseTargets;

    // Integer powers at most this long are unrolled; longer ones use the dense principal power.
    constexpr std::size_t kMaxUnrolledPower = 4096;

    cd expi(double a) { return std::polar(1.0, a); }

    Operation op(OpKind kind, QubitList controls, QubitList targets, std::vector<double> params = {})
    {
        Operation o;
        o.kind = kind;
        o.controls = std::move(controls);
        o.targets = std::move(targets);
        o.params = std::move(params);
        return o;
    }

    QubitList concat(const QubitList& a, const QubitList& b)
    {
        QubitList out = a;
        out.insert(out.end(), b.begin(), b.end());
        return out;
    }

    void requireDistinct(const QubitList& qubits, std::string_view what)
    {
        QubitList sorted = qubits;
        std::ranges::sort(sorted);
        if (const auto dup = std::ranges::adjacent_find(sorted); dup != sorted.end())
            throw ImportError("E9006", std::format("{} uses qubit {} more than once", what, *dup));
    }

    // Applies U (on `targets`, targets[0] = MSB) under active-high `controls` to every column of M,
    // all given as bit positions of M's row index.
    void applyToColumns(Eigen::MatrixXcd& M, const std::vector<int>& controls, const std::vector<int>& targets,
                        const Eigen::MatrixXcd& U)
    {
        const Eigen::Index dim = M.rows();
        const std::size_t m = targets.size();
        const Eigen::Index sub = Eigen::Index{1} << m;
        Eigen::Index cmask = 0, tmask = 0;
        for (const int b : controls) cmask |= Eigen::Index{1} << b;
        for (const int b : targets) tmask |= Eigen::Index{1} << b;
        std::vector<Eigen::Index> offset(static_cast<std::size_t>(sub), 0);
        for (Eigen::Index s = 0; s < sub; ++s)
            for (std::size_t j = 0; j < m; ++j)
                if ((s >> (m - 1 - j)) & 1) offset[static_cast<std::size_t>(s)] |= Eigen::Index{1} << targets[j];
        Eigen::VectorXcd a(sub);
        for (Eigen::Index base = 0; base < dim; ++base)
        {
            if ((base & tmask) != 0 || (base & cmask) != cmask) continue;
            for (Eigen::Index col = 0; col < M.cols(); ++col)
            {
                for (Eigen::Index s = 0; s < sub; ++s) a[s] = M(base | offset[static_cast<std::size_t>(s)], col);
                const Eigen::VectorXcd b = U * a;
                for (Eigen::Index s = 0; s < sub; ++s) M(base | offset[static_cast<std::size_t>(s)], col) = b[s];
            }
        }
    }

    // Principal power of a unitary: U = Q T Q† (Schur) with T diagonal because U is normal.
    Eigen::MatrixXcd principalPower(const Eigen::MatrixXcd& U, double r)
    {
        const Eigen::ComplexSchur<Eigen::MatrixXcd> schur(U);
        if (schur.info() != Eigen::Success) throw ImportError("E9003", "the Schur decomposition for a gate power did not converge");
        const Eigen::MatrixXcd& Q = schur.matrixU();
        const Eigen::MatrixXcd& T = schur.matrixT();
        Eigen::VectorXcd d(T.rows());
        for (Eigen::Index k = 0; k < T.rows(); ++k)
        {
            const cd lambda = T(k, k);
            double arg = std::arg(lambda);
            // arg ∈ (−π, π]: an eigenvalue −1 that rounds to just below the negative real axis
            // belongs to +π, so that pow(1/2) @ z is s rather than sdg.
            if (arg < -kPi + 1e-9) arg += 2 * kPi;
            d[k] = std::pow(std::abs(lambda), r) * expi(r * arg);
        }
        return Q * d.asDiagonal() * Q.adjoint();
    }

    bool isInteger(double r) { return std::isfinite(r) && std::nearbyint(r) == r; }

    // Phase angle of a diagonal gate diag(1, e^{iλ}), when g is one.
    std::optional<double> phaseAngle(const Gate& g)
    {
        switch (g.prim)
        {
            case Prim::Z: return kPi;
            case Prim::S: return kPi / 2;
            case Prim::Sdg: return -kPi / 2;
            case Prim::T: return kPi / 4;
            case Prim::Tdg: return -kPi / 4;
            case Prim::Phase: return g.params[0];
            default: return std::nullopt;
        }
    }

    // Multiple k of `step` when angle = k·step to within rounding, else nullopt.
    std::optional<long long> multipleOf(double angle, double step)
    {
        if (!std::isfinite(angle)) return std::nullopt;
        const double k = std::nearbyint(angle / step);
        if (std::abs(angle - k * step) > 1e-12 * std::max(1.0, std::abs(angle))) return std::nullopt;
        return static_cast<long long>(k);
    }

    int quarter(long long k) { return static_cast<int>(((k % 4) + 4) % 4); }
} // namespace


Gate makeGate(Prim prim, QubitList targets, std::vector<double> params, QubitList controls)
{
    Gate g;
    g.prim = prim;
    g.targets = std::move(targets);
    g.params = std::move(params);
    g.controls = std::move(controls);
    return g;
}

Gate denseGate(QubitList targets, Eigen::MatrixXcd matrix)
{
    Gate g;
    g.prim = Prim::Unitary;
    g.targets = std::move(targets);
    g.matrix = std::move(matrix);
    return g;
}


// ---- Gate algebra ----

Seq controlled(Seq body, const QubitList& controls, const std::vector<bool>& active)
{
    if (controls.empty()) return body;
    requireDistinct(controls, "a controlled gate");
    for (Gate& g : body)
    {
        for (const Qubit c : controls)
            if (std::ranges::find(g.controls, c) != g.controls.end() || std::ranges::find(g.targets, c) != g.targets.end())
                throw ImportError("E9006", std::format("qubit {} is both a control and an operand of the controlled gate", c));
        g.controls = concat(controls, g.controls);
    }
    Seq flips;
    for (std::size_t k = 0; k < active.size() && k < controls.size(); ++k)
        if (!active[k]) flips.push_back(makeGate(Prim::X, {controls[k]}));
    if (flips.empty()) return body;
    Seq out = flips;
    out.insert(out.end(), body.begin(), body.end());
    out.insert(out.end(), flips.begin(), flips.end());
    return out;
}

Seq inverse(const Seq& body)
{
    Seq out;
    out.reserve(body.size() + 1);
    for (auto it = body.rbegin(); it != body.rend(); ++it)
    {
        Gate g = *it;
        switch (g.prim)
        {
            case Prim::X: case Prim::Y: case Prim::Z: case Prim::H: case Prim::Swap: break;
            case Prim::S: g.prim = Prim::Sdg; break;
            case Prim::Sdg: g.prim = Prim::S; break;
            case Prim::T: g.prim = Prim::Tdg; break;
            case Prim::Tdg: g.prim = Prim::T; break;
            case Prim::SX:
            {
                // SX² = X, so SX⁻¹ = X·SX exactly.
                Gate x = g;
                x.prim = Prim::X;
                out.push_back(g);
                out.push_back(std::move(x));
                continue;
            }
            case Prim::RX: case Prim::RY: case Prim::RZ: case Prim::Phase: case Prim::GPhase: g.params[0] = -g.params[0]; break;
            case Prim::U3: g.params = {-g.params[0], -g.params[2], -g.params[1]}; break;
            case Prim::Unitary: g.matrix = g.matrix.adjoint().eval(); break;
        }
        out.push_back(std::move(g));
    }
    return out;
}

Seq power(const Seq& body, double exponent)
{
    if (!std::isfinite(exponent)) throw ImportError("E9001", std::format("gate power {} is not finite", exponent));
    if (body.empty()) return body;
    if (isInteger(exponent) && std::abs(exponent) * static_cast<double>(body.size()) <= static_cast<double>(kMaxUnrolledPower))
    {
        const Seq base = exponent < 0 ? inverse(body) : body;
        Seq out;
        for (long long k = 0; k < std::llabs(static_cast<long long>(exponent)); ++k) out.insert(out.end(), base.begin(), base.end());
        return out;
    }
    if (body.size() == 1)
    {
        Gate g = body.front();
        switch (g.prim)
        {
            case Prim::RX: case Prim::RY: case Prim::RZ: case Prim::Phase: case Prim::GPhase:
                g.params[0] *= exponent;
                return {g};
            default:
                break;
        }
        if (const auto lambda = phaseAngle(g))
        {
            // diag(1, e^{iλ}) with λ ∈ (−π, π]: the principal power is diag(1, e^{irλ}).
            g.prim = Prim::Phase;
            g.params = {*lambda * exponent};
            return {g};
        }
        // C(U)^r = C(U^r): the control-off subspace has eigenvalue 1, whose principal power is 1.
        Gate out = denseGate(g.targets, principalPower(baseMatrix(g), exponent));
        out.controls = g.controls;
        return {out};
    }
    const QubitList sup = supportOf(body);
    if (sup.size() > kMaxDense)
        throw ImportError("E9004", std::format("a non-integer power of a gate on {} qubits needs a dense matrix; at most {} qubits",
                                               sup.size(), kMaxDense));
    return {denseGate(sup, principalPower(matrixOf(body, sup), exponent))};
}

QubitList supportOf(const Seq& body)
{
    QubitList out;
    auto add = [&](Qubit q)
    {
        if (std::ranges::find(out, q) == out.end()) out.push_back(q);
    };
    for (const Gate& g : body)
    {
        for (const Qubit q : g.controls) add(q);
        for (const Qubit q : g.targets) add(q);
    }
    return out;
}

Eigen::MatrixXcd baseMatrix(const Gate& g)
{
    const double r = 1.0 / std::numbers::sqrt2;
    Eigen::MatrixXcd m(2, 2);
    switch (g.prim)
    {
        case Prim::X: m << 0, 1, 1, 0; break;
        case Prim::Y: m << 0, cd(0, -1), cd(0, 1), 0; break;
        case Prim::Z: m << 1, 0, 0, -1; break;
        case Prim::H: m << r, r, r, -r; break;
        case Prim::S: m << 1, 0, 0, cd(0, 1); break;
        case Prim::Sdg: m << 1, 0, 0, cd(0, -1); break;
        case Prim::T: m << 1, 0, 0, expi(kPi / 4); break;
        case Prim::Tdg: m << 1, 0, 0, expi(-kPi / 4); break;
        case Prim::SX: m << cd(0.5, 0.5), cd(0.5, -0.5), cd(0.5, -0.5), cd(0.5, 0.5); break;
        case Prim::RX:
        {
            const double c = std::cos(g.params[0] / 2), s = std::sin(g.params[0] / 2);
            m << c, cd(0, -s), cd(0, -s), c;
            break;
        }
        case Prim::RY:
        {
            const double c = std::cos(g.params[0] / 2), s = std::sin(g.params[0] / 2);
            m << c, -s, s, c;
            break;
        }
        case Prim::RZ: m << expi(-g.params[0] / 2), 0, 0, expi(g.params[0] / 2); break;
        case Prim::Phase: m << 1, 0, 0, expi(g.params[0]); break;
        case Prim::U3:
        {
            const double c = std::cos(g.params[0] / 2), s = std::sin(g.params[0] / 2);
            const double phi = g.params[1], lambda = g.params[2];
            m << c, -s * expi(lambda), s * expi(phi), c * expi(phi + lambda);
            break;
        }
        case Prim::Swap:
            m = Eigen::MatrixXcd::Zero(4, 4);
            m(0, 0) = m(1, 2) = m(2, 1) = m(3, 3) = 1;
            break;
        case Prim::Unitary: return g.matrix;
        case Prim::GPhase:
            m.resize(1, 1);
            m(0, 0) = expi(g.params[0]);
            break;
    }
    return m;
}

Eigen::MatrixXcd matrixOf(const Seq& body, const QubitList& support)
{
    const std::size_t k = support.size();
    if (k > kMaxDense) throw ImportError("E9004", std::format("a dense matrix on {} qubits; at most {}", k, kMaxDense));
    const auto dim = Eigen::Index{1} << k;
    Eigen::MatrixXcd M = Eigen::MatrixXcd::Identity(dim, dim);
    auto local = [&](Qubit q)
    {
        const auto it = std::ranges::find(support, q);
        if (it == support.end()) throw std::logic_error(std::format("matrixOf: qubit {} outside the support", q));
        return static_cast<int>(k - 1 - static_cast<std::size_t>(it - support.begin()));
    };
    for (const Gate& g : body)
    {
        std::vector<int> controls, targets;
        for (const Qubit q : g.controls) controls.push_back(local(q));
        for (const Qubit q : g.targets) targets.push_back(local(q));
        if (g.prim == Prim::GPhase)
        {
            if (controls.empty())
            {
                M *= expi(g.params[0]);
                continue;
            }
            // A controlled global phase is a phase gate on the last control.
            targets = {controls.back()};
            controls.pop_back();
            Eigen::MatrixXcd p(2, 2);
            p << 1, 0, 0, expi(g.params[0]);
            applyToColumns(M, controls, targets, p);
            continue;
        }
        applyToColumns(M, controls, targets, baseMatrix(g));
    }
    return M;
}


// ---- Circuit construction ----

void CircuitBuilder::push(Operation o, const std::optional<Condition>& condition)
{
    o.condition = condition;
    c.ops.push_back(std::move(o));
}

void CircuitBuilder::emit(const Gate& g, const std::optional<Condition>& condition)
{
    for (const double p : g.params)
        if (!std::isfinite(p)) throw ImportError("E9001", std::format("gate parameter {} is not finite", p));
    requireDistinct(concat(g.controls, g.targets), "a gate");
    const QubitList& ctl = g.controls;
    const QubitList& tgt = g.targets;
    const std::size_t nc = ctl.size();

    auto phaseOn = [&](const QubitList& qubits, double lambda)
    {
        if (qubits.size() == 1) push(op(OpKind::Phase, {}, qubits, {lambda}), condition);
        else if (qubits.size() == 2) push(op(OpKind::CPhase, {qubits[0]}, {qubits[1]}, {lambda}), condition);
        else push(op(OpKind::MCPhase, {}, qubits, {lambda}), condition);
    };
    auto dense = [&](const Eigen::MatrixXcd& m)
    {
        if (tgt.size() > kMaxDense)
            throw ImportError("E9004", std::format("a dense gate on {} targets; at most {}", tgt.size(), kMaxDense));
        Operation o = op(OpKind::Unitary, ctl, tgt);
        o.matrix = m;
        push(std::move(o), condition);
    };

    switch (g.prim)
    {
        case Prim::GPhase:
            if (nc > 0) phaseOn(ctl, g.params[0]);
            return;
        case Prim::X:
            if (nc == 0) push(op(OpKind::X, {}, tgt), condition);
            else if (nc == 1) push(op(OpKind::CNOT, ctl, tgt), condition);
            else if (nc == 2) push(op(OpKind::Toffoli, ctl, tgt), condition);
            else push(op(OpKind::MCX, ctl, tgt), condition);
            return;
        case Prim::Z:
            if (nc == 0) push(op(OpKind::Z, {}, tgt), condition);
            else if (nc == 1) push(op(OpKind::CZ, ctl, tgt), condition);
            else push(op(OpKind::MCZ, {}, concat(ctl, tgt)), condition);
            return;
        case Prim::Phase:
            if (nc == 0) push(op(OpKind::Phase, {}, tgt, g.params), condition);
            else phaseOn(concat(ctl, tgt), g.params[0]);
            return;
        case Prim::S: case Prim::Sdg: case Prim::T: case Prim::Tdg:
            if (nc == 0)
            {
                const OpKind k = g.prim == Prim::S ? OpKind::S : g.prim == Prim::Sdg ? OpKind::Sdg : g.prim == Prim::T ? OpKind::T : OpKind::Tdg;
                push(op(k, {}, tgt), condition);
            }
            else phaseOn(concat(ctl, tgt), *phaseAngle(g));
            return;
        case Prim::RZ:
            if (nc == 0) push(op(OpKind::RZ, {}, tgt, g.params), condition);
            else
            {
                // C(Rz(θ)) = C(e^{−iθ/2}) · C(P(θ)), keeping the native phase kernels.
                phaseOn(ctl, -g.params[0] / 2);
                phaseOn(concat(ctl, tgt), g.params[0]);
            }
            return;
        case Prim::Swap:
            if (nc == 0) push(op(OpKind::Swap, {}, tgt), condition);
            else if (nc == 1) push(op(OpKind::Fredkin, ctl, tgt), condition);
            else dense(baseMatrix(g));
            return;
        case Prim::Y: case Prim::H: case Prim::SX: case Prim::RX: case Prim::RY: case Prim::U3:
            if (nc > 0)
            {
                dense(baseMatrix(g));
                return;
            }
            switch (g.prim)
            {
                case Prim::Y: push(op(OpKind::Y, {}, tgt), condition); break;
                case Prim::H: push(op(OpKind::H, {}, tgt), condition); break;
                case Prim::SX: push(op(OpKind::SX, {}, tgt), condition); break;
                case Prim::RX: push(op(OpKind::RX, {}, tgt, g.params), condition); break;
                case Prim::RY: push(op(OpKind::RY, {}, tgt, g.params), condition); break;
                default: push(op(OpKind::U3, {}, tgt, g.params), condition); break;
            }
            return;
        case Prim::Unitary:
            if (!g.matrix.allFinite()) throw ImportError("E9001", "a dense gate matrix holds NaN or Inf");
            dense(g.matrix);
            return;
    }
}

void CircuitBuilder::apply(const Seq& seq, const std::optional<Condition>& condition)
{
    for (const Gate& g : seq) emit(g, condition);
}

void CircuitBuilder::measure(Qubit q, std::optional<std::size_t> clbit, bool invert, double flip)
{
    if (invert) push(op(OpKind::X, {}, {q}), std::nullopt);
    if (flip > 0.0) push(op(OpKind::PauliChannel, {}, {q}, {flip, 0.0, 0.0}), std::nullopt);
    Operation m = op(OpKind::Measure, {}, {q});
    m.clbit = clbit;
    push(std::move(m), std::nullopt);
    if (invert) push(op(OpKind::X, {}, {q}), std::nullopt);
}

void CircuitBuilder::basisChange(const std::vector<PauliFactor>& factors, bool undo)
{
    for (const PauliFactor& f : factors)
    {
        // V maps the letter onto Z: X by H, Y by S† then H (Y → X → Z).
        if (f.letter == 'X') push(op(OpKind::H, {}, {f.qubit}), std::nullopt);
        else if (f.letter == 'Y' && !undo)
        {
            push(op(OpKind::Sdg, {}, {f.qubit}), std::nullopt);
            push(op(OpKind::H, {}, {f.qubit}), std::nullopt);
        }
        else if (f.letter == 'Y')
        {
            push(op(OpKind::H, {}, {f.qubit}), std::nullopt);
            push(op(OpKind::S, {}, {f.qubit}), std::nullopt);
        }
        else if (f.letter != 'Z') throw ImportError("E9001", std::format("unknown Pauli letter `{}`", f.letter));
    }
}

void CircuitBuilder::measurePauli(const std::vector<PauliFactor>& factors, std::optional<std::size_t> clbit, bool invert,
                                  double flip)
{
    if (factors.empty()) throw ImportError("E9006", "a Pauli-product measurement needs at least one factor");
    QubitList qs;
    for (const PauliFactor& f : factors) qs.push_back(f.qubit);
    requireDistinct(qs, "a Pauli product");
    const Qubit last = qs.back();
    basisChange(factors, false);
    for (std::size_t k = 0; k + 1 < qs.size(); ++k) push(op(OpKind::CNOT, {qs[k]}, {last}), std::nullopt);
    measure(last, clbit, invert, flip);
    for (std::size_t k = qs.size() - 1; k-- > 0;) push(op(OpKind::CNOT, {qs[k]}, {last}), std::nullopt);
    basisChange(factors, true);
}

void CircuitBuilder::rotatePauli(const std::vector<PauliFactor>& factors, bool dagger)
{
    if (factors.empty()) throw ImportError("E9006", "a Pauli-product rotation needs at least one factor");
    QubitList qs;
    for (const PauliFactor& f : factors) qs.push_back(f.qubit);
    requireDistinct(qs, "a Pauli product");
    const Qubit last = qs.back();
    basisChange(factors, false);
    for (std::size_t k = 0; k + 1 < qs.size(); ++k) push(op(OpKind::CNOT, {qs[k]}, {last}), std::nullopt);
    push(op(dagger ? OpKind::Sdg : OpKind::S, {}, {last}), std::nullopt);
    for (std::size_t k = qs.size() - 1; k-- > 0;) push(op(OpKind::CNOT, {qs[k]}, {last}), std::nullopt);
    basisChange(factors, true);
}

void CircuitBuilder::reset(Qubit q, const std::optional<Condition>& condition) { push(op(OpKind::Reset, {}, {q}), condition); }

void CircuitBuilder::pauliChannel(QubitList targets, std::vector<double> probabilities, const std::optional<Condition>& condition)
{
    requireDistinct(targets, "a Pauli channel");
    push(op(OpKind::PauliChannel, {}, std::move(targets), std::move(probabilities)), condition);
}

void CircuitBuilder::kraus(QubitList targets, std::vector<Eigen::MatrixXcd> operators, const std::optional<Condition>& condition)
{
    requireDistinct(targets, "a Kraus channel");
    if (targets.size() > kMaxDense)
        throw ImportError("E9004", std::format("a Kraus channel on {} qubits; at most {}", targets.size(), kMaxDense));
    Operation o = op(OpKind::Kraus, {}, std::move(targets));
    o.kraus = std::move(operators);
    push(std::move(o), condition);
}


// ---- Operations read from circuit JSON ----

Operation makeOperation(OpKind kind, QubitList controls, QubitList targets, std::vector<double> params, Eigen::MatrixXcd matrix,
                        std::vector<Eigen::MatrixXcd> kraus, std::optional<std::size_t> clbit, std::optional<Condition> condition)
{
    Operation o = op(kind, std::move(controls), std::move(targets), std::move(params));
    o.matrix = std::move(matrix);
    o.kraus = std::move(kraus);
    o.clbit = clbit;
    o.condition = condition;
    return o;
}

std::string validateOperation(const Operation& o, std::size_t nQubits, std::size_t nClbits)
{
    constexpr int any = -1;
    struct Arity
    {
        int controls, targets, params;
    };
    auto arity = [](OpKind k) -> Arity
    {
        switch (k)
        {
            case OpKind::RX: case OpKind::RY: case OpKind::RZ: case OpKind::Phase: return {0, 1, 1};
            case OpKind::U3: return {0, 1, 3};
            case OpKind::CNOT: case OpKind::CZ: return {1, 1, 0};
            case OpKind::CPhase: return {1, 1, 1};
            case OpKind::Swap: return {0, 2, 0};
            case OpKind::Toffoli: return {2, 1, 0};
            case OpKind::Fredkin: return {1, 2, 0};
            case OpKind::MCX: return {any, 1, 0};
            case OpKind::MCZ: return {0, any, 0};
            case OpKind::MCPhase: return {0, any, 1};
            case OpKind::Unitary: return {any, any, 0};
            case OpKind::PauliChannel: return {0, any, any};
            case OpKind::Kraus: return {0, any, 0};
            default: return {0, 1, 0}; // the fixed 1-qubit gates, measure, reset
        }
    };
    const Arity a = arity(o.kind);
    const std::string_view name = Qputer::opName(o.kind);
    auto count = [&](std::string_view role, std::size_t got, int want) -> std::string
    {
        if (want == any || got == static_cast<std::size_t>(want)) return {};
        return std::format("`{}` takes {} {}, got {}", name, want, role, got);
    };
    for (const std::string& e : {count("controls", o.controls.size(), a.controls), count("targets", o.targets.size(), a.targets),
                                 count("params", o.params.size(), a.params)})
        if (!e.empty()) return e;
    if (o.targets.empty()) return std::format("`{}` needs at least one target", name);

    QubitList all = concat(o.controls, o.targets);
    for (const Qubit q : all)
        if (q >= nQubits) return std::format("qubit {} out of range [0, {})", q, nQubits);
    std::ranges::sort(all);
    if (const auto dup = std::ranges::adjacent_find(all); dup != all.end()) return std::format("qubit {} used more than once", *dup);
    for (const double p : o.params)
        if (!std::isfinite(p)) return std::format("non-finite parameter {}", p);

    if (o.kind == OpKind::Unitary || o.kind == OpKind::Kraus)
    {
        if (o.targets.size() > kMaxDense) return std::format("{} targets; dense operations take at most {}", o.targets.size(), kMaxDense);
        const auto dim = Eigen::Index{1} << o.targets.size();
        const double tol = Qputer::QuantumGate::kUnitaryTolerance;
        if (o.kind == OpKind::Unitary)
        {
            if (o.matrix.rows() != dim || o.matrix.cols() != dim)
                return std::format("matrix is {}x{}, expected {}x{}", o.matrix.rows(), o.matrix.cols(), dim, dim);
            if (!o.matrix.allFinite()) return "matrix holds NaN or Inf";
            const double err = (o.matrix.adjoint() * o.matrix - Eigen::MatrixXcd::Identity(dim, dim)).cwiseAbs().maxCoeff();
            if (!(err <= tol)) return std::format("matrix is not unitary (max |U†U − I| = {:.3e})", err);
        }
        else
        {
            if (o.kraus.empty()) return "a Kraus channel needs at least one operator";
            Eigen::MatrixXcd sum = Eigen::MatrixXcd::Zero(dim, dim);
            for (std::size_t k = 0; k < o.kraus.size(); ++k)
            {
                const Eigen::MatrixXcd& K = o.kraus[k];
                if (K.rows() != dim || K.cols() != dim)
                    return std::format("Kraus operator {} is {}x{}, expected {}x{}", k, K.rows(), K.cols(), dim, dim);
                if (!K.allFinite()) return std::format("Kraus operator {} holds NaN or Inf", k);
                sum.noalias() += K.adjoint() * K;
            }
            const double err = (sum - Eigen::MatrixXcd::Identity(dim, dim)).cwiseAbs().maxCoeff();
            if (!(err <= tol)) return std::format("Kraus operators are not trace preserving (max |ΣK†K − I| = {:.3e})", err);
        }
    }
    if (o.kind == OpKind::PauliChannel)
    {
        if (o.targets.size() > 2) return "a Pauli channel acts on 1 or 2 qubits";
        const std::size_t terms = o.targets.size() == 1 ? 3 : 15;
        if (o.params.size() != terms) return std::format("a Pauli channel on {} qubit(s) takes {} probabilities", o.targets.size(), terms);
        double total = 0.0;
        for (const double p : o.params)
        {
            if (!(p >= 0.0 && p <= 1.0)) return std::format("probability {} outside [0, 1]", p);
            total += p;
        }
        if (!(total <= 1.0 + 1e-12)) return std::format("probabilities sum to {} > 1", total);
    }
    if (o.clbit)
    {
        if (o.kind != OpKind::Measure) return "only `measure` writes a clbit";
        if (*o.clbit >= nClbits) return std::format("clbit {} out of range [0, {})", *o.clbit, nClbits);
    }
    if (o.condition)
    {
        if (o.kind == OpKind::Measure) return "a measurement cannot be conditioned";
        if (o.condition->mask == 0) return "a condition must name at least one clbit";
        if ((o.condition->value & ~o.condition->mask) != 0) return "condition value sets bits outside its mask";
        if (nClbits < 64 && (o.condition->mask >> nClbits) != 0) return std::format("condition clbit out of range [0, {})", nClbits);
    }
    return {};
}


// ---- Runner support ----

std::optional<std::vector<Operation>> cliffordForm(const Operation& o)
{
    if (o.kind == OpKind::CPhase)
    {
        const auto k = multipleOf(o.params[0], kPi);
        if (!k) return std::nullopt;
        std::vector<Operation> out;
        if (*k % 2 != 0) out.push_back(op(OpKind::CZ, o.controls, o.targets));
        for (Operation& x : out) x.condition = o.condition;
        return out;
    }
    if (o.kind != OpKind::RX && o.kind != OpKind::RY && o.kind != OpKind::RZ && o.kind != OpKind::Phase) return std::nullopt;
    const auto k = multipleOf(o.params[0], kPi / 2);
    if (!k) return std::nullopt;
    const Qubit q = o.targets[0];
    const int n = quarter(*k);
    std::vector<OpKind> seq;
    switch (o.kind)
    {
        case OpKind::RZ: case OpKind::Phase:
            if (n == 1) seq = {OpKind::S};
            else if (n == 2) seq = {OpKind::Z};
            else if (n == 3) seq = {OpKind::Sdg};
            break;
        case OpKind::RX:
            if (n == 1) seq = {OpKind::SX};
            else if (n == 2) seq = {OpKind::X};
            else if (n == 3) seq = {OpKind::SX, OpKind::X};
            break;
        default: // RY: Ry(π/2) ∝ X·H, Ry(π) ∝ Y, Ry(3π/2) ∝ Z·H
            if (n == 1) seq = {OpKind::H, OpKind::X};
            else if (n == 2) seq = {OpKind::Y};
            else if (n == 3) seq = {OpKind::H, OpKind::Z};
            break;
    }
    std::vector<Operation> out;
    for (const OpKind kind : seq)
    {
        out.push_back(op(kind, {}, {q}));
        out.back().condition = o.condition;
    }
    return out;
}

} // namespace Noether::Interop
