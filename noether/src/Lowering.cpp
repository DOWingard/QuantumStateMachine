#include "Lowering.hpp"

#include <QuantumGates.hpp>

#include <cmath>
#include <numbers>
#include <stdexcept>



namespace Noether
{

using Qputer::OpKind;
using Qputer::Operation;

namespace
{
    Operation op1(OpKind k, Qubit t, std::vector<double> params = {})
    {
        Operation o;
        o.kind = k;
        o.targets = {t};
        o.params = std::move(params);
        return o;
    }

    Operation op2(OpKind k, Qubit c, Qubit t, std::vector<double> params = {})
    {
        Operation o;
        o.kind = k;
        o.controls = {c};
        o.targets = {t};
        o.params = std::move(params);
        return o;
    }

    Operation dense(const QubitList& controls, const QubitList& targets, Eigen::MatrixXcd m)
    {
        Operation o;
        o.kind = OpKind::Unitary;
        o.controls = controls;
        o.targets = targets;
        o.matrix = std::move(m);
        return o;
    }

    // Quarter turns k (mod 4) of an exact angle in (π/2)ℤ; the stabilizer path only sees such angles.
    int quarterTurns(const Affine& a)
    {
        if (!a.isConst() || !a.c.exact()) throw std::logic_error("non-Clifford angle reached the stabilizer lowering");
        const Rational p = a.c.piCoef();
        const std::int64_t k = (p.n * 2) / p.d;
        return static_cast<int>(((k % 4) + 4) % 4);
    }

    // exp(-i θ/2 Z) up to global phase, θ = kπ/2.
    void cliffordRz(std::vector<Operation>& out, Qubit q, int k)
    {
        if (k == 1) out.push_back(op1(OpKind::S, q));
        else if (k == 2) out.push_back(op1(OpKind::Z, q));
        else if (k == 3) out.push_back(op1(OpKind::Sdg, q));
    }

    void cliffordRx(std::vector<Operation>& out, Qubit q, int k)
    {
        if (k == 1) out.push_back(op1(OpKind::SX, q));
        else if (k == 2) out.push_back(op1(OpKind::X, q));
        else if (k == 3)
        {
            out.push_back(op1(OpKind::SX, q));
            out.push_back(op1(OpKind::X, q));
        }
    }

    void cliffordRy(std::vector<Operation>& out, Qubit q, int k)
    {
        // Ry(π/2) = H·Z and Ry(-π/2) = Z·H as matrices.
        if (k == 1)
        {
            out.push_back(op1(OpKind::Z, q));
            out.push_back(op1(OpKind::H, q));
        }
        else if (k == 2) out.push_back(op1(OpKind::Y, q));
        else if (k == 3)
        {
            out.push_back(op1(OpKind::H, q));
            out.push_back(op1(OpKind::Z, q));
        }
    }

    // Basis change taking Pauli `letter` on q to Z (before), or back (after).
    void toZ(std::vector<Operation>& out, Qubit q, char letter, bool undo)
    {
        if (letter == 'X') out.push_back(op1(OpKind::H, q));
        else if (letter == 'Y')
        {
            if (!undo)
            {
                out.push_back(op1(OpKind::Sdg, q));
                out.push_back(op1(OpKind::H, q));
            }
            else
            {
                out.push_back(op1(OpKind::H, q));
                out.push_back(op1(OpKind::S, q));
            }
        }
    }

    // Parity of `qs` onto qs.back() by a CNOT ladder (or its undo).
    void ladder(std::vector<Operation>& out, const QubitList& qs, bool undo)
    {
        if (qs.size() < 2) return;
        if (!undo)
            for (std::size_t k = 0; k + 1 < qs.size(); ++k) out.push_back(op2(OpKind::CNOT, qs[k], qs.back()));
        else
            for (std::size_t k = qs.size() - 1; k-- > 0;) out.push_back(op2(OpKind::CNOT, qs[k], qs.back()));
    }

    Eigen::MatrixXcd swapMatrix() { return gateMatrix(GK::SWAP, {}); }

    // The op without controls, lowered for the given backend.
    void lowerBase(std::vector<Operation>& out, const IrOp& op, const std::vector<double>& a, bool stab)
    {
        const QubitList& t = op.targets;
        switch (op.kind)
        {
            case GK::I:
            case GK::GPhase: return;
            case GK::X: out.push_back(op1(OpKind::X, t[0])); return;
            case GK::Y: out.push_back(op1(OpKind::Y, t[0])); return;
            case GK::Z: out.push_back(op1(OpKind::Z, t[0])); return;
            case GK::H: out.push_back(op1(OpKind::H, t[0])); return;
            case GK::S: out.push_back(op1(OpKind::S, t[0])); return;
            case GK::Sdg: out.push_back(op1(OpKind::Sdg, t[0])); return;
            case GK::T: out.push_back(op1(OpKind::T, t[0])); return;
            case GK::Tdg: out.push_back(op1(OpKind::Tdg, t[0])); return;
            case GK::SX: out.push_back(op1(OpKind::SX, t[0])); return;
            case GK::SXdg: // √X† = X·√X
                out.push_back(op1(OpKind::SX, t[0]));
                out.push_back(op1(OpKind::X, t[0]));
                return;
            case GK::RX:
                if (stab) cliffordRx(out, t[0], quarterTurns(op.angles[0]));
                else out.push_back(op1(OpKind::RX, t[0], {a[0]}));
                return;
            case GK::RY:
                if (stab) cliffordRy(out, t[0], quarterTurns(op.angles[0]));
                else out.push_back(op1(OpKind::RY, t[0], {a[0]}));
                return;
            case GK::RZ:
                if (stab) cliffordRz(out, t[0], quarterTurns(op.angles[0]));
                else out.push_back(op1(OpKind::RZ, t[0], {a[0]}));
                return;
            case GK::P:
                if (stab) cliffordRz(out, t[0], quarterTurns(op.angles[0]));
                else out.push_back(op1(OpKind::Phase, t[0], {a[0]}));
                return;
            case GK::U3:
                if (stab)
                {
                    // U3(θ,φ,λ) = Rz(φ) Ry(θ) Rz(λ) up to global phase.
                    cliffordRz(out, t[0], quarterTurns(op.angles[2]));
                    cliffordRy(out, t[0], quarterTurns(op.angles[0]));
                    cliffordRz(out, t[0], quarterTurns(op.angles[1]));
                }
                else out.push_back(op1(OpKind::U3, t[0], {a[0], a[1], a[2]}));
                return;
            case GK::CNOT: out.push_back(op2(OpKind::CNOT, t[0], t[1])); return;
            case GK::CZ: out.push_back(op2(OpKind::CZ, t[0], t[1])); return;
            case GK::CP:
                if (stab)
                {
                    if (quarterTurns(op.angles[0]) == 2) out.push_back(op2(OpKind::CZ, t[0], t[1]));
                }
                else out.push_back(op2(OpKind::CPhase, t[0], t[1], {a[0]}));
                return;
            case GK::SWAP:
            {
                Operation o;
                o.kind = OpKind::Swap;
                o.targets = {t[0], t[1]};
                out.push_back(std::move(o));
                return;
            }
            case GK::Toffoli:
            {
                Operation o;
                o.kind = OpKind::Toffoli;
                o.controls = {t[0], t[1]};
                o.targets = {t[2]};
                out.push_back(std::move(o));
                return;
            }
            case GK::Fredkin:
            {
                Operation o;
                o.kind = OpKind::Fredkin;
                o.controls = {t[0]};
                o.targets = {t[1], t[2]};
                out.push_back(std::move(o));
                return;
            }
            case GK::PauliRot:
            {
                for (std::size_t k = 0; k < t.size(); ++k) toZ(out, t[k], op.paulis[k], false);
                ladder(out, t, false);
                if (stab) cliffordRz(out, t.back(), quarterTurns(op.angles[0]));
                else out.push_back(op1(OpKind::RZ, t.back(), {a[0]}));
                ladder(out, t, true);
                for (std::size_t k = 0; k < t.size(); ++k) toZ(out, t[k], op.paulis[k], true);
                return;
            }
            case GK::Matrix: out.push_back(dense({}, t, *op.matrix)); return;
            case GK::MeasureZ:
            {
                Operation o = op1(OpKind::Measure, t[0]);
                o.clbit = op.clbit;
                out.push_back(std::move(o));
                return;
            }
            case GK::MeasurePauli:
            {
                for (std::size_t k = 0; k < t.size(); ++k) toZ(out, t[k], op.paulis[k], false);
                ladder(out, t, false);
                if (op.negate) out.push_back(op1(OpKind::X, t.back()));
                Operation m = op1(OpKind::Measure, t.back());
                m.clbit = op.clbit;
                out.push_back(std::move(m));
                if (op.negate) out.push_back(op1(OpKind::X, t.back()));
                ladder(out, t, true);
                for (std::size_t k = 0; k < t.size(); ++k) toZ(out, t[k], op.paulis[k], true);
                return;
            }
            case GK::Reset: out.push_back(op1(OpKind::Reset, t[0])); return;
            case GK::Channel:
            {
                Operation o;
                o.targets = t;
                if (op.channel->pauli)
                {
                    o.kind = OpKind::PauliChannel;
                    o.params = op.channel->probs;
                }
                else
                {
                    o.kind = OpKind::Kraus;
                    o.kraus = op.channel->kraus;
                }
                out.push_back(std::move(o));
                return;
            }
        }
    }

    // The op with positive controls `c` (negative controls already conjugated by X).
    void lowerControlled(std::vector<Operation>& out, const IrOp& op, const QubitList& c, const std::vector<double>& a, bool stab)
    {
        const QubitList& t = op.targets;
        auto all = [&](const QubitList& extra)
        {
            QubitList q = c;
            q.insert(q.end(), extra.begin(), extra.end());
            return q;
        };
        auto mcx = [&](const QubitList& controls, Qubit target)
        {
            if (controls.size() == 1) out.push_back(op2(OpKind::CNOT, controls[0], target));
            else if (controls.size() == 2)
            {
                Operation o;
                o.kind = OpKind::Toffoli;
                o.controls = controls;
                o.targets = {target};
                out.push_back(std::move(o));
            }
            else
            {
                Operation o;
                o.kind = OpKind::MCX;
                o.controls = controls;
                o.targets = {target};
                out.push_back(std::move(o));
            }
        };
        auto mcphase = [&](const QubitList& qs, double lambda)
        {
            Operation o;
            if (qs.size() == 1) o = op1(OpKind::Phase, qs[0], {lambda});
            else if (qs.size() == 2) o = op2(OpKind::CPhase, qs[0], qs[1], {lambda});
            else
            {
                o.kind = OpKind::MCPhase;
                o.targets = qs;
                o.params = {lambda};
            }
            out.push_back(std::move(o));
        };
        auto mcz = [&](const QubitList& qs)
        {
            if (qs.size() == 2) out.push_back(op2(OpKind::CZ, qs[0], qs[1]));
            else
            {
                Operation o;
                o.kind = OpKind::MCZ;
                o.targets = qs;
                out.push_back(std::move(o));
            }
        };

        switch (op.kind)
        {
            case GK::I: return;
            case GK::X: mcx(c, t[0]); return;
            case GK::Z: mcz(all({t[0]})); return;
            case GK::Y:
                out.push_back(op1(OpKind::Sdg, t[0]));
                mcx(c, t[0]);
                out.push_back(op1(OpKind::S, t[0]));
                return;
            case GK::GPhase:
                if (stab) cliffordRz(out, c[0], quarterTurns(op.angles[0]));
                else mcphase(c, a[0]);
                return;
            case GK::P: mcphase(all({t[0]}), a[0]); return;
            case GK::CP: mcphase(all({t[0], t[1]}), a[0]); return;
            case GK::CZ: mcz(all({t[0], t[1]})); return;
            case GK::CNOT: mcx(all({t[0]}), t[1]); return;
            case GK::Toffoli: mcx(all({t[0], t[1]}), t[2]); return;
            case GK::SWAP:
                if (c.size() == 1)
                {
                    Operation o;
                    o.kind = OpKind::Fredkin;
                    o.controls = c;
                    o.targets = {t[0], t[1]};
                    out.push_back(std::move(o));
                }
                else out.push_back(dense(c, {t[0], t[1]}, swapMatrix()));
                return;
            case GK::Fredkin: out.push_back(dense(all({t[0]}), {t[1], t[2]}, swapMatrix())); return;
            case GK::PauliRot:
            {
                // The basis change and parity ladder commute with the controls; only the Rz is controlled.
                for (std::size_t k = 0; k < t.size(); ++k) toZ(out, t[k], op.paulis[k], false);
                ladder(out, t, false);
                out.push_back(dense(c, {t.back()}, gateMatrix(GK::RZ, {a[0]})));
                ladder(out, t, true);
                for (std::size_t k = 0; k < t.size(); ++k) toZ(out, t[k], op.paulis[k], true);
                return;
            }
            case GK::Matrix: out.push_back(dense(c, t, *op.matrix)); return;
            case GK::MeasureZ: case GK::MeasurePauli: case GK::Reset: case GK::Channel:
                throw std::logic_error("controlled non-unitary op");
            default: out.push_back(dense(c, t, gateMatrix(op.kind, a))); return;
        }
    }
} // namespace


std::vector<Operation> lower(const IrOp& op, const std::vector<double>& params, bool stabilizer)
{
    std::vector<double> a;
    a.reserve(op.angles.size());
    for (const Affine& x : op.angles) a.push_back(x.eval(params));

    std::vector<Operation> out;
    for (const Qubit q : op.negControls) out.push_back(op1(OpKind::X, q));
    if (op.controls.empty() && op.negControls.empty()) lowerBase(out, op, a, stabilizer);
    else
    {
        QubitList c = op.controls;
        c.insert(c.end(), op.negControls.begin(), op.negControls.end());
        lowerControlled(out, op, c, a, stabilizer);
    }
    for (const Qubit q : op.negControls) out.push_back(op1(OpKind::X, q));

    if (op.cond)
        for (Operation& o : out) o.condition = Qputer::Condition{op.cond->mask, op.cond->value};
    return out;
}

void applyUnitary(Qputer::QuantumStateVector& s, const Operation& op)
{
    using G = Qputer::QuantumGate;
    const QubitList& t = op.targets;
    const QubitList& c = op.controls;
    switch (op.kind)
    {
        case OpKind::X: G::x(s, t[0]); return;
        case OpKind::Y: G::y(s, t[0]); return;
        case OpKind::Z: G::z(s, t[0]); return;
        case OpKind::H: G::h(s, t[0]); return;
        case OpKind::S: G::s(s, t[0]); return;
        case OpKind::Sdg: G::sdg(s, t[0]); return;
        case OpKind::T: G::t(s, t[0]); return;
        case OpKind::Tdg: G::tdg(s, t[0]); return;
        case OpKind::SX: G::sx(s, t[0]); return;
        case OpKind::RX: G::rx(s, t[0], op.params[0]); return;
        case OpKind::RY: G::ry(s, t[0], op.params[0]); return;
        case OpKind::RZ: G::rz(s, t[0], op.params[0]); return;
        case OpKind::Phase: G::phase(s, t[0], op.params[0]); return;
        case OpKind::U3: G::u3(s, t[0], op.params[0], op.params[1], op.params[2]); return;
        case OpKind::CNOT: G::cnot(s, c[0], t[0]); return;
        case OpKind::CZ: G::cz(s, c[0], t[0]); return;
        case OpKind::CPhase: G::cphase(s, c[0], t[0], op.params[0]); return;
        case OpKind::Swap: G::swap(s, t[0], t[1]); return;
        case OpKind::Toffoli: G::toffoli(s, c[0], c[1], t[0]); return;
        case OpKind::Fredkin: G::fredkin(s, c[0], t[0], t[1]); return;
        case OpKind::MCX: G::mcx(s, c, t[0]); return;
        case OpKind::MCZ: G::mcz(s, t); return;
        case OpKind::MCPhase: G::mcphase(s, t, op.params[0]); return;
        case OpKind::Unitary:
            if (c.empty()) G::apply(s, t, op.matrix);
            else G::controlled(s, c, t, op.matrix);
            return;
        case OpKind::Measure: case OpKind::Reset: case OpKind::PauliChannel: case OpKind::Kraus: break;
    }
    throw std::logic_error("applyUnitary: not a unitary operation");
}

void applyMatrix(Qputer::QuantumStateVector& s, const QubitList& targets, const Eigen::MatrixXcd& m)
{
    const std::size_t k = targets.size();
    const std::size_t dim = std::size_t{1} << k;
    std::size_t mask = 0;
    for (const Qubit t : targets) mask |= std::size_t{1} << t;
    std::vector<std::size_t> offs(dim, 0);
    for (std::size_t j = 0; j < dim; ++j)
        for (std::size_t b = 0; b < k; ++b)
            if ((j >> (k - 1 - b)) & 1U) offs[j] |= std::size_t{1} << targets[b];
    Eigen::VectorXcd in(static_cast<Eigen::Index>(dim));
    for (std::size_t base = 0; base < s.size(); ++base)
    {
        if (base & mask) continue;
        for (std::size_t j = 0; j < dim; ++j) in[static_cast<Eigen::Index>(j)] = s[base | offs[j]];
        const Eigen::VectorXcd out = m * in;
        for (std::size_t j = 0; j < dim; ++j) s[base | offs[j]] = out[static_cast<Eigen::Index>(j)];
    }
}

Eigen::MatrixXcd unitaryOf(const std::vector<IrOp>& ops, const QubitList& support, const std::vector<double>& params)
{
    const std::size_t k = support.size();
    const Eigen::Index dim = Eigen::Index{1} << k;
    cd phase{1.0, 0.0};
    // Local qubit of each global qubit: support[0] is the MSB of the matrix index, i.e. local k-1.
    auto local = [&](Qubit q)
    {
        for (std::size_t j = 0; j < k; ++j)
            if (support[j] == q) return static_cast<Qubit>(k - 1 - j);
        throw std::logic_error("unitaryOf: op outside its support");
    };
    // A non-unitary matrix (an observable term) is multiplied in directly; everything else goes
    // through the gate kernels.
    struct Step
    {
        Operation op;
        bool raw = false;
    };
    std::vector<Step> steps;
    for (const IrOp& op : ops)
    {
        if (op.kind == GK::GPhase && op.controls.empty() && op.negControls.empty())
        {
            phase *= std::polar(1.0, op.angles[0].eval(params));
            continue;
        }
        if (op.kind == GK::Matrix && !op.unitaryMatrix && op.controls.empty() && op.negControls.empty())
        {
            Step st;
            st.raw = true;
            st.op.kind = OpKind::Unitary;
            for (const Qubit q : op.targets) st.op.targets.push_back(local(q));
            st.op.matrix = *op.matrix;
            steps.push_back(std::move(st));
            continue;
        }
        for (Operation o : lower(op, params, false))
        {
            for (Qubit& q : o.controls) q = local(q);
            for (Qubit& q : o.targets) q = local(q);
            steps.push_back({std::move(o), false});
        }
    }
    if (k == 0) return Eigen::MatrixXcd::Constant(1, 1, phase);

    Eigen::MatrixXcd u(dim, dim);
    for (Eigen::Index col = 0; col < dim; ++col)
    {
        Qputer::QuantumStateVector s(k);
        s[0] = 0.0;
        s[static_cast<std::size_t>(col)] = 1.0;
        for (const Step& st : steps)
        {
            if (st.raw) applyMatrix(s, st.op.targets, st.op.matrix);
            else applyUnitary(s, st.op);
        }
        u.col(col) = s.vector() * phase;
    }
    return u;
}

std::vector<Operation> productPreparation(const KetV& ket, std::uint64_t& basisLow)
{
    basisLow = 0;
    std::vector<Operation> out;
    Qubit q = 0;
    for (const KetFactor& f : ket.factors)
    {
        switch (f.kind)
        {
            case KetFactor::Kind::Basis:
                for (const char b : f.bits)
                {
                    if (b == '1')
                    {
                        if (q < 64) basisLow |= std::uint64_t{1} << q;
                        else out.push_back(op1(OpKind::X, q));
                    }
                    ++q;
                }
                break;
            case KetFactor::Kind::Plus: out.push_back(op1(OpKind::H, q++)); break;
            case KetFactor::Kind::Minus:
                out.push_back(op1(OpKind::X, q));
                out.push_back(op1(OpKind::H, q++));
                break;
            case KetFactor::Kind::PlusI:
                out.push_back(op1(OpKind::H, q));
                out.push_back(op1(OpKind::S, q++));
                break;
            case KetFactor::Kind::MinusI:
                out.push_back(op1(OpKind::X, q));
                out.push_back(op1(OpKind::H, q));
                out.push_back(op1(OpKind::S, q++));
                break;
            case KetFactor::Kind::Dense: throw std::logic_error("productPreparation: dense factor");
        }
    }
    return out;
}

} // namespace Noether
