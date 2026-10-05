#include "Tools.hpp"

#include "Lowering.hpp"

#include <QuantumStateMachine.hpp>

#include <format>
#include <random>



namespace Noether
{

using Qputer::OpKind;
using Qputer::Operation;

namespace
{
    // The unitary ops of a program in time order; error names the first event that is not one.
    std::vector<IrOp> unitaryOps(const Ir& ir, std::string& error)
    {
        std::vector<IrOp> ops;
        for (const Event& ev : ir.events)
        {
            if (ev.kind == EvK::Prepare)
            {
                if (!ev.ket->isBasis() || ev.ket->basisBits().find('1') != std::string::npos)
                {
                    error = "equiv compares operations from |0…0⟩; remove the prepare";
                    return {};
                }
                continue;
            }
            if (ev.kind != EvK::Op) continue;
            const IrOp& op = ev.op;
            if (op.kind == GK::MeasureZ || op.kind == GK::MeasurePauli || op.kind == GK::Reset || op.kind == GK::Channel || op.cond)
            {
                error = std::format("equiv compares unitary programs, but `{}` is a measurement, reset, channel or conditioned op",
                                    gkName(op.kind));
                return {};
            }
            if (op.kind == GK::Matrix && !op.unitaryMatrix)
            {
                error = "equiv found a non-unitary matrix";
                return {};
            }
            ops.push_back(op);
        }
        return ops;
    }

    // Inverse of a Clifford core op sequence (self-inverse gates, S ↔ S†, √X† = X·√X).
    std::vector<Operation> inverse(const std::vector<Operation>& ops)
    {
        std::vector<Operation> out;
        for (auto it = ops.rbegin(); it != ops.rend(); ++it)
        {
            Operation o = *it;
            if (o.kind == OpKind::S) o.kind = OpKind::Sdg;
            else if (o.kind == OpKind::Sdg) o.kind = OpKind::S;
            else if (o.kind == OpKind::SX)
            {
                Operation x = o;
                x.kind = OpKind::X;
                out.push_back(x);
            }
            out.push_back(std::move(o));
        }
        return out;
    }

    cd innerProduct(const Qputer::QuantumStateVector& a, const Qputer::QuantumStateVector& b)
    {
        cd acc = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) acc += std::conj(a[i]) * b[i];
        return acc;
    }
} // namespace

EquivResult equivalent(const Ir& a, const Ir& b, bool ignorePhase)
{
    EquivResult r;
    if (a.nQubits != b.nQubits)
    {
        r.error = std::format("the programs declare {} and {} qubits", a.nQubits, b.nQubits);
        return r;
    }
    const std::size_t n = a.nQubits;
    const std::vector<IrOp> oa = unitaryOps(a, r.error);
    if (!r.error.empty()) return r;
    const std::vector<IrOp> ob = unitaryOps(b, r.error);
    if (!r.error.empty()) return r;
    if (n == 0)
    {
        r.equivalent = true;
        r.method = "unitary";
        return r;
    }

    if (n <= 12)
    {
        r.method = "unitary";
        QubitList support;
        for (Qubit q = 0; q < n; ++q) support.push_back(q);
        const Eigen::MatrixXcd ua = unitaryOf(oa, support, a.paramValues);
        const Eigen::MatrixXcd ub = unitaryOf(ob, support, b.paramValues);
        cd phase = 1.0;
        if (ignorePhase)
        {
            const cd tr = (ua.adjoint() * ub).trace();
            phase = std::abs(tr) > 1e-12 ? tr / std::abs(tr) : cd(1.0);
        }
        r.maxDeviation = (ub - phase * ua).cwiseAbs().maxCoeff();
        r.equivalent = r.maxDeviation <= 1e-10;
        if (!r.equivalent)
        {
            // The basis input whose output column differs most.
            Eigen::Index worst = 0;
            double dev = -1.0;
            for (Eigen::Index c = 0; c < ua.cols(); ++c)
            {
                const double d = (ub.col(c) - phase * ua.col(c)).norm();
                if (d > dev) dev = d, worst = c;
            }
            std::string bits(n, '0');
            for (std::size_t q = 0; q < n; ++q)
                if ((static_cast<std::size_t>(worst) >> (n - 1 - q)) & 1U) bits[q] = '1';
            r.counterexample = "|" + bits + "⟩";
        }
        return r;
    }

    bool allClifford = true;
    for (const IrOp& op : oa) allClifford = allClifford && op.clifford;
    for (const IrOp& op : ob) allClifford = allClifford && op.clifford;
    if (allClifford && n <= 4096)
    {
        // Choi state: Bell pairs (q, n+q), then A and B† on the first half; A = B up to phase iff
        // every X_q X_{n+q} and Z_q Z_{n+q} still stabilises the result.
        r.method = "tableau";
        Qputer::QuantumStateMachine m(2 * n, 0, 1, Qputer::Backend::Stabilizer);
        for (Qubit q = 0; q < n; ++q)
        {
            m.h(q);
            m.cnot(q, n + q);
        }
        std::vector<Operation> lb;
        for (const IrOp& op : oa)
            for (const Operation& o : lower(op, a.paramValues, true)) m.append(o);
        for (const IrOp& op : ob)
            for (const Operation& o : lower(op, b.paramValues, true)) lb.push_back(o);
        for (const Operation& o : inverse(lb)) m.append(o);
        r.equivalent = true;
        for (Qubit q = 0; q < n && r.equivalent; ++q)
        {
            if (m.expectation("XX", {q, n + q}) < 0.5)
            {
                r.equivalent = false;
                r.counterexample = std::format("X on qubit {} maps differently", q);
            }
            else if (m.expectation("ZZ", {q, n + q}) < 0.5)
            {
                r.equivalent = false;
                r.counterexample = std::format("Z on qubit {} maps differently", q);
            }
        }
        r.maxDeviation = r.equivalent ? 0.0 : 1.0;
        return r;
    }

    if (n > Qputer::kMaxQubits)
    {
        r.error = std::format("no equivalence method for {} qubits with non-Clifford operations", n);
        return r;
    }
    r.method = "random-states";
    std::mt19937_64 rng(12345);
    const char* labels[] = {"0", "1", "+", "-", "+i", "-i"};
    std::optional<cd> phase;
    for (int trial = 0; trial < 8; ++trial)
    {
        KetV in;
        std::string text;
        for (std::size_t q = 0; q < n; ++q)
        {
            const int k = static_cast<int>(rng() % 6);
            KetFactor f;
            f.nq = 1;
            switch (k)
            {
                case 0: f.kind = KetFactor::Kind::Basis; f.bits = "0"; break;
                case 1: f.kind = KetFactor::Kind::Basis; f.bits = "1"; break;
                case 2: f.kind = KetFactor::Kind::Plus; break;
                case 3: f.kind = KetFactor::Kind::Minus; break;
                case 4: f.kind = KetFactor::Kind::PlusI; break;
                default: f.kind = KetFactor::Kind::MinusI; break;
            }
            in.factors.push_back(f);
            text += std::string(q ? "⊗" : "") + "|" + labels[k] + "⟩";
        }
        std::uint64_t basis = 0;
        const std::vector<Operation> prep = productPreparation(in, basis);
        auto runOn = [&](const std::vector<IrOp>& ops, const Ir& ir)
        {
            Qputer::QuantumStateVector s(n);
            s[0] = 0.0;
            s[basis] = 1.0;
            for (const Operation& o : prep) applyUnitary(s, o);
            for (const IrOp& op : ops)
                for (const Operation& o : lower(op, ir.paramValues, false)) applyUnitary(s, o);
            return s;
        };
        const Qputer::QuantumStateVector sa = runOn(oa, a), sb = runOn(ob, b);
        const cd ip = innerProduct(sa, sb);
        double dev = 0.0;
        if (ignorePhase)
        {
            if (!phase && std::abs(ip) > 1e-12) phase = ip / std::abs(ip);
            const cd ph = phase.value_or(1.0);
            for (std::size_t i = 0; i < sa.size(); ++i) dev = std::max(dev, std::abs(sb[i] - ph * sa[i]));
        }
        else
            for (std::size_t i = 0; i < sa.size(); ++i) dev = std::max(dev, std::abs(sb[i] - sa[i]));
        r.maxDeviation = std::max(r.maxDeviation, dev);
        if (dev > 1e-9 && r.counterexample.empty()) r.counterexample = text;
    }
    r.equivalent = r.maxDeviation <= 1e-9;
    return r;
}

} // namespace Noether
