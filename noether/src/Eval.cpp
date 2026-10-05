#include "CompilerImpl.hpp"

#include "Lowering.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <numbers>
#include <set>
#include <Eigen/Dense>



namespace Noether
{

namespace
{
    Num numInt(std::int64_t v) { return Num(Real::integer(v)); }

    DeferredV deferred(Type t) { return DeferredV{t, Stage::Run}; }

    bool isNumericType(Type t) { return t == Type::Int || t == Type::Real || t == Type::Complex; }

    // Result type of arithmetic on (possibly deferred) numbers.
    Type joinNumeric(Type a, Type b)
    {
        if (a == Type::Complex || b == Type::Complex) return Type::Complex;
        if (a == Type::Int && b == Type::Int) return Type::Int;
        return Type::Real;
    }

    const std::set<std::string_view> kFunctions{"sin", "cos", "tan", "exp", "log", "sqrt", "abs", "floor", "ceil",
                                                "min", "max", "trotter", "entropy", "fidelity", "marginal", "count",
                                                "depth"};
    const std::set<std::string_view> kChannels{"depolarize", "depolarize2", "dephase", "flip", "pauli", "ampdamp", "kraus"};

    struct BuiltinInfo
    {
        Builtin b;
        std::string_view roles; // "1", "c>t", "a,b", "c,c>t", "c>a,b", "C"
        std::size_t params;
    };

    const std::map<std::string_view, BuiltinInfo>& builtins()
    {
        static const std::map<std::string_view, BuiltinInfo> m{
            {"I", {Builtin::I, "1", 0}},       {"X", {Builtin::X, "1", 0}},         {"Y", {Builtin::Y, "1", 0}},
            {"Z", {Builtin::Z, "1", 0}},       {"H", {Builtin::H, "1", 0}},         {"S", {Builtin::S, "1", 0}},
            {"T", {Builtin::T, "1", 0}},       {"Rx", {Builtin::Rx, "1", 1}},       {"Ry", {Builtin::Ry, "1", 1}},
            {"Rz", {Builtin::Rz, "1", 1}},     {"P", {Builtin::P, "1", 1}},         {"U3", {Builtin::U3, "1", 3}},
            {"CNOT", {Builtin::CNOT, "c>t", 0}}, {"CZ", {Builtin::CZ, "a,b", 0}},   {"CP", {Builtin::CP, "a,b", 1}},
            {"SWAP", {Builtin::SWAP, "a,b", 0}}, {"Toffoli", {Builtin::Toffoli, "c,c>t", 0}},
            {"Fredkin", {Builtin::Fredkin, "c>a,b", 0}}, {"C", {Builtin::C, "C", 0}}};
        return m;
    }

    const BuiltinInfo& infoOf(Builtin b)
    {
        for (const auto& [name, info] : builtins())
            if (info.b == b) return info;
        static const BuiltinInfo sx{Builtin::SX, "1", 0};
        return sx;
    }

    std::string builtinName(Builtin b)
    {
        if (b == Builtin::SX) return "√X";
        for (const auto& [name, info] : builtins())
            if (info.b == b) return std::string(name);
        return "?";
    }

    GK kindOf(Builtin b, bool dagger)
    {
        switch (b)
        {
            case Builtin::I: return GK::I;
            case Builtin::X: return GK::X;
            case Builtin::Y: return GK::Y;
            case Builtin::Z: return GK::Z;
            case Builtin::H: return GK::H;
            case Builtin::S: return dagger ? GK::Sdg : GK::S;
            case Builtin::T: return dagger ? GK::Tdg : GK::T;
            case Builtin::SX: return dagger ? GK::SXdg : GK::SX;
            case Builtin::Rx: return GK::RX;
            case Builtin::Ry: return GK::RY;
            case Builtin::Rz: return GK::RZ;
            case Builtin::P: return GK::P;
            case Builtin::U3: return GK::U3;
            case Builtin::CNOT: return GK::CNOT;
            case Builtin::CZ: return GK::CZ;
            case Builtin::CP: return GK::CP;
            case Builtin::SWAP: return GK::SWAP;
            case Builtin::Toffoli: return GK::Toffoli;
            case Builtin::Fredkin: return GK::Fredkin;
            case Builtin::C: break;
        }
        return GK::I;
    }

    bool isRotation(GK k)
    {
        return k == GK::RX || k == GK::RY || k == GK::RZ || k == GK::P || k == GK::CP || k == GK::PauliRot ||
               k == GK::GPhase;
    }

    // e^{i·angle} when the value has unit modulus (exactly for 1, -1, i, -i or a tracked phase).
    std::optional<Affine> unitPhase(const Num& n)
    {
        if (n.phase) return n.phase;
        if (!n.isConst()) return std::nullopt;
        const auto re = n.re.c.asInt(), im = n.im.c.asInt();
        if (re && im)
        {
            if (*re == 1 && *im == 0) return Affine(Real::integer(0));
            if (*re == -1 && *im == 0) return Affine(Real::pi(Rational{1, 1}));
            if (*re == 0 && *im == 1) return Affine(Real::pi(Rational{1, 2}));
            if (*re == 0 && *im == -1) return Affine(Real::pi(Rational{-1, 2}));
        }
        const cd z = n.constValue();
        if (std::abs(std::abs(z) - 1.0) <= 1e-12) return Affine(Real::approx(std::arg(z)));
        return std::nullopt;
    }

    QubitList supportOf(const OpV& op)
    {
        std::set<Qubit> s;
        for (const IrOp& o : op.ops)
            for (const Qubit q : o.qubits()) s.insert(q);
        return {s.begin(), s.end()};
    }

    QubitList supportOf(const LinOpV& l)
    {
        std::set<Qubit> s;
        for (const PTerm& t : l.paulis)
            for (const auto& [q, c] : t.ps) s.insert(q);
        for (const auto& [c, op] : l.general)
            for (const Qubit q : supportOf(op)) s.insert(q);
        return {s.begin(), s.end()};
    }

    OpV pauliOp(const PTerm& t, Span at)
    {
        OpV op;
        for (const auto& [q, letter] : t.ps)
        {
            IrOp o;
            o.kind = letter == 'X' ? GK::X : letter == 'Y' ? GK::Y : GK::Z;
            o.targets = {q};
            o.span = at;
            op.ops.push_back(std::move(o));
        }
        return op;
    }

    bool isUnitaryMatrix(const Eigen::MatrixXcd& m)
    {
        if (m.rows() != m.cols()) return false;
        const double tol = 1e-12 * static_cast<double>(m.rows());
        return ((m.adjoint() * m) - Eigen::MatrixXcd::Identity(m.rows(), m.cols())).cwiseAbs().maxCoeff() <= std::max(tol, 1e-12);
    }

    bool isPowerOfTwo(Eigen::Index n) { return n > 0 && (n & (n - 1)) == 0; }

    std::size_t log2Size(Eigen::Index n)
    {
        std::size_t k = 0;
        while ((Eigen::Index{1} << k) < n) ++k;
        return k;
    }

    Eigen::MatrixXcd kron(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b)
    {
        Eigen::MatrixXcd out(a.rows() * b.rows(), a.cols() * b.cols());
        for (Eigen::Index i = 0; i < a.rows(); ++i)
            for (Eigen::Index j = 0; j < a.cols(); ++j) out.block(i * b.rows(), j * b.cols(), b.rows(), b.cols()) = a(i, j) * b;
        return out;
    }
} // namespace


// ---- Entry ----

Value Compiler::Impl::eval(const Expr& e, const ConstScopePtr& sc)
{
    switch (e.kind)
    {
        case EK::Int:
        case EK::Real: return literal(e);
        case EK::String: return e.name;
        case EK::Bool: return e.name == "true";
        case EK::Name: return lookup(e, sc);
        case EK::Ket: return ketLabel(e.name, e.span, sc);
        case EK::Bra:
        {
            const Value k = ketLabel(e.name, e.span, sc);
            return BraV{k.as<KetV>()};
        }
        case EK::Braket:
        {
            const Value bra = ketLabel(e.name, e.span, sc);
            const Value ket = ketLabel(e.label2, e.span, sc);
            return mul(BraV{bra.as<KetV>()}, ket, e.span);
        }
        case EK::List:
        {
            ListV l;
            for (const ExprPtr& k : e.kids) l.items.push_back(std::make_shared<Value>(eval(*k, sc)));
            return l;
        }
        case EK::Paren: return eval(*e.kids[0], sc);
        case EK::Unary:
        {
            const Value x = eval(*e.kids[0], sc);
            switch (e.op)
            {
                case Tok::Minus: return neg(x, e.span);
                case Tok::Plus:
                    if (!x.is<Num>() && !isNumericType(x.type()) && !isOperator(x))
                        fail("E4001", e.span, std::format("unary + needs a number, found {}", typeName(x.type())));
                    return x;
                case Tok::Not: return notValue(x, e.span);
                case Tok::Sqrt: return sqrtValue(x, e.span);
                default: break;
            }
            fail("E2001", e.span, "unknown prefix operator");
        }
        case EK::Binary:
        {
            if (e.op == Tok::KwAnd || e.op == Tok::KwOr) return logic(e, sc);
            const Value a = eval(*e.kids[0], sc);
            const Value b = eval(*e.kids[1], sc);
            switch (e.op)
            {
                case Tok::Plus: return add(a, b, e.span);
                case Tok::Minus: return add(a, neg(b, e.span), e.span);
                case Tok::Cdot: return mul(a, b, e.span);
                case Tok::Slash: return divide(a, b, e.span);
                case Tok::Otimes: return tensor(a, b, e.span);
                default: break;
            }
            fail("E2001", e.span, "unknown binary operator");
        }
        case EK::Product: return product(e, sc);
        case EK::Compare: return compare(e, sc);
        case EK::Sub: return applySub(e, sc);
        case EK::Call: return call(e, sc);
        case EK::Index: return index(e, sc);
        case EK::Slice: fail("E2001", e.span, "a range a..b is only valid in an index or a subscript");
        case EK::Dagger: return dagger(eval(*e.kids[0], sc), e.span);
        case EK::Power: return power(e, sc);
        case EK::TensorPow:
        {
            const Value base = eval(*e.kids[0], sc);
            const std::int64_t n = needInt(eval(*e.kids[1], sc), e.kids[1]->span, "a tensor power");
            if (n < 1) fail("E4003", e.kids[1]->span, "a tensor power must be at least 1");
            if (const auto* k = base.get<KetV>())
            {
                if (k->live) fail("E4001", e.span, "the live state |ψ⟩ has no tensor power");
                KetV out;
                out.scale = std::pow(k->scale, static_cast<double>(n));
                for (std::int64_t r = 0; r < n; ++r)
                    for (const KetFactor& f : k->factors)
                    {
                        if (!out.factors.empty() && f.kind == KetFactor::Kind::Basis &&
                            out.factors.back().kind == KetFactor::Kind::Basis)
                        {
                            out.factors.back().bits += f.bits;
                            out.factors.back().nq += f.nq;
                        }
                        else out.factors.push_back(f);
                    }
                return out;
            }
            if (const auto* g = base.get<GateV>(); g && oneQubitGate(*g))
            {
                GateV out = *g;
                out.tensorPow *= n;
                return out;
            }
            const MatV m = toMatrix(base, e.kids[0]->span);
            if (static_cast<double>(n) * static_cast<double>(m.nq()) > 12)
                fail("E5009", e.span, "a dense tensor power above 12 qubits is too large");
            MatV out = m;
            for (std::int64_t r = 1; r < n; ++r) out.m = kron(out.m, m.m);
            return out;
        }
        case EK::BigOp: return bigop(e, sc);
        case EK::Expval: return expval(e, sc);
        case EK::Abs: return absValue(e, sc);
        case EK::Set: fail("E2001", e.span, "a set {…} is only valid in spec statements");
    }
    fail("E2001", e.span, "unsupported expression");
}

Value Compiler::Impl::literal(const Expr& e)
{
    if (e.kind == EK::Int)
    {
        std::int64_t v = 0;
        const auto r = std::from_chars(e.name.data(), e.name.data() + e.name.size(), v);
        if (r.ec != std::errc{}) fail("E4007", e.span, std::format("integer literal {} overflows int64", e.name));
        return numInt(v);
    }
    if (const auto q = parseDecimal(e.name)) return Num(Real::rational(*q));
    return Num(Real::approx(std::stod(e.name)));
}


// ---- Names ----

Value Compiler::Impl::lookup(const Expr& e, const ConstScopePtr& sc)
{
    const std::string& n = e.name;
    const auto [sym, holder] = sc->find(n);
    if (sym)
    {
        sym->used = true;
        switch (sym->kind)
        {
            case Symbol::Kind::Runtime: return rt ? rt->slot(sym->slot) : sym->value;
            case Symbol::Kind::Def:
            case Symbol::Kind::Proc:
            {
                GateV g;
                g.kind = sym->kind == Symbol::Kind::Def ? GateV::Kind::Def : GateV::Kind::Proc;
                g.def = sym->stmt;
                g.name = n;
                g.span = e.span;
                return g;
            }
            case Symbol::Kind::LetGate:
            {
                GateV g;
                g.kind = GateV::Kind::LetGate;
                g.letGate = sym->binding;
                g.name = n;
                g.scope = holder->shared_from_this();
                g.span = e.span;
                return g;
            }
            default: return rt ? runtimeValue(*sym, *holder) : sym->value;
        }
    }
    if (n == "i") return Num::complex(Real::integer(0), Real::integer(1));
    if (n == "π") return Num(Real::pi(Rational{1, 1}));
    if (n == "e") return Num(Real::approx(std::numbers::e));
    if (n == "ψ") return liveKet(e.span);
    if (n == "ρ")
    {
        GateV g;
        g.kind = GateV::Kind::Rho;
        g.name = "ρ";
        g.span = e.span;
        return g;
    }
    if (const auto it = builtins().find(n); it != builtins().end())
    {
        GateV g;
        g.kind = GateV::Kind::Builtin;
        g.builtin = it->second.b;
        g.name = n;
        g.span = e.span;
        return g;
    }
    if (kFunctions.contains(n))
        fail("E4001", e.span, std::format("`{}` is a function; call it with arguments, e.g. {}(x)", n, n));
    if (kChannels.contains(n))
        fail("E4001", e.span, std::format("`{}` is a channel; give its parameter, e.g. {}(0.01)_0", n, n));
    unknownName(n, e.span, sc);
}

void Compiler::Impl::unknownName(const std::string& name, Span at, const ConstScopePtr& sc)
{
    // Defined later at top level: no hoisting.
    if (const auto it = laterNames.find(name); it != laterNames.end() && it->second.begin > at.begin &&
                                                it->second.file == at.file)
    {
        d.error("E3004", at, std::format("`{}` is used before its definition", name)).note(it->second, "defined here");
        throw CompileAbort{};
    }
    // A global hidden by def/proc hygiene.
    for (const Scope* s = sc.get(); s; s = s->parentScope())
        if (s->isGlobal())
            if (const Symbol* g = s->localLookup(name))
            {
                if (g->kind == Symbol::Kind::QReg)
                    fail("E5009", at, std::format("qubit register `{}` is not visible inside a def or proc; pass it as a "
                                                  "subscript parameter", name));
                if (g->kind == Symbol::Kind::BReg)
                    fail("E5009", at, std::format("bit register `{}` is visible in procs but not in defs", name));
                if (g->kind == Symbol::Kind::Runtime)
                    fail("E4004", at, std::format("`{}` is a run-time value and is not visible inside a def or proc", name));
                d.error("E3004", at, std::format("`{}` is defined after this def, so it is not visible here", name))
                    .note(g->span, "defined here");
                throw CompileAbort{};
            }

    std::vector<std::string> candidates;
    for (const Scope* s = sc.get(); s; s = s->parentScope())
        for (const auto& [k, v] : s->symbols()) candidates.push_back(k);
    for (const auto& [k, v] : builtins()) candidates.emplace_back(k);
    for (const auto& k : kFunctions) candidates.emplace_back(k);
    for (const auto& k : kChannels) candidates.emplace_back(k);

    auto resolvable = [&](const std::string& part)
    {
        return std::ranges::find(candidates, part) != candidates.end() || part == "i" || part == "π" || part == "e";
    };
    Diagnostic& diag = d.error("E3001", at, std::format("unknown name `{}`", name));
    for (std::size_t k = 1; k < name.size(); ++k)
    {
        const std::string a = name.substr(0, k), b = name.substr(k);
        if (resolvable(a) && resolvable(b))
        {
            diag.fix(std::format("did you mean the product `{} {}`?", a, b), at, a + " " + b);
            break;
        }
    }
    std::string best;
    std::size_t bestD = name.size() <= 3 ? 2 : 3;
    for (const std::string& cand : candidates)
        if (const std::size_t dist = editDistance(name, cand); dist < bestD && cand != name)
        {
            bestD = dist;
            best = cand;
        }
    if (!best.empty()) diag.fix(std::format("did you mean `{}`?", best), at, best);
    throw CompileAbort{};
}

// At run time params have values: affine numbers are evaluated, and a let whose value depends on a
// param non-affinely (sin θ) is re-evaluated from its expression.
Value Compiler::Impl::runtimeValue(const Symbol& sym, const Scope& holder)
{
    bool nonlinear = false;
    auto bind = [&](auto&& self, const Value& v) -> Value
    {
        if (const auto* n = v.get<Num>())
        {
            if (n->isConst()) return v;
            if (n->nonlinear)
            {
                nonlinear = true;
                return v;
            }
            std::vector<double> p(ir.paramNames.size());
            for (std::size_t k = 0; k < p.size(); ++k) p[k] = rt->param(static_cast<std::uint32_t>(k));
            return Num::fromDouble(n->eval(p));
        }
        if (const auto* l = v.get<ListV>())
        {
            ListV out;
            for (const ValuePtr& item : l->items) out.items.push_back(std::make_shared<Value>(self(self, *item)));
            return out;
        }
        return v;
    };
    Value out = bind(bind, sym.value);
    if (!nonlinear) return out;
    if (sym.kind == Symbol::Kind::Let && sym.binding) return eval(*sym.binding->value, holder.shared_from_this());
    fail("E4004", sym.span, "this value depends on a param non-affinely and cannot be re-evaluated here");
}

Value Compiler::Impl::liveKet(Span at)
{
    if (rt) return rt->liveKet(at);
    KetV k;
    k.live = true;
    return k;
}

Value Compiler::Impl::ketLabel(const std::string& label, Span at, const ConstScopePtr& sc)
{
    KetV k;
    if (!label.empty() && std::ranges::all_of(label, [](char c) { return c == '0' || c == '1'; }))
    {
        KetFactor f;
        f.kind = KetFactor::Kind::Basis;
        f.bits = label;
        f.nq = label.size();
        k.factors.push_back(std::move(f));
        return k;
    }
    auto single = [&](KetFactor::Kind kind)
    {
        KetFactor f;
        f.kind = kind;
        f.nq = 1;
        k.factors.push_back(f);
        return k;
    };
    if (label == "+") return single(KetFactor::Kind::Plus);
    if (label == "-") return single(KetFactor::Kind::Minus);
    if (label == "+i") return single(KetFactor::Kind::PlusI);
    if (label == "-i") return single(KetFactor::Kind::MinusI);
    if (label == "ψ")
    {
        if (rt)
        {
            KetV live = rt->liveKet(at);
            live.live = true; // keep the marker so overlaps know which side is the live state
            return live;
        }
        k.live = true;
        return k;
    }
    auto name = std::make_shared<Expr>();
    name->kind = EK::Name;
    name->name = label;
    name->span = at;
    const Value v = lookup(*name, sc);
    if (const auto* kv = v.get<KetV>()) return *kv;
    if (const auto* bv = v.get<BraV>()) return bv->ket;
    fail("E4001", at, std::format("`{}` in a ket must name a Ket, found {}", label, typeName(v.type())));
}


// ---- Conversions ----

std::int64_t Compiler::Impl::needInt(const Value& v, Span at, std::string_view what)
{
    if (const auto* n = v.get<Num>())
    {
        if (!n->isConst()) fail("E4004", at, std::format("{} must be known at compile time, but it depends on a param", what));
        if (!n->im.isZero()) fail("E4001", at, std::format("{} must be an integer, found a complex number", what));
        if (const auto k = n->asInt()) return *k;
        fail("E4003", at, std::format("{} must be an exact integer, found {}; use floor(…) or ceil(…)", what, n->re.c.str()));
    }
    if (v.stage() == Stage::Run)
        fail("E4004", at, std::format("{} must be known at compile time, but it depends on a measurement or readout", what));
    fail("E4001", at, std::format("{} must be an integer, found {}", what, typeName(v.type())));
}

Num Compiler::Impl::needNum(const Value& v, Span at, std::string_view what)
{
    if (const auto* n = v.get<Num>()) return *n;
    if (v.stage() == Stage::Run || v.is<BitsV>() || v.is<CondV>())
        fail("E4004", at, std::format("{} cannot depend on a measurement or readout; branch on bits with `if c[k]:`", what));
    fail("E4001", at, std::format("{} must be a number, found {}", what, typeName(v.type())));
}

Affine Compiler::Impl::needAngle(const Value& v, Span at, std::string_view what)
{
    const Num n = needNum(v, at, what);
    if (n.nonlinear) fail("E4005", at, std::format("{} must depend on params affinely (a·θ + b)", what));
    if (!n.im.isZero()) fail("E4001", at, std::format("{} must be real", what));
    return n.re;
}

double Compiler::Impl::needReal(const Value& v, Span at, std::string_view what)
{
    const Num n = needNum(v, at, what);
    if (!n.isConst()) fail("E4004", at, std::format("{} must be known at compile time, but it depends on a param", what));
    if (!n.im.isZero()) fail("E4001", at, std::format("{} must be real", what));
    return n.re.c.value();
}

bool Compiler::Impl::isOperator(const Value& v) const { return v.is<OpV>() || v.is<LinOpV>(); }

std::optional<PTerm> Compiler::Impl::asPauli(const OpV& op) const
{
    PTerm acc;
    acc.coef = numInt(1);
    for (auto it = op.ops.rbegin(); it != op.ops.rend(); ++it)
    {
        const IrOp& o = *it;
        if (!o.controls.empty() || !o.negControls.empty() || o.cond) return std::nullopt;
        char letter = 0;
        switch (o.kind)
        {
            case GK::I: continue;
            case GK::X: letter = 'X'; break;
            case GK::Y: letter = 'Y'; break;
            case GK::Z: letter = 'Z'; break;
            default: return std::nullopt;
        }
        PTerm t;
        t.coef = numInt(1);
        t.ps = {{o.targets[0], letter}};
        acc = multiply(acc, t);
    }
    if (!op.gphase.isZero()) acc.coef = acc.coef * expi(op.gphase);
    return acc;
}

LinOpV Compiler::Impl::toLin(const Value& v, Span at)
{
    if (const auto* l = v.get<LinOpV>()) return *l;
    if (const auto* n = v.get<Num>())
    {
        LinOpV out;
        out.paulis.push_back({*n, {}});
        return out;
    }
    if (const auto* op = v.get<OpV>())
    {
        LinOpV out;
        if (const auto p = asPauli(*op)) out.paulis.push_back(*p);
        else out.general.emplace_back(numInt(1), *op);
        return out;
    }
    fail("E4001", at, std::format("expected an operator, found {}", typeName(v.type())));
}

OpV Compiler::Impl::toOp(const Value& v, Span at)
{
    if (const auto* op = v.get<OpV>()) return *op;
    const LinOpV l = simplify(toLin(v, at));
    if (l.paulis.size() == 1 && l.general.empty())
    {
        const auto ph = unitPhase(l.paulis[0].coef);
        if (!ph) fail("E5003", at, "this operator is not unitary: its coefficient does not have modulus 1");
        OpV op = pauliOp(l.paulis[0], at);
        op.gphase = *ph;
        return op;
    }
    if (l.paulis.empty() && l.general.size() == 1)
    {
        const auto ph = unitPhase(l.general[0].first);
        if (!ph) fail("E5003", at, "this operator is not unitary: its coefficient does not have modulus 1");
        OpV op = l.general[0].second;
        op.gphase = op.gphase + *ph;
        return op;
    }
    if (l.paulis.empty() && l.general.empty()) fail("E5003", at, "the zero operator is not unitary");
    QubitList support;
    const Eigen::MatrixXcd m = denseOf(l, support, at);
    if (support.empty()) fail("E5003", at, "a scalar is not an operation");
    if (!isUnitaryMatrix(m)) fail("E5003", at, "this operator is not unitary");
    IrOp o;
    o.kind = GK::Matrix;
    o.targets = support;
    o.matrix = std::make_shared<const Eigen::MatrixXcd>(m);
    o.span = at;
    OpV op;
    op.ops.push_back(std::move(o));
    return op;
}

MatV Compiler::Impl::toMatrix(const Value& v, Span at)
{
    if (const auto* m = v.get<MatV>()) return *m;
    if (const auto* l = v.get<ListV>())
    {
        const std::size_t rows = l->items.size();
        if (rows == 0) fail("E5009", at, "an empty matrix");
        Eigen::MatrixXcd m(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(rows));
        for (std::size_t r = 0; r < rows; ++r)
        {
            const auto* row = l->items[r]->get<ListV>();
            if (!row || row->items.size() != rows) fail("E5009", at, "a matrix literal must be square: [[a, b], [c, d]]");
            for (std::size_t c = 0; c < rows; ++c)
            {
                const Num n = needNum(*row->items[c], at, "a matrix entry");
                if (!n.isConst()) fail("E4004", at, "matrix entries cannot depend on params; use rotations for parameterised gates");
                m(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = n.constValue();
            }
        }
        if (!isPowerOfTwo(m.rows())) fail("E5009", at, std::format("a {}×{} matrix is not a qubit operator (size must be 2^k)", rows, rows));
        return MatV{m};
    }
    if (const auto* g = v.get<GateV>())
    {
        if (g->kind != GateV::Kind::Builtin || g->builtin == Builtin::C)
            fail("E4001", at, std::format("`{}` has no matrix without qubits; apply it with a subscript", g->name));
        const BuiltinInfo& info = infoOf(g->builtin);
        if (g->args.size() != info.params)
            fail("E4001", at, std::format("`{}` takes {} parameter{}", g->name, info.params, info.params == 1 ? "" : "s"));
        std::vector<double> ps;
        for (const auto& a : g->args) ps.push_back(needReal(*a, at, "a gate parameter"));
        Eigen::MatrixXcd m = builtinMatrix(g->builtin, ps, g->dagger);
        if (g->power)
        {
            const auto k = g->power->asInt();
            if (!k || *k < 0) fail("E4001", at, "a gate matrix power must be a non-negative integer");
            Eigen::MatrixXcd r = Eigen::MatrixXcd::Identity(m.rows(), m.cols());
            for (std::int64_t t = 0; t < *k; ++t) r = m * r;
            m = r;
        }
        if (g->tensorPow > 12) fail("E5009", at, "a dense tensor power above 12 qubits is too large");
        const Eigen::MatrixXcd one = m;
        for (std::int64_t t = 1; t < g->tensorPow; ++t) m = kron(m, one);
        return MatV{m};
    }
    fail("E4001", at, std::format("expected a matrix, found {}", typeName(v.type())));
}

Eigen::MatrixXcd Compiler::Impl::denseOf(const OpV& op, const QubitList& support, Span at)
{
    for (const IrOp& o : op.ops)
        if (o.kind == GK::MeasureZ || o.kind == GK::MeasurePauli || o.kind == GK::Reset || o.kind == GK::Channel || o.cond)
            fail("E4001", at, "a measurement, reset, channel or condition has no matrix");
    std::vector<double> params(ir.paramNames.size(), 0.0);
    for (const IrOp& o : op.ops)
        for (const Affine& a : o.angles)
            if (!a.isConst()) fail("E4005", at, "a dense matrix cannot depend on params");
    Eigen::MatrixXcd m = unitaryOf(op.ops, support, params);
    if (!op.gphase.isZero()) m *= std::polar(1.0, op.gphase.c.value());
    return m;
}

Eigen::MatrixXcd Compiler::Impl::denseOf(const LinOpV& l, QubitList& support, Span at)
{
    support = supportOf(l);
    if (support.size() > 12) fail("E5009", at, std::format("a dense operator on {} qubits is too large (limit 12)", support.size()));
    const Eigen::Index dim = Eigen::Index{1} << support.size();
    Eigen::MatrixXcd m = Eigen::MatrixXcd::Zero(dim, dim);
    for (const PTerm& t : l.paulis)
    {
        if (!t.coef.isConst()) fail("E4005", at, "operator coefficients cannot depend on params here");
        Eigen::MatrixXcd term = Eigen::MatrixXcd::Identity(1, 1);
        for (const Qubit q : support)
        {
            char letter = 'I';
            for (const auto& [pq, c] : t.ps)
                if (pq == q) letter = c;
            term = kron(term, pauliMatrix(letter));
        }
        m += t.coef.constValue() * term;
    }
    for (const auto& [c, op] : l.general)
    {
        if (!c.isConst()) fail("E4005", at, "operator coefficients cannot depend on params here");
        m += c.constValue() * denseOf(op, support, at);
    }
    return m;
}

std::shared_ptr<const CondNode> Compiler::Impl::toCond(const Value& v, Span at)
{
    if (const auto* c = v.get<CondV>()) return c->root;
    if (const auto* b = v.get<bool>())
    {
        auto n = std::make_shared<CondNode>();
        n->kind = CondNode::Kind::Const;
        n->value = *b;
        return n;
    }
    if (const auto* bits = v.get<BitsV>())
    {
        if (bits->clbits.size() != 1)
            fail("E4001", at, std::format("a whole bit register is not a Bool; compare it with a string, e.g. {} == \"01\"", bits->reg));
        auto n = std::make_shared<CondNode>();
        n->kind = CondNode::Kind::Var;
        n->clbit = bits->clbits[0];
        return n;
    }
    fail("E4001", at, std::format("expected a condition over classical bits, found {}", typeName(v.type())));
}


// ---- Algebra ----

Value Compiler::Impl::neg(const Value& a, Span at)
{
    if (const auto* n = a.get<Num>()) return -*n;
    if (const auto* dv = a.get<DeferredV>())
    {
        if (!isNumericType(dv->type)) fail("E4001", at, std::format("cannot negate {}", typeName(dv->type)));
        return *dv;
    }
    return mul(numInt(-1), a, at);
}

Value Compiler::Impl::add(const Value& a, const Value& b, Span at)
{
    const auto* da = a.get<DeferredV>();
    const auto* db = b.get<DeferredV>();
    if (da || db)
    {
        const Type ta = a.type(), tb = b.type();
        if (!isNumericType(ta) || !isNumericType(tb))
            fail("E4001", at, std::format("cannot add {} and {}", typeName(ta), typeName(tb)));
        return deferred(joinNumeric(ta, tb));
    }
    const auto* na = a.get<Num>();
    const auto* nb = b.get<Num>();
    if (na && nb) return *na + *nb;
    if (a.is<ListV>() || b.is<ListV>()) return add(a.is<ListV>() ? Value(toMatrix(a, at)) : a, b.is<ListV>() ? Value(toMatrix(b, at)) : b, at);
    if (const auto* ka = a.get<KetV>())
        if (const auto* kb = b.get<KetV>())
        {
            if (ka->live || kb->live) fail("E4001", at, "the live state |ψ⟩ cannot be added to");
            if (ka->nq() != kb->nq()) fail("E5009", at, std::format("cannot add kets on {} and {} qubits", ka->nq(), kb->nq()));
            if (ka->nq() > 25) fail("E5009", at, "a superposition above 25 qubits is too large");
            KetV out;
            KetFactor f;
            f.kind = KetFactor::Kind::Dense;
            f.amp = ka->dense() + kb->dense();
            f.nq = ka->nq();
            out.factors.push_back(std::move(f));
            return out;
        }
    if (const auto* ba = a.get<BraV>())
        if (const auto* bb = b.get<BraV>())
        {
            const Value k = add(ba->ket, bb->ket, at);
            return BraV{k.as<KetV>()};
        }
    if (const auto* ma = a.get<MatV>())
        if (const auto* mb = b.get<MatV>())
        {
            if (ma->m.rows() != mb->m.rows()) fail("E5009", at, "cannot add matrices of different sizes");
            return MatV{ma->m + mb->m};
        }
    if ((isOperator(a) || na) && (isOperator(b) || nb))
    {
        LinOpV la = toLin(a, at);
        const LinOpV lb = toLin(b, at);
        la.paulis.insert(la.paulis.end(), lb.paulis.begin(), lb.paulis.end());
        la.general.insert(la.general.end(), lb.general.begin(), lb.general.end());
        return simplify(std::move(la));
    }
    fail("E4001", at, std::format("cannot add {} and {}", typeName(a.type()), typeName(b.type())));
}

Value Compiler::Impl::mul(const Value& a, const Value& b, Span at)
{
    if (a.is<DeferredV>() || b.is<DeferredV>())
    {
        const Type ta = a.type(), tb = b.type();
        if (!isNumericType(ta) || !isNumericType(tb))
            fail("E4001", at, std::format("cannot multiply {} by {}", typeName(ta), typeName(tb)));
        return deferred(joinNumeric(ta, tb));
    }
    if (a.is<ListV>()) return mul(toMatrix(a, at), b, at);
    if (b.is<ListV>()) return mul(a, toMatrix(b, at), at);
    if (const auto* g = a.get<GateV>(); g && g->kind == GateV::Kind::Builtin && g->builtin != Builtin::C)
        return mul(toMatrix(a, at), b, at);
    if (const auto* g = b.get<GateV>(); g && g->kind == GateV::Kind::Builtin && g->builtin != Builtin::C)
        return mul(a, toMatrix(b, at), at);

    const auto* na = a.get<Num>();
    const auto* nb = b.get<Num>();
    if (na && nb) return *na * *nb;
    if (na || nb)
    {
        const Num& s = na ? *na : *nb;
        const Value& x = na ? b : a;
        if (const auto* k = x.get<KetV>())
        {
            if (k->live) fail("E4001", at, "the live state |ψ⟩ cannot be scaled");
            if (!s.isConst()) fail("E4004", at, "kets cannot depend on params");
            KetV out = *k;
            out.scale *= s.constValue();
            return out;
        }
        if (const auto* bv = x.get<BraV>())
        {
            if (!s.isConst()) fail("E4004", at, "bras cannot depend on params");
            BraV out = *bv;
            out.ket.scale *= std::conj(s.constValue());
            return out;
        }
        if (const auto* m = x.get<MatV>())
        {
            if (!s.isConst()) fail("E4004", at, "matrices cannot depend on params");
            return MatV{m->m * s.constValue()};
        }
        if (const auto* op = x.get<OpV>())
        {
            if (const auto ph = unitPhase(s))
            {
                OpV out = *op;
                out.gphase = out.gphase + *ph;
                return out;
            }
            LinOpV l = toLin(x, at);
            for (PTerm& t : l.paulis) t.coef = s * t.coef;
            for (auto& [c, o] : l.general) c = s * c;
            return l;
        }
        if (const auto* l0 = x.get<LinOpV>())
        {
            LinOpV l = *l0;
            for (PTerm& t : l.paulis) t.coef = s * t.coef;
            for (auto& [c, o] : l.general) c = s * c;
            return simplify(std::move(l));
        }
        if (const auto* bo = x.get<BraOpV>())
        {
            BraOpV out = *bo;
            for (PTerm& t : out.op.paulis) t.coef = s * t.coef;
            for (auto& [c, o] : out.op.general) c = s * c;
            return out;
        }
        fail("E4001", at, std::format("cannot multiply a number by {}", typeName(x.type())));
    }

    if (const auto* bra = a.get<BraV>())
    {
        if (const auto* ket = b.get<KetV>())
        {
            if (ket->live && !bra->ket.live)
            {
                if (rt) return Num::fromDouble(rt->overlap(bra->ket, at));
                readout(bra->ket.isBasis() ? "amplitude" : "overlap", true, at);
                return deferred(Type::Complex);
            }
            if (ket->live || bra->ket.live) fail("E4001", at, "⟨ψ|ψ⟩ is always 1");
            if (bra->ket.nq() != ket->nq()) fail("E5009", at, std::format("⟨…| on {} qubits and |…⟩ on {} qubits", bra->ket.nq(), ket->nq()));
            if (ket->nq() > 25) fail("E5009", at, "an inner product above 25 qubits is too large");
            return Num::fromDouble(bra->ket.dense().dot(ket->dense()));
        }
        if (isOperator(b)) return BraOpV{bra->ket, toLin(b, at)};
    }
    if (const auto* bo = a.get<BraOpV>())
    {
        if (const auto* ket = b.get<KetV>())
        {
            if (bo->bra.live && ket->live)
            {
                if (rt) return Num::fromDouble(rt->expectation(bo->op, at));
                readout("matrix-element", !bo->op.isPauli(), at, bo->op.paulis.size() + bo->op.general.size());
                return deferred(Type::Complex);
            }
            if (bo->bra.live || ket->live)
                fail("E4001", at, "a matrix element needs the live state on both sides (⟨ψ|A|ψ⟩) or on neither");
            const std::size_t n = ket->nq();
            if (bo->bra.nq() != n) fail("E5009", at, "bra and ket sizes differ");
            QubitList support = supportOf(bo->op);
            for (const Qubit q : support)
                if (q >= n) fail("E5009", at, std::format("the operator acts on qubit {} outside the {}-qubit states", q, n));
            QubitList all;
            for (Qubit q = 0; q < n; ++q) all.push_back(q);
            LinOpV padded = bo->op;
            // Identity padding: denseOf builds over the operator's own support, so embed manually.
            Eigen::MatrixXcd full = Eigen::MatrixXcd::Zero(Eigen::Index{1} << n, Eigen::Index{1} << n);
            for (const PTerm& t : padded.paulis)
            {
                Eigen::MatrixXcd term = Eigen::MatrixXcd::Identity(1, 1);
                for (const Qubit q : all)
                {
                    char letter = 'I';
                    for (const auto& [pq, c] : t.ps)
                        if (pq == q) letter = c;
                    term = kron(term, pauliMatrix(letter));
                }
                full += t.coef.constValue() * term;
            }
            for (const auto& [c, op] : padded.general) full += c.constValue() * denseOf(op, all, at);
            return Num::fromDouble(bo->bra.dense().dot(full * ket->dense()));
        }
        if (isOperator(b))
        {
            const Value prod = mul(bo->op, b, at);
            return BraOpV{bo->bra, toLin(prod, at)};
        }
    }
    if (const auto* ka = a.get<KetV>())
        if (const auto* bb = b.get<BraV>())
        {
            if (ka->live || bb->ket.live) fail("E4001", at, "|ψ⟩⟨ψ| of the live state is not supported; use ρ_{…}");
            if (ka->nq() > 12) fail("E5009", at, "an outer product above 12 qubits is too large");
            return MatV{ka->dense() * bb->ket.dense().adjoint()};
        }

    if (isOperator(a) && isOperator(b))
    {
        if (const auto* oa = a.get<OpV>())
            if (const auto* ob = b.get<OpV>())
            {
                OpV out = *ob;
                out.ops.insert(out.ops.end(), oa->ops.begin(), oa->ops.end());
                out.gphase = oa->gphase + ob->gphase;
                return out;
            }
        const LinOpV la = toLin(a, at), lb = toLin(b, at);
        LinOpV out;
        for (const PTerm& x : la.paulis)
            for (const PTerm& y : lb.paulis) out.paulis.push_back(multiply(x, y));
        auto general = [&](const Num& c, const OpV& left, const OpV& right)
        {
            OpV o = right;
            o.ops.insert(o.ops.end(), left.ops.begin(), left.ops.end());
            o.gphase = left.gphase + right.gphase;
            out.general.emplace_back(c, std::move(o));
        };
        for (const PTerm& x : la.paulis)
            for (const auto& [c, op] : lb.general) general(x.coef * c, pauliOp(x, at), op);
        for (const auto& [c, op] : la.general)
            for (const PTerm& y : lb.paulis) general(c * y.coef, op, pauliOp(y, at));
        for (const auto& [c1, op1] : la.general)
            for (const auto& [c2, op2] : lb.general) general(c1 * c2, op1, op2);
        return simplify(std::move(out));
    }
    if (const auto* ma = a.get<MatV>())
    {
        if (const auto* mb = b.get<MatV>())
        {
            if (ma->m.cols() != mb->m.rows()) fail("E5009", at, "matrix sizes do not match");
            return MatV{ma->m * mb->m};
        }
        if (const auto* k = b.get<KetV>())
        {
            if (k->live) fail("E4001", at, "apply operators to the live state with statements, not in expressions");
            if (ma->m.cols() != (Eigen::Index{1} << k->nq())) fail("E5009", at, "matrix and ket sizes do not match");
            KetV out;
            KetFactor f;
            f.kind = KetFactor::Kind::Dense;
            f.amp = ma->m * k->dense();
            f.nq = k->nq();
            out.factors.push_back(std::move(f));
            return out;
        }
    }
    if (isOperator(a))
        if (const auto* k = b.get<KetV>())
        {
            if (k->live) fail("E4001", at, "apply operators to the live state with statements, not in expressions");
            QubitList all;
            for (Qubit q = 0; q < k->nq(); ++q) all.push_back(q);
            const OpV op = toOp(a, at);
            for (const Qubit q : supportOf(op))
                if (q >= k->nq()) fail("E5009", at, "the operator acts outside the ket's qubits");
            KetV out;
            KetFactor f;
            f.kind = KetFactor::Kind::Dense;
            f.amp = denseOf(op, all, at) * k->dense();
            f.nq = k->nq();
            out.factors.push_back(std::move(f));
            return out;
        }
    if (a.is<bool>() || b.is<bool>() || a.is<CondV>() || b.is<CondV>())
        fail("E4001", at, "use `and` to combine conditions, not juxtaposition");
    fail("E4001", at, std::format("cannot multiply {} by {}", typeName(a.type()), typeName(b.type())));
}

Value Compiler::Impl::divide(const Value& a, const Value& b, Span at)
{
    if (a.is<DeferredV>() || b.is<DeferredV>())
    {
        const Type ta = a.type(), tb = b.type();
        if (!isNumericType(ta) || !isNumericType(tb))
            fail("E4001", at, std::format("cannot divide {} by {}", typeName(ta), typeName(tb)));
        return deferred(ta == Type::Complex || tb == Type::Complex ? Type::Complex : Type::Real);
    }
    const auto* nb = b.get<Num>();
    if (!nb) fail("E4001", at, std::format("cannot divide by {}", typeName(b.type())));
    if (const auto* na = a.get<Num>())
    {
        const auto q = Noether::divide(*na, *nb);
        if (!q) fail("E4001", at, "division by zero");
        return *q;
    }
    const auto inv = Noether::divide(numInt(1), *nb);
    if (!inv) fail("E4001", at, "division by zero");
    return mul(*inv, a, at);
}

Value Compiler::Impl::tensor(const Value& a, const Value& b, Span at)
{
    if (const auto* ka = a.get<KetV>())
        if (const auto* kb = b.get<KetV>())
        {
            if (ka->live || kb->live) fail("E4001", at, "the live state |ψ⟩ cannot be tensored");
            KetV out = *ka;
            out.scale *= kb->scale;
            for (const KetFactor& f : kb->factors)
            {
                if (!out.factors.empty() && f.kind == KetFactor::Kind::Basis && out.factors.back().kind == KetFactor::Kind::Basis)
                {
                    out.factors.back().bits += f.bits;
                    out.factors.back().nq += f.nq;
                }
                else out.factors.push_back(f);
            }
            return out;
        }
    if (const auto* ba = a.get<BraV>())
        if (const auto* bb = b.get<BraV>()) return BraV{tensor(ba->ket, bb->ket, at).as<KetV>()};
    if (isOperator(a) && isOperator(b))
    {
        const QubitList sa = a.is<OpV>() ? supportOf(a.as<OpV>()) : supportOf(a.as<LinOpV>());
        const QubitList sb = b.is<OpV>() ? supportOf(b.as<OpV>()) : supportOf(b.as<LinOpV>());
        for (const Qubit q : sa)
            if (std::ranges::find(sb, q) != sb.end())
                fail("E5002", at, std::format("the factors of ⊗ overlap on qubit {}", q));
        return mul(a, b, at);
    }
    const bool am = a.is<MatV>() || a.is<ListV>() || a.is<GateV>();
    const bool bm = b.is<MatV>() || b.is<ListV>() || b.is<GateV>();
    if (am && bm)
    {
        const MatV ma = toMatrix(a, at), mb = toMatrix(b, at);
        if (ma.nq() + mb.nq() > 12) fail("E5009", at, "a dense tensor product above 12 qubits is too large");
        return MatV{kron(ma.m, mb.m)};
    }
    if ((isOperator(a) && bm) || (am && isOperator(b)))
        fail("E5009", at, "⊗ of an applied operator and a matrix without qubits: give both explicit support, or neither");
    fail("E4001", at, std::format("cannot take ⊗ of {} and {}", typeName(a.type()), typeName(b.type())));
}

Value Compiler::Impl::dagger(const Value& a, Span at)
{
    if (const auto* op = a.get<OpV>()) return daggerOp(*op, at);
    if (const auto* l = a.get<LinOpV>())
    {
        LinOpV out;
        for (const PTerm& t : l->paulis) out.paulis.push_back({conj(t.coef), t.ps});
        for (const auto& [c, op] : l->general) out.general.emplace_back(conj(c), daggerOp(op, at));
        return out;
    }
    if (const auto* m = a.get<MatV>()) return MatV{m->m.adjoint()};
    if (a.is<ListV>()) return MatV{toMatrix(a, at).m.adjoint()};
    if (const auto* k = a.get<KetV>()) return BraV{*k};
    if (const auto* b = a.get<BraV>()) return b->ket;
    if (const auto* n = a.get<Num>()) return conj(*n);
    if (const auto* g = a.get<GateV>())
    {
        if (g->kind == GateV::Kind::Proc) fail("E5006", at, std::format("`{}` is a proc and has no adjoint", g->name));
        if (g->kind == GateV::Kind::Channel || g->kind == GateV::Kind::Rho)
            fail("E4001", at, std::format("`{}` has no adjoint", g->name));
        GateV out = *g;
        out.dagger = !out.dagger;
        return out;
    }
    if (const auto* dv = a.get<DeferredV>())
    {
        if (dv->type == Type::Matrix || isNumericType(dv->type)) return *dv;
    }
    fail("E4001", at, std::format("{} has no adjoint", typeName(a.type())));
}

Value Compiler::Impl::raise(const Value& base, const Value& exp, Span at)
{
    if (base.is<DeferredV>() || exp.is<DeferredV>())
    {
        const Type tb = base.type(), te = exp.type();
        if (!isNumericType(tb) || !isNumericType(te))
            fail("E4001", at, std::format("cannot raise {} to {}", typeName(tb), typeName(te)));
        return deferred(tb == Type::Complex || te == Type::Complex ? Type::Complex : tb == Type::Int && te == Type::Int ? Type::Int : Type::Real);
    }
    const Num n = needNum(exp, at, "an exponent");
    if (const auto* b = base.get<Num>())
    {
        const auto k = n.asInt();
        if (k)
        {
            if (*k == 0) return numInt(1);
            if (*k == 1) return *b;
            if (b->nonlinear || !b->isConst())
            {
                Num out;
                out.nonlinear = true;
                const cd v = std::pow(b->constValue(), static_cast<double>(*k));
                out.re = Affine(Real::approx(v.real()));
                out.im = Affine(Real::approx(v.imag()));
                return out;
            }
            // Exact integer powers; int64 overflow of an integer power is E4007.
            const bool intBase = b->asInt().has_value();
            Num acc = numInt(1);
            Num sq = *b;
            std::int64_t e = *k < 0 ? -*k : *k;
            if (intBase && *k > 0)
            {
                std::int64_t r = 1, x = *b->asInt();
                for (std::int64_t t = 0; t < e; ++t)
                    if (__builtin_mul_overflow(r, x, &r))
                        fail("E4007", at, "integer power overflows int64; use a real base such as 2.0");
                acc = numInt(r);
            }
            else
            {
                while (e > 0)
                {
                    if (e & 1) acc = acc * sq;
                    sq = sq * sq;
                    e >>= 1;
                }
                if (*k < 0)
                {
                    const auto inv = Noether::divide(numInt(1), acc);
                    if (!inv) fail("E4001", at, "zero to a negative power");
                    acc = *inv;
                }
            }
            if (b->phase) acc.phase = b->phase->scaled(Real::integer(*k));
            return acc;
        }
        if (!b->isConst() || !n.isConst())
        {
            Num out;
            out.nonlinear = true;
            return out;
        }
        if (b->phase && n.isReal()) return expi(b->phase->scaled(n.re.c));
        const cd v = std::pow(b->constValue(), n.constValue());
        if (b->isReal() && n.isReal() && b->re.c.value() >= 0) return Num(Real::approx(v.real()));
        return Num::fromDouble(v);
    }
    if (const auto* op = base.get<OpV>()) return powerOp(*op, n, at);
    if (const auto* g = base.get<GateV>())
    {
        if (g->kind == GateV::Kind::Proc && n.asInt() != 1) fail("E5006", at, std::format("`{}` is a proc and has no power", g->name));
        GateV out = *g;
        out.power = out.power ? *out.power * n : n;
        return out;
    }
    if (base.is<MatV>() || base.is<ListV>())
    {
        const MatV m = toMatrix(base, at);
        const auto k = n.asInt();
        if (!k) fail("E4001", at, "a matrix power must be an integer");
        Eigen::MatrixXcd x = *k < 0 ? Eigen::MatrixXcd(m.m.inverse()) : m.m;
        std::int64_t e = *k < 0 ? -*k : *k;
        Eigen::MatrixXcd r = Eigen::MatrixXcd::Identity(m.m.rows(), m.m.cols());
        while (e > 0)
        {
            if (e & 1) r = x * r;
            x = x * x;
            e >>= 1;
        }
        return MatV{r};
    }
    if (const auto* l = base.get<LinOpV>())
    {
        const auto k = n.asInt();
        if (!k || *k < 0) fail("E4001", at, "an operator power must be a non-negative integer");
        Value acc = numInt(1);
        for (std::int64_t t = 0; t < *k; ++t) acc = mul(acc, *l, at);
        return acc;
    }
    fail("E4001", at, std::format("cannot raise {} to a power", typeName(base.type())));
}

Value Compiler::Impl::sqrtValue(const Value& a, Span at)
{
    if (const auto* n = a.get<Num>())
    {
        if (n->isReal() && n->isConst())
        {
            const Real& r = n->re.c;
            if (r.exact() && r.piCoef().isZero() && r.rat().n >= 0)
            {
                auto isq = [](std::int64_t v) -> std::optional<std::int64_t>
                {
                    const auto s = static_cast<std::int64_t>(std::llround(std::sqrt(static_cast<double>(v))));
                    for (std::int64_t c = std::max<std::int64_t>(0, s - 1); c <= s + 1; ++c)
                        if (c * c == v) return c;
                    return std::nullopt;
                };
                const auto a1 = isq(r.rat().n), b1 = isq(r.rat().d);
                if (a1 && b1)
                    if (const auto q = Rational::make(*a1, *b1)) return Num(Real::rational(*q));
            }
            const double v = r.value();
            if (v >= 0) return Num(Real::approx(std::sqrt(v)));
            return Num::complex(Real::integer(0), Real::approx(std::sqrt(-v)));
        }
        if (!n->isConst())
        {
            Num out;
            out.nonlinear = true;
            return out;
        }
        return Num::fromDouble(std::sqrt(n->constValue()));
    }
    if (const auto* dv = a.get<DeferredV>())
    {
        if (!isNumericType(dv->type)) fail("E4001", at, "√ needs a number");
        return deferred(dv->type == Type::Complex ? Type::Complex : Type::Real);
    }
    if (const auto* g = a.get<GateV>(); g && g->kind == GateV::Kind::Builtin && g->builtin == Builtin::X && !g->dagger && !g->power)
    {
        GateV out = *g;
        out.builtin = Builtin::SX;
        out.name = "√X";
        return out;
    }
    if (const auto* op = a.get<OpV>())
    {
        bool allX = !op->ops.empty() && op->gphase.isZero();
        for (const IrOp& o : op->ops)
            if (o.kind != GK::X || !o.controls.empty() || !o.negControls.empty() || o.cond) allX = false;
        if (allX)
        {
            OpV out = *op;
            for (IrOp& o : out.ops) o.kind = GK::SX;
            return out;
        }
    }
    fail("E4008", at, "√ of an operator is only defined for X (the √X gate)");
}

Value Compiler::Impl::expValue(const Value& a, Span at)
{
    if (const auto* n = a.get<Num>())
    {
        if (!n->nonlinear && n->re.isZero()) return expi(n->im);
        if (!n->isConst())
        {
            Num out;
            out.nonlinear = true;
            return out;
        }
        if (n->isReal())
        {
            if (n->re.c.isZero()) return numInt(1);
            return Num(Real::approx(std::exp(n->re.c.value())));
        }
        return Num::fromDouble(std::exp(n->constValue()));
    }
    if (const auto* dv = a.get<DeferredV>())
    {
        if (!isNumericType(dv->type)) fail("E4001", at, "exp needs a number or an operator");
        return deferred(dv->type == Type::Complex ? Type::Complex : Type::Real);
    }
    if (isOperator(a)) return expOperator(simplify(toLin(a, at)), at);
    fail("E4001", at, std::format("exp needs a number or an operator, found {}", typeName(a.type())));
}

Value Compiler::Impl::notValue(const Value& a, Span at)
{
    if (const auto* b = a.get<bool>()) return !*b;
    if (a.is<CondV>() || a.is<BitsV>())
    {
        auto n = std::make_shared<CondNode>();
        n->kind = CondNode::Kind::Not;
        n->a = toCond(a, at);
        return CondV{n};
    }
    if (const auto* dv = a.get<DeferredV>(); dv && dv->type == Type::Bool) return *dv;
    fail("E4001", at, std::format("¬ needs a Bool, found {}", typeName(a.type())));
}


// ---- Composite expressions ----

Value Compiler::Impl::product(const Expr& e, const ConstScopePtr& sc)
{
    // `f (x)` with a space before the parenthesis: a function value cannot be multiplied, so the
    // only meaning is the call; warn and evaluate it as one.
    auto spacedCall = [&](std::size_t k) -> std::optional<Value>
    {
        const Expr& f = *e.kids[k];
        if (k + 1 >= e.kids.size() || f.kind != EK::Name || !kFunctions.contains(f.name) || sc->lookup(f.name)) return std::nullopt;
        const Expr& arg = *e.kids[k + 1];
        if (arg.kind != EK::Paren) return std::nullopt;
        warn("W0009", Span::join(f.span, arg.span),
             std::format("`{} (…)` has a space before the parenthesis; write the call `{}(…)`", f.name, f.name));
        auto call = std::make_shared<Expr>();
        call->kind = EK::Call;
        call->span = Span::join(f.span, arg.span);
        call->kids = {e.kids[k], arg.kids[0]};
        call->argNames = {""};
        return eval(*call, sc);
    };
    std::size_t first = 1;
    Value acc;
    if (auto v = spacedCall(0))
    {
        acc = std::move(*v);
        first = 2;
    }
    else acc = eval(*e.kids[0], sc);
    for (std::size_t k = first; k < e.kids.size(); ++k)
    {
        const Expr& prev = *e.kids[k - 1];
        const Expr& cur = *e.kids[k];
        if (auto v = spacedCall(k))
        {
            acc = mul(acc, *v, e.span);
            ++k;
            continue;
        }
        if (!rt)
        {
            if (cur.kind == EK::Paren && !cur.touching && prev.kind == EK::Name && builtins().contains(prev.name))
                warn("W0009", Span::join(prev.span, cur.span),
                     std::format("`{} (…)` with a space is a product; did you mean the call `{}(…)`?", prev.name, prev.name));
            if (cur.touching && prev.kind == EK::Binary && prev.op == Tok::Slash &&
                (prev.kids[1]->kind == EK::Int || prev.kids[1]->kind == EK::Real) &&
                (cur.kind == EK::Name || cur.kind == EK::Product))
            {
                const Span den{prev.kids[1]->span.file, prev.kids[1]->span.begin, cur.span.end};
                d.warning("W0001", Span::join(prev.span, cur.span),
                          std::format("`/` binds tighter than juxtaposition: this is ({})·{}", formatExpr(prev, false),
                                      formatExpr(cur, false)))
                    .fix("divide by the whole product", den,
                         std::format("({}{})", formatExpr(*prev.kids[1], false), formatExpr(cur, false)));
            }
        }
        acc = mul(acc, eval(cur, sc), Span::join(e.kids[0]->span, cur.span));
    }
    return acc;
}

Value Compiler::Impl::logic(const Expr& e, const ConstScopePtr& sc)
{
    const Value a = eval(*e.kids[0], sc);
    const Value b = eval(*e.kids[1], sc);
    const bool isAnd = e.op == Tok::KwAnd;
    if (const auto* ba = a.get<bool>())
        if (const auto* bb = b.get<bool>()) return isAnd ? (*ba && *bb) : (*ba || *bb);
    auto condLike = [](const Value& v) { return v.is<CondV>() || v.is<BitsV>() || v.is<bool>(); };
    if (condLike(a) && condLike(b))
    {
        auto n = std::make_shared<CondNode>();
        n->kind = isAnd ? CondNode::Kind::And : CondNode::Kind::Or;
        n->a = toCond(a, e.kids[0]->span);
        n->b = toCond(b, e.kids[1]->span);
        return CondV{n};
    }
    if (a.type() == Type::Bool && b.type() == Type::Bool) return deferred(Type::Bool);
    fail("E4001", e.span, std::format("`{}` needs Bool operands, found {} and {}", isAnd ? "and" : "or",
                                      typeName(a.type()), typeName(b.type())));
}

Value Compiler::Impl::compare(const Expr& e, const ConstScopePtr& sc)
{
    const Value a = eval(*e.kids[0], sc);
    const Value b = eval(*e.kids[1], sc);
    const Tok op = e.op;

    // Bits against a pattern or 0/1: a run-time condition.
    auto bitsCond = [&](const BitsV& bits, const Value& other) -> std::optional<Value>
    {
        std::string pattern;
        if (const auto* s = other.get<std::string>()) pattern = *s;
        else if (const auto* n = other.get<Num>(); n && n->asInt() && (*n->asInt() == 0 || *n->asInt() == 1))
            pattern = std::string(bits.clbits.size(), '0'), pattern.back() = *n->asInt() ? '1' : '0';
        else if (const auto* bl = other.get<bool>()) pattern = *bl ? "1" : "0";
        else return std::nullopt;
        if (pattern.size() != bits.clbits.size() || !std::ranges::all_of(pattern, [](char c) { return c == '0' || c == '1'; }))
            fail("E5009", e.span, std::format("comparing {} bit{} with \"{}\"", bits.clbits.size(), bits.clbits.size() == 1 ? "" : "s", pattern));
        std::shared_ptr<const CondNode> acc;
        for (std::size_t k = 0; k < pattern.size(); ++k)
        {
            auto var = std::make_shared<CondNode>();
            var->kind = CondNode::Kind::Var;
            var->clbit = bits.clbits[k];
            std::shared_ptr<const CondNode> lit = var;
            if (pattern[k] == '0')
            {
                auto nn = std::make_shared<CondNode>();
                nn->kind = CondNode::Kind::Not;
                nn->a = var;
                lit = nn;
            }
            if (!acc) acc = lit;
            else
            {
                auto andN = std::make_shared<CondNode>();
                andN->kind = CondNode::Kind::And;
                andN->a = acc;
                andN->b = lit;
                acc = andN;
            }
        }
        if (op == Tok::NotEq)
        {
            auto nn = std::make_shared<CondNode>();
            nn->kind = CondNode::Kind::Not;
            nn->a = acc;
            acc = nn;
        }
        return Value(CondV{acc});
    };
    if ((op == Tok::EqEq || op == Tok::NotEq))
    {
        if (const auto* bits = a.get<BitsV>())
            if (auto c = bitsCond(*bits, b)) return *c;
        if (const auto* bits = b.get<BitsV>())
            if (auto c = bitsCond(*bits, a)) return *c;
    }

    if (a.is<DeferredV>() || b.is<DeferredV>())
    {
        const Type ta = a.type(), tb = b.type();
        const bool ok = (isNumericType(ta) && isNumericType(tb)) || (ta == tb && (op == Tok::EqEq || op == Tok::NotEq));
        if (!ok) fail("E4001", e.span, std::format("cannot compare {} with {}", typeName(ta), typeName(tb)));
        if ((op != Tok::EqEq && op != Tok::NotEq && op != Tok::Approx) && (ta == Type::Complex || tb == Type::Complex))
            fail("E4001", e.span, "complex numbers have no order; compare abs(…) or a real part");
        if (e.kids.size() > 2) needReal(eval(*e.kids[2], sc), e.kids[2]->span, "a tolerance");
        return deferred(Type::Bool);
    }
    if (const auto* sa = a.get<std::string>())
    {
        const auto* sb = b.get<std::string>();
        if (!sb || (op != Tok::EqEq && op != Tok::NotEq)) fail("E4001", e.span, "strings compare only with == and ≠");
        return op == Tok::EqEq ? *sa == *sb : *sa != *sb;
    }
    if (const auto* ba = a.get<bool>())
    {
        const auto* bb = b.get<bool>();
        if (!bb || (op != Tok::EqEq && op != Tok::NotEq)) fail("E4001", e.span, "Bools compare only with == and ≠");
        return op == Tok::EqEq ? *ba == *bb : *ba != *bb;
    }
    const auto* na = a.get<Num>();
    const auto* nb = b.get<Num>();
    if (!na || !nb) fail("E4001", e.span, std::format("cannot compare {} with {}", typeName(a.type()), typeName(b.type())));
    const std::vector<double> params = rt ? [&] {
        std::vector<double> p(ir.paramNames.size());
        for (std::size_t k = 0; k < p.size(); ++k) p[k] = rt->param(static_cast<std::uint32_t>(k));
        return p;
    }() : ir.paramValues;
    const cd x = na->isConst() ? na->constValue() : na->eval(params);
    const cd y = nb->isConst() ? nb->constValue() : nb->eval(params);
    if (!rt && (!na->isConst() || !nb->isConst())) return deferred(Type::Bool);
    switch (op)
    {
        case Tok::Approx:
        {
            const double tol = e.kids.size() > 2 ? needReal(eval(*e.kids[2], sc), e.kids[2]->span, "a tolerance") : 1e-9;
            return std::abs(x - y) <= tol;
        }
        case Tok::EqEq:
        {
            // Exact comparison when both sides are exact.
            if (na->isConst() && nb->isConst() && na->re.c.exact() && nb->re.c.exact() && na->im.c.exact() && nb->im.c.exact())
                return (na->re.c - nb->re.c).isZero() && (na->im.c - nb->im.c).isZero();
            return x == y;
        }
        case Tok::NotEq:
            if (na->isConst() && nb->isConst() && na->re.c.exact() && nb->re.c.exact() && na->im.c.exact() && nb->im.c.exact())
                return !((na->re.c - nb->re.c).isZero() && (na->im.c - nb->im.c).isZero());
            return x != y;
        default: break;
    }
    if (x.imag() != 0.0 || y.imag() != 0.0) fail("E4001", e.span, "complex numbers have no order");
    switch (op)
    {
        case Tok::Less: return x.real() < y.real();
        case Tok::LessEq: return x.real() <= y.real();
        case Tok::Greater: return x.real() > y.real();
        case Tok::GreaterEq: return x.real() >= y.real();
        default: break;
    }
    fail("E2001", e.span, "unknown comparison");
}

Value Compiler::Impl::index(const Expr& e, const ConstScopePtr& sc)
{
    const Value base = eval(*e.kids[0], sc);
    const Expr& ix = *e.kids[1];
    if (ix.kind == EK::Slice)
    {
        const std::int64_t lo = needInt(eval(*ix.kids[0], sc), ix.kids[0]->span, "a range bound");
        const std::int64_t hi = needInt(eval(*ix.kids[1], sc), ix.kids[1]->span, "a range bound");
        auto range = [&](std::size_t size) -> std::pair<std::size_t, std::size_t>
        {
            if (hi < lo) return {0, 0};
            if (lo < 0 || static_cast<std::size_t>(hi) >= size)
                fail("E5008", ix.span, std::format("range {}..{} is out of bounds for size {}", lo, hi, size));
            return {static_cast<std::size_t>(lo), static_cast<std::size_t>(hi - lo + 1)};
        };
        if (const auto* q = base.get<QubitsV>())
        {
            const auto [off, n] = range(q->q.size());
            QubitsV out;
            out.q.assign(q->q.begin() + static_cast<std::ptrdiff_t>(off), q->q.begin() + static_cast<std::ptrdiff_t>(off + n));
            return out;
        }
        if (const auto* b = base.get<BitsV>())
        {
            const auto [off, n] = range(b->clbits.size());
            BitsV out;
            out.reg = b->reg;
            out.clbits.assign(b->clbits.begin() + static_cast<std::ptrdiff_t>(off), b->clbits.begin() + static_cast<std::ptrdiff_t>(off + n));
            return out;
        }
        if (const auto* l = base.get<ListV>())
        {
            const auto [off, n] = range(l->items.size());
            ListV out;
            out.items.assign(l->items.begin() + static_cast<std::ptrdiff_t>(off), l->items.begin() + static_cast<std::ptrdiff_t>(off + n));
            return out;
        }
        fail("E4001", e.span, std::format("{} cannot be sliced", typeName(base.type())));
    }

    const Value iv = eval(ix, sc);
    if (const auto* c = base.get<CountsV>())
    {
        const auto* key = iv.get<std::string>();
        if (!key) fail("E4001", ix.span, "counts are indexed by a bitstring such as \"0101\"");
        const auto it = c->counts.find(*key);
        return numInt(it == c->counts.end() ? 0 : static_cast<std::int64_t>(it->second));
    }
    if (const auto* dv = base.get<DeferredV>())
    {
        if (dv->type != Type::Counts) fail("E4001", e.span, std::format("{} cannot be indexed", typeName(dv->type)));
        if (iv.type() != Type::String) fail("E4001", ix.span, "counts are indexed by a bitstring such as \"0101\"");
        return deferred(Type::Int);
    }
    const std::int64_t k = needInt(iv, ix.span, "an index");
    auto check = [&](std::size_t size)
    {
        if (k < 0 || static_cast<std::size_t>(k) >= size)
            fail("E5008", ix.span, std::format("index {} is out of range for size {}", k, size));
        return static_cast<std::size_t>(k);
    };
    if (const auto* q = base.get<QubitsV>())
    {
        QubitsV out;
        out.q = {q->q[check(q->q.size())]};
        out.single = true;
        return out;
    }
    if (const auto* b = base.get<BitsV>())
    {
        BitsV out;
        out.reg = b->reg;
        out.clbits = {b->clbits[check(b->clbits.size())]};
        out.single = true;
        if (rt) return out;
        return out;
    }
    if (const auto* l = base.get<ListV>()) return *l->items[check(l->items.size())];
    fail("E4001", e.span, std::format("{} cannot be indexed", typeName(base.type())));
}

Value Compiler::Impl::power(const Expr& e, const ConstScopePtr& sc)
{
    const Expr& b = *e.kids[0];
    const Expr& x = *e.kids[1];
    if (b.kind == EK::Name && b.name == "e" && !sc->lookup("e")) return expValue(eval(x, sc), e.span);
    // |⟨b|ψ⟩|^2: a basis probability, which both backends read.
    if (b.kind == EK::Abs && b.kids[0]->kind == EK::Braket && b.kids[0]->label2 == "ψ" && x.kind == EK::Int && x.name == "2")
    {
        const std::string& label = b.kids[0]->name;
        if (!label.empty() && std::ranges::all_of(label, [](char c) { return c == '0' || c == '1'; }))
            return probabilityOf(label, e.span);
    }
    return raise(eval(b, sc), eval(x, sc), e.span);
}

Value Compiler::Impl::probabilityOf(const std::string& bits, Span at)
{
    if (bits.size() != ir.nQubits)
        fail("E5009", at, std::format("⟨{}| has {} qubits but {} are declared", bits, bits.size(), ir.nQubits));
    if (rt) return Num(Real::approx(rt->probability(bits, at)));
    readout("probability", false, at);
    return deferred(Type::Real);
}

Value Compiler::Impl::bigop(const Expr& e, const ConstScopePtr& sc)
{
    const std::int64_t lo = needInt(eval(*e.kids[0], sc), e.kids[0]->span, "a bound");
    const std::int64_t hi = needInt(eval(*e.kids[1], sc), e.kids[1]->span, "a bound");
    checkBindable(e.name, e.span);
    auto term = [&](std::int64_t j)
    {
        auto inner = std::make_shared<Scope>(std::const_pointer_cast<Scope>(sc));
        Symbol sym;
        sym.kind = Symbol::Kind::Loop;
        sym.value = numInt(j);
        sym.used = true;
        inner->define(e.name, std::move(sym));
        return eval(*e.kids[2], inner);
    };
    if (hi < lo) return e.op == Tok::Sum ? numInt(0) : Value(OpV{});
    std::vector<Value> terms;
    terms.reserve(static_cast<std::size_t>(hi - lo + 1));
    for (std::int64_t j = lo; j <= hi; ++j) terms.push_back(term(j));
    // Pairwise reduction keeps the order (∏_{a}^{b} U_j = U_b ⋯ U_a) and costs O(n log n) merges
    // instead of O(n²) for long Pauli sums and products.
    while (terms.size() > 1)
    {
        std::vector<Value> next;
        next.reserve((terms.size() + 1) / 2);
        for (std::size_t k = 0; k + 1 < terms.size(); k += 2)
            next.push_back(e.op == Tok::Sum ? add(terms[k], terms[k + 1], e.span) : mul(terms[k + 1], terms[k], e.span));
        if (terms.size() % 2) next.push_back(std::move(terms.back()));
        terms = std::move(next);
    }
    return std::move(terms.front());
}

Value Compiler::Impl::expval(const Expr& e, const ConstScopePtr& sc)
{
    const Value v = eval(*e.kids[0], sc);
    if (!isOperator(v)) fail("E4001", e.span, std::format("⟨…⟩ needs an observable, found {}", typeName(v.type())));
    const LinOpV l = simplify(toLin(v, e.span));
    for (const PTerm& t : l.paulis)
    {
        if (!t.coef.im.isZero())
        {
            const bool tiny = t.coef.im.isConst() && std::abs(t.coef.im.c.value()) <= 1e-12;
            if (!tiny) fail("E4006", e.span, "⟨A⟩ needs a Hermitian A; for a general operator use ⟨ψ|A|ψ⟩");
        }
    }
    if (!l.general.empty())
    {
        QubitList support;
        const Eigen::MatrixXcd m = denseOf(l, support, e.span);
        if ((m - m.adjoint()).cwiseAbs().maxCoeff() > 1e-12)
            fail("E4006", e.span, "⟨A⟩ needs a Hermitian A; for a general operator use ⟨ψ|A|ψ⟩");
        if (rt) return Num(Real::approx(rt->expectation(l, e.span).real()));
        readout("dense-observable", true, e.span, 1, support.size());
        return deferred(Type::Real);
    }
    if (rt) return Num(Real::approx(rt->expectation(l, e.span).real()));
    readout("pauli", false, e.span, l.paulis.size());
    return deferred(Type::Real);
}

Value Compiler::Impl::absValue(const Expr& e, const ConstScopePtr& sc)
{
    const Expr& inner = *e.kids[0];
    if (inner.kind == EK::Braket && inner.label2 == "ψ" && !inner.name.empty() &&
        std::ranges::all_of(inner.name, [](char c) { return c == '0' || c == '1'; }))
    {
        const Value p = probabilityOf(inner.name, e.span);
        if (const auto* n = p.get<Num>()) return Num(Real::approx(std::sqrt(n->re.c.value())));
        return p;
    }
    const Value v = eval(inner, sc);
    if (const auto* n = v.get<Num>())
    {
        if (!n->isConst())
        {
            Num out;
            out.nonlinear = true;
            return out;
        }
        if (n->isReal()) return n->re.c.value() < 0 ? Num(-n->re.c) : *n;
        return Num(Real::approx(std::abs(n->constValue())));
    }
    if (const auto* dv = v.get<DeferredV>())
    {
        if (!isNumericType(dv->type)) fail("E4001", e.span, "|…| needs a number");
        return deferred(dv->type == Type::Int ? Type::Int : Type::Real);
    }
    fail("E4001", e.span, std::format("|…| needs a number, found {}", typeName(v.type())));
}


// ---- Calls ----

Value Compiler::Impl::call(const Expr& e, const ConstScopePtr& sc)
{
    const Expr& callee = *e.kids[0];
    if (callee.kind == EK::Name && !sc->lookup(callee.name))
    {
        const std::string& n = callee.name;
        if (kFunctions.contains(n)) return callFunction(n, e, sc);
        if (kChannels.contains(n)) return callFunction(n, e, sc);
    }
    for (const std::string& an : e.argNames)
        if (!an.empty()) fail("E4001", e.span, std::format("named argument `{}` is only accepted by trotter", an));

    const Value f = eval(callee, sc);
    if (const auto* cv = f.get<ControlV>())
    {
        if (e.kids.size() != 2) fail("E4001", e.span, "C_{…}(U) takes exactly one operation");
        const Value u = eval(*e.kids[1], sc);
        if (!isOperator(u))
            fail("E4001", e.kids[1]->span, std::format("C_{{…}}(…) needs an operation with explicit support, found {}", typeName(u.type())));
        return controlled(*cv, toOp(u, e.kids[1]->span), e.span);
    }
    if (const auto* g = f.get<GateV>())
    {
        if (!g->args.empty()) fail("E4002", e.span, std::format("`{}` already has its parameters", g->name));
        GateV out = *g;
        if (g->kind == GateV::Kind::Builtin)
        {
            if (g->builtin == Builtin::C) fail("E2001", e.span, "C needs its controls first: C_{0}(U)");
            const BuiltinInfo& info = infoOf(g->builtin);
            if (info.params == 0) fail("E4002", e.span, std::format("`{}` takes no parameters", g->name));
            if (e.kids.size() - 1 != info.params)
                fail("E4001", e.span, std::format("`{}` takes {} parameter{}", g->name, info.params, info.params == 1 ? "" : "s"));
        }
        else if (g->kind == GateV::Kind::LetGate) fail("E4002", e.span, std::format("`{}` takes no parameters", g->name));
        else if (g->kind == GateV::Kind::Rho || g->kind == GateV::Kind::Channel)
            fail("E4002", e.span, std::format("`{}` is not callable", g->name));
        for (std::size_t k = 1; k < e.kids.size(); ++k)
        {
            const Value a = eval(*e.kids[k], sc);
            if (g->kind == GateV::Kind::Builtin)
            {
                const Affine ang = needAngle(a, e.kids[k]->span, "a gate angle");
                // W0002: a decimal or computed angle within rounding of a multiple of π/4 (exact
                // multiples of π are written with π and never warn).
                const bool piFree = !ang.c.exact() || (ang.c.piCoef().isZero() && !ang.c.rat().isZero());
                if (!rt && ang.isConst() && piFree)
                {
                    const double v = ang.c.value();
                    const double q = v / (std::numbers::pi / 4);
                    const double r = std::round(q);
                    if (r != 0 && std::abs(q - r) <= 1e-5 && std::abs(v - r * std::numbers::pi / 4) <= 1e-5 * std::max(1.0, std::abs(v)))
                    {
                        const auto k4 = static_cast<long long>(r);
                        const auto rr = Rational::make(k4, 4).value_or(Rational{k4, 4});
                        const std::string exact = Real::pi(rr).str();
                        Diagnostic& diag = d.warning("W0002", e.kids[k]->span,
                                                     std::format("{} is within rounding of {}; write the exact angle", v, exact));
                        if (e.kids[k]->kind == EK::Real) diag.fix("use the exact angle", e.kids[k]->span, exact);
                    }
                }
            }
            out.args.push_back(std::make_shared<Value>(a));
        }
        return out;
    }
    fail("E4002", callee.span, std::format("{} is not callable", typeName(f.type())));
}

Value Compiler::Impl::callFunction(const std::string& name, const Expr& e, const ConstScopePtr& sc)
{
    std::vector<Value> args;
    std::vector<Span> spans;
    std::map<std::string, Value> named;
    for (std::size_t k = 1; k < e.kids.size(); ++k)
    {
        const std::string& an = e.argNames[k - 1];
        if (!an.empty())
        {
            if (name != "trotter") fail("E4001", e.kids[k]->span, std::format("`{}` takes no named arguments", name));
            named[an] = eval(*e.kids[k], sc);
            continue;
        }
        if (name == "count" && k == 1)
        {
            args.emplace_back();
            spans.push_back(e.kids[k]->span);
            continue;
        }
        args.push_back(eval(*e.kids[k], sc));
        spans.push_back(e.kids[k]->span);
    }
    auto arity = [&](std::size_t n)
    {
        if (args.size() != n)
            fail("E4001", e.span, std::format("`{}` takes {} argument{}", name, n, n == 1 ? "" : "s"));
    };

    if (name == "sin" || name == "cos" || name == "tan" || name == "log" || name == "floor" || name == "ceil")
    {
        arity(1);
        const Value& a = args[0];
        if (const auto* dv = a.get<DeferredV>())
        {
            if (!isNumericType(dv->type)) fail("E4001", spans[0], std::format("`{}` needs a number", name));
            return deferred(name == "floor" || name == "ceil" ? Type::Int : Type::Real);
        }
        const Num n = needNum(a, spans[0], "the argument");
        if (!n.isConst())
        {
            if (name == "floor" || name == "ceil")
                fail("E4004", spans[0], std::format("`{}` of a param is not known at compile time", name));
            Num out;
            out.nonlinear = true;
            return out;
        }
        if (!n.im.isZero()) fail("E4001", spans[0], std::format("`{}` needs a real argument", name));
        const Real& r = n.re.c;
        if (name == "floor" || name == "ceil")
        {
            if (r.exact() && r.piCoef().isZero())
            {
                const Rational q = r.rat();
                std::int64_t fl = q.n / q.d;
                if ((q.n % q.d != 0) && (q.n < 0)) --fl;
                if (name == "ceil" && q.n % q.d != 0) ++fl;
                return numInt(fl);
            }
            const double v = name == "floor" ? std::floor(r.value()) : std::ceil(r.value());
            return numInt(static_cast<std::int64_t>(v));
        }
        if (r.isZero())
        {
            if (name == "cos") return numInt(1);
            if (name == "log") fail("E4001", spans[0], "log(0) is undefined");
            return numInt(0);
        }
        const double v = r.value();
        if (name == "sin") return Num(Real::approx(std::sin(v)));
        if (name == "cos") return Num(Real::approx(std::cos(v)));
        if (name == "tan") return Num(Real::approx(std::tan(v)));
        if (v <= 0) fail("E4001", spans[0], "log needs a positive argument");
        return Num(Real::approx(std::log(v)));
    }
    if (name == "exp") return arity(1), expValue(args[0], e.span);
    if (name == "sqrt") return arity(1), sqrtValue(args[0], e.span);
    if (name == "abs")
    {
        arity(1);
        auto p = std::make_shared<Expr>();
        p->kind = EK::Abs;
        p->span = e.span;
        p->kids = {e.kids[1]};
        return absValue(*p, sc);
    }
    if (name == "min" || name == "max")
    {
        if (args.empty()) fail("E4001", e.span, std::format("`{}` needs at least one argument", name));
        bool anyDeferred = false;
        for (const Value& a : args)
            if (a.is<DeferredV>()) anyDeferred = true;
        if (anyDeferred) return deferred(Type::Real);
        Num best = needNum(args[0], spans[0], "an argument");
        for (std::size_t k = 1; k < args.size(); ++k)
        {
            const Num n = needNum(args[k], spans[k], "an argument");
            if (!n.isConst() || !best.isConst())
            {
                Num out;
                out.nonlinear = true;
                return out;
            }
            if ((name == "min") == (n.re.c.value() < best.re.c.value())) best = n;
        }
        return best;
    }
    if (name == "trotter")
    {
        if (args.size() != 2) fail("E4001", e.span, "trotter(Ham, t, steps=n, order=1|2|4) takes two positional arguments");
        if (!isOperator(args[0])) fail("E4001", spans[0], "trotter needs a Hamiltonian (a Pauli sum)");
        const Num t = needNum(args[1], spans[1], "the evolution time");
        std::int64_t steps = 1, order = 1;
        for (const auto& [k, v] : named)
        {
            if (k == "steps") steps = needInt(v, e.span, "steps");
            else if (k == "order") order = needInt(v, e.span, "order");
            else fail("E4001", e.span, std::format("trotter has no argument `{}` (use steps= and order=)", k));
        }
        if (steps < 1) fail("E4003", e.span, "trotter needs steps ≥ 1");
        if (order != 1 && order != 2 && order != 4) fail("E4001", e.span, "trotter order must be 1, 2 or 4");
        return trotter(simplify(toLin(args[0], spans[0])), t, steps, order, e.span);
    }
    if (name == "entropy")
    {
        arity(1);
        const auto* r = args[0].get<RhoV>();
        if (!r) fail("E4001", spans[0], "entropy needs a reduced state ρ_{…}");
        if (rt) return Num(Real::approx(rt->entropy(r->q, e.span)));
        readout("entropy", false, e.span, 1, r->q.size());
        return deferred(Type::Real);
    }
    if (name == "fidelity")
    {
        arity(1);
        const auto* k = args[0].get<KetV>();
        if (!k || k->live) fail("E4001", spans[0], "fidelity needs a fixed ket |φ⟩");
        if (k->nq() != ir.nQubits) fail("E5009", spans[0], std::format("|φ⟩ has {} qubits but {} are declared", k->nq(), ir.nQubits));
        if (k->isBasis())
        {
            const Value p = probabilityOf(k->basisBits(), e.span);
            if (const auto* n = p.get<Num>()) return Num(Real::approx(n->re.c.value() * std::norm(k->scale)));
            return p;
        }
        if (rt) return Num(Real::approx(std::norm(rt->overlap(*k, e.span))));
        readout("fidelity", true, e.span);
        return deferred(Type::Real);
    }
    if (name == "marginal")
    {
        arity(2);
        const auto* reg = args[1].get<BitsV>();
        if (!reg) fail("E4001", spans[1], "marginal(counts, reg) needs a bit register as its second argument");
        if (const auto* c = args[0].get<CountsV>())
        {
            CountsV out;
            out.layout = {{reg->reg, reg->clbits}};
            // Position of each clbit in the key (registers joined by '|').
            std::map<std::size_t, std::size_t> pos;
            std::size_t p = 0;
            for (std::size_t r = 0; r < c->layout.size(); ++r)
            {
                for (const std::size_t cb : c->layout[r].second) pos[cb] = p++;
                ++p; // separator
            }
            for (const auto& [key, n] : c->counts)
            {
                std::string k2;
                for (const std::size_t cb : reg->clbits)
                {
                    const auto it = pos.find(cb);
                    if (it == pos.end()) fail("E5009", spans[1], "the register is not part of these counts");
                    k2 += key[it->second];
                }
                out.counts[k2] += n;
            }
            return out;
        }
        if (args[0].type() != Type::Counts) fail("E4001", spans[0], "marginal needs counts from `run` as its first argument");
        return deferred(Type::Counts);
    }
    if (name == "count" || name == "depth")
    {
        std::optional<GK> gate;
        if (name == "count")
        {
            if (e.kids.size() != 2) fail("E4001", e.span, "count(G) takes one gate name");
            const Expr* g = e.kids[1].get();
            bool dag = false;
            if (g->kind == EK::Dagger)
            {
                dag = true;
                g = g->kids[0].get();
            }
            if (g->kind == EK::Unary && g->op == Tok::Sqrt && g->kids[0]->kind == EK::Name && g->kids[0]->name == "X")
                gate = dag ? GK::SXdg : GK::SX;
            else if (g->kind == EK::Name && builtins().contains(g->name) && g->name != "C")
                gate = kindOf(builtins().at(g->name).b, dag);
            else if (g->kind == EK::Name && g->name == "measure")
                gate = GK::MeasureZ;
            else fail("E4001", g->span, "count(G) needs a gate name such as CNOT or T†");
        }
        else if (!args.empty()) fail("E4001", e.span, "depth() takes no arguments");
        if (rt) return numInt(static_cast<std::int64_t>(rt->metric(name, gate)));
        return deferred(Type::Int);
    }

    // Channels
    GateV g;
    g.kind = GateV::Kind::Channel;
    g.name = name;
    g.span = e.span;
    auto spec = std::make_shared<ChannelSpec>();
    spec->name = name;
    auto prob = [&](std::size_t k)
    {
        const double p = needReal(args.at(k), spans.at(k), "a probability");
        if (p < 0 || p > 1) fail("E4001", spans[k], std::format("probability {} lies outside [0, 1]", p));
        return p;
    };
    if (name == "depolarize") { arity(1); const double p = prob(0); spec->probs = {p / 3, p / 3, p / 3}; }
    else if (name == "depolarize2") { arity(1); const double p = prob(0); spec->arity = 2; spec->probs.assign(15, p / 15); }
    else if (name == "dephase") { arity(1); spec->probs = {0, 0, prob(0)}; }
    else if (name == "flip") { arity(1); spec->probs = {prob(0), 0, 0}; }
    else if (name == "pauli")
    {
        arity(3);
        spec->probs = {prob(0), prob(1), prob(2)};
        if (spec->probs[0] + spec->probs[1] + spec->probs[2] > 1 + 1e-12)
            fail("E4001", e.span, "pauli(px, py, pz) needs px + py + pz ≤ 1");
    }
    else if (name == "ampdamp")
    {
        arity(1);
        const double gamma = prob(0);
        spec->pauli = false;
        Eigen::MatrixXcd k0 = Eigen::MatrixXcd::Zero(2, 2), k1 = Eigen::MatrixXcd::Zero(2, 2);
        k0(0, 0) = 1;
        k0(1, 1) = std::sqrt(1 - gamma);
        k1(0, 1) = std::sqrt(gamma);
        spec->kraus = {k0, k1};
    }
    else if (name == "kraus")
    {
        arity(1);
        const auto* l = args[0].get<ListV>();
        if (!l || l->items.empty()) fail("E4001", spans[0], "kraus([K1, K2, …]) needs a list of matrices");
        spec->pauli = false;
        for (const ValuePtr& item : l->items) spec->kraus.push_back(toMatrix(*item, spans[0]).m);
        const Eigen::Index dim = spec->kraus[0].rows();
        Eigen::MatrixXcd sum = Eigen::MatrixXcd::Zero(dim, dim);
        for (const Eigen::MatrixXcd& k : spec->kraus)
        {
            if (k.rows() != dim) fail("E5009", spans[0], "Kraus operators must all have the same size");
            sum += k.adjoint() * k;
        }
        if ((sum - Eigen::MatrixXcd::Identity(dim, dim)).cwiseAbs().maxCoeff() > 1e-10 * static_cast<double>(dim))
            fail("E5003", spans[0], "the Kraus operators are not complete (Σ K†K ≠ I)");
        spec->arity = log2Size(dim);
    }
    g.channel = spec;
    return g;
}


// ---- Subscripts and gates ----

QArgs Compiler::Impl::qargs(const QList& ql, const ConstScopePtr& sc, Span at)
{
    QArgs out;
    out.seps = ql.seps;
    out.span = at;
    const bool top = ctx.back() == Ctx::Top;
    for (const QItem& it : ql.items)
    {
        QArg a;
        a.negated = it.negated;
        a.span = it.span;
        if (it.expr->kind == EK::Slice)
        {
            if (!top) fail("E5009", it.span, "inside a def or proc, qubits are reachable only through its subscript parameters");
            const std::int64_t lo = needInt(eval(*it.expr->kids[0], sc), it.expr->kids[0]->span, "a qubit index");
            const std::int64_t hi = needInt(eval(*it.expr->kids[1], sc), it.expr->kids[1]->span, "a qubit index");
            for (std::int64_t q = lo; q <= hi; ++q)
            {
                if (q < 0) fail("E5008", it.span, std::format("qubit {} is out of range", q));
                a.q.push_back(static_cast<Qubit>(q));
            }
        }
        else
        {
            const Value v = eval(*it.expr, sc);
            if (const auto* q = v.get<QubitsV>())
            {
                a.q = q->q;
                a.single = q->single;
            }
            else if (v.is<Num>())
            {
                if (!top)
                    fail("E5009", it.span, "inside a def or proc, qubits are reachable only through its subscript parameters");
                const std::int64_t qi = needInt(v, it.span, "a qubit index");
                if (qi < 0) fail("E5008", it.span, std::format("qubit {} is out of range", qi));
                a.q = {static_cast<Qubit>(qi)};
                a.single = true;
            }
            else if (v.is<BitsV>()) fail("E4001", it.span, "a bit register is not a qubit");
            else fail("E4001", it.span, std::format("a subscript needs qubits, found {}", typeName(v.type())));
        }
        for (const Qubit q : a.q) checkQubit(q, it.span);
        out.items.push_back(std::move(a));
    }
    return out;
}

void Compiler::Impl::checkQubit(Qubit q, Span at)
{
    if (q >= ir.nQubits)
        fail("E5008", at, ir.nQubits == 0 ? std::format("qubit {} used but no qubits are declared (add `qubits q[n]`)", q)
                                          : std::format("qubit {} is out of range: {} qubit{} declared", q, ir.nQubits,
                                                        ir.nQubits == 1 ? " is" : "s are"));
}

void Compiler::Impl::checkDistinct(const QubitList& q, Span at)
{
    std::set<Qubit> seen;
    for (const Qubit x : q)
        if (!seen.insert(x).second) fail("E5001", at, std::format("qubit {} appears twice in one operation (no-cloning)", x));
}

Value Compiler::Impl::applySub(const Expr& e, const ConstScopePtr& sc)
{
    const Expr& base = *e.kids[0];
    // `my_var` used as a name.
    if (base.kind == EK::Name && !sc->lookup(base.name) && !builtins().contains(base.name) && base.name != "ρ" &&
        !e.qlist.braced && e.qlist.items.size() == 1 && e.qlist.items[0].expr->kind == EK::Name &&
        !sc->lookup(e.qlist.items[0].expr->name))
    {
        const std::string& a = base.name;
        const std::string& b = e.qlist.items[0].expr->name;
        auto ascii = [](const std::string& s) { return std::ranges::all_of(s, [](char c) { return static_cast<unsigned char>(c) < 0x80; }); };
        if (ascii(a) && ascii(b) && !laterNames.contains(a))
        {
            std::string camel = a + b;
            camel[a.size()] = static_cast<char>(std::toupper(static_cast<unsigned char>(b[0])));
            d.error("E3005", e.span, std::format("names cannot contain `_` (it always means subscript): `{}_{}`", a, b))
                .fix("use camelCase", e.span, camel);
            throw CompileAbort{};
        }
    }

    const Value b = eval(base, sc);
    const QArgs qa = qargs(e.qlist, sc, e.span);
    const auto* g = b.get<GateV>();
    if (!(g && g->kind == GateV::Kind::Builtin && g->builtin == Builtin::C))
        for (const QArg& a : qa.items)
            if (a.negated) fail("E2001", a.span, "¬ marks a negative control and is only valid in C_{…}");

    if (g)
    {
        if (g->kind == GateV::Kind::Rho)
        {
            RhoV r;
            for (const QArg& a : qa.items) r.q.insert(r.q.end(), a.q.begin(), a.q.end());
            checkDistinct(r.q, e.span);
            return r;
        }
        if (g->kind == GateV::Kind::Builtin && g->builtin == Builtin::C)
        {
            ControlV cv;
            for (const QArg& a : qa.items) (a.negated ? cv.neg : cv.pos).insert((a.negated ? cv.neg : cv.pos).end(), a.q.begin(), a.q.end());
            QubitList all = cv.pos;
            all.insert(all.end(), cv.neg.begin(), cv.neg.end());
            checkDistinct(all, e.span);
            return cv;
        }
        return applyGate(*g, qa, e.span);
    }
    if (b.is<MatV>() || b.is<ListV>())
    {
        const MatV m = toMatrix(b, base.span);
        QubitList targets;
        for (const QArg& a : qa.items) targets.insert(targets.end(), a.q.begin(), a.q.end());
        if ((Eigen::Index{1} << targets.size()) != m.m.rows())
            fail("E5009", e.span, std::format("a {}×{} matrix needs {} qubit{}, given {}", m.m.rows(), m.m.rows(), m.nq(),
                                              m.nq() == 1 ? "" : "s", targets.size()));
        if (targets.size() > Qputer::QuantumGate::kMaxDenseTargets)
            fail("E5009", e.span, std::format("dense matrices act on at most {} qubits", Qputer::QuantumGate::kMaxDenseTargets));
        checkDistinct(targets, e.span);
        IrOp o;
        o.kind = GK::Matrix;
        o.targets = targets;
        o.matrix = std::make_shared<const Eigen::MatrixXcd>(m.m);
        o.unitaryMatrix = isUnitaryMatrix(m.m);
        o.span = e.span;
        OpV op;
        op.ops.push_back(std::move(o));
        return op;
    }
    if (isOperator(b)) fail("E4001", e.span, "this operator already acts on explicit qubits");
    fail("E4001", e.span, std::format("{} cannot take a subscript", typeName(b.type())));
}

bool Compiler::Impl::oneQubitGate(const GateV& g) const
{
    if (g.kind == GateV::Kind::Builtin) return g.builtin != Builtin::C && infoOf(g.builtin).roles == "1";
    if (g.kind == GateV::Kind::Def || g.kind == GateV::Kind::LetGate)
    {
        const SubParams& sub = g.kind == GateV::Kind::LetGate ? g.letGate->sub : g.def->sub;
        return sub.params.size() == 1 && sub.params[0].sizeName.empty();
    }
    return false;
}

OpV Compiler::Impl::applyGate(const GateV& g, const QArgs& qa, Span at)
{
    if (g.tensorPow > 1)
    {
        QubitList all;
        for (const QArg& a : qa.items) all.insert(all.end(), a.q.begin(), a.q.end());
        if (all.size() != static_cast<std::size_t>(g.tensorPow))
            fail("E5009", at, std::format("`{}^⊗{}` acts on {} qubits, given {}", g.name, g.tensorPow, g.tensorPow, all.size()));
        checkDistinct(all, at);
        GateV one = g;
        one.tensorPow = 1;
        OpV out;
        for (const Qubit q : all)
        {
            QArgs single;
            single.items.push_back(QArg{{q}, true, false, at});
            single.span = at;
            OpV part = applyGate(one, single, at);
            out.ops.insert(out.ops.end(), part.ops.begin(), part.ops.end());
            out.gphase = out.gphase + part.gphase;
        }
        return out;
    }
    OpV op;
    switch (g.kind)
    {
        case GateV::Kind::Builtin: op = applyBuiltin(g, qa, at); break;
        case GateV::Kind::Def:
        case GateV::Kind::Proc:
        case GateV::Kind::LetGate: op = expandDef(g, qa, at); break;
        case GateV::Kind::Channel: return applyChannel(g, qa, at);
        default: fail("E4001", at, std::format("`{}` cannot be applied to qubits", g.name));
    }
    if (g.dagger) op = daggerOp(op, at);
    if (g.power) op = powerOp(op, *g.power, at);
    return op;
}

OpV Compiler::Impl::applyBuiltin(const GateV& g, const QArgs& qa, Span at)
{
    const BuiltinInfo& info = infoOf(g.builtin);
    const std::string name = builtinName(g.builtin);
    if (g.args.size() != info.params)
        fail("E4001", at, info.params == 1 ? std::format("`{}` needs an angle, e.g. {}(π/2)_0", name, name)
                                           : std::format("`{}` takes {} parameters", name, info.params));
    std::vector<Affine> angles;
    for (const auto& a : g.args) angles.push_back(needAngle(*a, at, "a gate angle"));
    const GK kind = kindOf(g.builtin, false);

    OpV out;
    auto make = [&](QubitList targets)
    {
        checkDistinct(targets, at);
        IrOp o;
        o.kind = kind;
        o.targets = std::move(targets);
        o.angles = angles;
        o.span = at;
        out.ops.push_back(std::move(o));
    };

    if (info.roles == "1")
    {
        for (const Tok s : qa.seps)
            if (s != Tok::Comma) fail("E2008", at, std::format("`{}` acts on single qubits; separate them with commas", name));
        for (const QArg& a : qa.items)
            for (const Qubit q : a.q) make({q});
        return out;
    }

    std::vector<Tok> want;
    if (info.roles == "c>t") want = {Tok::Arrow};
    else if (info.roles == "a,b") want = {Tok::Comma};
    else if (info.roles == "c,c>t") want = {Tok::Comma, Tok::Arrow};
    else if (info.roles == "c>a,b") want = {Tok::Arrow, Tok::Comma};
    if (qa.items.size() != want.size() + 1)
        fail("E5009", at, std::format("`{}` takes {} qubit slots ({}_{{{}}})", name, want.size() + 1, name,
                                      info.roles == "c>t" ? "c→t" : info.roles == "a,b" ? "a,b" : info.roles == "c,c>t" ? "c,c→t" : "c→a,b"));
    if (qa.seps != want)
    {
        std::string fixed;
        const SourceFile& f = sm.file(at.file);
        for (std::size_t k = 0; k < qa.items.size(); ++k)
        {
            if (k) fixed += want[k - 1] == Tok::Arrow ? "→" : ",";
            fixed += f.slice(qa.items[k].span);
        }
        Diagnostic& diag = d.error("E2008", at, info.roles.find('>') != std::string_view::npos
                                                   ? std::format("`{}` needs `→` between control and target", name)
                                                   : std::format("`{}` is symmetric: separate its qubits with commas", name));
        const Span inner{at.file, qa.items.front().span.begin, qa.items.back().span.end};
        diag.fix("use the role syntax", inner, fixed);
        throw CompileAbort{};
    }
    std::size_t len = 1;
    for (const QArg& a : qa.items)
        if (!a.single && a.q.size() != 1)
        {
            if (len != 1 && a.q.size() != len) fail("E5009", at, "broadcast registers must have equal sizes");
            len = a.q.size();
        }
    for (std::size_t k = 0; k < len; ++k)
    {
        QubitList t;
        for (const QArg& a : qa.items) t.push_back(a.q.size() == 1 ? a.q[0] : a.q[k]);
        make(std::move(t));
    }
    return out;
}

OpV Compiler::Impl::expandDef(const GateV& g, const QArgs& qa, Span at)
{
    const bool isLet = g.kind == GateV::Kind::LetGate;
    const SubParams& sub = isLet ? g.letGate->sub : g.def->sub;
    const Span defSpan = isLet ? g.letGate->nameSpan : g.def->span;
    if (g.kind == GateV::Kind::Proc && ctx.back() == Ctx::Def)
        fail("E5006", at, std::format("a def is unitary and cannot call the proc `{}`", g.name));
    if (!isLet && g.args.size() != g.def->cparams.size())
        fail("E4001", at, std::format("`{}` takes {} parameter{}", g.name, g.def->cparams.size(), g.def->cparams.size() == 1 ? "" : "s"));
    if (qa.items.size() != sub.params.size())
        fail("E5009", at, std::format("`{}` takes {} qubit argument{}", g.name, sub.params.size(), sub.params.size() == 1 ? "" : "s"));
    if (qa.seps != sub.seps)
    {
        std::string sig;
        for (std::size_t k = 0; k < sub.params.size(); ++k)
            sig += (k ? (sub.seps[k - 1] == Tok::Arrow ? "→" : ",") : "") + sub.params[k].name;
        fail("E2008", at, std::format("`{}` is declared {}_{{{}}}; use the same separators", g.name, g.name, sig));
    }
    const Stmt* key = isLet ? g.letGate == nullptr ? nullptr : reinterpret_cast<const Stmt*>(g.letGate) : g.def;
    if (std::ranges::find(expanding, key) != expanding.end())
        fail("E3004", at, std::format("`{}` refers to itself; recursion is not allowed", g.name));

    // Broadcast: single-qubit slots given registers apply elementwise.
    std::size_t len = 1;
    for (std::size_t k = 0; k < sub.params.size(); ++k)
    {
        const QArg& a = qa.items[k];
        if (sub.params[k].sizeName.empty() && !a.single && a.q.size() != 1)
        {
            if (len != 1 && a.q.size() != len) fail("E5009", at, "broadcast registers must have equal sizes");
            len = a.q.size();
        }
    }

    OpV out;
    for (std::size_t l = 0; l < len; ++l)
    {
        ConstScopePtr parent = isLet ? g.scope : ConstScopePtr(owner.global);
        auto scope = std::make_shared<Scope>(parent, g.kind == GateV::Kind::Proc ? Scope::Boundary::Proc : Scope::Boundary::Def,
                                             isLet ? Span{} : defSpan);
        QubitList all;
        for (std::size_t k = 0; k < sub.params.size(); ++k)
        {
            const QParam& p = sub.params[k];
            const QArg& a = qa.items[k];
            QubitsV q;
            if (p.sizeName.empty())
            {
                q.q = {a.q.size() == 1 ? a.q[0] : a.q[l]};
                q.single = true;
            }
            else
            {
                q.q = a.q;
                if (std::isdigit(static_cast<unsigned char>(p.sizeName[0])))
                {
                    if (std::to_string(a.q.size()) != p.sizeName)
                        fail("E5009", a.span, std::format("`{}` needs a register of size {} for `{}`, given {}", g.name, p.sizeName, p.name, a.q.size()));
                }
                else
                {
                    Symbol n;
                    n.kind = Symbol::Kind::CParam;
                    n.value = numInt(static_cast<std::int64_t>(a.q.size()));
                    n.used = true;
                    scope->define(p.sizeName, std::move(n));
                }
            }
            all.insert(all.end(), q.q.begin(), q.q.end());
            Symbol s;
            s.kind = Symbol::Kind::QParam;
            s.value = q;
            s.used = true;
            s.span = p.span;
            scope->define(p.name, std::move(s));
        }
        checkDistinct(all, at);
        if (!isLet)
            for (std::size_t k = 0; k < g.def->cparams.size(); ++k)
            {
                Symbol s;
                s.kind = Symbol::Kind::CParam;
                s.value = *g.args[k];
                s.used = true;
                scope->define(g.def->cparams[k], std::move(s));
            }

        expanding.push_back(key);
        ctx.push_back(g.kind == GateV::Kind::Proc ? Ctx::Proc : Ctx::Def);
        Sink inner;
        inner.ctx = ctx.back();
        try
        {
            if (isLet)
            {
                const Value v = eval(*g.letGate->value, scope);
                if (!isOperator(v))
                    fail("E4001", g.letGate->value->span, std::format("`{}` must be an operation, found {}", g.name, typeName(v.type())));
                const OpV body = toOp(v, g.letGate->value->span);
                emitOps(inner, body, at);
            }
            else block(g.def->body, scope, inner);
        }
        catch (...)
        {
            ctx.pop_back();
            expanding.pop_back();
            throw;
        }
        ctx.pop_back();
        expanding.pop_back();
        out.ops.insert(out.ops.end(), inner.ops.begin(), inner.ops.end());
    }
    return out;
}

OpV Compiler::Impl::applyChannel(const GateV& g, const QArgs& qa, Span at)
{
    if (ctx.back() == Ctx::Def) fail("E5006", at, "a def is unitary; channels belong in a proc or at top level");
    OpV out;
    auto make = [&](QubitList targets)
    {
        checkDistinct(targets, at);
        IrOp o;
        o.kind = GK::Channel;
        o.channel = g.channel;
        o.targets = std::move(targets);
        o.span = at;
        out.ops.push_back(std::move(o));
    };
    const std::size_t arity = g.channel->arity;
    if (arity == 1)
    {
        for (const QArg& a : qa.items)
            for (const Qubit q : a.q) make({q});
        return out;
    }
    if (g.channel->pauli)
    {
        if (qa.items.size() != 2) fail("E5009", at, std::format("`{}` acts on two qubits: {}(p)_{{a,b}}", g.name, g.name));
        std::size_t len = 1;
        for (const QArg& a : qa.items)
            if (a.q.size() != 1)
            {
                if (len != 1 && a.q.size() != len) fail("E5009", at, "broadcast registers must have equal sizes");
                len = a.q.size();
            }
        for (std::size_t k = 0; k < len; ++k)
            make({qa.items[0].q.size() == 1 ? qa.items[0].q[0] : qa.items[0].q[k],
                  qa.items[1].q.size() == 1 ? qa.items[1].q[0] : qa.items[1].q[k]});
        return out;
    }
    QubitList targets;
    for (const QArg& a : qa.items) targets.insert(targets.end(), a.q.begin(), a.q.end());
    if (targets.size() != arity)
        fail("E5009", at, std::format("these Kraus operators act on {} qubit{}, given {}", arity, arity == 1 ? "" : "s", targets.size()));
    make(std::move(targets));
    return out;
}

OpV Compiler::Impl::daggerOp(const OpV& op, Span at)
{
    OpV out;
    out.gphase = -op.gphase;
    for (auto it = op.ops.rbegin(); it != op.ops.rend(); ++it)
    {
        IrOp o = *it;
        switch (o.kind)
        {
            case GK::S: o.kind = GK::Sdg; break;
            case GK::Sdg: o.kind = GK::S; break;
            case GK::T: o.kind = GK::Tdg; break;
            case GK::Tdg: o.kind = GK::T; break;
            case GK::SX: o.kind = GK::SXdg; break;
            case GK::SXdg: o.kind = GK::SX; break;
            case GK::RX: case GK::RY: case GK::RZ: case GK::P: case GK::CP: case GK::PauliRot: case GK::GPhase:
                for (Affine& a : o.angles) a = -a;
                break;
            case GK::U3: o.angles = {-o.angles[0], -o.angles[2], -o.angles[1]}; break;
            case GK::Matrix: o.matrix = std::make_shared<const Eigen::MatrixXcd>(o.matrix->adjoint()); break;
            case GK::MeasureZ: case GK::MeasurePauli: case GK::Reset: case GK::Channel:
                fail("E5006", at, "measurements, resets and channels have no adjoint");
            default: break;
        }
        out.ops.push_back(std::move(o));
    }
    return out;
}

OpV Compiler::Impl::powerOp(const OpV& op, const Num& n, Span at)
{
    if (!op.unitary()) fail("E5006", at, "only unitary operations can be raised to a power");
    if (n.nonlinear) fail("E4005", at, "a power must depend on params affinely");
    if (!n.im.isZero()) fail("E4001", at, "a power must be real");
    // A single rotation (or T/S, which are phase rotations) scales its angle: exact for any real power.
    if (op.ops.size() == 1 && op.gphase.isZero())
    {
        IrOp o = op.ops[0];
        auto scaleAngles = [&](IrOp& x)
        {
            for (Affine& a : x.angles)
            {
                if (!a.isConst() && !n.re.isConst()) fail("E4005", at, "a param angle raised to a param power is not affine");
                a = a.isConst() ? n.re.scaled(a.c) : a.scaled(n.re.c);
            }
        };
        if (isRotation(o.kind))
        {
            scaleAngles(o);
            OpV out;
            out.ops.push_back(std::move(o));
            return out;
        }
        if (o.kind == GK::T || o.kind == GK::Tdg || o.kind == GK::S || o.kind == GK::Sdg || o.kind == GK::Z)
        {
            const Rational base = o.kind == GK::T ? Rational{1, 4} : o.kind == GK::Tdg ? Rational{-1, 4}
                                : o.kind == GK::S ? Rational{1, 2} : o.kind == GK::Sdg ? Rational{-1, 2} : Rational{1, 1};
            o.kind = GK::P;
            o.angles = {Affine(Real::pi(base))};
            scaleAngles(o);
            OpV out;
            out.ops.push_back(std::move(o));
            return out;
        }
    }
    if (!n.isConst()) fail("E4005", at, "only a single rotation can be raised to a param power");
    const auto k = n.asInt();
    if (!k) fail("E4001", at, "only a single rotation can be raised to a non-integer power");
    if (*k == 0) return OpV{};
    const OpV base = *k < 0 ? daggerOp(op, at) : op;
    const std::int64_t m = *k < 0 ? -*k : *k;
    if (static_cast<double>(m) * static_cast<double>(base.ops.size()) > 1e6)
        fail("E4007", at, std::format("power {} would repeat {} operations; use a matrix or a rotation instead", m, base.ops.size()));
    // A single dense matrix multiplies out instead of repeating.
    if (base.ops.size() == 1 && base.ops[0].kind == GK::Matrix && base.ops[0].controls.empty() && base.ops[0].negControls.empty())
    {
        Eigen::MatrixXcd x = *base.ops[0].matrix, r = Eigen::MatrixXcd::Identity(x.rows(), x.cols());
        std::int64_t e = m;
        while (e > 0)
        {
            if (e & 1) r = x * r;
            x = x * x;
            e >>= 1;
        }
        OpV out = base;
        out.ops[0].matrix = std::make_shared<const Eigen::MatrixXcd>(r);
        out.gphase = base.gphase.scaled(Real::integer(m));
        return out;
    }
    OpV out;
    for (std::int64_t r = 0; r < m; ++r) out.ops.insert(out.ops.end(), base.ops.begin(), base.ops.end());
    out.gphase = base.gphase.scaled(Real::integer(m));
    return out;
}

OpV Compiler::Impl::controlled(const ControlV& cv, const OpV& op, Span at)
{
    if (!op.unitary()) fail("E5006", at, "C_{…}(…) needs a unitary operation; a proc, measurement or channel has no controlled form");
    const QubitList sup = supportOf(op);
    for (const Qubit q : sup)
        if (std::ranges::find(cv.pos, q) != cv.pos.end() || std::ranges::find(cv.neg, q) != cv.neg.end())
            fail("E5001", at, std::format("qubit {} is both a control and a target of C_{{…}}(…)", q));
    OpV out;
    for (IrOp o : op.ops)
    {
        o.controls.insert(o.controls.end(), cv.pos.begin(), cv.pos.end());
        o.negControls.insert(o.negControls.end(), cv.neg.begin(), cv.neg.end());
        out.ops.push_back(std::move(o));
    }
    if (!op.gphase.isZero())
    {
        IrOp ph;
        ph.kind = GK::GPhase;
        ph.angles = {op.gphase};
        ph.controls = cv.pos;
        ph.negControls = cv.neg;
        ph.span = at;
        out.ops.push_back(std::move(ph));
    }
    return out;
}

IrOp Compiler::Impl::pauliRotation(const PTerm& term, const Affine& angle, Span at)
{
    IrOp o;
    o.span = at;
    o.angles = {angle};
    if (term.ps.size() == 1)
    {
        const char c = term.ps[0].second;
        o.kind = c == 'X' ? GK::RX : c == 'Y' ? GK::RY : GK::RZ;
        o.targets = {term.ps[0].first};
        return o;
    }
    o.kind = GK::PauliRot;
    for (const auto& [q, c] : term.ps)
    {
        o.targets.push_back(q);
        o.paulis += c;
    }
    return o;
}

OpV Compiler::Impl::expOperator(const LinOpV& a, Span at)
{
    if (!a.general.empty())
    {
        // Dense case: exp(A) for anti-Hermitian A on a small support (state vector only).
        QubitList support;
        const Eigen::MatrixXcd m = denseOf(a, support, at);
        if ((m + m.adjoint()).cwiseAbs().maxCoeff() > 1e-12)
            fail("E5005", at, "exp of an operator needs -i·r·O with O Hermitian");
        const Eigen::MatrixXcd h = cd(0, 1) * m; // A = -iH
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(h);
        Eigen::VectorXcd phases(es.eigenvalues().size());
        for (Eigen::Index k = 0; k < phases.size(); ++k) phases[k] = std::polar(1.0, -es.eigenvalues()[k]);
        IrOp o;
        o.kind = GK::Matrix;
        o.targets = support;
        o.matrix = std::make_shared<const Eigen::MatrixXcd>(es.eigenvectors() * phases.asDiagonal() * es.eigenvectors().adjoint());
        o.span = at;
        OpV out;
        out.ops.push_back(std::move(o));
        return out;
    }
    for (std::size_t x = 0; x < a.paulis.size(); ++x)
        for (std::size_t y = x + 1; y < a.paulis.size(); ++y)
            if (!commute(a.paulis[x], a.paulis[y]))
                fail("E5005", at, "exp of a sum of non-commuting Pauli strings is not exact; use trotter(Ham, t, steps=n, order=2)");
    OpV out;
    for (const PTerm& t : a.paulis)
    {
        if (t.coef.nonlinear) fail("E4005", at, "the exponent must depend on params affinely");
        const bool reZero = t.coef.re.isZero() || (t.coef.re.isConst() && !t.coef.re.c.exact() && std::abs(t.coef.re.c.value()) <= 1e-14);
        if (!reZero) fail("E5005", at, "exp of an operator needs an anti-Hermitian exponent -i·r·O (r real)");
        // c = -iα  →  exp(-iα P) = R_P(2α), α = -Im c.
        const Affine alpha = -t.coef.im;
        if (t.ps.empty())
        {
            out.gphase = out.gphase - alpha;
            continue;
        }
        out.ops.push_back(pauliRotation(t, alpha.scaled(Real::integer(2)), at));
    }
    return out;
}

OpV Compiler::Impl::trotter(const LinOpV& ham, const Num& t, std::int64_t steps, std::int64_t order, Span at)
{
    if (!ham.general.empty()) fail("E4001", at, "trotter needs a sum of Pauli strings");
    std::vector<PTerm> terms;
    Num idCoef = numInt(0);
    for (const PTerm& p : ham.paulis)
    {
        if (!p.coef.im.isZero()) fail("E5004", at, "trotter needs a Hermitian Hamiltonian (real coefficients)");
        if (p.ps.empty()) idCoef = idCoef + p.coef;
        else terms.push_back(p);
    }
    if (!t.im.isZero()) fail("E4001", at, "the evolution time must be real");

    // Sequence of (term, fraction of one step), merged where neighbours repeat a term.
    std::vector<std::pair<std::size_t, Real>> seq;
    auto push = [&](std::size_t k, Real f)
    {
        if (!seq.empty() && seq.back().first == k) seq.back().second = seq.back().second + f;
        else seq.emplace_back(k, f);
    };
    const std::size_t m = terms.size();
    auto s2 = [&](const Real& f)
    {
        const Real half = f / Real::integer(2);
        for (std::size_t k = 0; k + 1 < m; ++k) push(k, half);
        if (m) push(m - 1, f);
        for (std::size_t k = m - 1; k-- > 0;) push(k, half);
    };
    for (std::int64_t s = 0; s < steps; ++s)
    {
        if (order == 1)
            for (std::size_t k = 0; k < m; ++k) push(k, Real::integer(1));
        else if (order == 2) s2(Real::integer(1));
        else
        {
            const double p = 1.0 / (4.0 - std::cbrt(4.0));
            const Real rp = Real::approx(p), rq = Real::approx(1.0 - 4.0 * p);
            s2(rp); s2(rp); s2(rq); s2(rp); s2(rp);
        }
    }
    OpV out;
    const Num dt = *Noether::divide(t, numInt(steps));
    for (const auto& [k, f] : seq)
    {
        const Num ang = terms[k].coef * dt * Num(f) * numInt(2);
        if (ang.nonlinear) fail("E4005", at, "trotter angles must depend on params affinely");
        out.ops.push_back(pauliRotation(terms[k], ang.re, at));
    }
    const Num ph = -(idCoef * t);
    if (!ph.nonlinear) out.gphase = ph.re;
    return out;
}

void Compiler::Impl::readout(std::string kind, bool svOnly, Span at, std::size_t terms, std::size_t width)
{
    if (rt || !readoutContext) return;
    ir.readouts.push_back({at, std::move(kind), svOnly, terms, width});
}

} // namespace Noether
