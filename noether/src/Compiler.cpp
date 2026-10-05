#include "CompilerImpl.hpp"

#include "Backend.hpp"
#include "Parser.hpp"
#include "Sha256.hpp"
#include "StdLib.hpp"

#include <StabilizerState.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <set>
#include <sstream>



namespace Noether
{

// ---- Scope ----

std::pair<const Symbol*, const Scope*> Scope::find(std::string_view name) const
{
    const Scope* s = this;
    Boundary crossed = Boundary::None;
    Span def;
    while (s)
    {
        if (const auto it = s->names.find(name); it != s->names.end())
        {
            const Symbol& sym = it->second;
            if (crossed != Boundary::None && s->isGlobal())
            {
                using K = Symbol::Kind;
                if (sym.kind == K::QReg || sym.kind == K::Runtime) return {nullptr, nullptr};
                if (sym.kind == K::BReg && crossed == Boundary::Def) return {nullptr, nullptr};
                if (sym.span.file == def.file && sym.span.begin >= def.begin && sym.kind != K::BReg)
                    return {nullptr, nullptr};
            }
            return {&sym, s};
        }
        if (s->boundary != Boundary::None)
        {
            crossed = s->boundary;
            def = s->defSpan;
        }
        s = s->parent.get();
    }
    return {nullptr, nullptr};
}

const Symbol* Scope::localLookup(std::string_view name) const
{
    const auto it = names.find(name);
    return it == names.end() ? nullptr : &it->second;
}

Symbol& Scope::define(const std::string& name, Symbol s)
{
    return names.insert_or_assign(name, std::move(s)).first->second;
}


// ---- Errors ----

void Compiler::Impl::fail(std::string code, Span s, std::string msg)
{
    if (rt) throw RuntimeError{std::move(code), std::move(msg), s};
    d.fatal(std::move(code), s, std::move(msg));
}

void Compiler::Impl::warn(std::string code, Span s, std::string msg)
{
    if (!rt) d.warning(std::move(code), s, std::move(msg));
}

namespace
{
    const std::set<std::string_view> kReserved{
        "i", "e", "π", "ψ", "ρ",
        "I", "X", "Y", "Z", "H", "S", "T", "Rx", "Ry", "Rz", "P", "U3", "CNOT", "CZ", "CP", "SWAP", "Toffoli",
        "Fredkin", "C",
        "sin", "cos", "tan", "exp", "log", "sqrt", "abs", "floor", "ceil", "min", "max", "trotter", "entropy",
        "fidelity", "marginal", "count", "depth",
        "depolarize", "depolarize2", "dephase", "flip", "pauli", "ampdamp", "kraus"};

    std::uint32_t lineOf(const SourceManager& sm, Span s)
    {
        return s.file < sm.size() ? sm.file(s.file).lineCol(s.begin).line : 0;
    }

    bool isMeasure(const IrOp& op) { return op.kind == GK::MeasureZ || op.kind == GK::MeasurePauli; }
} // namespace

void Compiler::Impl::checkBindable(const std::string& name, Span at)
{
    if (kReserved.contains(name))
        fail("E3002", at, std::format("`{}` is reserved and cannot be bound{}", name,
                                      name == "i" ? "; use k or j as a loop variable"
                                      : name == "H" ? "; it is the Hadamard (name Hamiltonians Ham or 𝓗)"
                                                    : ""));
}

Symbol& Compiler::Impl::define(const ScopePtr& sc, const std::string& name, Symbol s, Span at)
{
    checkBindable(name, at);
    if (const Symbol* prev = sc->localLookup(name))
    {
        Diagnostic& diag = d.error("E3003", at, std::format("`{}` is already bound in this scope", name));
        diag.note(prev->span, "first bound here");
        throw CompileAbort{};
    }
    if (!sc->isGlobal())
        if (const Symbol* outer = sc->parentScope() ? sc->parentScope()->lookup(name) : nullptr)
            warn("W0003", at, std::format("`{}` shadows the binding at line {}", name, lineOf(sm, outer->span)));
    s.span = at;
    s.global = sc->isGlobal();
    return sc->define(name, std::move(s));
}

int Compiler::Impl::newSlot() { return ir.slots++; }


// ---- Statements ----

void Compiler::Impl::block(const Block& b, const ScopePtr& sc, Sink& sink)
{
    for (const StmtPtr& s : b)
    {
        try
        {
            stmt(*s, sc, sink);
        }
        catch (const CompileAbort&)
        {
        }
    }
}

void Compiler::Impl::stmt(const Stmt& s, const ScopePtr& sc, Sink& sink)
{
    const bool top = sink.ctx == Ctx::Top && sc->isGlobal();
    auto topOnly = [&](std::string_view what)
    {
        if (sink.ctx != Ctx::Top)
            fail("E2001", s.head, std::format("`{}` is only allowed at top level, not inside a def or proc", what));
    };
    // Readouts and runs happen on the live pass, outside the condition cubes of a run-time `if`.
    auto unguarded = [&](std::string_view what)
    {
        if (sink.guard)
            fail("E5012", s.head, std::format("`{}` inside a run-time `if` is not supported; only operations can be conditioned", what));
    };
    auto globalOnly = [&](std::string_view what)
    {
        if (!top) fail("E2001", s.head, std::format("`{}` must appear at the top level of the file", what));
    };

    switch (s.kind)
    {
        case SK::Header: return;
        case SK::Qubits:
        case SK::Bits: globalOnly(s.kind == SK::Qubits ? "qubits" : "bits"); return regStmt(s, sc);
        case SK::Seed:
        {
            globalOnly("seed");
            if (seedSet) fail("E3003", s.head, "the seed is already set");
            seedSet = true;
            const std::int64_t v = needInt(eval(*s.expr, sc), s.expr->span, "a seed");
            if (v < 0 || v >= (std::int64_t{1} << 53)) fail("E4003", s.expr->span, "the seed must be an integer in [0, 2^53)");
            ir.seed = static_cast<std::uint64_t>(v);
            return;
        }
        case SK::Backend:
            globalOnly("backend");
            if (backendSet) fail("E3003", s.head, "the backend is already set");
            backendSet = true;
            ir.backendRequest = s.text;
            return;
        case SK::Trajectories:
        {
            globalOnly("trajectories");
            if (trajSet) fail("E3003", s.head, "trajectories is already set");
            trajSet = true;
            const std::int64_t v = needInt(eval(*s.expr, sc), s.expr->span, "a trajectory count");
            if (v < 1) fail("E4003", s.expr->span, "trajectories must be at least 1");
            ir.trajectories = static_cast<std::size_t>(v);
            return;
        }
        case SK::Let: return letStmt(s, sc, sink);
        case SK::Param: globalOnly("param"); return paramStmt(s, sc);
        case SK::Prepare: topOnly("prepare"); unguarded("prepare"); return prepareStmt(s, sc);
        case SK::Measure: return measureStmt(s, sc, sink);
        case SK::Run: topOnly("run"); unguarded("run"); return runStmt(s, sc);
        case SK::Reset:
        {
            if (sink.ctx == Ctx::Def)
                fail("E5006", s.head, "a def is unitary; reset is only allowed in a proc or at top level");
            const QArgs qa = qargs(s.qlist, sc, s.span);
            for (const QArg& a : qa.items)
                for (const Qubit q : a.q)
                {
                    IrOp op;
                    op.kind = GK::Reset;
                    op.targets = {q};
                    op.span = s.span;
                    emit(sink, std::move(op));
                }
            return;
        }
        case SK::Print: topOnly("print"); unguarded("print"); return printStmt(s, sc);
        case SK::Assert: topOnly("assert"); unguarded("assert"); return assertStmt(s, sc);
        case SK::Import: globalOnly("import"); return importStmt(s, sc);
        case SK::Apply: return applyStmt(s, sc, sink);
        case SK::For: return forStmt(s, sc, sink);
        case SK::If: return ifStmt(s, sc, sink);
        case SK::Def:
        case SK::Proc: globalOnly(s.kind == SK::Def ? "def" : "proc"); return defStmt(s, sc);
        case SK::Noise: globalOnly("noise"); return noiseStmt(s, sc);
        case SK::Spec: return specStmt(s, sc);
    }
}

void Compiler::Impl::regStmt(const Stmt& s, const ScopePtr& sc)
{
    const bool qubits = s.kind == SK::Qubits;
    for (const RegDecl& r : s.regs)
    {
        const std::int64_t n = needInt(eval(*r.size, sc), r.size->span, "a register size");
        if (n < 1) fail("E4003", r.size->span, "a register needs at least one element");
        RegInfo info{r.name, qubits ? ir.nQubits : ir.nClbits, static_cast<std::size_t>(n), r.span};
        Symbol sym;
        if (qubits)
        {
            if (ir.nQubits + info.size > Qputer::kMaxStabilizerQubits)
                fail("E6003", r.span, std::format("{} qubits exceed the largest backend ({} qubits on the stabilizer "
                                                  "tableau)", ir.nQubits + info.size, Qputer::kMaxStabilizerQubits));
            QubitsV q;
            for (std::size_t k = 0; k < info.size; ++k) q.q.push_back(info.offset + k);
            sym.kind = Symbol::Kind::QReg;
            sym.value = q;
            ir.nQubits += info.size;
            ir.qregs.push_back(info);
        }
        else
        {
            if (ir.nClbits + info.size > 64)
                fail("E6003", r.span, std::format("{} classical bits exceed the limit of 64", ir.nClbits + info.size));
            BitsV b;
            b.reg = r.name;
            for (std::size_t k = 0; k < info.size; ++k) b.clbits.push_back(info.offset + k);
            sym.kind = Symbol::Kind::BReg;
            sym.value = b;
            ir.nClbits += info.size;
            ir.bregs.push_back(info);
        }
        define(sc, r.name, std::move(sym), r.span);
    }
}

void Compiler::Impl::letStmt(const Stmt& s, const ScopePtr& sc, Sink& sink)
{
    for (const Binding& b : s.bindings)
    {
        if (b.hasSub)
        {
            // `let a_b = 1`: a subscripted let is a gate definition, so an expression that never uses its
            // qubit parameter is a name with an underscore.
            if (!b.sub.braced && b.sub.params.size() == 1 && b.sub.params[0].sizeName.empty())
            {
                const std::string& qp = b.sub.params[0].name;
                bool mentions = false;
                auto walk = [&](auto&& self, const Expr& e) -> void
                {
                    if (e.kind == EK::Name && e.name == qp) mentions = true;
                    for (const QItem& it : e.qlist.items) self(self, *it.expr);
                    for (const ExprPtr& k : e.kids) self(self, *k);
                };
                walk(walk, *b.value);
                if (!mentions)
                {
                    std::string camel = b.name + qp;
                    if (!qp.empty()) camel[b.name.size()] = static_cast<char>(std::toupper(static_cast<unsigned char>(qp[0])));
                    d.error("E3005", Span::join(b.nameSpan, b.sub.params[0].span),
                            std::format("names cannot contain `_` (it always means subscript); `{}_{}` is a gate "
                                        "definition", b.name, qp))
                        .fix("use camelCase", Span::join(b.nameSpan, b.sub.params[0].span), camel);
                    throw CompileAbort{};
                }
            }
            Symbol sym;
            sym.kind = Symbol::Kind::LetGate;
            sym.binding = &b;
            sym.stmt = &s;
            define(sc, b.name, std::move(sym), b.nameSpan);
            continue;
        }

        const std::size_t readoutsBefore = ir.readouts.size();
        readoutContext = true;
        Value v;
        try
        {
            const auto over = owner.opt.lets.find(b.name);
            if (over != owner.opt.lets.end() && sc->isGlobal() && s.span.file == 0)
            {
                const std::string& text = over->second;
                std::int64_t iv = 0;
                const auto r = std::from_chars(text.data(), text.data() + text.size(), iv);
                if (r.ec == std::errc{} && r.ptr == text.data() + text.size()) v = Num(Real::integer(iv));
                else if (const auto q = parseDecimal(text)) v = Num(Real::rational(*q));
                else fail("E4001", b.nameSpan, std::format("instance value `{}` for `{}` is not a number", text, b.name));
            }
            else v = eval(*b.value, sc);
            noteStateReadout(v, b.value ? b.value->span : b.nameSpan);
        }
        catch (...)
        {
            readoutContext = false;
            throw;
        }
        readoutContext = false;

        Symbol sym;
        sym.stmt = &s;
        if (v.stage() == Stage::Run)
        {
            if (sink.ctx != Ctx::Top)
                fail("E4004", b.value->span, "a value known only at run time cannot be bound inside a def or proc");
            sym.kind = Symbol::Kind::Runtime;
            sym.slot = newSlot();
            sym.value = DeferredV{v.type(), Stage::Run};
            Event ev;
            ev.kind = EvK::Let;
            ev.span = s.span;
            ev.expr = b.value;
            ev.scope = sc;
            ev.slot = sym.slot;
            ev.name = b.name;
            ir.events.push_back(std::move(ev));
            if (noiseEmitted && ir.readouts.size() > readoutsBefore) noisyReadouts.push_back(b.value->span);
        }
        else
        {
            ir.readouts.resize(readoutsBefore);
            sym.kind = Symbol::Kind::Let;
            sym.binding = &b; // re-evaluated at run time when the value depends on params non-affinely
            sym.value = std::move(v);
        }
        define(sc, b.name, std::move(sym), b.nameSpan);
    }
}

void Compiler::Impl::paramStmt(const Stmt& s, const ScopePtr& sc)
{
    ParamInfo p;
    p.name = s.name;
    p.span = s.nameSpan;
    p.doc = s.doc;
    p.base = static_cast<std::uint32_t>(ir.paramNames.size());
    if (s.paramSize)
    {
        const std::int64_t n = needInt(eval(*s.paramSize, sc), s.paramSize->span, "a param size");
        if (n < 1) fail("E4003", s.paramSize->span, "a param vector needs at least one element");
        p.size = static_cast<std::size_t>(n);
        p.isVector = true;
    }
    p.lo = needReal(eval(*s.lo, sc), s.lo->span, "the interval's lower bound");
    p.hi = needReal(eval(*s.hi, sc), s.hi->span, "the interval's upper bound");
    if (!(p.lo <= p.hi)) fail("E4001", Span::join(s.lo->span, s.hi->span), "the interval needs lo ≤ hi");

    std::vector<double> init(p.size, p.lo <= 0.0 && 0.0 <= p.hi ? 0.0 : p.lo);
    if (s.init)
    {
        const Value iv = eval(*s.init, sc);
        if (const auto* l = iv.get<ListV>())
        {
            if (l->items.size() != p.size)
                fail("E5009", s.init->span, std::format("{} initial values for a param of size {}", l->items.size(), p.size));
            for (std::size_t k = 0; k < p.size; ++k) init[k] = needReal(*l->items[k], s.init->span, "an initial value");
        }
        else std::ranges::fill(init, needReal(iv, s.init->span, "an initial value"));
    }
    for (const double v : init)
        if (v < p.lo || v > p.hi) fail("E4001", s.init ? s.init->span : s.span, std::format("initial value {} lies outside [{}, {}]", v, p.lo, p.hi));
    p.init = init;

    Symbol sym;
    sym.kind = Symbol::Kind::Param;
    if (p.isVector)
    {
        ListV l;
        for (std::size_t k = 0; k < p.size; ++k)
        {
            l.items.push_back(std::make_shared<Value>(Num(Affine::param(p.base + static_cast<std::uint32_t>(k)))));
            ir.paramNames.push_back(std::format("{}[{}]", p.name, k));
        }
        sym.value = l;
    }
    else
    {
        sym.value = Num(Affine::param(p.base));
        ir.paramNames.push_back(p.name);
    }
    ir.paramValues.insert(ir.paramValues.end(), init.begin(), init.end());
    ir.params.push_back(p);
    define(sc, s.name, std::move(sym), s.nameSpan);
}

void Compiler::Impl::prepareStmt(const Stmt& s, const ScopePtr& sc)
{
    const Value v = eval(*s.expr, sc);
    const KetV* k = v.get<KetV>();
    if (!k || k->live) fail("E4001", s.expr->span, std::format("prepare needs a ket, found {}", typeName(v.type())));
    if (k->nq() != ir.nQubits)
        fail("E5009", s.expr->span, std::format("the ket has {} qubit{} but {} {} declared", k->nq(), k->nq() == 1 ? "" : "s",
                                                ir.nQubits, ir.nQubits == 1 ? "qubit is" : "qubits are"));
    const double n2 = k->norm2();
    if (std::abs(n2 - 1.0) > 1e-12)
    {
        d.error("E5007", s.expr->span, std::format("the prepared ket has squared norm {} (must be 1 within 1e-12)", n2))
            .fix("divide by the norm", s.expr->span, std::format("({})/√({})", formatExpr(*s.expr, false), n2));
        throw CompileAbort{};
    }
    Event ev;
    ev.kind = EvK::Prepare;
    ev.span = s.span;
    ev.ket = std::make_shared<const KetV>(*k);
    ir.events.push_back(std::move(ev));
    measuredSincePrepare = false;

    for (const NoiseRuleIr& r : noise)
        if (r.after && r.event == "prepare")
        {
            IrOp marker;
            marker.kind = GK::Reset; // only its qubits are read by insertNoise
            for (Qubit q = 0; q < ir.nQubits; ++q) marker.targets.push_back(q);
            NoiseRuleIr only = r;
            only.event = "reset";
            const auto saved = noise;
            noise = {only};
            insertNoise(marker, true);
            noise = saved;
        }
}

void Compiler::Impl::measureStmt(const Stmt& s, const ScopePtr& sc, Sink& sink)
{
    if (sink.ctx == Ctx::Def)
        fail("E5006", s.head, "a def is unitary; measure is only allowed in a proc or at top level");
    if (sink.guard) fail("E5012", s.head, "measurement inside `if` is not supported; measure unconditionally and branch on the result");

    const Value lv = eval(*s.lvalue, sc);
    const BitsV* bits = lv.get<BitsV>();
    if (!bits) fail("E4001", s.lvalue->span, "a measurement result must go to a bit register (declare `bits c[n]`)");

    if (s.measureSub)
    {
        const QArgs qa = qargs(s.qlist, sc, s.span);
        QubitList qs;
        for (const QArg& a : qa.items) qs.insert(qs.end(), a.q.begin(), a.q.end());
        if (qs.size() != bits->clbits.size())
            fail("E5009", s.span, std::format("{} qubit{} measured into {} bit{}", qs.size(), qs.size() == 1 ? "" : "s",
                                              bits->clbits.size(), bits->clbits.size() == 1 ? "" : "s"));
        checkDistinct(qs, s.span);
        for (std::size_t k = 0; k < qs.size(); ++k)
        {
            IrOp op;
            op.kind = GK::MeasureZ;
            op.targets = {qs[k]};
            op.clbit = bits->clbits[k];
            op.span = s.span;
            emit(sink, std::move(op));
        }
        return;
    }

    const Value pv = eval(*s.expr, sc);
    if (!isOperator(pv)) fail("E5010", s.expr->span, std::format("measure needs a Pauli string, found {}", typeName(pv.type())));
    const LinOpV lin = simplify(toLin(pv, s.expr->span));
    if (lin.paulis.size() != 1 || !lin.general.empty() || lin.paulis[0].ps.empty())
        fail("E5010", s.expr->span, "measure needs one Pauli string such as Z_0 Z_1 or X_2");
    const PTerm& t = lin.paulis[0];
    const auto sign = t.coef.asInt();
    if (!sign || (*sign != 1 && *sign != -1))
        fail("E5010", s.expr->span, "a measured Pauli string must have coefficient +1 or -1");
    if (bits->clbits.size() != 1)
        fail("E5009", s.lvalue->span, "a Pauli measurement records one bit; index the register, e.g. c[0]");

    IrOp op;
    op.span = s.span;
    op.clbit = bits->clbits[0];
    if (t.ps.size() == 1 && t.ps[0].second == 'Z' && *sign == 1)
    {
        op.kind = GK::MeasureZ;
        op.targets = {t.ps[0].first};
    }
    else
    {
        op.kind = GK::MeasurePauli;
        for (const auto& [q, letter] : t.ps)
        {
            op.targets.push_back(q);
            op.paulis += letter;
        }
        op.negate = *sign == -1;
    }
    emit(sink, std::move(op));
}

void Compiler::Impl::runStmt(const Stmt& s, const ScopePtr& sc)
{
    if (s.lvalue->kind != EK::Name) fail("E4001", s.lvalue->span, "run results bind a plain name, e.g. counts ← run 1000");
    const std::int64_t shots = needInt(eval(*s.expr, sc), s.expr->span, "a shot count");
    if (shots < 1) fail("E4003", s.expr->span, "run needs at least one shot");
    if (!measuredSincePrepare)
        fail("E5011", s.span, "run tallies classical bits, but nothing has been measured into one since the last prepare");
    Symbol sym;
    sym.kind = Symbol::Kind::Runtime;
    sym.slot = newSlot();
    sym.value = DeferredV{Type::Counts, Stage::Run};
    Event ev;
    ev.kind = EvK::Run;
    ev.span = s.span;
    ev.shots = static_cast<std::uint64_t>(shots);
    ev.slot = sym.slot;
    ev.name = s.lvalue->name;
    ev.scope = sc;
    define(sc, s.lvalue->name, std::move(sym), s.lvalue->span);
    ir.events.push_back(std::move(ev));
}

void Compiler::Impl::printStmt(const Stmt& s, const ScopePtr& sc)
{
    Event ev;
    ev.kind = EvK::Print;
    ev.span = s.span;
    ev.scope = sc;
    const std::size_t before = ir.readouts.size();
    for (const PrintItem& item : s.items)
    {
        readoutContext = true;
        try
        {
            noteStateReadout(eval(*item.expr, sc), item.expr->span);
        }
        catch (...)
        {
            readoutContext = false;
            throw;
        }
        readoutContext = false;
        ev.items.push_back({item.expr, item.label, formatExpr(*item.expr, false), item.span});
    }
    bool quantum = false;
    for (std::size_t k = before; k < ir.readouts.size(); ++k)
        if (ir.readouts[k].kind != "bits" && ir.readouts[k].kind != "counts") quantum = true;
    if (noiseEmitted && quantum) noisyReadouts.push_back(s.span);
    ir.events.push_back(std::move(ev));
}

// A readout whose value is the state itself (print |ψ⟩, let s = |ψ⟩, print ρ_A) reads amplitudes
// only the state vector holds; overlaps such as ⟨0|ψ⟩ and entropy(ρ_A) register their own readouts.
void Compiler::Impl::noteStateReadout(const Value& v, Span at)
{
    const KetV* k = v.get<KetV>();
    if (const auto* b = v.get<BraV>()) k = &b->ket;
    if (k && k->live)
    {
        readout("amplitudes", true, at, 1, ir.nQubits);
        if (ir.nQubits > 12)
            warn("W0006", at, std::format("|ψ⟩ on {} qubits prints up to 2^{} amplitudes; print probabilities, marginals or "
                                          "use --top", ir.nQubits, ir.nQubits));
    }
    if (const auto* r = v.get<RhoV>()) readout("density", true, at, 1, r->q.size());
}

void Compiler::Impl::assertStmt(const Stmt& s, const ScopePtr& sc)
{
    const std::size_t before = ir.readouts.size();
    readoutContext = true;
    Value v;
    try
    {
        v = eval(*s.expr, sc);
    }
    catch (...)
    {
        readoutContext = false;
        throw;
    }
    readoutContext = false;
    if (v.is<BitsV>()) (void)toCond(v, s.expr->span); // one clbit reads as a Bool, as in `if c[0]:`
    else if (v.type() != Type::Bool)
        fail("E4001", s.expr->span, std::format("assert needs a Bool, found {}", typeName(v.type())));
    bool quantum = false;
    for (std::size_t k = before; k < ir.readouts.size(); ++k)
        if (ir.readouts[k].kind != "bits" && ir.readouts[k].kind != "counts") quantum = true;
    if (noiseEmitted && quantum) noisyReadouts.push_back(s.span);
    Event ev;
    ev.kind = EvK::Assert;
    ev.span = s.span;
    ev.expr = s.expr;
    ev.text = formatExpr(*s.expr, false);
    ev.scope = sc;
    ir.events.push_back(std::move(ev));
}

void Compiler::Impl::forStmt(const Stmt& s, const ScopePtr& sc, Sink& sink)
{
    std::vector<std::int64_t> values;
    if (s.expr)
    {
        const Value lv = eval(*s.expr, sc);
        const ListV* l = lv.get<ListV>();
        if (!l) fail("E4001", s.expr->span, "a for loop iterates over a range a..b or a list of integers");
        for (const ValuePtr& item : l->items) values.push_back(needInt(*item, s.expr->span, "a loop value"));
    }
    else
    {
        const std::int64_t lo = needInt(eval(*s.lo, sc), s.lo->span, "a loop bound");
        const std::int64_t hi = needInt(eval(*s.hi, sc), s.hi->span, "a loop bound");
        const std::int64_t step = s.step ? needInt(eval(*s.step, sc), s.step->span, "a loop step") : 1;
        if (step == 0) fail("E4003", s.step->span, "a loop step cannot be zero");
        if (step > 0)
            for (std::int64_t k = lo; k <= hi; k += step) values.push_back(k);
        else
            for (std::int64_t k = lo; k >= hi; k += step) values.push_back(k);
    }
    checkBindable(s.name, s.nameSpan);
    for (const std::int64_t k : values)
    {
        auto inner = std::make_shared<Scope>(sc);
        Symbol sym;
        sym.kind = Symbol::Kind::Loop;
        sym.value = Num(Real::integer(k));
        sym.used = true;
        define(inner, s.name, std::move(sym), s.nameSpan);
        block(s.body, inner, sink);
    }
}

void Compiler::Impl::ifStmt(const Stmt& s, const ScopePtr& sc, Sink& sink)
{
    const Value cv = eval(*s.expr, sc);
    if (const bool* b = cv.get<bool>())
    {
        block(*b ? s.body : s.orelse, std::make_shared<Scope>(sc), sink);
        return;
    }
    if (!cv.is<CondV>() && !cv.is<BitsV>())
    {
        if (cv.type() == Type::Bool && cv.stage() == Stage::Run)
            fail("E4004", s.expr->span, "an if condition may test only classical bits (measurement results)");
        fail("E4001", s.expr->span, std::format("an if condition needs a Bool, found {}", typeName(cv.type())));
    }
    if (sink.ctx == Ctx::Def)
        fail("E5006", s.head, "a def is unitary and cannot branch on measurement results; use a proc");
    const std::shared_ptr<const CondNode> cond = toCond(cv, s.expr->span);

    auto conj = [](std::shared_ptr<const CondNode> a, std::shared_ptr<const CondNode> b)
    {
        if (!a) return b;
        auto n = std::make_shared<CondNode>();
        n->kind = CondNode::Kind::And;
        n->a = std::move(a);
        n->b = std::move(b);
        return std::shared_ptr<const CondNode>(n);
    };
    auto negate = [](std::shared_ptr<const CondNode> a)
    {
        auto n = std::make_shared<CondNode>();
        n->kind = CondNode::Kind::Not;
        n->a = std::move(a);
        return std::shared_ptr<const CondNode>(n);
    };

    const auto saved = sink.guard;
    auto run = [&](const Block& b, std::shared_ptr<const CondNode> g)
    {
        std::size_t count = 0;
        toCubes(g, count);
        if (count > 64)
            fail("E6005", s.expr->span, "the condition lowers to more than 64 disjoint cubes");
        sink.guard = g;
        try
        {
            block(b, std::make_shared<Scope>(sc), sink);
        }
        catch (...)
        {
            sink.guard = saved;
            throw;
        }
        sink.guard = saved;
    };
    run(s.body, conj(saved, cond));
    if (!s.orelse.empty()) run(s.orelse, conj(saved, negate(cond)));
}

void Compiler::Impl::defStmt(const Stmt& s, const ScopePtr& sc)
{
    std::set<std::string> seen;
    for (const std::string& p : s.cparams)
    {
        checkBindable(p, s.nameSpan);
        if (!seen.insert(p).second) fail("E3003", s.head, std::format("parameter `{}` is listed twice", p));
    }
    for (const QParam& q : s.sub.params)
    {
        checkBindable(q.name, q.span);
        if (!seen.insert(q.name).second) fail("E3003", q.span, std::format("parameter `{}` is listed twice", q.name));
        if (!q.sizeName.empty() && !std::isdigit(static_cast<unsigned char>(q.sizeName[0])))
        {
            checkBindable(q.sizeName, q.span);
            if (!seen.insert(q.sizeName).second)
                fail("E3003", q.span, std::format("parameter `{}` is listed twice", q.sizeName));
        }
    }
    Symbol sym;
    sym.kind = s.kind == SK::Def ? Symbol::Kind::Def : Symbol::Kind::Proc;
    sym.stmt = &s;
    sym.used = true;
    define(sc, s.name, std::move(sym), s.nameSpan);
}

void Compiler::Impl::noiseStmt(const Stmt& s, const ScopePtr& sc)
{
    for (const NoiseRule& r : s.rules)
    {
        const Value v = eval(*r.channel, sc);
        const GateV* g = v.get<GateV>();
        if (!g || g->kind != GateV::Kind::Channel)
            fail("E4001", r.channel->span, "a noise rule needs a channel such as depolarize(1e-3)");
        if ((r.event == "gate2") != (g->channel->arity == 2) && g->channel->arity == 2)
            fail("E5009", r.channel->span, "a 2-qubit channel can only follow gate2");
        noise.push_back({r.after, r.event, *g, r.span});
    }
    ir.hasNoise = true;
}

void Compiler::Impl::applyStmt(const Stmt& s, const ScopePtr& sc, Sink& sink)
{
    const Value v = eval(*s.expr, sc);
    if (const auto* g = v.get<GateV>())
    {
        if (g->kind == GateV::Kind::Rho) fail("E4001", s.expr->span, "ρ_A is a readout; print it or use entropy(ρ_A)");
        fail("E5009", s.expr->span, std::format("`{}` needs explicit support, e.g. {}_0", g->name.empty() ? "the gate" : g->name,
                                                g->name.empty() ? "U" : g->name));
    }
    if (v.is<MatV>() || v.is<ListV>())
    {
        std::size_t nq = 1;
        if (const auto* m = v.get<MatV>()) nq = m->nq();
        std::string example = "U_0";
        if (nq > 1)
        {
            example = "U_{0";
            for (std::size_t k = 1; k < nq; ++k) example += std::format(", {}", k);
            example += "}";
        }
        fail("E5009", s.expr->span, std::format("a matrix needs explicit support before it can be applied, e.g. {}", example));
    }
    if (v.is<ControlV>()) fail("E2001", s.expr->span, "C_{…} needs the controlled operation in parentheses: C_{0}(X_1)");
    if (!isOperator(v))
        fail("E4001", s.expr->span, std::format("this statement is a {}, not an operation to apply", typeName(v.type())));
    const OpV op = toOp(v, s.expr->span);
    if (sink.ctx == Ctx::Def && !op.unitary())
        fail("E5006", s.expr->span, "a def is unitary; measurements, resets and channels belong in a proc");
    emitOps(sink, op, s.expr->span);
}

void Compiler::Impl::specStmt(const Stmt& s, const ScopePtr& sc)
{
    owner.specInfo.stmts.push_back(&s.spec);
    owner.specInfo.spans.push_back(s.span);
    owner.specInfo.scope = sc;
}


// ---- Emission ----

void Compiler::Impl::emitOps(Sink& sink, const OpV& op, Span at)
{
    for (const IrOp& o : op.ops) emit(sink, o);
    if (!op.gphase.isZero() && sink.ctx != Ctx::Top)
    {
        IrOp g;
        g.kind = GK::GPhase;
        g.angles = {op.gphase};
        g.span = at;
        emit(sink, std::move(g));
    }
}

void Compiler::Impl::emit(Sink& sink, IrOp op)
{
    auto push = [&](IrOp o)
    {
        if (sink.ctx == Ctx::Top) emitTop(std::move(o));
        else sink.ops.push_back(std::move(o));
    };
    if (!sink.guard)
    {
        push(std::move(op));
        return;
    }
    if (isMeasure(op)) fail("E5012", op.span, "measurement inside `if` is not supported");
    std::size_t count = 0;
    const std::vector<Cube> cubes = toCubes(sink.guard, count);
    for (const Cube& cube : cubes)
    {
        IrOp copy = op;
        if (copy.cond)
        {
            const std::uint64_t both = copy.cond->mask & cube.mask;
            if ((copy.cond->value & both) != (cube.value & both)) continue; // disjoint: never fires
            copy.cond = Cube{copy.cond->mask | cube.mask, copy.cond->value | cube.value};
        }
        else copy.cond = cube;
        push(std::move(copy));
    }
}

void Compiler::Impl::emitTop(IrOp op)
{
    if (op.kind == GK::GPhase && op.controls.empty() && op.negControls.empty()) return; // unobservable
    insertNoise(op, false);
    if (isMeasure(op) && op.clbit) measuredSincePrepare = true;
    if (op.kind == GK::Channel) noiseEmitted = true;
    Event ev;
    ev.kind = EvK::Op;
    ev.span = op.span;
    ev.op = op;
    ir.events.push_back(std::move(ev));
    insertNoise(op, true);
}

void Compiler::Impl::insertNoise(const IrOp& op, bool after)
{
    if (noise.empty() || op.noise || op.kind == GK::Channel) return;
    const QubitList qs = op.qubits();
    const bool gate = !isMeasure(op) && op.kind != GK::Reset && op.kind != GK::GPhase && op.kind != GK::I;
    for (const NoiseRuleIr& r : noise)
    {
        if (r.after != after) continue;
        const bool match = (r.event == "gate1" && gate && qs.size() == 1) || (r.event == "gate2" && gate && qs.size() == 2) ||
                           (r.event == "measure" && isMeasure(op)) || (r.event == "reset" && op.kind == GK::Reset);
        if (!match) continue;
        auto chan = [&](QubitList targets)
        {
            IrOp c;
            c.kind = GK::Channel;
            c.channel = r.channel.channel;
            c.targets = std::move(targets);
            c.cond = op.cond;
            c.noise = true;
            c.span = r.span;
            Event ev;
            ev.kind = EvK::Op;
            ev.span = r.span;
            ev.op = std::move(c);
            ir.events.push_back(std::move(ev));
            noiseEmitted = true;
        };
        if (r.channel.channel->arity == 2)
        {
            if (qs.size() == 2) chan(qs);
        }
        else
            for (const Qubit q : qs) chan({q});
    }
}


// ---- Imports ----

void Compiler::Impl::importStmt(const Stmt& s, const ScopePtr& sc)
{
    std::string key;
    std::string text;
    if (s.text.starts_with("std/"))
    {
        const std::optional<std::string_view> src = stdSource(s.text);
        if (!src) fail("E3001", s.span, std::format("no standard library module `{}` (available: {})", s.text, stdModuleList()));
        key = s.text;
        text = std::string(*src);
    }
    else
    {
        const std::string dir = s.span.file < fileDirs.size() ? fileDirs[s.span.file] : std::string(".");
        const std::filesystem::path p = std::filesystem::path(dir) / s.text;
        std::error_code ec;
        key = std::filesystem::weakly_canonical(p, ec).string();
        bool ok = false;
        text = readFileText(p.string(), ok);
        if (!ok) fail("E3001", s.span, std::format("cannot read imported file `{}`", p.string()));
    }
    if (!importedPaths.insert(key).second) return;

    const std::uint32_t id = sm.add(key, std::move(text));
    fileDirs.resize(sm.size());
    fileDirs[id] = s.text.starts_with("std/") ? std::string("std") : std::filesystem::path(key).parent_path().string();
    auto prog = std::make_unique<Program>(parse(sm.file(id), id, d));
    const Program& p = *prog;
    imported.push_back(std::move(prog));
    for (const StmtPtr& st : p.stmts)
    {
        try
        {
            const SK k = st->kind;
            if (k != SK::Let && k != SK::Param && k != SK::Def && k != SK::Proc && k != SK::Import)
                fail("E8002", st->head, "an imported file may contain only let, param, def, proc and import");
            Sink top;
            stmt(*st, sc, top);
        }
        catch (const CompileAbort&)
        {
        }
    }
}


// ---- Conditions -> disjoint cubes (Shannon expansion) ----

namespace
{
    // Partial evaluation of f under the assignment so far: 0 false, 1 true, -1 undetermined.
    int evalCond(const CondNode* n, std::uint64_t mask, std::uint64_t value)
    {
        switch (n->kind)
        {
            case CondNode::Kind::Const: return n->value ? 1 : 0;
            case CondNode::Kind::Var:
                if ((mask >> n->clbit) & 1U) return static_cast<int>((value >> n->clbit) & 1U);
                return -1;
            case CondNode::Kind::Not:
            {
                const int a = evalCond(n->a.get(), mask, value);
                return a < 0 ? -1 : 1 - a;
            }
            case CondNode::Kind::And:
            {
                const int a = evalCond(n->a.get(), mask, value), b = evalCond(n->b.get(), mask, value);
                if (a == 0 || b == 0) return 0;
                return a == 1 && b == 1 ? 1 : -1;
            }
            case CondNode::Kind::Or:
            {
                const int a = evalCond(n->a.get(), mask, value), b = evalCond(n->b.get(), mask, value);
                if (a == 1 || b == 1) return 1;
                return a == 0 && b == 0 ? 0 : -1;
            }
        }
        return -1;
    }

    // The next clbit to split on: the first unassigned variable, in formula order, of an undecided
    // subformula. Following the formula's structure keeps a conjunction of k two-literal clauses at
    // 2^k cubes, where a fixed variable order can blow up.
    std::optional<std::size_t> pickVar(const CondNode* n, std::uint64_t mask, std::uint64_t value)
    {
        if (!n || evalCond(n, mask, value) >= 0) return std::nullopt;
        if (n->kind == CondNode::Kind::Var) return n->clbit;
        if (auto v = pickVar(n->a.get(), mask, value)) return v;
        return pickVar(n->b.get(), mask, value);
    }
} // namespace

std::vector<Cube> toCubes(const std::shared_ptr<const CondNode>& f, std::size_t& count)
{
    std::vector<Cube> out;
    if (!f)
    {
        out.push_back({0, 0});
        count = 1;
        return out;
    }
    auto rec = [&](auto&& self, std::uint64_t mask, std::uint64_t value) -> void
    {
        if (out.size() > 64) return;
        const int r = evalCond(f.get(), mask, value);
        if (r == 0) return;
        if (r == 1)
        {
            out.push_back({mask, value});
            return;
        }
        const auto v = pickVar(f.get(), mask, value);
        if (!v) return;
        const std::uint64_t b = std::uint64_t{1} << *v;
        self(self, mask | b, value);
        self(self, mask | b, value | b);
    };
    rec(rec, 0, 0);
    count = out.size();
    return out;
}


// ---- Compiler ----

Compiler::Compiler(SourceManager& sources, Diagnostics& diags, CompileOptions options)
    : impl(nullptr), sm(sources), d(diags), opt(std::move(options)), global(std::make_shared<Scope>())
{
    impl = std::make_unique<Impl>(*this);
}

Compiler::~Compiler() = default;

void Compiler::compile(const Program& prog)
{
    impl->fileDirs.resize(sm.size());
    if (impl->fileDirs[prog.file].empty())
        impl->fileDirs[prog.file] = std::filesystem::path(sm.file(prog.file).path()).parent_path().string();
    if (impl->fileDirs[prog.file].empty()) impl->fileDirs[prog.file] = ".";

    for (const StmtPtr& s : prog.stmts)
    {
        auto note = [&](const std::string& n, Span sp) { impl->laterNames.emplace(n, sp); };
        switch (s->kind)
        {
            case SK::Let:
                for (const Binding& b : s->bindings) note(b.name, b.nameSpan);
                break;
            case SK::Param:
            case SK::Def:
            case SK::Proc: note(s->name, s->nameSpan); break;
            case SK::Qubits:
            case SK::Bits:
                for (const RegDecl& r : s->regs) note(r.name, r.span);
                break;
            case SK::Run: note(s->lvalue->name, s->lvalue->span); break;
            default: break;
        }
    }

    if (opt.definitionsOnly) return compileDefinitions(prog);

    Sink top;
    impl->block(prog.stmts, global, top);
}

void Compiler::compileDefinitions(const Program& prog)
{
    impl->fileDirs.resize(sm.size());
    if (impl->fileDirs[prog.file].empty())
        impl->fileDirs[prog.file] = std::filesystem::path(sm.file(prog.file).path()).parent_path().string();
    if (impl->fileDirs[prog.file].empty()) impl->fileDirs[prog.file] = ".";
    for (const StmtPtr& st : prog.stmts)
    {
        try
        {
            const SK k = st->kind;
            if (k != SK::Let && k != SK::Param && k != SK::Def && k != SK::Proc && k != SK::Import)
                impl->fail("E8002", st->head, "a candidate may contain only the header, import, let, param, def and proc");
            Sink top;
            impl->stmt(*st, global, top);
        }
        catch (const CompileAbort&)
        {
        }
    }
}

void Compiler::compileStatement(const Stmt& s)
{
    try
    {
        Sink top;
        impl->stmt(s, global, top);
    }
    catch (const CompileAbort&)
    {
    }
}

void Compiler::applyByName(const std::string& name, const QubitList& qubits, Span at)
{
    try
    {
        auto e = std::make_shared<Expr>();
        e->kind = EK::Name;
        e->name = name;
        e->span = at;
        const Value g = impl->lookup(*e, global);
        const GateV* gv = g.get<GateV>();
        if (!gv) impl->fail("E8005", at, std::format("`{}` is not a def or proc", name));
        QArgs qa;
        qa.span = at;
        QArg a;
        a.q = qubits;
        a.single = false;
        a.span = at;
        qa.items.push_back(a);
        const OpV op = impl->applyGate(*gv, qa, at);
        Sink top;
        impl->emitOps(top, op, at);
    }
    catch (const CompileAbort&)
    {
    }
}

void Compiler::finish()
{
    // --set overrides of param values: name=v, name[k]=v, or name=[v, …].
    for (const auto& [key, val] : opt.sets)
    {
        std::string name = key;
        std::optional<std::size_t> idx;
        if (const auto lb = key.find('['); lb != std::string::npos && key.back() == ']')
        {
            name = key.substr(0, lb);
            idx = static_cast<std::size_t>(std::stoul(key.substr(lb + 1, key.size() - lb - 2)));
        }
        if (const std::size_t a = name.find_first_not_of(' '); a != std::string::npos) name = name.substr(a);
        // ASCII spellings of Greek names work too: --set theta=0.3
        auto canon = [](const std::string& n)
        {
            SourceManager tmp;
            const std::uint32_t id = tmp.add("--set", n);
            Diagnostics dd(tmp);
            const LexResult lr = lex(tmp.file(id), id, dd);
            return !lr.tokens.empty() && lr.tokens[0].kind == Tok::Ident ? lr.tokens[0].text : n;
        };
        name = canon(name);
        const auto it = std::ranges::find_if(irv.params, [&](const ParamInfo& p) { return p.name == name; });
        if (it == irv.params.end())
        {
            d.error("E3001", Span{UINT32_MAX, 0, 0}, std::format("--set {}: no param named `{}`", key, name));
            continue;
        }
        std::vector<double> vals;
        try
        {
            if (!val.empty() && val.front() == '[')
            {
                std::stringstream ss(val.substr(1, val.size() - 2));
                std::string tok;
                while (std::getline(ss, tok, ',')) vals.push_back(std::stod(tok));
            }
            else vals.push_back(std::stod(val));
        }
        catch (const std::exception&)
        {
            d.error("E4001", Span{UINT32_MAX, 0, 0}, std::format("--set {}={}: not a number", key, val));
            continue;
        }
        auto assign = [&](std::size_t k, double v)
        {
            if (v < it->lo || v > it->hi)
                d.error("E4001", it->span, std::format("--set {}: {} lies outside [{}, {}]", key, v, it->lo, it->hi));
            irv.paramValues[it->base + k] = v;
        };
        if (idx)
        {
            if (*idx >= it->size) d.error("E5008", it->span, std::format("--set {}: index out of range", key));
            else assign(*idx, vals.at(0));
        }
        else if (vals.size() == it->size)
            for (std::size_t k = 0; k < vals.size(); ++k) assign(k, vals[k]);
        else if (vals.size() == 1)
            for (std::size_t k = 0; k < it->size; ++k) assign(k, vals[0]);
        else d.error("E5009", it->span, std::format("--set {}: {} values for a param of size {}", key, vals.size(), it->size));
    }

    // Unused lets and params (W0004), only for the main program's globals, and only for programs
    // without errors: a use inside a statement that failed to compile is not seen.
    if (!d.hasErrors())
        for (const auto& [name, sym] : global->symbols())
            if (!sym.used && (sym.kind == Symbol::Kind::Let || sym.kind == Symbol::Kind::Param) && sym.span.file == 0)
                d.warning("W0004", sym.span, std::format("`{}` is never used", name));

    if (irv.trajectories == 0)
        for (const Span& s : impl->noisyReadouts)
            d.warning("W0005", s, "this readout follows noise and samples a single trajectory; set `trajectories K` to average");

    if (opt.backend) irv.backendRequest = *opt.backend;
    if (opt.seed) irv.seed = opt.seed;
    if (!d.hasErrors()) selectBackend(irv, d);
}

Value Compiler::evalRuntime(const Expr& e, const ConstScopePtr& scope, RuntimeHooks& hooks)
{
    RuntimeHooks* saved = impl->rt;
    impl->rt = &hooks;
    try
    {
        Value v = impl->eval(e, scope);
        impl->rt = saved;
        return v;
    }
    catch (...)
    {
        impl->rt = saved;
        throw;
    }
}

std::string Compiler::text(const Expr& e) const { return formatExpr(e, false); }


// ---- Files ----

std::string readFileText(const std::string& path, bool& ok)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        ok = false;
        return {};
    }
    std::stringstream ss;
    ss << in.rdbuf();
    ok = true;
    return ss.str();
}

std::unique_ptr<Compilation> compileText(const std::string& path, std::string text, const CompileOptions& options)
{
    auto c = std::make_unique<Compilation>();
    c->sources = std::make_unique<SourceManager>();
    c->diags = std::make_unique<Diagnostics>(*c->sources);
    const std::string sha = sha256Hex(text);
    c->mainFile = c->sources->add(path, std::move(text));
    c->programs.push_back(std::make_unique<Program>(parse(c->sources->file(c->mainFile), c->mainFile, *c->diags)));
    c->compiler = std::make_unique<Compiler>(*c->sources, *c->diags, options);
    c->compiler->ir().sourceSha256 = sha;
    c->compiler->compile(*c->programs[0]);
    c->compiler->finish();
    return c;
}

std::unique_ptr<Compilation> compileFile(const std::string& path, const CompileOptions& options)
{
    bool ok = false;
    std::string text = readFileText(path, ok);
    if (!ok)
    {
        auto c = std::make_unique<Compilation>();
        c->sources = std::make_unique<SourceManager>();
        c->diags = std::make_unique<Diagnostics>(*c->sources);
        c->readFailed = true;
        c->readError = std::format("cannot read {}", path);
        return c;
    }
    return compileText(path, std::move(text), options);
}

std::unique_ptr<Compilation> parseFile(const std::string& path)
{
    auto c = std::make_unique<Compilation>();
    c->sources = std::make_unique<SourceManager>();
    c->diags = std::make_unique<Diagnostics>(*c->sources);
    bool ok = false;
    std::string text = readFileText(path, ok);
    if (!ok)
    {
        c->readFailed = true;
        c->readError = std::format("cannot read {}", path);
        return c;
    }
    c->mainFile = c->sources->add(path, std::move(text));
    c->programs.push_back(std::make_unique<Program>(parse(c->sources->file(c->mainFile), c->mainFile, *c->diags)));
    return c;
}

} // namespace Noether
