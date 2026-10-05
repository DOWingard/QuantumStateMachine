#include "Tools.hpp"

#include "Backend.hpp"
#include "Cli.hpp"
#include "Decompose.hpp"
#include "Estimate.hpp"
#include "Formatter.hpp"
#include "Lowering.hpp"
#include "Parser.hpp"
#include "Sha256.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <sstream>



namespace Noether
{

namespace fs = std::filesystem;

namespace
{
    using Clock = std::chrono::steady_clock;

    // ---- Spec statements gathered from the parsed task file ----
    struct TaskSpec
    {
        std::string task;
        const SpecStmt* candidate = nullptr;
        Span candidateSpan;
        const SpecStmt* target = nullptr;
        std::vector<const SpecStmt*> metrics;
        std::vector<std::string> gates;
        bool hasGates = false;
        std::vector<std::pair<std::string, bool>> forbid;
        bool allowMatrix = false;
        const SpecStmt* coupling = nullptr;
        ExprPtr shots;
        std::vector<std::pair<std::string, ExprPtr>> optimize;
        bool hasOptimize = false;
        std::vector<ExprPtr> requirements;
        std::vector<std::pair<bool, ExprPtr>> objectives; // maximize?, expression
        std::vector<ExprPtr> goals;
        std::string instVar, holdVar;
        ExprPtr instSet, holdSet;
        std::string aggregate = "max";
        std::vector<std::pair<std::string, ExprPtr>> budget;
    };

    TaskSpec gather(const Program& p)
    {
        TaskSpec t;
        for (const StmtPtr& s : p.stmts)
        {
            if (s->kind != SK::Spec) continue;
            const SpecStmt& sp = s->spec;
            const std::string& k = sp.keyword;
            if (k == "task") t.task = sp.name;
            else if (k == "candidate") t.candidate = &sp, t.candidateSpan = s->span;
            else if (k == "target") t.target = &sp;
            else if (k == "metric") t.metrics.push_back(&sp);
            else if (k == "gates")
            {
                t.hasGates = true;
                for (const auto& [g, dag] : sp.gates) t.gates.push_back(g + (dag ? "†" : ""));
            }
            else if (k == "forbid") t.forbid.insert(t.forbid.end(), sp.gates.begin(), sp.gates.end());
            else if (k == "allow") t.allowMatrix = sp.word == "matrix";
            else if (k == "coupling") t.coupling = &sp;
            else if (k == "readout") t.shots = sp.word == "shots" ? sp.expr : nullptr;
            else if (k == "optimize") t.hasOptimize = true, t.optimize = sp.options;
            else if (k == "require") t.requirements.push_back(sp.expr);
            else if (k == "minimize" || k == "maximize") t.objectives.emplace_back(k == "maximize", sp.expr);
            else if (k == "goal") t.goals.push_back(sp.expr);
            else if (k == "instances") t.instVar = sp.name, t.instSet = sp.expr;
            else if (k == "holdout") t.holdVar = sp.name, t.holdSet = sp.expr;
            else if (k == "aggregate") t.aggregate = sp.word;
            else if (k == "budget") t.budget = sp.options;
        }
        return t;
    }

    // Run-time hooks for classical evaluation of spec expressions: structural metrics only.
    class MetricHooks final : public RuntimeHooks
    {
        public:
        explicit MetricHooks(const std::vector<IrOp>* ops, std::size_t n) : range(ops), nq(n) {}
        double param(std::uint32_t) const override { return 0.0; }
        const Value& slot(int) const override { return none; }
        std::uint64_t creg() const override { return 0; }
        cd expectation(const LinOpV&, Span at) override { throw RuntimeError{"E4004", "spec constraints cannot read the state", at}; }
        cd overlap(const KetV&, Span at) override { throw RuntimeError{"E4004", "spec constraints cannot read the state", at}; }
        double probability(const std::string&, Span at) override { throw RuntimeError{"E4004", "spec constraints cannot read the state", at}; }
        double entropy(const QubitList&, Span at) override { throw RuntimeError{"E4004", "spec constraints cannot read the state", at}; }
        Eigen::MatrixXcd density(const QubitList&, Span at) override { throw RuntimeError{"E4004", "spec constraints cannot read the state", at}; }
        KetV liveKet(Span at) override { throw RuntimeError{"E4004", "spec constraints cannot read the state", at}; }
        double metric(std::string_view name, std::optional<GK> gate) override
        {
            if (!range) return 0.0;
            if (name == "count")
            {
                std::size_t c = 0;
                for (const IrOp& op : *range)
                    if (gate && op.kind == *gate) ++c;
                return static_cast<double>(c);
            }
            std::vector<std::size_t> level(nq, 0);
            std::size_t depth = 0;
            for (const IrOp& op : *range)
            {
                std::size_t l = 0;
                for (const Qubit q : op.qubits()) l = std::max(l, level[q]);
                ++l;
                for (const Qubit q : op.qubits()) level[q] = l;
                depth = std::max(depth, l);
            }
            return static_cast<double>(depth);
        }

        private:
        const std::vector<IrOp>* range;
        std::size_t nq;
        Value none;
    };

    std::optional<double> realOf(const Value& v)
    {
        if (const auto* n = v.get<Num>(); n && n->isConst() && n->constValue().imag() == 0.0) return n->constValue().real();
        if (const auto* b = v.get<bool>()) return *b ? 1.0 : 0.0;
        return std::nullopt;
    }

    std::string numberText(double v)
    {
        if (std::floor(v) == v && std::abs(v) < 1e15) return std::to_string(static_cast<long long>(v));
        return jsonNumber(v);
    }

    // Ground energy of a Pauli sum: dense diagonalisation up to 8 qubits, Lanczos above (≤ 22).
    std::optional<double> groundEnergy(const LinOpV& ham, std::size_t n, std::string& error)
    {
        if (!ham.general.empty())
        {
            error = "target ground needs a sum of Pauli strings";
            return std::nullopt;
        }
        if (n > 22)
        {
            error = std::format("target ground supports up to 22 qubits, not {}", n);
            return std::nullopt;
        }
        const std::size_t dim = std::size_t{1} << n;
        struct Term
        {
            std::uint64_t flip = 0, zmask = 0, ymask = 0;
            cd coef;
        };
        std::vector<Term> terms;
        for (const PTerm& t : ham.paulis)
        {
            Term x;
            x.coef = t.coef.constValue();
            for (const auto& [q, l] : t.ps)
            {
                if (l == 'X' || l == 'Y') x.flip |= std::uint64_t{1} << q;
                if (l == 'Z' || l == 'Y') x.zmask |= std::uint64_t{1} << q;
                if (l == 'Y') x.ymask |= std::uint64_t{1} << q;
            }
            terms.push_back(x);
        }
        // P|i⟩ = phase(i)·|i ⊕ flip⟩ with Y = iXZ: phase = i^{#Y} (−1)^{popcount(i & zmask)}.
        auto apply = [&](const Eigen::VectorXcd& v, Eigen::VectorXcd& w)
        {
            w.setZero(static_cast<Eigen::Index>(dim));
            for (const Term& t : terms)
            {
                cd base = t.coef;
                for (int k = 0; k < std::popcount(t.ymask); ++k) base *= cd(0, 1);
                for (std::size_t i = 0; i < dim; ++i)
                {
                    const double sgn = (std::popcount(i & t.zmask) % 2) ? -1.0 : 1.0;
                    w[static_cast<Eigen::Index>(i ^ t.flip)] += base * sgn * v[static_cast<Eigen::Index>(i)];
                }
            }
        };
        if (n <= 8)
        {
            Eigen::MatrixXcd h(static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));
            Eigen::VectorXcd e(static_cast<Eigen::Index>(dim)), w;
            for (std::size_t c = 0; c < dim; ++c)
            {
                e.setZero();
                e[static_cast<Eigen::Index>(c)] = 1.0;
                apply(e, w);
                h.col(static_cast<Eigen::Index>(c)) = w;
            }
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(h, Eigen::EigenvaluesOnly);
            return es.eigenvalues()[0];
        }
        // Lanczos from a fixed pseudo-random start; the lowest Ritz value converges first.
        const std::size_t m = std::min<std::size_t>(dim, 300);
        Eigen::VectorXcd v(static_cast<Eigen::Index>(dim)), vPrev = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(dim)), w;
        std::uint64_t x = 88172645463325252ULL;
        for (std::size_t i = 0; i < dim; ++i)
        {
            x ^= x << 13, x ^= x >> 7, x ^= x << 17;
            v[static_cast<Eigen::Index>(i)] = static_cast<double>(x % 1000003) / 1000003.0 - 0.5;
        }
        v.normalize();
        std::vector<double> alpha, beta;
        double prev = 1e300;
        for (std::size_t k = 0; k < m; ++k)
        {
            apply(v, w);
            const double a = w.dot(v).real();
            alpha.push_back(a);
            w -= a * v;
            if (k > 0) w -= beta.back() * vPrev;
            const double b = w.norm();
            Eigen::MatrixXd tri = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(alpha.size()), static_cast<Eigen::Index>(alpha.size()));
            for (std::size_t j = 0; j < alpha.size(); ++j)
            {
                tri(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(j)) = alpha[j];
                if (j + 1 < alpha.size())
                    tri(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(j + 1)) = tri(static_cast<Eigen::Index>(j + 1), static_cast<Eigen::Index>(j)) = beta[j];
            }
            const double e0 = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(tri, Eigen::EigenvaluesOnly).eigenvalues()[0];
            if (std::abs(e0 - prev) < 1e-12 || b < 1e-12) return e0;
            prev = e0;
            beta.push_back(b);
            vPrev = v;
            v = w / b;
        }
        return prev;
    }

    std::string isoNow()
    {
        const std::time_t t = std::time(nullptr);
        std::tm tm{};
        gmtime_r(&t, &tm);
        char buf[32];
        std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
        return buf;
    }

    std::string gitCommit(const fs::path& dir)
    {
        const std::string cmd = std::format("git -C '{}' rev-parse --short HEAD 2>/dev/null", dir.string());
        FILE* p = popen(cmd.c_str(), "r");
        if (!p) return "-";
        char buf[64] = {};
        const bool ok = std::fgets(buf, sizeof buf, p) != nullptr;
        pclose(p);
        std::string s = ok ? buf : "-";
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        return s.empty() ? "-" : s;
    }

    bool writeText(const fs::path& p, const std::string& text)
    {
        std::ofstream o(p, std::ios::binary | std::ios::trunc);
        o << text;
        return static_cast<bool>(o);
    }

    double durationSeconds(const ExprPtr& e, double fallback)
    {
        if (!e) return fallback;
        std::string t = e->kind == EK::String ? e->name : formatExpr(*e, true);
        double scale = 1.0;
        if (!t.empty() && (t.back() == 's' || t.back() == 'm' || t.back() == 'h'))
        {
            scale = t.back() == 'm' ? 60.0 : t.back() == 'h' ? 3600.0 : 1.0;
            t.pop_back();
        }
        try
        {
            return std::stod(t) * scale;
        }
        catch (...)
        {
            return fallback;
        }
    }

    ExprPtr option(const std::vector<std::pair<std::string, ExprPtr>>& opts, std::string_view key)
    {
        for (const auto& [k, v] : opts)
            if (k == key) return v;
        return nullptr;
    }


    // ---- One evaluation of a candidate against one instance of the spec ----
    struct InstanceResult
    {
        std::string status = "ok"; // ok | error | timeout
        std::string instanceValue;
        std::map<std::string, double> metrics;
        std::map<std::string, double> stderrs;
        bool feasible = false;
        bool goalMet = false;
        double violation = 0.0;
        std::vector<double> objectives;
        std::size_t nops = 0;
        Json params = Json::object();
        Json diagnostics = Json::array();
        Json violations = Json::array();
    };

    struct Evaluator
    {
        std::string specPath, candPath, specText, candText;
        const ResearchOptions& opt;
        Json savedParams; // params.json content, if any
        Clock::time_point deadline;

        Evaluator(const ResearchOptions& o) : opt(o) {}

        InstanceResult run(const std::string& var, const std::string& value, bool checkOnly)
        {
            InstanceResult res;
            res.instanceValue = value;
            SourceManager sm;
            Diagnostics d(sm);
            const std::uint32_t specId = sm.add(specPath, specText);
            const Program specProg = parse(sm.file(specId), specId, d);
            const TaskSpec spec = gather(specProg);
            CompileOptions copt;
            if (!var.empty()) copt.lets[var] = value;
            Compiler comp(sm, d, copt);
            comp.compile(specProg);
            const std::uint32_t candId = sm.add(candPath, candText);
            const Program candProg = parse(sm.file(candId), candId, d);
            comp.compileDefinitions(candProg);
            auto finishWithError = [&](std::string status)
            {
                res.status = std::move(status);
                res.diagnostics = d.toJson();
                return res;
            };
            if (d.hasErrors()) return finishWithError("error");
            if (!spec.candidate)
            {
                d.error("E8005", Span{specId, 0, 0}, "the task spec has no `candidate def Name_{…}` statement");
                return finishWithError("error");
            }

            // Signature (E8005)
            const SpecStmt& sig = *spec.candidate;
            const Symbol* sym = comp.globals()->lookup(sig.name);
            if (!sym || (sym->kind != Symbol::Kind::Def && sym->kind != Symbol::Kind::Proc))
            {
                d.error("E8005", Span{candId, 0, 0}, std::format("the candidate must define `{} {}`", sig.word, sig.name));
                return finishWithError("error");
            }
            if (sig.word == "def" && sym->kind != Symbol::Kind::Def)
            {
                d.error("E8005", sym->stmt->head, std::format("the task asks for `def {}`; a proc may measure and is not allowed", sig.name));
                return finishWithError("error");
            }
            const SubParams& have = sym->stmt->sub;
            bool shapeOk = have.params.size() == sig.sub.params.size() && have.seps == sig.sub.seps;
            for (std::size_t k = 0; shapeOk && k < have.params.size(); ++k)
                shapeOk = have.params[k].sizeName.empty() == sig.sub.params[k].sizeName.empty();
            if (!shapeOk)
            {
                std::string want;
                for (std::size_t k = 0; k < sig.sub.params.size(); ++k)
                {
                    const QParam& p = sig.sub.params[k];
                    if (k) want += sig.sub.seps[k - 1] == Tok::Arrow ? "→" : ",";
                    want += p.name + (p.sizeName.empty() ? "" : "[" + p.sizeName + "]");
                }
                d.error("E8005", sym->stmt->head, std::format("`{}` must have the signature {}_{{{}}}", sig.name, sig.name, want))
                    .note(spec.candidateSpan, "required here");
                return finishWithError("error");
            }

            // Forbidden constructs (E8007)
            std::set<std::string> forbidden, forbiddenDag;
            for (const auto& [g, dag] : spec.forbid) (dag ? forbiddenDag : forbidden).insert(g);
            auto scan = [&](auto&& self, const Expr& e, bool daggered) -> void
            {
                if (e.kind == EK::Name && (forbidden.contains(e.name) || (daggered && forbiddenDag.contains(e.name))))
                    d.error("E8007", e.span, std::format("`{}{}` is forbidden by the task", e.name, daggered ? "†" : ""));
                if (e.kind == EK::Unary && e.op == Tok::Sqrt && e.kids[0]->kind == EK::Name && forbidden.contains("√" + e.kids[0]->name))
                    d.error("E8007", e.span, "`√X` is forbidden by the task");
                for (const ExprPtr& k : e.kids)
                    if (k) self(self, *k, e.kind == EK::Dagger);
                for (const QItem& q : e.qlist.items)
                    if (q.expr) self(self, *q.expr, false);
            };
            auto scanBlock = [&](auto&& self, const Block& b) -> void
            {
                for (const StmtPtr& s : b)
                {
                    for (const ExprPtr* e : {&s->expr, &s->lo, &s->hi, &s->step, &s->init})
                        if (*e) scan(scan, **e, false);
                    for (const Binding& bd : s->bindings) scan(scan, *bd.value, false);
                    self(self, s->body);
                    self(self, s->orelse);
                }
            };
            scanBlock(scanBlock, candProg.stmts);
            if (d.hasErrors()) return finishWithError("error");

            // Apply the candidate to the spec's registers, then the metric readouts.
            std::vector<std::shared_ptr<Expr>> keep;
            auto name = [&](const std::string& n, Span at)
            {
                auto e = std::make_shared<Expr>();
                e->kind = EK::Name;
                e->name = n;
                e->span = at;
                keep.push_back(e);
                return e;
            };
            auto apply = std::make_shared<Expr>();
            apply->kind = EK::Sub;
            apply->span = spec.candidateSpan;
            apply->kids = {name(sig.name, spec.candidateSpan)};
            apply->qlist.braced = true;
            for (const QParam& p : sig.sub.params) apply->qlist.items.push_back({false, name(p.name, spec.candidateSpan), spec.candidateSpan});
            apply->qlist.seps = sig.sub.seps;
            auto applyStmt = std::make_shared<Stmt>();
            applyStmt->kind = SK::Apply;
            applyStmt->span = applyStmt->head = spec.candidateSpan;
            applyStmt->expr = apply;
            Ir& ir = comp.ir();
            const std::size_t e0 = ir.events.size();
            comp.compileStatement(*applyStmt);
            const std::size_t e1 = ir.events.size();

            auto print = std::make_shared<Stmt>();
            print->kind = SK::Print;
            print->span = print->head = spec.candidateSpan;
            const std::string targetKind = spec.target ? spec.target->word : "";
            if (targetKind == "state")
            {
                auto call = std::make_shared<Expr>();
                call->kind = EK::Call;
                call->span = spec.candidateSpan;
                call->kids = {name("fidelity", spec.candidateSpan), spec.target->expr};
                call->argNames = {""};
                print->items.push_back({call, "fidelity", spec.candidateSpan});
            }
            else if (targetKind == "ground")
            {
                auto ev = std::make_shared<Expr>();
                ev->kind = EK::Expval;
                ev->span = spec.candidateSpan;
                ev->kids = {spec.target->expr};
                print->items.push_back({ev, "energy", spec.candidateSpan});
            }
            for (const SpecStmt* m : spec.metrics) print->items.push_back({m->expr, m->name, spec.candidateSpan});
            if (!print->items.empty()) comp.compileStatement(*print);
            comp.finish();
            if (d.hasErrors()) return finishWithError("error");

            // Custom matrices (E8007 unless `allow matrix`)
            if (!spec.allowMatrix)
                for (std::size_t k = e0; k < e1; ++k)
                    if (ir.events[k].kind == EvK::Op && ir.events[k].op.kind == GK::Matrix)
                        d.error("E8007", ir.events[k].op.span, "custom matrices are forbidden unless the task says `allow matrix`");
            if (d.hasErrors()) return finishWithError("error");

            // Lower into the target gate set (E8003); metrics are computed on the lowered circuit.
            std::vector<IrOp> candidateOps;
            if (spec.hasGates)
            {
                std::vector<std::string> unknown;
                const BasisLowering bl(spec.gates, unknown);
                for (const std::string& u : unknown) d.error("E8003", spec.candidateSpan, std::format("`{}` in `gates` is not a gate", u));
                std::vector<Event> lowered;
                std::size_t l0 = 0, l1 = 0;
                for (std::size_t k = 0; k < ir.events.size(); ++k)
                {
                    if (k == e0) l0 = lowered.size();
                    if (k == e1) l1 = lowered.size();
                    const Event& ev = ir.events[k];
                    if (ev.kind != EvK::Op || ev.op.noise)
                    {
                        lowered.push_back(ev);
                        continue;
                    }
                    std::vector<IrOp> ops;
                    std::string error;
                    if (!bl.lower(ev.op, ops, error))
                    {
                        d.error("E8003", ev.op.span, error);
                        continue;
                    }
                    for (IrOp& o : ops)
                    {
                        Event x = ev;
                        x.op = std::move(o);
                        lowered.push_back(std::move(x));
                    }
                }
                if (e1 == ir.events.size()) l1 = lowered.size();
                if (d.hasErrors()) return finishWithError("error");
                ir.events = std::move(lowered);
                for (std::size_t k = l0; k < l1; ++k)
                    if (ir.events[k].kind == EvK::Op) candidateOps.push_back(ir.events[k].op);
                selectBackend(ir, d);
                if (d.hasErrors()) return finishWithError("error");
            }
            else
                for (std::size_t k = e0; k < e1; ++k)
                    if (ir.events[k].kind == EvK::Op) candidateOps.push_back(ir.events[k].op);
            for (IrOp& o : candidateOps) o.clifford = isClifford(o);

            // Coupling (E8004): every multi-qubit op must act on an allowed pair.
            if (spec.coupling && spec.coupling->word != "all")
            {
                std::set<std::pair<Qubit, Qubit>> pairs;
                const std::size_t n = ir.nQubits;
                const std::string& w = spec.coupling->word;
                MetricHooks none(nullptr, n);
                auto intOf = [&](const ExprPtr& e)
                {
                    const auto v = realOf(comp.evalRuntime(*e, comp.globals(), none));
                    return static_cast<Qubit>(v.value_or(0));
                };
                if (w == "line" || w == "ring")
                {
                    for (Qubit q = 0; q + 1 < n; ++q) pairs.insert({q, q + 1});
                    if (w == "ring" && n > 2) pairs.insert({0, static_cast<Qubit>(n - 1)});
                }
                else if (w == "grid")
                {
                    const Qubit rows = intOf(spec.coupling->expr->kids[0]), cols = intOf(spec.coupling->expr->kids[1]);
                    for (Qubit r = 0; r < rows; ++r)
                        for (Qubit c = 0; c < cols; ++c)
                        {
                            if (c + 1 < cols) pairs.insert({r * cols + c, r * cols + c + 1});
                            if (r + 1 < rows) pairs.insert({r * cols + c, (r + 1) * cols + c});
                        }
                }
                else if (w == "set")
                    for (const ExprPtr& pr : spec.coupling->expr->kids)
                    {
                        const Qubit a = intOf(pr->kids[0]), b = intOf(pr->kids[1]);
                        pairs.insert({std::min(a, b), std::max(a, b)});
                    }
                for (const IrOp& op : candidateOps)
                {
                    const QubitList qs = op.qubits();
                    if (qs.size() < 2 || op.noise) continue;
                    bool ok = qs.size() == 2 && pairs.contains({std::min(qs[0], qs[1]), std::max(qs[0], qs[1])});
                    if (!ok)
                    {
                        Json v = Json::object();
                        v["code"] = "E8004";
                        std::string list;
                        for (const Qubit q : qs) list += (list.empty() ? "" : ", ") + std::to_string(q);
                        v["message"] = std::format("`{}` on qubits {{{}}} is not an allowed pair of the `{}` coupling", basisName(op), list, w);
                        v["span"] = spanJson(sm, op.span);
                        res.violations.push(std::move(v));
                    }
                }
            }

            // Structural metrics on the lowered candidate circuit.
            {
                Ir view;
                view.nQubits = ir.nQubits;
                for (const IrOp& o : candidateOps)
                {
                    Event x;
                    x.kind = EvK::Op;
                    x.op = o;
                    view.events.push_back(std::move(x));
                }
                const Resources r = resources(view);
                res.metrics["count2q"] = static_cast<double>(r.count2q);
                res.metrics["tcount"] = static_cast<double>(r.tcount);
                res.metrics["tdepth"] = static_cast<double>(r.tdepth);
                res.metrics["depth"] = static_cast<double>(r.depth);
                res.metrics["depth2q"] = static_cast<double>(r.depth2q);
                res.metrics["nops"] = static_cast<double>(r.ops);
                res.nops = r.ops;
            }
            std::vector<std::uint32_t> candParams;
            for (const ParamInfo& p : ir.params)
                if (p.span.file == candId)
                    for (std::size_t k = 0; k < p.size; ++k) candParams.push_back(p.base + static_cast<std::uint32_t>(k));
            res.metrics["nparams"] = static_cast<double>(candParams.size());
            if (checkOnly)
            {
                res.diagnostics = d.toJson();
                return res;
            }

            // Ground energy and target unitary fidelity do not depend on execution.
            MetricHooks hooks(&candidateOps, ir.nQubits);
            std::optional<double> ground;
            if (targetKind == "ground")
            {
                const Value hv = comp.evalRuntime(*spec.target->expr, comp.globals(), hooks);
                const LinOpV* ham = hv.get<LinOpV>();
                std::string error;
                if (ham) ground = groundEnergy(*ham, ir.nQubits, error);
                else error = "target ground needs an observable";
                if (!ground)
                {
                    d.error("E8006", spec.candidateSpan, error);
                    return finishWithError("error");
                }
                res.metrics["ground"] = *ground;
            }
            std::optional<double> unitaryFidelity;
            if (targetKind == "unitary")
            {
                const Value uv = comp.evalRuntime(*spec.target->expr, comp.globals(), hooks);
                const OpV* u = uv.get<OpV>();
                if (!u || !u->unitary())
                {
                    d.error("E4001", spec.candidateSpan, "target unitary needs an operation with explicit qubits, e.g. Toffoli_{0,1→2}");
                    return finishWithError("error");
                }
                if (ir.nQubits <= 12)
                {
                    QubitList all;
                    for (Qubit q = 0; q < ir.nQubits; ++q) all.push_back(q);
                    const Eigen::MatrixXcd U = unitaryOf(u->ops, all, ir.paramValues);
                    const Eigen::MatrixXcd V = unitaryOf(candidateOps, all, ir.paramValues);
                    const double dd = static_cast<double>(U.rows());
                    const double tr = std::norm((U.adjoint() * V).trace());
                    unitaryFidelity = (tr / dd + 1.0) / (dd + 1.0);
                }
                else
                {
                    Ir ia, ib;
                    ia.nQubits = ib.nQubits = ir.nQubits;
                    auto opEvent = [](const IrOp& o)
                    {
                        Event x;
                        x.kind = EvK::Op;
                        x.span = o.span;
                        x.op = o;
                        return x;
                    };
                    for (const IrOp& o : u->ops) ia.events.push_back(opEvent(o));
                    for (const IrOp& o : candidateOps) ib.events.push_back(opEvent(o));
                    tagClifford(ia);
                    tagClifford(ib);
                    const EquivResult er = equivalent(ia, ib, true);
                    if (!er.error.empty())
                    {
                        d.error("E8006", spec.candidateSpan, er.error);
                        return finishWithError("error");
                    }
                    unitaryFidelity = er.equivalent ? 1.0 : 0.0;
                }
                res.metrics["fidelity"] = *unitaryFidelity;
            }

            // Executes with params x and fills state metrics; false on failure.
            ExecOptions eo = opt.exec;
            eo.stopOnAssert = false;
            eo.timing = false;
            if (spec.shots)
            {
                const auto n = realOf(comp.evalRuntime(*spec.shots, comp.globals(), hooks));
                if (n && *n >= 1) eo.readoutShots = static_cast<std::uint64_t>(*n);
            }
            auto measure = [&](const std::vector<double>& x, std::map<std::string, double>& m, std::map<std::string, double>& se) -> bool
            {
                const double left = std::chrono::duration<double>(deadline - Clock::now()).count();
                if (left <= 0) return false;
                eo.timeoutSeconds = left;
                eo.params = x;
                const ExecResult r = execute(comp, eo);
                if (r.exitCode != 0 && r.exitCode != 3) return false;
                for (const PrintRecord& p : r.prints)
                {
                    if (p.number) m[p.label] = *p.number;
                    if (p.stderrValue) se[p.label] = *p.stderrValue;
                }
                if (ground && m.contains("energy")) m["gap"] = m["energy"] - *ground;
                if (se.contains("energy")) se["gap"] = se["energy"];
                return true;
            };

            // Requirements: violation = Σ max(0, shortfall) / max(|threshold|, 1e-12), with
            // confidence bounds for stochastic metrics.
            auto score = [&](const std::map<std::string, double>& m, const std::map<std::string, double>& se, double& violation,
                             bool& goalMet, std::vector<double>& objectives) -> bool
            {
                auto scope = std::make_shared<Scope>(comp.globals());
                for (const auto& [k, v] : m)
                {
                    Symbol s;
                    s.kind = Symbol::Kind::Let;
                    s.value = Num(Real::approx(v));
                    s.used = true;
                    scope->define(k, std::move(s));
                }
                auto valueOf = [&](const Expr& e) { return comp.evalRuntime(e, scope, hooks); };
                violation = 0.0;
                try
                {
                    for (const ExprPtr& r : spec.requirements)
                    {
                        if (r->kind == EK::Compare)
                        {
                            const auto a = realOf(valueOf(*r->kids[0]));
                            const auto b = realOf(valueOf(*r->kids[1]));
                            if (!a || !b) return false;
                            double lhs = *a;
                            const double s = r->kids[0]->kind == EK::Name && se.contains(r->kids[0]->name) ? se.at(r->kids[0]->name) : 0.0;
                            double shortfall = 0.0;
                            switch (r->op)
                            {
                                case Tok::GreaterEq: case Tok::Greater: lhs -= 2 * s; shortfall = *b - lhs; break;
                                case Tok::LessEq: case Tok::Less: lhs += 2 * s; shortfall = lhs - *b; break;
                                case Tok::Approx:
                                {
                                    const double tol = r->kids.size() > 2 ? realOf(valueOf(*r->kids[2])).value_or(1e-9) : 1e-9;
                                    shortfall = std::abs(*a - *b) - tol - 2 * s;
                                    break;
                                }
                                case Tok::EqEq: shortfall = std::abs(*a - *b); break;
                                case Tok::NotEq: shortfall = *a == *b ? 1.0 : 0.0; break;
                                default: break;
                            }
                            if (shortfall > 0) violation += shortfall / std::max(std::abs(*b), 1e-12);
                        }
                        else if (realOf(valueOf(*r)).value_or(0.0) == 0.0) violation += 1.0;
                    }
                    // A goal counts only for a feasible candidate (every require holds).
                    goalMet = violation <= 0.0;
                    for (const ExprPtr& g : spec.goals) goalMet = goalMet && realOf(valueOf(*g)).value_or(0.0) != 0.0;
                    objectives.clear();
                    for (const auto& [maximize, e] : spec.objectives)
                    {
                        const auto v = realOf(valueOf(*e));
                        if (!v) return false;
                        objectives.push_back(maximize ? -*v : *v);
                    }
                }
                catch (const RuntimeError& e)
                {
                    d.error(e.code, e.span, e.message);
                    return false;
                }
                catch (const CompileAbort&)
                {
                    return false;
                }
                return true;
            };

            // Inner loop over the candidate's params.
            std::vector<double> x = ir.paramValues;
            if (const Json* inst = savedParams.find("instances"); inst && inst->find(value.empty() ? "default" : value))
                loadParams(*inst->find(value.empty() ? "default" : value), ir, x);
            else if (const Json* ps = savedParams.find("params")) loadParams(*ps, ir, x);
            if (!candParams.empty() && spec.hasOptimize)
            {
                const std::string method = [&]
                {
                    const ExprPtr e = option(spec.optimize, "method");
                    return e ? (e->kind == EK::Name ? e->name : formatExpr(*e, true)) : std::string("nelder-mead");
                }();
                auto intOpt = [&](std::string_view k, std::size_t def)
                {
                    const ExprPtr e = option(spec.optimize, k);
                    if (!e) return def;
                    const auto v = realOf(comp.evalRuntime(*e, comp.globals(), hooks));
                    return v ? static_cast<std::size_t>(*v) : def;
                };
                std::vector<double> lo, hi, x0;
                for (const std::uint32_t idx : candParams)
                {
                    for (const ParamInfo& p : ir.params)
                        if (idx >= p.base && idx < p.base + p.size) lo.push_back(p.lo), hi.push_back(p.hi);
                    x0.push_back(x[idx]);
                }
                auto f = [&](const std::vector<double>& y)
                {
                    std::vector<double> full = x;
                    for (std::size_t k = 0; k < candParams.size(); ++k) full[candParams[k]] = y[k];
                    std::map<std::string, double> m = res.metrics, se;
                    double viol = 0.0;
                    bool goal = false;
                    std::vector<double> objs;
                    if (!measure(full, m, se) || !score(m, se, viol, goal, objs)) return 1e300;
                    // Violation first, then the first objective.
                    return viol * 1e6 + (objs.empty() ? 0.0 : objs[0]);
                };
                try
                {
                    const MinimizeResult mr = minimizeBlackBox(f, x0, lo, hi, method, intOpt("maxiter", 200), intOpt("restarts", 1),
                                                               ir.seed.value_or(0), [&] { return Clock::now() >= deadline; });
                    for (std::size_t k = 0; k < candParams.size(); ++k) x[candParams[k]] = mr.x[k];
                }
                catch (const std::invalid_argument& e)
                {
                    d.error("E2001", spec.candidateSpan, e.what());
                    return finishWithError("error");
                }
            }

            std::map<std::string, double> se;
            if (!measure(x, res.metrics, se))
                return finishWithError(Clock::now() >= deadline ? "timeout" : "error");
            res.stderrs = se;
            if (!score(res.metrics, se, res.violation, res.goalMet, res.objectives)) return finishWithError("error");
            if (!res.violations.asArray().empty()) res.violation += static_cast<double>(res.violations.size());
            res.feasible = res.violation == 0.0;
            for (const ParamInfo& p : ir.params)
            {
                if (p.span.file != candId) continue;
                if (!p.isVector) res.params[p.name] = x[p.base];
                else
                {
                    Json a = Json::array();
                    for (std::size_t k = 0; k < p.size; ++k) a.push(x[p.base + k]);
                    res.params[p.name] = a;
                }
            }
            res.diagnostics = d.toJson();
            return res;
        }

        static void loadParams(const Json& ps, const Ir& ir, std::vector<double>& x)
        {
            if (!ps.isObject()) return;
            for (const ParamInfo& p : ir.params)
            {
                const Json* v = ps.find(p.name);
                if (!v) continue;
                if (!p.isVector && v->isNumber()) x[p.base] = std::clamp(v->asDouble(), p.lo, p.hi);
                else if (p.isVector && v->isArray() && v->size() == p.size)
                    for (std::size_t k = 0; k < p.size; ++k)
                        if (v->asArray()[k].isNumber()) x[p.base + k] = std::clamp(v->asArray()[k].asDouble(), p.lo, p.hi);
            }
        }
    };

    std::size_t tokenCount(const std::string& path, const std::string& text)
    {
        SourceManager sm;
        Diagnostics d(sm);
        const std::uint32_t id = sm.add(path, text);
        std::size_t n = 0;
        for (const Token& t : lex(sm.file(id), id, d).tokens)
            if (t.kind != Tok::Newline && t.kind != Tok::Indent && t.kind != Tok::Dedent && t.kind != Tok::End) ++n;
        return n;
    }

    // Instance values of `instances` / `holdout`, evaluated in the default compilation of the spec.
    std::vector<std::string> setValues(const std::string& specPath, const std::string& specText, const ExprPtr& set)
    {
        std::vector<std::string> out;
        if (!set) return out;
        SourceManager sm;
        Diagnostics d(sm);
        const std::uint32_t id = sm.add(specPath, specText);
        const Program p = parse(sm.file(id), id, d);
        Compiler comp(sm, d, {});
        comp.compile(p);
        MetricHooks none(nullptr, 0);
        for (const ExprPtr& e : set->kids)
        {
            try
            {
                const auto v = realOf(comp.evalRuntime(*e, comp.globals(), none));
                out.push_back(v ? numberText(*v) : formatExpr(*e, true));
            }
            catch (...)
            {
                out.push_back(formatExpr(*e, true));
            }
        }
        return out;
    }

    struct EvalSummary
    {
        std::string status = "ok";
        bool feasible = false, goalMet = false;
        double violation = 0.0;
        std::vector<double> objectives;
        std::size_t nops = 0, tokens = 0;
        Json key, metrics, instances, params, diagnostics, violations;
        std::vector<std::string> objectiveNames;
        double wallS = 0.0;
    };

    EvalSummary evaluateAll(const std::string& specPath, const std::string& candPath, const ResearchOptions& opt,
                            const Json& savedParams, bool holdout)
    {
        EvalSummary s;
        const auto t0 = Clock::now();
        bool ok = false;
        Evaluator ev(opt);
        ev.specPath = specPath;
        ev.candPath = candPath;
        ev.specText = readFileText(specPath, ok);
        if (!ok) throw std::runtime_error(std::format("cannot read {}", specPath));
        ev.candText = readFileText(candPath, ok);
        if (!ok) throw std::runtime_error(std::format("cannot read {}", candPath));
        ev.savedParams = savedParams;

        SourceManager sm;
        Diagnostics d(sm);
        const std::uint32_t id = sm.add(specPath, ev.specText);
        const Program prog = parse(sm.file(id), id, d);
        const TaskSpec spec = gather(prog);
        for (const auto& [maximize, e] : spec.objectives) s.objectiveNames.push_back(formatExpr(*e, true));
        const double evalBudget = durationSeconds(option(spec.budget, "eval"), 60.0);
        ev.deadline = t0 + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(evalBudget));

        const std::string var = holdout ? spec.holdVar : spec.instVar;
        std::vector<std::string> values = setValues(specPath, ev.specText, holdout ? spec.holdSet : spec.instSet);
        if (values.empty()) values.emplace_back();

        s.instances = Json::array();
        s.params = Json::object();
        s.diagnostics = Json::array();
        s.violations = Json::array();
        s.feasible = true;
        s.goalMet = true;
        std::vector<std::vector<double>> objs;
        std::map<std::string, std::vector<double>> metricValues;
        for (const std::string& v : values)
        {
            const InstanceResult r = ev.run(v.empty() ? std::string() : var, v, false);
            Json inst = Json::object();
            if (!var.empty() && !v.empty()) inst[var] = v;
            Json m = Json::object();
            for (const auto& [k, x] : r.metrics) m[k] = x;
            inst["metrics"] = m;
            inst["status"] = r.status;
            inst["feasible"] = r.feasible;
            if (!r.stderrs.empty())
            {
                Json se = Json::object();
                for (const auto& [k, x] : r.stderrs) se[k] = x;
                inst["stderr"] = se;
            }
            s.instances.push(inst);
            for (const Json& dj : r.diagnostics.asArray()) s.diagnostics.push(dj);
            for (const Json& vj : r.violations.asArray()) s.violations.push(vj);
            s.params[v.empty() ? "default" : v] = r.params;
            if (r.status != "ok")
            {
                s.status = r.status;
                s.feasible = false;
                s.goalMet = false;
                break;
            }
            s.feasible = s.feasible && r.feasible;
            s.goalMet = s.goalMet && r.goalMet && r.feasible;
            s.violation = std::max(s.violation, r.violation);
            s.nops = std::max(s.nops, r.nops);
            objs.push_back(r.objectives);
            for (const auto& [k, x] : r.metrics) metricValues[k].push_back(x);
        }
        s.tokens = tokenCount(candPath, ev.candText);
        s.metrics = Json::object();
        for (const auto& [k, xs] : metricValues)
        {
            double agg = spec.aggregate == "mean" ? 0.0 : -std::numeric_limits<double>::infinity();
            for (const double x : xs) agg = spec.aggregate == "mean" ? agg + x / static_cast<double>(xs.size()) : std::max(agg, x);
            s.metrics[k] = agg;
        }
        if (s.status == "ok")
            for (std::size_t j = 0; j < s.objectiveNames.size(); ++j)
            {
                double agg = spec.aggregate == "mean" ? 0.0 : -std::numeric_limits<double>::infinity();
                for (const auto& o : objs)
                    agg = spec.aggregate == "mean" ? agg + o[j] / static_cast<double>(objs.size()) : std::max(agg, o[j]);
                s.objectives.push_back(agg);
            }
        s.key = Json::array();
        if (s.status == "ok")
        {
            s.key.push(s.feasible ? 0 : 1);
            s.key.push(s.violation);
            for (const double o : s.objectives) s.key.push(o);
            s.key.push(s.nops);
            s.key.push(s.tokens);
        }
        s.wallS = std::chrono::duration<double>(Clock::now() - t0).count();
        return s;
    }

    // Lexicographic comparison of ranking keys (−1, 0, 1).
    int compareKeys(const Json& a, const Json& b)
    {
        const std::size_t n = std::min(a.size(), b.size());
        for (std::size_t k = 0; k < n; ++k)
        {
            const double x = a.asArray()[k].asDouble(), y = b.asArray()[k].asDouble();
            if (x < y) return -1;
            if (x > y) return 1;
        }
        return a.size() < b.size() ? -1 : a.size() > b.size() ? 1 : 0;
    }

    // ---- Ledger (results.tsv) ----
    struct Row
    {
        std::map<std::string, std::string> f;
        Json key;
    };

    std::vector<std::string> splitTabs(const std::string& line)
    {
        std::vector<std::string> out;
        std::string cur;
        for (const char c : line)
        {
            if (c == '\t')
            {
                out.push_back(cur);
                cur.clear();
            }
            else cur += c;
        }
        out.push_back(cur);
        return out;
    }

    std::vector<Row> readLedger(const fs::path& p, std::vector<std::string>& header)
    {
        std::vector<Row> rows;
        std::ifstream in(p);
        std::string line;
        if (!std::getline(in, line)) return rows;
        header = splitTabs(line);
        while (std::getline(in, line))
        {
            if (line.empty()) continue;
            const auto cells = splitTabs(line);
            Row r;
            for (std::size_t k = 0; k < header.size() && k < cells.size(); ++k) r.f[header[k]] = cells[k];
            r.key = Json::parse(r.f["key"]).value_or(Json::array());
            rows.push_back(std::move(r));
        }
        return rows;
    }

    std::vector<std::string> ledgerHeader(const std::vector<std::string>& objectiveNames)
    {
        std::vector<std::string> h{"n", "timestamp", "commit", "status", "feasible", "goalMet", "violation"};
        for (const std::string& o : objectiveNames) h.push_back(o);
        for (const char* c : {"nops", "wallS", "candidateSha", "key", "description"}) h.emplace_back(c);
        return h;
    }

    std::string joinTabs(const std::vector<std::string>& v)
    {
        std::string s;
        for (std::size_t k = 0; k < v.size(); ++k) s += (k ? "\t" : "") + v[k];
        return s;
    }

    std::string sanitize(std::string s)
    {
        for (char& c : s)
            if (c == '\t' || c == '\n' || c == '\r') c = ' ';
        return s;
    }

    const Row* bestRow(const std::vector<Row>& rows)
    {
        const Row* best = nullptr;
        for (const Row& r : rows)
        {
            const std::string& st = r.f.at("status");
            if ((st != "baseline" && st != "keep") || r.key.size() == 0) continue;
            if (!best || compareKeys(r.key, best->key) < 0) best = &r;
        }
        return best;
    }

    Json lockJson(const std::string& specText, const std::string& created)
    {
        Json j = Json::object();
        j["schema"] = "noether.lock/1";
        j["specSha256"] = sha256Hex(specText);
        j["noetherVersion"] = kVersion;
        j["rulesVersion"] = kDecompositionRulesVersion;
        j["createdAt"] = created;
        j["createdUnix"] = static_cast<std::int64_t>(std::time(nullptr));
        return j;
    }

    std::optional<Json> readJsonFile(const fs::path& p)
    {
        bool ok = false;
        const std::string t = readFileText(p.string(), ok);
        if (!ok) return std::nullopt;
        return Json::parse(t);
    }

    Json evalHeader(bool ok)
    {
        Json j = Json::object();
        j["schema"] = "noether.eval/1";
        j["version"] = kVersion;
        j["lang"] = kLangVersion;
        j["ok"] = ok;
        return j;
    }

    // An evaluation that ran reports its outcome in `status`; the exit code follows the CLI table:
    // compile or spec errors 1, timeout 4, an evaluated candidate (feasible or not) 0.
    int statusExit(const std::string& status)
    {
        if (status == "error") return 1;
        if (status == "timeout") return 4;
        return 0;
    }

    // Writes the eval document, appends the ledger row (record) and returns the verdict.
    ResearchOutcome evaluateInWorkspace(const fs::path& dir, const ResearchOptions& opt, const std::string& status0)
    {
        ResearchOutcome out;
        const fs::path specPath = dir / "spec.ntr", candPath = dir / "algo.ntr", lockPath = dir / "spec.lock";
        bool ok = false;
        const std::string specText = readFileText(specPath.string(), ok);
        if (!ok)
        {
            out.exitCode = 2;
            out.text = std::format("noether: no spec.ntr in {}\n", dir.string());
            out.json = evalHeader(false);
            out.json["error"] = out.text;
            return out;
        }
        const auto lock = readJsonFile(lockPath);
        if (!lock || !lock->find("specSha256") || lock->find("specSha256")->asString() != sha256Hex(specText))
        {
            out.exitCode = 6;
            out.json = evalHeader(false);
            // The whole spec file is the span: any edit to it breaks the lock.
            SourceManager sm;
            const std::uint32_t id = sm.add(specPath.string(), specText);
            Diagnostic diag;
            diag.code = "E8001";
            diag.message = "spec.ntr does not match spec.lock: the evaluator was modified";
            diag.span = Span{id, 0, static_cast<std::uint32_t>(specText.size())};
            if (lock && lock->find("specSha256"))
                diag.note(diag.span, std::format("spec.lock records sha256 {}, spec.ntr has {}", lock->find("specSha256")->asString(),
                                                 sha256Hex(specText)));
            Json ds = Json::array();
            ds.push(diagnosticJson(sm, diag));
            out.json["diagnostics"] = ds;
            out.text = "error[E8001]: spec.ntr does not match spec.lock: the evaluator was modified\n";
            return out;
        }
        if (!opt.noFormat)
        {
            std::string err;
            canonicalizeFile(candPath.string(), err);
        }
        const Json saved = readJsonFile(dir / "params.json").value_or(Json::object());
        EvalSummary s;
        try
        {
            s = evaluateAll(specPath.string(), candPath.string(), opt, saved, false);
        }
        catch (const std::exception& e)
        {
            out.exitCode = 2;
            out.text = std::format("noether: {}\n", e.what());
            out.json = evalHeader(false);
            out.json["error"] = e.what();
            return out;
        }
        const std::string candText = readFileText(candPath.string(), ok);
        std::vector<std::string> header;
        std::vector<Row> rows = readLedger(dir / "results.tsv", header);
        if (header.empty()) header = ledgerHeader(s.objectiveNames);
        const Row* best = bestRow(rows);

        std::string verdict;
        if (s.status != "ok") verdict = s.status == "timeout" ? "error" : "error";
        else if (!best || compareKeys(s.key, best->key) < 0) verdict = "improved";
        else if (!s.feasible) verdict = "infeasible";
        else if (compareKeys(s.key, best->key) == 0) verdict = "equal";
        else verdict = "worse";

        Json j = evalHeader(s.status == "ok");
        j["task"] = gather([&] {
            SourceManager sm;
            Diagnostics d(sm);
            const std::uint32_t id = sm.add(specPath.string(), specText);
            return parse(sm.file(id), id, d);
        }()).task;
        j["specSha256"] = sha256Hex(specText);
        j["candidateSha256"] = sha256Hex(candText);
        j["status"] = s.status;
        j["feasible"] = s.feasible;
        j["goalMet"] = s.goalMet;
        j["key"] = s.key;
        j["verdict"] = verdict;
        if (best)
        {
            Json b = Json::object();
            b["n"] = std::stoll(best->f.at("n"));
            b["key"] = best->key;
            j["best"] = b;
        }
        else j["best"] = nullptr;
        j["metrics"] = s.metrics;
        j["instances"] = s.instances;
        j["params"] = s.params;
        j["violations"] = s.violations;
        j["diagnostics"] = s.diagnostics;
        j["wallS"] = s.wallS;

        const std::size_t n = rows.size();
        fs::create_directories(dir / "runs");
        writeText(dir / "runs" / std::format("{}.json", n), j.dump() + "\n");
        if (opt.record || !status0.empty())
        {
            std::string status = status0;
            if (status.empty())
                status = s.status == "timeout" ? "timeout" : s.status != "ok" ? "error" : verdict == "improved" ? "keep"
                         : verdict == "infeasible" ? "infeasible" : "discard";
            std::vector<std::string> cells{std::to_string(n), isoNow(), gitCommit(dir), status, s.feasible ? "1" : "0",
                                           s.goalMet ? "1" : "0", jsonNumber(s.violation)};
            for (std::size_t k = 0; k < s.objectiveNames.size(); ++k)
                cells.push_back(k < s.objectives.size() ? jsonNumber(s.objectives[k]) : "");
            cells.push_back(std::to_string(s.nops));
            cells.push_back(jsonNumber(std::round(s.wallS * 1000) / 1000));
            cells.push_back(sha256Hex(candText).substr(0, 12));
            cells.push_back(s.key.dump(-1));
            cells.push_back(sanitize(opt.record.value_or(status0)));
            const fs::path ledger = dir / "results.tsv";
            const bool fresh = !fs::exists(ledger);
            std::ofstream o(ledger, std::ios::app);
            if (fresh) o << joinTabs(header) << "\n";
            o << joinTabs(cells) << "\n";
            if (verdict == "improved" || status0 == "baseline")
            {
                Json pj = Json::object();
                pj["schema"] = "noether.params/1";
                pj["instances"] = s.params;
                writeText(dir / "params.json", pj.dump() + "\n");
            }
            j["recorded"] = n;
        }
        out.json = j;
        out.exitCode = statusExit(s.status);
        out.text = std::format("{}: key {} — {} (metrics {})\n", verdict, s.key.dump(-1), s.status, s.metrics.dump(-1));
        for (const Json& dj : s.diagnostics.asArray())
            if (const Json* m = dj.find("message")) out.text += std::format("  {}: {}\n", dj.find("code")->asString(), m->asString());
        for (const Json& vj : s.violations.asArray()) out.text += std::format("  E8004: {}\n", vj.find("message")->asString());
        return out;
    }
} // namespace


ResearchOutcome evaluateCommand(const ResearchOptions& opt)
{
    if (!opt.dir.empty()) return evaluateInWorkspace(opt.dir, opt, "");
    ResearchOutcome out;
    try
    {
        if (!opt.noFormat)
        {
            std::string err;
            canonicalizeFile(opt.candidate, err);
        }
        const EvalSummary s = evaluateAll(opt.spec, opt.candidate, opt, Json::object(), false);
        bool ok = false;
        Json j = evalHeader(s.status == "ok");
        j["specSha256"] = sha256Hex(readFileText(opt.spec, ok));
        j["candidateSha256"] = sha256Hex(readFileText(opt.candidate, ok));
        j["status"] = s.status;
        j["feasible"] = s.feasible;
        j["goalMet"] = s.goalMet;
        j["key"] = s.key;
        j["metrics"] = s.metrics;
        j["instances"] = s.instances;
        j["params"] = s.params;
        j["violations"] = s.violations;
        j["diagnostics"] = s.diagnostics;
        j["wallS"] = s.wallS;
        out.json = j;
        out.exitCode = statusExit(s.status);
        out.text = std::format("{}: key {} feasible={} goalMet={}\nmetrics {}\n", s.status, s.key.dump(-1), s.feasible, s.goalMet,
                               s.metrics.dump(-1));
        for (const Json& dj : s.diagnostics.asArray()) out.text += std::format("  {}: {}\n", dj.find("code")->asString(), dj.find("message")->asString());
        for (const Json& vj : s.violations.asArray()) out.text += std::format("  E8004: {}\n", vj.find("message")->asString());
    }
    catch (const std::exception& e)
    {
        out.exitCode = 2;
        out.json = evalHeader(false);
        out.json["error"] = e.what();
        out.text = std::format("noether: {}\n", e.what());
    }
    return out;
}

ResearchOutcome checkCandidate(const ResearchOptions& opt)
{
    ResearchOutcome out;
    bool ok = false;
    Evaluator ev(opt);
    ev.specPath = opt.spec;
    ev.candPath = opt.candidate;
    ev.specText = readFileText(opt.spec, ok);
    if (ok) ev.candText = readFileText(opt.candidate, ok);
    if (!ok)
    {
        out.exitCode = 2;
        out.json = evalHeader(false);
        out.json["error"] = "cannot read the spec or the candidate";
        out.text = "noether: cannot read the spec or the candidate\n";
        return out;
    }
    ev.deadline = Clock::now() + std::chrono::seconds(60);
    SourceManager sm;
    Diagnostics d(sm);
    const std::uint32_t id = sm.add(opt.spec, ev.specText);
    const TaskSpec spec = gather(parse(sm.file(id), id, d));
    const std::vector<std::string> values = setValues(opt.spec, ev.specText, spec.instSet);
    const InstanceResult r = ev.run(values.empty() ? "" : spec.instVar, values.empty() ? "" : values.front(), true);
    Json j = Json::object();
    j["schema"] = "noether.check/1";
    j["version"] = kVersion;
    j["lang"] = kLangVersion;
    // `check` is the pre-flight gate, so coupling violations are errors here, not only metrics.
    Json diags = r.diagnostics.isArray() ? r.diagnostics : Json::array();
    for (const Json& v : r.violations.isArray() ? r.violations.asArray() : Json::Array{})
    {
        Json dj = v;
        dj["severity"] = "error";
        dj["notes"] = Json::array();
        dj["fixes"] = Json::array();
        diags.push(std::move(dj));
    }
    const bool passed = r.status == "ok" && r.violations.size() == 0;
    j["ok"] = passed;
    j["diagnostics"] = diags;
    j["violations"] = r.violations;
    Json m = Json::object();
    for (const auto& [k, x] : r.metrics) m[k] = x;
    j["metrics"] = m;
    out.json = j;
    out.exitCode = passed ? 0 : 1;
    out.text = passed ? std::format("ok: structural metrics {}\n", m.dump(-1)) : "";
    for (const Json& dj : diags.asArray()) out.text += std::format("{}: {}\n", dj.find("code")->asString(), dj.find("message")->asString());
    return out;
}

ResearchOutcome researchInit(const std::string& tag, const ResearchOptions& opt)
{
    ResearchOutcome out;
    const fs::path dir = fs::path(opt.dir.empty() ? "research" : opt.dir) / tag;
    bool ok = false;
    const std::string specText = readFileText(opt.spec, ok);
    if (!ok)
    {
        out.exitCode = 2;
        out.text = std::format("noether: cannot read {}\n", opt.spec);
        out.json = evalHeader(false);
        out.json["error"] = out.text;
        return out;
    }
    if (fs::exists(dir / "spec.lock"))
    {
        out.exitCode = 2;
        out.text = std::format("noether: {} is already a research workspace\n", dir.string());
        out.json = evalHeader(false);
        out.json["error"] = out.text;
        return out;
    }
    SourceManager sm;
    Diagnostics d(sm);
    const std::uint32_t id = sm.add(opt.spec, specText);
    const Program prog = parse(sm.file(id), id, d);
    const TaskSpec spec = gather(prog);
    if (d.hasErrors() || !spec.candidate)
    {
        out.exitCode = 1;
        out.text = d.render() + (spec.candidate ? "" : "noether: the spec has no `candidate` statement\n");
        out.json = evalHeader(false);
        out.json["diagnostics"] = d.toJson();
        return out;
    }
    fs::create_directories(dir / "runs");
    writeText(dir / "spec.ntr", specText);
    writeText(dir / "spec.lock", lockJson(specText, isoNow()).dump() + "\n");
    writeText(dir / ".gitignore", "results.tsv\nnotes.md\nruns/\n");
    writeText(dir / "notes.md", std::format("# Notes: {}\n\nHypotheses and lessons, one entry per experiment.\n", spec.task));
    writeText(dir / "program.md",
              std::format("# Task brief: {}\n\nThe frozen evaluator is `spec.ntr`; the only editable file is `algo.ntr`.\n\n"
                          "## Context\n\n(what the task is for, what is known)\n\n## Hints\n\n(constructions to try or avoid, references)\n\n"
                          "## Ideas\n\n(open directions)\n",
                          spec.task));
    if (!opt.from.empty())
    {
        const std::string from = readFileText(opt.from, ok);
        if (!ok)
        {
            out.exitCode = 2;
            out.text = std::format("noether: cannot read {}\n", opt.from);
            out.json = evalHeader(false);
            out.json["error"] = out.text;
            return out;
        }
        writeText(dir / "algo.ntr", from);
    }
    else
    {
        const SpecStmt& sig = *spec.candidate;
        std::string params;
        for (std::size_t k = 0; k < sig.sub.params.size(); ++k)
        {
            const QParam& p = sig.sub.params[k];
            if (k) params += sig.sub.seps[k - 1] == Tok::Arrow ? "→" : ",";
            params += p.name + (p.sizeName.empty() ? "" : "[" + p.sizeName + "]");
        }
        const QParam& first = sig.sub.params.front();
        const std::string body = first.sizeName.empty() ? "I_" + first.name : "I_{" + first.name + "[0]}";
        writeText(dir / "algo.ntr", std::format("noether 0.1\n## Candidate for task \"{}\". This is the only file the research loop edits.\n"
                                                "{} {}_{{{}}}:\n    {}\n",
                                                spec.task, sig.word, sig.name, params, body));
    }
    out = evaluateInWorkspace(dir, opt, "baseline");
    out.text = std::format("initialised {}\nbaseline: {}", dir.string(), out.text);
    if (out.json.isObject()) out.json["workspace"] = dir.string();
    return out;
}

ResearchOutcome researchStatus(const ResearchOptions& opt)
{
    ResearchOutcome out;
    const fs::path dir = opt.dir;
    std::vector<std::string> header;
    const std::vector<Row> rows = readLedger(dir / "results.tsv", header);
    bool ok = false;
    const std::string specText = readFileText((dir / "spec.ntr").string(), ok);
    const auto lock = readJsonFile(dir / "spec.lock");
    if (!ok || !lock)
    {
        out.exitCode = 2;
        out.text = std::format("noether: {} is not a research workspace\n", dir.string());
        out.json = evalHeader(false);
        out.json["error"] = out.text;
        return out;
    }
    SourceManager sm;
    Diagnostics d(sm);
    const std::uint32_t id = sm.add("spec.ntr", specText);
    const TaskSpec spec = gather(parse(sm.file(id), id, d));
    const double experimentsBudget = [&] {
        const ExprPtr e = option(spec.budget, "experiments");
        return e && e->kind == EK::Int ? std::stod(e->name) : 0.0;
    }();
    const double wallBudget = durationSeconds(option(spec.budget, "wall"), 0.0);
    const double used = rows.empty() ? 0.0 : static_cast<double>(rows.size() - 1);
    const double wallUsed = static_cast<double>(std::time(nullptr) - (lock->find("createdUnix") ? lock->find("createdUnix")->asInt() : 0));
    const Row* best = bestRow(rows);
    const bool goalMet = best && best->f.at("goalMet") == "1";
    std::string reason;
    if (goalMet) reason = "goal met";
    else if (experimentsBudget > 0 && used >= experimentsBudget) reason = "experiment budget exhausted";
    else if (wallBudget > 0 && wallUsed >= wallBudget) reason = "wall-clock budget exhausted";

    Json j = Json::object();
    j["schema"] = "noether.status/1";
    j["version"] = kVersion;
    j["ok"] = true;
    j["task"] = spec.task;
    j["rows"] = rows.size();
    j["goalMet"] = goalMet;
    j["stop"] = !reason.empty();
    j["reason"] = reason.empty() ? Json(nullptr) : Json(reason);
    Json budget = Json::object();
    budget["experimentsUsed"] = used;
    budget["experimentsBudget"] = experimentsBudget > 0 ? Json(experimentsBudget) : Json(nullptr);
    budget["wallUsedS"] = wallUsed;
    budget["wallBudgetS"] = wallBudget > 0 ? Json(wallBudget) : Json(nullptr);
    j["budget"] = budget;
    auto rowJson = [&](const Row& r)
    {
        Json x = Json::object();
        for (const auto& [k, v] : r.f) x[k] = v;
        x["key"] = r.key;
        return x;
    };
    j["best"] = best ? rowJson(*best) : Json(nullptr);
    // Pareto frontier over the objectives of feasible rows.
    Json pareto = Json::array();
    std::vector<const Row*> feasible;
    for (const Row& r : rows)
        if (r.f.at("feasible") == "1" && r.key.size() >= 2) feasible.push_back(&r);
    const std::size_t nObj = spec.objectives.size();
    for (const Row* a : feasible)
    {
        bool dominated = false;
        for (const Row* b : feasible)
        {
            if (a == b) continue;
            bool allLe = true, anyLt = false;
            for (std::size_t k = 0; k < nObj; ++k)
            {
                const double x = b->key.asArray()[2 + k].asDouble(), y = a->key.asArray()[2 + k].asDouble();
                allLe = allLe && x <= y;
                anyLt = anyLt || x < y;
            }
            dominated = dominated || (allLe && anyLt);
        }
        if (!dominated) pareto.push(std::stoll(a->f.at("n")));
    }
    j["pareto"] = pareto;
    Json last = Json::array();
    for (std::size_t k = rows.size() > 5 ? rows.size() - 5 : 0; k < rows.size(); ++k) last.push(rowJson(rows[k]));
    j["last"] = last;
    out.json = j;
    out.text = std::format("task {}: {} rows, best {}, goal {}, {}\n", spec.task, rows.size(), best ? best->key.dump(-1) : "none",
                           goalMet ? "met" : "not met", reason.empty() ? "continue" : "stop: " + reason);
    return out;
}

ResearchOutcome researchReport(const ResearchOptions& opt)
{
    ResearchOutcome out;
    const fs::path dir = opt.dir;
    std::vector<std::string> header;
    const std::vector<Row> rows = readLedger(dir / "results.tsv", header);
    bool ok = false;
    const std::string specText = readFileText((dir / "spec.ntr").string(), ok);
    if (!ok)
    {
        out.exitCode = 2;
        out.text = std::format("noether: {} is not a research workspace\n", dir.string());
        out.json = evalHeader(false);
        out.json["error"] = out.text;
        return out;
    }
    SourceManager sm;
    Diagnostics d(sm);
    const std::uint32_t id = sm.add("spec.ntr", specText);
    const TaskSpec spec = gather(parse(sm.file(id), id, d));
    const Row* best = bestRow(rows);

    std::string md = std::format("# Research report: {}\n\n", spec.task);
    md += std::format("{} experiments after the baseline.\n\n", rows.empty() ? 0 : rows.size() - 1);
    if (best)
        md += std::format("**Best:** row {} (commit {}), key `{}`, feasible {}, goal met {}.\n\n", best->f.at("n"), best->f.at("commit"),
                          best->key.dump(-1), best->f.at("feasible") == "1" ? "yes" : "no", best->f.at("goalMet") == "1" ? "yes" : "no");
    md += "## Trajectory\n\n| n | status | feasible | violation | key | description |\n|---|---|---|---|---|---|\n";
    for (const Row& r : rows)
        md += std::format("| {} | {} | {} | {} | `{}` | {} |\n", r.f.at("n"), r.f.at("status"), r.f.at("feasible"), r.f.at("violation"),
                          r.key.dump(-1), r.f.count("description") ? r.f.at("description") : "");

    // Holdout: the current candidate on the holdout instances (or those of --holdout).
    std::string holdSpec = (dir / "spec.ntr").string();
    if (!opt.holdout.empty()) holdSpec = opt.holdout;
    Json hold = Json::array();
    try
    {
        const std::string hs = readFileText(holdSpec, ok);
        SourceManager sm2;
        Diagnostics d2(sm2);
        const std::uint32_t id2 = sm2.add(holdSpec, hs);
        const TaskSpec hspec = gather(parse(sm2.file(id2), id2, d2));
        if (hspec.holdSet)
        {
            const Json saved = readJsonFile(dir / "params.json").value_or(Json::object());
            const EvalSummary s = evaluateAll(holdSpec, (dir / "algo.ntr").string(), opt, saved, true);
            hold = s.instances;
            md += std::format("\n## Holdout\n\nstatus {}, feasible {}, metrics `{}`\n\n", s.status, s.feasible ? "yes" : "no", s.metrics.dump(-1));
            for (const Json& inst : s.instances.asArray()) md += std::format("- `{}`\n", inst.dump(-1));
        }
        else md += "\n## Holdout\n\nThe spec declares no holdout instances.\n";
    }
    catch (const std::exception& e)
    {
        md += std::format("\n## Holdout\n\nfailed: {}\n", e.what());
    }
    const std::string notes = readFileText((dir / "notes.md").string(), ok);
    if (ok) md += "\n## Lessons (notes.md)\n\n" + notes + "\n";
    Json j = Json::object();
    j["schema"] = "noether.report/1";
    j["version"] = kVersion;
    j["ok"] = true;
    j["markdown"] = md;
    j["holdout"] = hold;
    out.json = j;
    out.text = md;
    return out;
}

} // namespace Noether
