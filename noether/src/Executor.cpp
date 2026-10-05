#include "Executor.hpp"

#include "Backend.hpp"
#include "Estimate.hpp"
#include "Lowering.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <format>
#include <random>
#include <stdexcept>



namespace Noether
{

using Qputer::Operation;
using Qputer::Outcome;
using Qputer::QuantumStateMachine;

namespace
{
    using Clock = std::chrono::steady_clock;

    struct Timeout
    {
    };

    std::uint64_t splitmix(std::uint64_t x)
    {
        x += 0x9E3779B97F4A7C15ULL;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
        return x ^ (x >> 31);
    }

    bool evalCondNode(const CondNode& n, std::uint64_t creg)
    {
        switch (n.kind)
        {
            case CondNode::Kind::Const: return n.value;
            case CondNode::Kind::Var: return ((creg >> n.clbit) & 1U) != 0;
            case CondNode::Kind::Not: return !evalCondNode(*n.a, creg);
            case CondNode::Kind::And: return evalCondNode(*n.a, creg) && evalCondNode(*n.b, creg);
            case CondNode::Kind::Or: return evalCondNode(*n.a, creg) || evalCondNode(*n.b, creg);
        }
        return false;
    }

    std::string bitsText(const BitsV& b, std::uint64_t creg)
    {
        std::string s;
        for (const std::size_t c : b.clbits) s += ((creg >> c) & 1U) ? '1' : '0';
        return s;
    }

    // Kronecker-order index (q0 = MSB) of a state-vector index (qubit q = bit q).
    std::size_t kronIndex(std::size_t i, std::size_t n)
    {
        std::size_t r = 0;
        for (std::size_t q = 0; q < n; ++q)
            if ((i >> q) & 1U) r |= std::size_t{1} << (n - 1 - q);
        return r;
    }

    Json complexJson(cd z) { return Json(Json::Array{Json(z.real()), Json(z.imag())}); }

    Json numJson(const Num& n)
    {
        if (const auto k = n.asInt()) return Json(*k);
        const cd z = n.constValue();
        if (z.imag() == 0.0) return Json(z.real());
        return complexJson(z);
    }

    Json ketJson(const KetV& k, std::size_t top)
    {
        const std::size_t n = k.nq();
        const Eigen::VectorXcd a = k.dense();
        std::vector<std::pair<std::size_t, cd>> nz;
        for (Eigen::Index i = 0; i < a.size(); ++i)
            if (std::abs(a[i]) > 1e-12) nz.emplace_back(static_cast<std::size_t>(i), a[i]);
        if (top && nz.size() > top)
        {
            std::ranges::stable_sort(nz, [](const auto& x, const auto& y) { return std::abs(x.second) > std::abs(y.second); });
            nz.resize(top);
        }
        Json amps = Json::object();
        for (const auto& [i, z] : nz)
        {
            std::string key(n, '0');
            for (std::size_t q = 0; q < n; ++q)
                if ((i >> (n - 1 - q)) & 1U) key[q] = '1';
            amps[key] = complexJson(z);
        }
        Json j = Json::object();
        j["basis"] = "q0-leftmost";
        j["amplitudes"] = amps;
        return j;
    }

    Json matrixJson(const Eigen::MatrixXcd& m)
    {
        Json rows = Json::array();
        for (Eigen::Index r = 0; r < m.rows(); ++r)
        {
            Json row = Json::array();
            for (Eigen::Index c = 0; c < m.cols(); ++c) row.push(complexJson(m(r, c)));
            rows.push(std::move(row));
        }
        Json j = Json::object();
        j["matrix"] = rows;
        return j;
    }

    Json countsJson(const std::map<std::string, std::uint64_t>& counts, std::size_t top)
    {
        std::vector<std::pair<std::string, std::uint64_t>> v(counts.begin(), counts.end());
        if (top && v.size() > top)
        {
            std::ranges::stable_sort(v, [](const auto& a, const auto& b) { return a.second > b.second; });
            v.resize(top);
        }
        Json j = Json::object();
        for (const auto& [k, n] : v) j[k] = n;
        return j;
    }

    std::string jsonText(const Json& j)
    {
        if (j.isString()) return j.asString();
        return j.dump(-1);
    }

    std::uint32_t lineOf(const SourceManager& sm, Span s)
    {
        if (s.file >= sm.size()) return 0;
        return sm.file(s.file).lineCol(s.begin).line;
    }


    class Executor final : public RuntimeHooks
    {
        public:
        Executor(Compiler& c, const ExecOptions& o) : comp(c), ir(c.ir()), opt(o) {}

        ExecResult run()
        {
            const auto t0 = Clock::now();
            start = t0;
            params = opt.params ? *opt.params : ir.paramValues;
            slots.assign(static_cast<std::size_t>(ir.slots), Value{});
            // A drawn seed stays below 2^53 so the JSON report (and any reader parsing doubles) keeps it exact for --seed.
            result.seed = ir.seed ? *ir.seed
                                  : splitmix(std::random_device{}() ^ (std::uint64_t{std::random_device{}()} << 32)) & ((std::uint64_t{1} << 53) - 1);
            stab = ir.backend == "stabilizer";

            if (ir.nQubits > 0)
            {
                const std::uint64_t bytes = registerBytes(ir.backend, ir.nQubits);
                if (bytes > opt.maxMemBytes)
                {
                    comp.diagnostics().error("E6004", ir.qregs.empty() ? Span{UINT32_MAX, 0, 0} : ir.qregs.front().span,
                                             std::format("the {} register needs {} bytes, over --max-mem {}", ir.backend, bytes,
                                                         opt.maxMemBytes));
                    result.exitCode = 4;
                    return result;
                }
                live = makeMachine(result.seed);
                cur = live.get();
            }

            try
            {
                for (std::size_t k = 0; k < ir.events.size(); ++k)
                {
                    eventIndex = k;
                    if (!step(ir.events[k])) break;
                }
                if (result.exitCode == 0 && live && !stab)
                {
                    // Checked once at the end (a norm is O(2^N)); the last operation is where the drift
                    // became visible.
                    const double n2 = live->state().norm();
                    if (std::abs(n2 - 1.0) > 1e-8)
                    {
                        Span at = ir.events.empty() ? Span{} : ir.events.back().span;
                        for (const Event& ev : ir.events)
                            if (ev.kind == EvK::Op) at = ev.span;
                        comp.diagnostics().error("E7003", at, std::format("the state's norm drifted to {} during execution", n2));
                        result.exitCode = 5;
                    }
                }
            }
            catch (const Timeout&)
            {
                comp.diagnostics().error("E7002", ir.events[eventIndex].span,
                                         std::format("execution exceeded --timeout {}s", *opt.timeoutSeconds));
                result.exitCode = 4;
            }
            catch (const CompileAbort&)
            {
                result.exitCode = 1;
            }
            catch (const RuntimeError& e)
            {
                comp.diagnostics().error(e.code, e.span, e.message);
                result.exitCode = e.code == "E7001" ? 3 : 1;
            }
            catch (const std::bad_alloc&)
            {
                comp.diagnostics().error("E6004", ir.events[eventIndex].span, "out of memory");
                result.exitCode = 4;
            }
            catch (const std::exception& e)
            {
                result.internalError = e.what();
                comp.diagnostics().error("E7003", ir.events[eventIndex].span, std::format("internal error: {}", e.what()));
                result.exitCode = 5;
            }
            result.executeMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            return result;
        }

        // ---- RuntimeHooks ----
        double param(std::uint32_t index) const override { return params.at(index); }
        const Value& slot(int index) const override { return slots.at(static_cast<std::size_t>(index)); }
        std::uint64_t creg() const override { return cur ? cur->classical_register() : 0; }

        cd expectation(const LinOpV& op, Span at) override
        {
            needState(at);
            if (opt.readoutShots) return sampledExpectation(op, at);
            cd total = 0.0;
            for (const PTerm& t : op.paulis)
            {
                const cd c = t.coef.eval(params);
                if (t.ps.empty())
                {
                    total += c;
                    continue;
                }
                std::string letters;
                QubitList qs;
                for (const auto& [q, l] : t.ps)
                {
                    letters += l;
                    qs.push_back(q);
                }
                total += c * cur->expectation(letters, qs);
            }
            if (!op.general.empty())
            {
                const Qputer::QuantumStateVector& s = svState(at, "a dense observable");
                for (const auto& [c, o] : op.general) total += c.eval(params) * matrixElement(s, o);
            }
            return total;
        }

        cd overlap(const KetV& bra, Span at) override
        {
            if (opt.readoutShots) throw RuntimeError{"E8006", "an overlap or fidelity cannot be estimated from measurement shots (readout shots)", at};
            const Qputer::QuantumStateVector& s = svState(at, "an overlap ⟨φ|ψ⟩");
            const std::size_t n = ir.nQubits;
            const Eigen::VectorXcd phi = bra.dense();
            cd acc = 0.0;
            for (std::size_t i = 0; i < s.size(); ++i) acc += std::conj(phi[static_cast<Eigen::Index>(kronIndex(i, n))]) * s[i];
            return acc;
        }

        double probability(const std::string& bits, Span at) override
        {
            needState(at);
            if (opt.readoutShots)
            {
                QubitList all;
                for (Qubit q = 0; q < ir.nQubits; ++q) all.push_back(q);
                QuantumStateMachine m = *cur;
                m.reseed(splitmix(result.seed ^ ++shotCounter));
                const std::vector<Outcome> outs = m.sample(all, *opt.readoutShots);
                std::uint64_t hits = 0;
                for (const Outcome o : outs)
                {
                    bool match = true;
                    for (std::size_t q = 0; q < bits.size() && match; ++q) match = (((o >> q) & 1U) != 0) == (bits[q] == '1');
                    hits += match ? 1 : 0;
                }
                const double p = static_cast<double>(hits) / static_cast<double>(outs.size());
                shotVar += p * (1 - p) / static_cast<double>(outs.size());
                return p;
            }
            Outcome idx = 0;
            for (std::size_t q = 0; q < bits.size(); ++q)
                if (bits[q] == '1')
                {
                    if (q >= 64)
                        throw RuntimeError{"E6002", "basis probabilities with a 1 beyond qubit 63 are not supported", at};
                    idx |= Outcome{1} << q;
                }
            return cur->probability(idx);
        }

        double entropy(const QubitList& q, Span at) override
        {
            if (opt.readoutShots) throw RuntimeError{"E8006", "entropy cannot be estimated from measurement shots (readout shots)", at};
            needState(at);
            return cur->entropy(q);
        }

        Eigen::MatrixXcd density(const QubitList& q, Span at) override
        {
            if (opt.readoutShots) throw RuntimeError{"E8006", "ρ_A cannot be estimated from measurement shots (readout shots)", at};
            svState(at, "ρ_A");
            if (q.size() > 13) throw RuntimeError{"E6002", "ρ_A is limited to 13 qubits", at};
            return cur->reduced_density_matrix(q);
        }

        KetV liveKet(Span at) override
        {
            if (opt.readoutShots) throw RuntimeError{"E8006", "|ψ⟩ amplitudes cannot be estimated from measurement shots (readout shots)", at};
            const Qputer::QuantumStateVector& s = svState(at, "|ψ⟩");
            const std::size_t n = ir.nQubits;
            KetFactor f;
            f.kind = KetFactor::Kind::Dense;
            f.nq = n;
            f.amp = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(s.size()));
            for (std::size_t i = 0; i < s.size(); ++i) f.amp[static_cast<Eigen::Index>(kronIndex(i, n))] = s[i];
            KetV k;
            k.factors.push_back(std::move(f));
            k.live = true;
            return k;
        }

        double metric(std::string_view name, std::optional<GK> gate) override
        {
            if (name == "count")
            {
                std::size_t n = 0;
                for (const IrOp* op : irHistory)
                    if (gate && op->kind == *gate) ++n;
                return static_cast<double>(n);
            }
            std::vector<std::size_t> level(ir.nQubits, 0);
            std::size_t depth = 0;
            for (const IrOp* op : irHistory)
            {
                const QubitList qs = op->qubits();
                std::size_t l = 0;
                for (const Qubit q : qs) l = std::max(l, level[q]);
                ++l;
                for (const Qubit q : qs) level[q] = l;
                depth = std::max(depth, l);
            }
            return static_cast<double>(depth);
        }

        private:
        // ⟨A⟩ from measurement shots: each Pauli term is measured in its eigenbasis on a copy of
        // the state; the variance of the estimate accumulates into shotVar.
        cd sampledExpectation(const LinOpV& op, Span at)
        {
            if (!op.general.empty())
                throw RuntimeError{"E8006", "a dense observable cannot be estimated from measurement shots", at};
            const std::uint64_t shots = *opt.readoutShots;
            cd total = 0.0;
            for (const PTerm& t : op.paulis)
            {
                const cd c = t.coef.eval(params);
                if (t.ps.empty())
                {
                    total += c;
                    continue;
                }
                QuantumStateMachine m = *cur;
                m.reseed(splitmix(result.seed ^ ++shotCounter));
                QubitList qs;
                for (const auto& [q, l] : t.ps)
                {
                    if (l == 'X') m.h(q);
                    else if (l == 'Y')
                    {
                        m.sdg(q);
                        m.h(q);
                    }
                    qs.push_back(q);
                }
                const std::vector<Outcome> outs = m.sample(qs, shots);
                double sum = 0.0;
                for (const Outcome o : outs) sum += (std::popcount(o) % 2 == 0) ? 1.0 : -1.0;
                const double mean = sum / static_cast<double>(shots);
                total += c * mean;
                shotVar += std::norm(c) * std::max(0.0, 1.0 - mean * mean) / static_cast<double>(shots);
            }
            return total;
        }

        std::unique_ptr<QuantumStateMachine> makeMachine(std::uint64_t seed) const
        {
            return std::make_unique<QuantumStateMachine>(
                ir.nQubits, ir.nClbits, seed, stab ? Qputer::Backend::Stabilizer : Qputer::Backend::StateVector);
        }

        void needState(Span at) const
        {
            if (!cur) throw RuntimeError{"E5008", "this readout needs qubits, but none are declared", at};
        }

        const Qputer::QuantumStateVector& svState(Span at, std::string_view what) const
        {
            needState(at);
            if (stab) throw RuntimeError{"E6002", std::format("{} needs the state vector", what), at};
            return cur->state();
        }

        // ⟨ψ|O|ψ⟩ for a product of IR ops (dense matrices may be Hermitian rather than unitary).
        cd matrixElement(const Qputer::QuantumStateVector& s, const OpV& o) const
        {
            Qputer::QuantumStateVector phi = s;
            for (const IrOp& op : o.ops)
            {
                if (op.kind == GK::Matrix && !op.unitaryMatrix && op.controls.empty() && op.negControls.empty())
                {
                    applyMatrix(phi, op.targets, *op.matrix);
                    continue;
                }
                for (const Operation& x : lower(op, params, false)) applyUnitary(phi, x);
            }
            cd acc = 0.0;
            for (std::size_t i = 0; i < s.size(); ++i) acc += std::conj(s[i]) * phi[i];
            return acc * std::polar(1.0, o.gphase.eval(params));
        }

        void checkTime()
        {
            if (opt.timeoutSeconds && std::chrono::duration<double>(Clock::now() - start).count() > *opt.timeoutSeconds)
                throw Timeout{};
        }

        void preparePrep(QuantumStateMachine& m) const
        {
            if (prep.dense) m.prepare_state(prep.amps);
            else m.prepare_basis(prep.basis);
            for (const Operation& o : prep.ops) m.append(o);
        }

        bool step(const Event& ev)
        {
            checkTime();
            switch (ev.kind)
            {
                case EvK::Op:
                {
                    IrOp op = ev.op;
                    if (opt.shift && opt.shift->event == eventIndex)
                        op.angles[opt.shift->angle] = op.angles[opt.shift->angle] + Affine(Real::approx(opt.shift->delta));
                    for (const Operation& o : lower(op, params, stab))
                    {
                        live->append(o);
                        history.push_back(o);
                    }
                    irHistory.push_back(&ev.op);
                    if (ev.op.kind == GK::Channel) noisy = true;
                    return true;
                }
                case EvK::Prepare:
                {
                    prep = {};
                    const KetV& k = *ev.ket;
                    if (k.isStabilizerProduct()) prep.ops = productPreparation(k, prep.basis);
                    else
                    {
                        // Kronecker order (q0 = MSB) to the register's order (qubit q = bit q).
                        const Eigen::VectorXcd a = k.dense();
                        prep.dense = true;
                        prep.amps.resize(a.size());
                        for (Eigen::Index i = 0; i < a.size(); ++i)
                            prep.amps[i] = a[static_cast<Eigen::Index>(kronIndex(static_cast<std::size_t>(i), ir.nQubits))];
                    }
                    preparePrep(*live);
                    history.clear(); // preparePrep replays prep.ops itself
                    irHistory.clear();
                    noisy = false;
                    return true;
                }
                case EvK::Print:
                {
                    for (const PrintItemIr& item : ev.items)
                    {
                        PrintRecord rec;
                        rec.span = item.span;
                        rec.label = item.label;
                        rec.text = item.text;
                        const Sampled s = sample(*item.expr, ev.scope, nullptr);
                        rec.value = s.json;
                        rec.number = s.number;
                        rec.stderrValue = s.stderrValue;
                        result.prints.push_back(std::move(rec));
                    }
                    return true;
                }
                case EvK::Assert: return assertion(ev);
                case EvK::Run:
                {
                    const Qputer::Counts counts = live->run(ev.shots);
                    CountsV cv;
                    for (const RegInfo& r : ir.bregs)
                    {
                        std::vector<std::size_t> cl;
                        for (std::size_t k = 0; k < r.size; ++k) cl.push_back(r.offset + k);
                        cv.layout.emplace_back(r.name, std::move(cl));
                    }
                    for (const auto& [outcome, n] : counts) cv.counts[countsKey(ir, outcome)] += n;
                    RunRecord rec;
                    rec.span = ev.span;
                    rec.name = ev.name;
                    rec.shots = ev.shots;
                    rec.counts = cv.counts;
                    result.runs.push_back(std::move(rec));
                    slots[static_cast<std::size_t>(ev.slot)] = std::move(cv);
                    return true;
                }
                case EvK::Let:
                    slots[static_cast<std::size_t>(ev.slot)] = comp.evalRuntime(*ev.expr, ev.scope, *this);
                    return true;
            }
            return true;
        }

        struct Sampled
        {
            Json json;
            std::optional<double> number;
            std::optional<double> stderrValue;
            Value value;
        };

        static std::optional<cd> numeric(const Value& v)
        {
            if (const auto* n = v.get<Num>(); n && n->isConst()) return n->constValue();
            return std::nullopt;
        }

        // Evaluates an expression on the live pass, or averaged over trajectories when noise has
        // been applied since the last prepare and `trajectories K` is set. `rhs` (asserts) is
        // evaluated on the same trajectory, and the statistics are of lhs − rhs.
        Sampled sample(const Expr& e, const ConstScopePtr& sc, const Expr* rhs)
        {
            Sampled out;
            shotVar = 0.0;
            out.value = comp.evalRuntime(e, sc, *this);
            // A bare ρ_A readout is the reduced density matrix itself.
            if (const auto* r = out.value.get<RhoV>()) out.value = MatV{density(r->q, e.span)};
            if (opt.readoutShots && shotVar > 0.0) out.stderrValue = std::sqrt(shotVar);
            out.json = valueJson(out.value, creg(), opt.top);
            if (const auto z = numeric(out.value); z && z->imag() == 0.0) out.number = z->real();
            const std::size_t k = ir.trajectories;
            if (k == 0 || !noisy || !numeric(out.value) || !live) return out;

            std::vector<cd> vals;
            vals.reserve(k);
            std::vector<cd> diffs;
            for (std::size_t t = 0; t < k; ++t)
            {
                checkTime();
                auto m = makeMachine(splitmix(result.seed ^ splitmix(eventIndex * 1000003ULL + t + 1)));
                preparePrep(*m);
                for (const Operation& o : history) m->append(o);
                cur = m.get();
                try
                {
                    const auto a = numeric(comp.evalRuntime(e, sc, *this));
                    vals.push_back(a.value_or(0.0));
                    if (rhs) diffs.push_back(a.value_or(0.0) - numeric(comp.evalRuntime(*rhs, sc, *this)).value_or(0.0));
                }
                catch (...)
                {
                    cur = live.get();
                    throw;
                }
                cur = live.get();
            }
            // A value no trajectory changes (a classical expression) keeps the live pass's exact value.
            auto constant = [](const std::vector<cd>& v) { return std::ranges::all_of(v, [&](const cd& x) { return x == v.front(); }); };
            if (!vals.empty() && constant(vals) && (!rhs || constant(diffs)) && numeric(out.value) == vals.front()) return out;
            auto stats = [&](const std::vector<cd>& v, cd& mean, double& se)
            {
                if (constant(v))
                {
                    mean = v.empty() ? cd{} : v.front();
                    se = 0.0;
                    return;
                }
                mean = 0.0;
                for (const cd& x : v) mean += x;
                mean /= static_cast<double>(v.size());
                double var = 0.0;
                for (const cd& x : v) var += std::norm(x - mean);
                se = v.size() > 1 ? std::sqrt(var / static_cast<double>(v.size() - 1) / static_cast<double>(v.size())) : 0.0;
            };
            cd mean;
            double se = 0.0;
            stats(vals, mean, se);
            out.value = Num::fromDouble(mean);
            out.json = mean.imag() == 0.0 ? Json(mean.real()) : complexJson(mean);
            out.number = mean.imag() == 0.0 ? std::optional<double>(mean.real()) : std::nullopt;
            out.stderrValue = se > 0.0 ? std::optional<double>(se) : std::nullopt;
            if (rhs)
            {
                cd dm;
                double dse = 0.0;
                stats(diffs, dm, dse);
                out.stderrValue = dse > 0.0 ? std::optional<double>(dse) : std::nullopt;
            }
            return out;
        }

        bool truth(const Value& v) const
        {
            if (const auto* b = v.get<bool>()) return *b;
            if (const auto* c = v.get<CondV>()) return evalCondNode(*c->root, creg());
            if (const auto* bits = v.get<BitsV>()) return bitsText(*bits, creg()).find('1') != std::string::npos;
            throw RuntimeError{"E4001", "assert needs a Bool", {}};
        }

        // One comparison of an assert. Numeric comparisons of noisy readouts are averaged over the
        // trajectories (sample); lhs, rhs and tol are recorded for the report.
        bool compareOnce(const Expr& e, const ConstScopePtr& sc, AssertRecord& rec)
        {
            const Sampled l = sample(*e.kids[0], sc, e.kids[1].get());
            const Value r = comp.evalRuntime(*e.kids[1], sc, *this);
            rec.lhs = l.json;
            rec.rhs = valueJson(r, creg(), opt.top);
            rec.stderrValue = l.stderrValue;
            rec.tol.reset();
            const auto lz = numeric(l.value), rz = numeric(r);
            if (!lz || !rz) return truth(comp.evalRuntime(e, sc, *this));
            const cd d = *lz - *rz;
            const double se = l.stderrValue.value_or(0.0);
            switch (e.op)
            {
                case Tok::Approx:
                {
                    double tol = 1e-9;
                    if (e.kids.size() > 2)
                    {
                        const auto tz = numeric(comp.evalRuntime(*e.kids[2], sc, *this));
                        if (!tz) throw RuntimeError{"E4001", "the tolerance must be a number", e.kids[2]->span};
                        tol = tz->real();
                    }
                    rec.tol = tol;
                    return std::abs(d) <= tol + 2.0 * se;
                }
                case Tok::EqEq: case Tok::NotEq: return truth(comp.evalRuntime(e, sc, *this));
                case Tok::Less: return lz->real() < rz->real();
                case Tok::LessEq: return lz->real() <= rz->real();
                case Tok::Greater: return lz->real() > rz->real();
                case Tok::GreaterEq: return lz->real() >= rz->real();
                default: return truth(comp.evalRuntime(e, sc, *this));
            }
        }

        // `and` / `or` of comparisons check each comparison on its own, so every noisy readout in a
        // compound assert gets its trajectory average; the record keeps the deciding comparison.
        bool checkAssert(const Expr& e, const ConstScopePtr& sc, AssertRecord& rec)
        {
            if (e.kind == EK::Paren) return checkAssert(*e.kids[0], sc, rec);
            if (e.kind == EK::Binary && (e.op == Tok::KwAnd || e.op == Tok::KwOr))
            {
                const bool a = checkAssert(*e.kids[0], sc, rec);
                if (e.op == Tok::KwAnd ? !a : a) return a;
                return checkAssert(*e.kids[1], sc, rec);
            }
            if (e.kind == EK::Compare) return compareOnce(e, sc, rec);
            return truth(comp.evalRuntime(e, sc, *this));
        }

        bool assertion(const Event& ev)
        {
            AssertRecord rec;
            rec.span = ev.span;
            rec.text = ev.text;
            rec.passed = checkAssert(*ev.expr, ev.scope, rec);

            const bool passed = rec.passed;
            const std::string lhsText = rec.lhs.isNull() ? "" : jsonText(rec.lhs);
            const std::string rhsText = rec.rhs.isNull() ? "" : jsonText(rec.rhs);
            result.asserts.push_back(std::move(rec));
            if (passed) return true;
            std::string msg = std::format("assertion failed: {}", ev.text);
            if (!lhsText.empty()) msg += std::format(" (left = {}, right = {})", lhsText, rhsText);
            comp.diagnostics().error("E7001", ev.span, msg);
            result.exitCode = 3;
            return !opt.stopOnAssert;
        }

        Compiler& comp;
        Ir& ir;
        const ExecOptions& opt;
        ExecResult result;
        Clock::time_point start;
        std::vector<double> params;
        std::vector<Value> slots;
        std::unique_ptr<QuantumStateMachine> live;
        QuantumStateMachine* cur = nullptr;
        bool stab = false;
        std::size_t eventIndex = 0;

        struct Prep
        {
            bool dense = false;
            Eigen::VectorXcd amps;
            std::uint64_t basis = 0;
            std::vector<Operation> ops;
        };
        Prep prep;
        std::vector<Operation> history;     // core ops since the last prepare (for trajectories)
        std::vector<const IrOp*> irHistory; // IR ops since the last prepare (count, depth)
        bool noisy = false;
        std::uint64_t shotCounter = 0;
        double shotVar = 0.0;
    };
} // namespace


ExecResult execute(Compiler& compiler, const ExecOptions& options) { return Executor(compiler, options).run(); }

std::string countsKey(const Ir& ir, Outcome value)
{
    std::string key;
    for (std::size_t r = 0; r < ir.bregs.size(); ++r)
    {
        if (r) key += '|';
        for (std::size_t k = 0; k < ir.bregs[r].size; ++k) key += ((value >> (ir.bregs[r].offset + k)) & 1U) ? '1' : '0';
    }
    return key;
}

Json valueJson(const Value& v, std::uint64_t creg, std::size_t top)
{
    if (const auto* n = v.get<Num>()) return n->isConst() ? numJson(*n) : Json(nullptr);
    if (const auto* b = v.get<bool>()) return Json(*b);
    if (const auto* s = v.get<std::string>()) return Json(*s);
    if (const auto* bits = v.get<BitsV>()) return Json(bitsText(*bits, creg));
    if (const auto* c = v.get<CondV>()) return Json(evalCondNode(*c->root, creg));
    if (const auto* c = v.get<CountsV>()) return countsJson(c->counts, top);
    if (const auto* k = v.get<KetV>()) return ketJson(*k, top);
    if (const auto* bra = v.get<BraV>()) return ketJson(bra->ket, top);
    if (const auto* m = v.get<MatV>()) return matrixJson(m->m);
    if (const auto* l = v.get<ListV>())
    {
        Json a = Json::array();
        for (const ValuePtr& x : l->items) a.push(valueJson(*x, creg, top));
        return a;
    }
    return Json(std::string(typeName(v.type())));
}

Json documentHeader(std::string_view kind, bool ok, const Diagnostics& diags)
{
    Json j = Json::object();
    j["schema"] = std::format("noether.{}/1", kind);
    j["version"] = kVersion;
    j["lang"] = kLangVersion;
    j["ok"] = ok;
    j["diagnostics"] = diags.toJson();
    return j;
}

Json runJson(Compiler& compiler, const ExecResult& r, std::optional<double> compileMs, const ExecOptions& options)
{
    const Ir& ir = compiler.ir();
    const SourceManager& sm = compiler.sources();
    Json j = documentHeader("run", r.exitCode == 0 && !compiler.diagnostics().hasErrors(), compiler.diagnostics());
    j["sourceSha256"] = ir.sourceSha256;
    j["seed"] = r.seed;
    j["backend"] = ir.backend;
    j["backendReason"] = ir.backendReason;
    Json params = Json::object();
    const std::vector<double>& pv = options.params ? *options.params : ir.paramValues;
    for (const ParamInfo& p : ir.params)
    {
        if (!p.isVector) params[p.name] = pv[p.base];
        else
        {
            Json a = Json::array();
            for (std::size_t k = 0; k < p.size; ++k) a.push(pv[p.base + k]);
            params[p.name] = a;
        }
    }
    j["params"] = params;
    Json prints = Json::array();
    for (const PrintRecord& p : r.prints)
    {
        Json x = Json::object();
        x["line"] = lineOf(sm, p.span);
        x["label"] = p.label.empty() ? Json(nullptr) : Json(p.label);
        x["text"] = p.text;
        x["value"] = p.value;
        x["stderr"] = p.stderrValue ? Json(*p.stderrValue) : Json(nullptr);
        prints.push(std::move(x));
    }
    j["prints"] = prints;
    Json asserts = Json::array();
    for (const AssertRecord& a : r.asserts)
    {
        Json x = Json::object();
        x["line"] = lineOf(sm, a.span);
        x["text"] = a.text;
        x["passed"] = a.passed;
        x["lhs"] = a.lhs;
        x["rhs"] = a.rhs;
        x["tol"] = a.tol ? Json(*a.tol) : Json(nullptr);
        if (a.stderrValue) x["stderr"] = *a.stderrValue;
        asserts.push(std::move(x));
    }
    j["asserts"] = asserts;
    Json runs = Json::array();
    for (const RunRecord& run : r.runs)
    {
        Json x = Json::object();
        x["name"] = run.name;
        x["line"] = lineOf(sm, run.span);
        x["shots"] = run.shots;
        x["keys"] = "c0-leftmost";
        x["counts"] = countsJson(run.counts, options.top);
        runs.push(std::move(x));
    }
    j["runs"] = runs;
    j["resources"] = resourcesJson(resources(ir));
    if (options.timing)
    {
        Json t = Json::object();
        if (compileMs) t["compileMs"] = *compileMs;
        t["executeMs"] = r.executeMs;
        j["timing"] = t;
    }
    return j;
}

std::string runText(Compiler& compiler, const ExecResult& r, const ExecOptions& options)
{
    const SourceManager& sm = compiler.sources();
    std::string out;
    for (const PrintRecord& p : r.prints)
    {
        out += std::format("{} = {}", p.label.empty() ? p.text : p.label, jsonText(p.value));
        if (p.stderrValue) out += std::format(" ± {}", jsonNumber(*p.stderrValue));
        out += "\n";
    }
    for (const RunRecord& run : r.runs)
        out += std::format("{} ← run {} (line {}): {}\n", run.name, run.shots, lineOf(sm, run.span),
                           countsJson(run.counts, options.top).dump(-1));
    std::size_t passed = 0;
    for (const AssertRecord& a : r.asserts) passed += a.passed ? 1 : 0;
    if (!r.asserts.empty()) out += std::format("asserts: {}/{} passed\n", passed, r.asserts.size());
    return out;
}

} // namespace Noether
