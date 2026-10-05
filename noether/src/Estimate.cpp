#include "Estimate.hpp"

#include "Backend.hpp"

#include <algorithm>
#include <cmath>
#include <format>



namespace Noether
{

bool isTGate(const IrOp& op)
{
    if (!op.controls.empty() || !op.negControls.empty()) return false;
    if (op.kind == GK::T || op.kind == GK::Tdg) return true;
    if (op.kind != GK::RZ && op.kind != GK::P && op.kind != GK::PauliRot) return false;
    if (op.kind == GK::PauliRot && op.paulis.find_first_not_of('Z') != std::string::npos) return false;
    const Affine& a = op.angles[0];
    if (!a.isConst() || !a.c.exact() || !a.c.rat().isZero()) return false;
    const Rational p = a.c.piCoef();
    return p.d == 4 && p.n % 2 != 0;
}

Resources resources(const Ir& ir)
{
    Resources r;
    r.qubits = ir.nQubits;
    r.clbits = ir.nClbits;
    r.nparams = ir.paramNames.size();
    std::vector<std::size_t> level(ir.nQubits, 0), level2(ir.nQubits, 0), levelT(ir.nQubits, 0);
    for (const Event& ev : ir.events)
    {
        if (ev.kind == EvK::Prepare)
        {
            std::ranges::fill(level, 0);
            std::ranges::fill(level2, 0);
            std::ranges::fill(levelT, 0);
            continue;
        }
        if (ev.kind != EvK::Op || ev.op.noise) continue;
        const IrOp& op = ev.op;
        ++r.ops;
        ++r.opsByKind[std::string(gkName(op.kind))];
        if (!op.clifford) r.clifford = false;
        if (op.kind == GK::MeasureZ || op.kind == GK::MeasurePauli) ++r.measurements;
        const QubitList qs = op.qubits();
        if (qs.empty()) continue;
        std::size_t l = 0, l2 = 0, lt = 0;
        for (const Qubit q : qs)
        {
            l = std::max(l, level[q]);
            l2 = std::max(l2, level2[q]);
            lt = std::max(lt, levelT[q]);
        }
        ++l;
        if (qs.size() >= 2)
        {
            ++l2;
            ++r.count2q;
        }
        if (isTGate(op))
        {
            ++lt;
            ++r.tcount;
        }
        for (const Qubit q : qs)
        {
            level[q] = l;
            level2[q] = l2;
            levelT[q] = lt;
        }
        r.depth = std::max(r.depth, l);
        r.depth2q = std::max(r.depth2q, l2);
        r.tdepth = std::max(r.tdepth, lt);
    }
    r.peakBytes = ir.backend.empty() ? 0 : registerBytes(ir.backend, ir.nQubits);
    return r;
}

Json resourcesJson(const Resources& r)
{
    Json j = Json::object();
    j["qubits"] = r.qubits;
    j["clbits"] = r.clbits;
    j["ops"] = r.ops;
    j["depth"] = r.depth;
    j["depth2q"] = r.depth2q;
    j["count2q"] = r.count2q;
    j["tcount"] = r.tcount;
    j["clifford"] = r.clifford;
    j["peakBytes"] = r.peakBytes;
    return j;
}

namespace
{
    std::string readoutCost(const ReadoutUse& u, const std::string& backend)
    {
        const bool stab = backend == "stabilizer";
        if (u.kind == "pauli") return stab ? std::format("O({}·N²/64)", u.terms) : std::format("O({}·2^N)", u.terms);
        if (u.kind == "probability") return stab ? "O(N²/64)" : "O(1)";
        if (u.kind == "entropy") return stab ? "O(N³/64)" : std::format("O(2^(N+{}))", u.width);
        if (u.kind == "dense-observable" || u.kind == "density") return std::format("O(2^(N+{}))", u.width);
        if (u.kind == "bits" || u.kind == "counts") return "O(1)";
        return "O(2^N)";
    }
} // namespace

Json estimateJson(const Ir& ir, const SourceManager& sm)
{
    const Resources r = resources(ir);
    Json j = Json::object();
    j["resources"] = resourcesJson(r);
    j["backend"] = ir.backend;
    j["backendReason"] = ir.backendReason;
    Json kinds = Json::object();
    for (const auto& [k, n] : r.opsByKind) kinds[k] = n;
    j["opsByKind"] = kinds;
    j["tdepth"] = r.tdepth;
    j["nparams"] = r.nparams;
    Json reads = Json::array();
    for (const ReadoutUse& u : ir.readouts)
    {
        Json x = Json::object();
        x["line"] = u.span.file < sm.size() ? static_cast<std::int64_t>(sm.file(u.span.file).lineCol(u.span.begin).line) : 0;
        x["kind"] = u.kind;
        x["cost"] = readoutCost(u, ir.backend);
        x["backendOk"] = !(ir.backend == "stabilizer" && u.statevectorOnly);
        reads.push(std::move(x));
    }
    j["readouts"] = reads;

    // Rough single-thread model: a state-vector gate streams 2^N amplitudes (~1 ns each); a tableau
    // gate touches N/64 words and a measurement N²/64.
    const double n = static_cast<double>(ir.nQubits);
    double seconds = 0.0;
    if (ir.backend == "stabilizer")
        seconds = static_cast<double>(r.ops - r.measurements) * (n / 64.0) * 2e-9 +
                  static_cast<double>(r.measurements) * (n * n / 64.0) * 1e-9;
    else seconds = static_cast<double>(r.ops) * std::ldexp(1.0, static_cast<int>(ir.nQubits)) * 1e-9;
    j["estimatedSeconds"] = seconds;
    j["estimatedBytes"] = r.peakBytes;
    return j;
}

} // namespace Noether
