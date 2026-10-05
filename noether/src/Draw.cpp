#include "Tools.hpp"

#include <algorithm>
#include <format>



namespace Noether
{

namespace
{
    std::uint32_t lineOf(const SourceManager& sm, Span s) { return s.file < sm.size() ? sm.file(s.file).lineCol(s.begin).line : 0; }

    Json qubitsJson(const QubitList& q)
    {
        Json a = Json::array();
        for (const Qubit x : q) a.push(static_cast<std::uint64_t>(x));
        return a;
    }

    Json angleJson(const Affine& a, const std::vector<std::string>& names)
    {
        Json j = Json::object();
        j["exact"] = a.c.exact();
        if (a.c.exact())
        {
            j["piNum"] = a.c.piCoef().n;
            j["piDen"] = a.c.piCoef().d;
            j["remNum"] = a.c.rat().n;
            j["remDen"] = a.c.rat().d;
        }
        else j["rem"] = a.c.value();
        Json ps = Json::object();
        for (const auto& [idx, coef] : a.terms) ps[names.at(idx)] = coef.value();
        j["params"] = ps;
        j["text"] = a.str(names);
        return j;
    }

    std::string angleText(const IrOp& op, const std::vector<std::string>& names)
    {
        if (op.angles.empty()) return {};
        std::string s = "(";
        for (std::size_t k = 0; k < op.angles.size(); ++k) s += (k ? ", " : "") + op.angles[k].str(names);
        return s + ")";
    }

    std::string clbitName(const Ir& ir, std::size_t c)
    {
        for (const RegInfo& r : ir.bregs)
            if (c >= r.offset && c < r.offset + r.size) return std::format("{}[{}]", r.name, c - r.offset);
        return std::format("c{}", c);
    }

    std::string qubitName(const Ir& ir, Qubit q)
    {
        for (const RegInfo& r : ir.qregs)
            if (q >= r.offset && q < r.offset + r.size) return std::format("{}[{}]", r.name, q - r.offset);
        return std::format("q{}", q);
    }

    std::string condText(const Ir& ir, const Cube& c)
    {
        std::string s;
        for (std::size_t b = 0; b < 64; ++b)
            if ((c.mask >> b) & 1U) s += (s.empty() ? "" : ",") + clbitName(ir, b) + "=" + (((c.value >> b) & 1U) ? "1" : "0");
        return s;
    }

    std::string opLabel(const IrOp& op, const std::vector<std::string>& names)
    {
        std::string base(gkName(op.kind));
        if (op.kind == GK::PauliRot) base = "exp(" + op.paulis + ")";
        if (op.kind == GK::MeasurePauli) base = std::string(op.negate ? "M-" : "M") + op.paulis;
        if (op.kind == GK::Channel) base = op.channel->name;
        return base + angleText(op, names);
    }
} // namespace


Json irJson(Compiler& compiler)
{
    const Ir& ir = compiler.ir();
    const SourceManager& sm = compiler.sources();
    Json events = Json::array();
    Json ops = Json::array();
    for (const Event& ev : ir.events)
    {
        Json e = Json::object();
        e["line"] = lineOf(sm, ev.span);
        switch (ev.kind)
        {
            case EvK::Op:
            {
                const IrOp& op = ev.op;
                Json o = Json::object();
                o["kind"] = gkName(op.kind);
                o["controls"] = qubitsJson(op.controls);
                o["negControls"] = qubitsJson(op.negControls);
                o["targets"] = qubitsJson(op.targets);
                Json angles = Json::array();
                for (const Affine& a : op.angles) angles.push(angleJson(a, ir.paramNames));
                o["angles"] = angles;
                if (!op.paulis.empty()) o["pauli"] = std::string(op.negate ? "-" : "") + op.paulis;
                o["clbit"] = op.clbit ? Json(static_cast<std::uint64_t>(*op.clbit)) : Json(nullptr);
                if (op.cond)
                {
                    Json c = Json::object();
                    c["mask"] = op.cond->mask;
                    c["value"] = op.cond->value;
                    o["condition"] = c;
                }
                else o["condition"] = nullptr;
                o["clifford"] = op.clifford;
                o["noise"] = op.noise;
                if (op.kind == GK::Channel) o["channel"] = op.channel->name;
                o["line"] = lineOf(sm, ev.span);
                ops.push(o);
                e["kind"] = "op";
                e["op"] = ops.size() - 1;
                break;
            }
            case EvK::Prepare: e["kind"] = "prepare"; break;
            case EvK::Print:
            {
                e["kind"] = "print";
                Json items = Json::array();
                for (const PrintItemIr& it : ev.items) items.push(it.label.empty() ? Json(it.text) : Json(it.label));
                e["items"] = items;
                break;
            }
            case EvK::Assert: e["kind"] = "assert"; e["text"] = ev.text; break;
            case EvK::Run: e["kind"] = "run"; e["name"] = ev.name; e["shots"] = ev.shots; break;
            case EvK::Let: e["kind"] = "let"; e["name"] = ev.name; break;
        }
        events.push(std::move(e));
    }
    Json j = Json::object();
    j["qubits"] = ir.nQubits;
    j["clbits"] = ir.nClbits;
    j["backend"] = ir.backend;
    j["backendReason"] = ir.backendReason;
    Json params = Json::array();
    for (std::size_t k = 0; k < ir.paramNames.size(); ++k)
    {
        Json p = Json::object();
        p["name"] = ir.paramNames[k];
        p["value"] = ir.paramValues[k];
        params.push(std::move(p));
    }
    j["params"] = params;
    j["ops"] = ops;
    j["events"] = events;
    return j;
}

std::string irText(Compiler& compiler)
{
    const Ir& ir = compiler.ir();
    const SourceManager& sm = compiler.sources();
    std::string out = std::format("# {} qubits, {} clbits, backend {} ({})\n", ir.nQubits, ir.nClbits, ir.backend, ir.backendReason);
    for (const Event& ev : ir.events)
    {
        const std::uint32_t line = lineOf(sm, ev.span);
        switch (ev.kind)
        {
            case EvK::Op:
            {
                const IrOp& op = ev.op;
                std::string s = std::format("{:>5}  {}", line, opLabel(op, ir.paramNames));
                auto list = [&](const QubitList& q)
                {
                    std::string t;
                    for (const Qubit x : q) t += (t.empty() ? "" : ",") + qubitName(ir, x);
                    return t;
                };
                if (!op.controls.empty() || !op.negControls.empty())
                {
                    s += " C{" + list(op.controls);
                    if (!op.negControls.empty()) s += (op.controls.empty() ? "¬" : ",¬") + list(op.negControls);
                    s += "}";
                }
                s += " " + list(op.targets);
                if (op.clbit) s += " → " + clbitName(ir, *op.clbit);
                if (op.cond) s += " if " + condText(ir, *op.cond);
                if (op.clifford) s += "  [clifford]";
                if (op.noise) s += "  [noise]";
                out += s + "\n";
                break;
            }
            case EvK::Prepare: out += std::format("{:>5}  prepare\n", line); break;
            case EvK::Print: out += std::format("{:>5}  print\n", line); break;
            case EvK::Assert: out += std::format("{:>5}  assert {}\n", line, ev.text); break;
            case EvK::Run: out += std::format("{:>5}  {} ← run {}\n", line, ev.name, ev.shots); break;
            case EvK::Let: out += std::format("{:>5}  let {}\n", line, ev.name); break;
        }
    }
    return out;
}


namespace
{
    struct Placed
    {
        const IrOp* op = nullptr;
        bool barrier = false;
    };

    // ASAP layers. An op occupies every wire between its lowest and highest qubit so that its
    // connector can be drawn; conditions wait for the measurements of their clbits.
    std::vector<std::vector<Placed>> layers(const Ir& ir)
    {
        std::vector<std::vector<Placed>> out;
        std::vector<std::size_t> qlevel(ir.nQubits, 0);
        std::vector<std::size_t> clevel(64, 0);
        for (const Event& ev : ir.events)
        {
            if (ev.kind == EvK::Prepare)
            {
                std::size_t l = 0;
                for (const std::size_t x : qlevel) l = std::max(l, x);
                if (l == 0) continue;
                out.resize(l + 1);
                out[l].push_back({nullptr, true});
                std::ranges::fill(qlevel, l + 1);
                continue;
            }
            if (ev.kind != EvK::Op) continue;
            const IrOp& op = ev.op;
            const QubitList qs = op.qubits();
            if (qs.empty()) continue;
            const auto [lo, hi] = std::ranges::minmax(qs);
            std::size_t l = 0;
            for (Qubit q = lo; q <= hi; ++q) l = std::max(l, qlevel[q]);
            if (op.cond)
                for (std::size_t b = 0; b < 64; ++b)
                    if ((op.cond->mask >> b) & 1U) l = std::max(l, clevel[b]);
            if (op.clbit && *op.clbit < 64) l = std::max(l, clevel[*op.clbit]);
            if (out.size() <= l) out.resize(l + 1);
            out[l].push_back({&op, false});
            for (Qubit q = lo; q <= hi; ++q) qlevel[q] = l + 1;
            if (op.clbit && *op.clbit < 64) clevel[*op.clbit] = l + 1;
        }
        return out;
    }
} // namespace

Json drawJson(const Ir& ir)
{
    Json out = Json::array();
    for (const auto& layer : layers(ir))
    {
        Json l = Json::array();
        for (const Placed& p : layer)
        {
            Json x = Json::object();
            if (p.barrier)
            {
                x["op"] = "prepare";
                x["qubits"] = Json::array();
            }
            else
            {
                x["op"] = opLabel(*p.op, ir.paramNames);
                x["qubits"] = qubitsJson(p.op->qubits());
                if (p.op->cond) x["condition"] = condText(ir, *p.op->cond);
                if (p.op->clbit) x["clbit"] = static_cast<std::uint64_t>(*p.op->clbit);
            }
            l.push(std::move(x));
        }
        out.push(std::move(l));
    }
    return out;
}

std::string drawText(const Ir& ir, bool ascii)
{
    const auto ls = layers(ir);
    if (ir.nQubits == 0) return "(no qubits)\n";
    if (ir.nQubits > 64 || ls.size() > 400)
        return std::format("circuit too large to draw as text: {} qubits, {} layers (use --json)\n", ir.nQubits, ls.size());

    const std::string wire = ascii ? "-" : "─";
    const std::string vert = ascii ? "|" : "│";
    const std::string cross = ascii ? "+" : "┼";
    const std::string dot = ascii ? "*" : "●";
    const std::string odot = ascii ? "o" : "○";
    const std::string plus = ascii ? "(+)" : "⊕";
    const std::string swp = ascii ? "x" : "×";
    const std::string bar = ascii ? "#" : "┃";

    std::vector<std::string> rows(ir.nQubits);
    std::size_t nameWidth = 0;
    for (Qubit q = 0; q < ir.nQubits; ++q) nameWidth = std::max(nameWidth, qubitName(ir, q).size());
    for (Qubit q = 0; q < ir.nQubits; ++q)
    {
        const std::string n = qubitName(ir, q);
        rows[q] = n + std::string(nameWidth - n.size(), ' ') + ": " + wire;
    }

    for (const auto& layer : ls)
    {
        std::vector<std::string> cell(ir.nQubits);
        std::vector<bool> through(ir.nQubits, false);
        for (const Placed& p : layer)
        {
            if (p.barrier)
            {
                for (Qubit q = 0; q < ir.nQubits; ++q) cell[q] = bar;
                continue;
            }
            const IrOp& op = *p.op;
            std::string label = opLabel(op, ir.paramNames);
            if (ascii) label = asciiSpelling(label);
            if (op.clbit) label += "->" + clbitName(ir, *op.clbit);
            if (op.cond) label += "[" + condText(ir, *op.cond) + "]";
            for (const Qubit c : op.controls) cell[c] = dot;
            for (const Qubit c : op.negControls) cell[c] = odot;
            const QubitList& t = op.targets;
            switch (op.kind)
            {
                case GK::CNOT: cell[t[0]] = dot; cell[t[1]] = plus; break;
                case GK::CZ: cell[t[0]] = dot; cell[t[1]] = dot; break;
                case GK::SWAP: cell[t[0]] = swp; cell[t[1]] = swp; break;
                case GK::Toffoli: cell[t[0]] = dot; cell[t[1]] = dot; cell[t[2]] = plus; break;
                case GK::Fredkin: cell[t[0]] = dot; cell[t[1]] = swp; cell[t[2]] = swp; break;
                case GK::X:
                    if (!op.controls.empty() || !op.negControls.empty()) cell[t[0]] = plus;
                    else cell[t[0]] = label;
                    break;
                default:
                    for (const Qubit q : t) cell[q] = label;
                    break;
            }
            if (op.kind == GK::GPhase && !op.controls.empty()) cell[op.controls.back()] = label;
            const QubitList qs = op.qubits();
            const auto [lo, hi] = std::ranges::minmax(qs);
            for (Qubit q = lo + 1; q < hi; ++q)
                if (cell[q].empty()) through[q] = true;
        }
        std::size_t w = 1;
        for (const std::string& c : cell) w = std::max(w, codePointCount(c));
        for (Qubit q = 0; q < ir.nQubits; ++q)
        {
            std::string c = cell[q];
            if (c.empty()) c = through[q] ? cross : wire;
            const std::size_t pad = w - codePointCount(c);
            std::string fill;
            for (std::size_t k = 0; k < pad; ++k) fill += wire;
            rows[q] += wire + c + fill + wire;
        }
    }
    std::string out;
    for (const std::string& r : rows) out += r + "\n";
    return out;
}

} // namespace Noether
