#include "Tools.hpp"

#include "Lowering.hpp"

#include <format>



namespace Noether
{

using Qputer::OpKind;
using Qputer::Operation;

namespace
{
    std::string num(double v) { return jsonNumber(v); }
} // namespace

// OpenQASM 3 over stdgates.inc. Each Noether register becomes a register of the same name; params
// are exported with their bound values; the q0-leftmost text convention needs no conversion here
// because QASM addresses qubits by index.
std::string qasmExport(const Ir& ir, std::string& error)
{
    std::string out = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\n";
    auto qname = [&](Qubit q)
    {
        for (const RegInfo& r : ir.qregs)
            if (q >= r.offset && q < r.offset + r.size) return std::format("{}[{}]", asciiSpelling(r.name), q - r.offset);
        return std::format("q[{}]", q);
    };
    auto cname = [&](std::size_t c)
    {
        for (const RegInfo& r : ir.bregs)
            if (c >= r.offset && c < r.offset + r.size) return std::format("{}[{}]", asciiSpelling(r.name), c - r.offset);
        return std::format("c[{}]", c);
    };
    for (const RegInfo& r : ir.qregs) out += std::format("qubit[{}] {};\n", r.size, asciiSpelling(r.name));
    for (const RegInfo& r : ir.bregs) out += std::format("bit[{}] {};\n", r.size, asciiSpelling(r.name));

    auto list = [&](const QubitList& qs)
    {
        std::string s;
        for (const Qubit q : qs) s += (s.empty() ? "" : ", ") + qname(q);
        return s;
    };
    for (const Event& ev : ir.events)
    {
        if (ev.kind == EvK::Prepare)
        {
            if (!ev.ket->isStabilizerProduct())
            {
                error = "a superposition prepare has no OpenQASM form";
                return {};
            }
            for (Qubit q = 0; q < ir.nQubits; ++q) out += std::format("reset {};\n", qname(q));
            std::uint64_t basis = 0;
            std::vector<Operation> ops = productPreparation(*ev.ket, basis);
            for (Qubit q = 0; q < std::min<std::size_t>(ir.nQubits, 64); ++q)
                if ((basis >> q) & 1U) out += std::format("x {};\n", qname(q));
            for (const Operation& o : ops)
            {
                const char* g = o.kind == OpKind::H ? "h" : o.kind == OpKind::S ? "s" : "x";
                out += std::format("{} {};\n", g, qname(o.targets[0]));
            }
            continue;
        }
        if (ev.kind != EvK::Op) continue;
        const IrOp& irop = ev.op;
        if (irop.kind == GK::Channel)
        {
            out += std::format("// noise channel {} on {} omitted\n", irop.channel->name, list(irop.targets));
            continue;
        }
        if (irop.kind == GK::Matrix)
        {
            error = "a dense matrix gate has no OpenQASM form";
            return {};
        }
        for (const Operation& o : lower(irop, ir.paramValues, false))
        {
            std::string stmt;
            const QubitList& t = o.targets;
            const QubitList& c = o.controls;
            const std::string p0 = o.params.empty() ? "" : num(o.params[0]);
            switch (o.kind)
            {
                case OpKind::X: stmt = "x " + list(t); break;
                case OpKind::Y: stmt = "y " + list(t); break;
                case OpKind::Z: stmt = "z " + list(t); break;
                case OpKind::H: stmt = "h " + list(t); break;
                case OpKind::S: stmt = "s " + list(t); break;
                case OpKind::Sdg: stmt = "sdg " + list(t); break;
                case OpKind::T: stmt = "t " + list(t); break;
                case OpKind::Tdg: stmt = "tdg " + list(t); break;
                case OpKind::SX: stmt = "sx " + list(t); break;
                case OpKind::RX: stmt = std::format("rx({}) {}", p0, list(t)); break;
                case OpKind::RY: stmt = std::format("ry({}) {}", p0, list(t)); break;
                case OpKind::RZ: stmt = std::format("rz({}) {}", p0, list(t)); break;
                case OpKind::Phase: stmt = std::format("p({}) {}", p0, list(t)); break;
                case OpKind::U3: stmt = std::format("U({}, {}, {}) {}", p0, num(o.params[1]), num(o.params[2]), list(t)); break;
                case OpKind::CNOT: stmt = std::format("cx {}, {}", qname(c[0]), qname(t[0])); break;
                case OpKind::CZ: stmt = std::format("cz {}, {}", qname(c[0]), qname(t[0])); break;
                case OpKind::CPhase: stmt = std::format("cp({}) {}, {}", p0, qname(c[0]), qname(t[0])); break;
                case OpKind::Swap: stmt = "swap " + list(t); break;
                case OpKind::Toffoli: stmt = std::format("ccx {}, {}", list(c), qname(t[0])); break;
                case OpKind::Fredkin: stmt = std::format("cswap {}, {}", qname(c[0]), list(t)); break;
                case OpKind::MCX: stmt = std::format("ctrl({}) @ x {}, {}", c.size(), list(c), qname(t[0])); break;
                case OpKind::MCZ: stmt = std::format("ctrl({}) @ z {}", t.size() - 1, list(t)); break;
                case OpKind::MCPhase: stmt = std::format("ctrl({}) @ p({}) {}", t.size() - 1, p0, list(t)); break;
                case OpKind::Measure:
                    stmt = o.clbit ? std::format("{} = measure {}", cname(*o.clbit), qname(t[0])) : "measure " + qname(t[0]);
                    break;
                case OpKind::Reset: stmt = "reset " + qname(t[0]); break;
                case OpKind::Unitary:
                    error = "a controlled dense gate has no OpenQASM form";
                    return {};
                case OpKind::PauliChannel:
                case OpKind::Kraus: continue;
            }
            if (o.condition)
            {
                std::string cond;
                for (std::size_t b = 0; b < 64; ++b)
                    if ((o.condition->mask >> b) & 1U)
                        cond += (cond.empty() ? "" : " && ") + std::string(((o.condition->value >> b) & 1U) ? "" : "!") + cname(b);
                stmt = std::format("if ({}) {{ {}; }}", cond, stmt);
                out += stmt + "\n";
            }
            else out += stmt + ";\n";
        }
    }
    return out;
}

} // namespace Noether
