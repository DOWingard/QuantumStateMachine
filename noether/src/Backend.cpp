#include "Backend.hpp"

#include <QuantumState.hpp>
#include <StabilizerState.hpp>

#include <format>



namespace Noether
{

namespace
{
    // Exact multiple of π·step (step = 1/2 or 1) with no rational remainder.
    bool multipleOfPi(const Affine& a, std::int64_t denominator)
    {
        if (!a.isConst() || !a.c.exact() || !a.c.rat().isZero()) return false;
        const Rational p = a.c.piCoef();
        return (p.n * denominator) % p.d == 0;
    }

    bool isPauliKind(GK k) { return k == GK::X || k == GK::Y || k == GK::Z; }

    std::string opLabel(const IrOp& op)
    {
        std::string s(gkName(op.kind));
        if (!op.controls.empty() || !op.negControls.empty())
            s = std::format("C_{{{} control{}}}({})", op.controls.size() + op.negControls.size(),
                            op.controls.size() + op.negControls.size() == 1 ? "" : "s", s);
        return s;
    }
} // namespace

bool isClifford(const IrOp& op)
{
    const std::size_t nControls = op.controls.size() + op.negControls.size();
    if (nControls > 1) return false;
    if (nControls == 1)
    {
        // CX, CY, CZ; a controlled global phase is a phase gate on the control.
        if (isPauliKind(op.kind)) return true;
        if (op.kind == GK::I) return true;
        if (op.kind == GK::GPhase) return multipleOfPi(op.angles[0], 2);
        return false;
    }
    switch (op.kind)
    {
        case GK::I: case GK::X: case GK::Y: case GK::Z: case GK::H: case GK::S: case GK::Sdg: case GK::SX: case GK::SXdg:
        case GK::CNOT: case GK::CZ: case GK::SWAP: case GK::GPhase:
        case GK::MeasureZ: case GK::MeasurePauli: case GK::Reset:
            return true;
        case GK::T: case GK::Tdg: case GK::Toffoli: case GK::Fredkin: case GK::Matrix:
            return false;
        case GK::RX: case GK::RY: case GK::RZ: case GK::P: case GK::PauliRot:
            return multipleOfPi(op.angles[0], 2);
        case GK::CP:
            return multipleOfPi(op.angles[0], 1);
        case GK::U3:
            return multipleOfPi(op.angles[0], 2) && multipleOfPi(op.angles[1], 2) && multipleOfPi(op.angles[2], 2);
        case GK::Channel:
            return op.channel && op.channel->pauli;
    }
    return false;
}

void tagClifford(Ir& ir)
{
    for (Event& ev : ir.events)
        if (ev.kind == EvK::Op) ev.op.clifford = isClifford(ev.op);
}

std::uint64_t registerBytes(const std::string& backend, std::size_t nQubits)
{
    if (backend == "stabilizer")
    {
        const std::uint64_t n = nQubits;
        return std::max<std::uint64_t>(1, n * n / 2);
    }
    if (nQubits >= 60) return UINT64_MAX;
    return std::uint64_t{16} << nQubits;
}

void selectBackend(Ir& ir, Diagnostics& d)
{
    tagClifford(ir);
    const std::size_t n = ir.nQubits;

    const IrOp* firstNonClifford = nullptr;
    Span firstNonCliffordSpan;
    const Event* firstDensePrepare = nullptr;
    for (const Event& ev : ir.events)
    {
        if (ev.kind == EvK::Op && !ev.op.clifford && !firstNonClifford)
        {
            firstNonClifford = &ev.op;
            firstNonCliffordSpan = ev.op.span;
        }
        if (ev.kind == EvK::Prepare && ev.ket && !ev.ket->isStabilizerProduct() && !firstDensePrepare) firstDensePrepare = &ev;
    }
    const ReadoutUse* firstSvReadout = nullptr;
    for (const ReadoutUse& r : ir.readouts)
        if (r.statevectorOnly)
        {
            firstSvReadout = &r;
            break;
        }

    auto useStatevector = [&](std::string reason)
    {
        if (n > Qputer::kMaxQubits)
        {
            Diagnostic& diag = d.error("E6003", firstNonClifford   ? firstNonCliffordSpan
                                                : firstDensePrepare ? firstDensePrepare->span
                                                : firstSvReadout    ? firstSvReadout->span
                                                : !ir.qregs.empty() ? ir.qregs.front().span
                                                                    : Span{UINT32_MAX, 0, 0},
                                       std::format("{} qubits need the state vector ({}), which holds at most {}", n, reason,
                                                   Qputer::kMaxQubits));
            if (!ir.qregs.empty()) diag.note(ir.qregs.front().span, "qubits declared here");
            return;
        }
        ir.backend = "statevector";
        ir.backendReason = std::move(reason);
    };

    if (ir.backendRequest == "statevector")
    {
        useStatevector("requested");
        return;
    }
    if (ir.backendRequest == "stabilizer")
    {
        bool ok = true;
        if (firstNonClifford)
        {
            d.error("E6001", firstNonCliffordSpan,
                    std::format("`{}` is not a Clifford operation, so the stabilizer backend cannot run it", opLabel(*firstNonClifford)));
            ok = false;
        }
        if (firstDensePrepare)
        {
            d.error("E6001", firstDensePrepare->span, "the stabilizer backend prepares only products of |0⟩ |1⟩ |±⟩ |±i⟩");
            ok = false;
        }
        for (const ReadoutUse& r : ir.readouts)
            if (r.statevectorOnly)
            {
                d.error("E6002", r.span, std::format("the {} readout needs the state vector", r.kind));
                ok = false;
            }
        if (ok)
        {
            ir.backend = "stabilizer";
            ir.backendReason = "requested";
        }
        return;
    }

    // auto
    if (!firstNonClifford && !firstDensePrepare && !firstSvReadout)
    {
        ir.backend = "stabilizer";
        ir.backendReason = "every operation is Clifford";
        return;
    }
    std::string why;
    if (firstNonClifford) why = std::format("`{}` is not Clifford", opLabel(*firstNonClifford));
    else if (firstDensePrepare) why = "a superposition is prepared";
    else why = std::format("the {} readout needs it", firstSvReadout->kind);
    if (n > Qputer::kMaxQubits)
    {
        const Span at = firstNonClifford   ? firstNonCliffordSpan
                        : firstDensePrepare ? firstDensePrepare->span
                                            : firstSvReadout->span;
        const std::string_view code = !firstNonClifford && !firstDensePrepare ? "E6002" : "E6001";
        Diagnostic& diag = d.error(std::string(code), at,
                                   std::format("{}, so {} qubits need the state vector, which holds at most {}", why, n,
                                               Qputer::kMaxQubits));
        if (!ir.qregs.empty()) diag.note(ir.qregs.front().span, "qubits declared here");
        return;
    }
    useStatevector(why);
}

} // namespace Noether
