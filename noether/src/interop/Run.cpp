#include "interop/Run.hpp"

#include "interop/Lowering.hpp"

#include <QuantumState.hpp>

#include <algorithm>
#include <bit>
#include <format>
#include <map>
#include <random>



namespace Noether::Interop
{

using Qputer::OpKind;
using Qputer::Operation;

namespace
{
    bool isChannel(const Operation& o) { return o.kind == OpKind::PauliChannel || o.kind == OpKind::Kraus; }

    // Same rule as QuantumStateMachine::terminal_measurements_only, decided before anything runs.
    bool terminalOnly(const std::vector<Operation>& ops, std::size_t nQubits)
    {
        std::vector<char> measured(nQubits, 0);
        for (const Operation& o : ops)
        {
            if (o.kind == OpKind::Reset || isChannel(o) || o.condition) return false;
            if (o.kind == OpKind::Measure)
            {
                measured[o.targets[0]] = 1;
                continue;
            }
            for (const Qubit q : o.controls)
                if (measured[q]) return false;
            for (const Qubit q : o.targets)
                if (measured[q]) return false;
        }
        return true;
    }

    std::string describe(const Operation& o)
    {
        std::string s(Qputer::opName(o.kind));
        if (!o.controls.empty()) s += std::format(" with {} control{}", o.controls.size(), o.controls.size() == 1 ? "" : "s");
        return s;
    }

    void append(Qputer::QuantumStateMachine& m, const Operation& o, std::size_t index)
    {
        try
        {
            m.append(o);
        }
        catch (const std::exception& e)
        {
            throw ImportError("E9003", std::format("operation {} ({}) was rejected by the state machine: {}", index, describe(o), e.what()));
        }
    }

    // Counts of the classical register from the terminal measurements, sampled from one simulation.
    Qputer::Counts sampleTerminal(Qputer::QuantumStateMachine& m, const std::vector<Operation>& ops, std::size_t shots)
    {
        std::vector<std::pair<Qubit, std::size_t>> writes; // (qubit, clbit) in program order
        QubitList qubits;
        for (std::size_t k = 0; k < ops.size(); ++k)
        {
            const Operation& o = ops[k];
            if (o.kind != OpKind::Measure)
            {
                append(m, o, k);
                continue;
            }
            if (!o.clbit) continue;
            writes.emplace_back(o.targets[0], *o.clbit);
            if (std::ranges::find(qubits, o.targets[0]) == qubits.end()) qubits.push_back(o.targets[0]);
        }
        std::map<Qubit, std::size_t> position;
        for (std::size_t j = 0; j < qubits.size(); ++j) position[qubits[j]] = j;
        Qputer::Counts counts;
        for (const auto& [sample, n] : m.sample_counts(qubits, shots))
        {
            Outcome reg = 0;
            for (const auto& [q, c] : writes)
            {
                const Outcome bit = Outcome{1} << c;
                reg = (reg & ~bit) | (((sample >> position[q]) & 1U) << c);
            }
            counts[reg] += n;
        }
        return counts;
    }

    Qputer::Counts countsOf(const ImportedCircuit& c, const std::vector<Operation>& ops, Qputer::Backend backend, std::uint64_t seed,
                            std::size_t shots, bool sample, std::string& method)
    {
        Qputer::QuantumStateMachine m(c.numQubits, c.numClbits, seed, backend);
        if (sample && terminalOnly(ops, c.numQubits))
        {
            method = "sampled";
            return sampleTerminal(m, ops, shots);
        }
        method = "trajectories";
        for (std::size_t k = 0; k < ops.size(); ++k) append(m, ops[k], k);
        return m.run(shots);
    }

    std::size_t parity(Outcome x) { return static_cast<std::size_t>(std::popcount(x) & 1); }

    // Below 2^53, so the reported seed is exact in JSON and accepted back by --seed.
    std::uint64_t drawSeed()
    {
        std::random_device rd;
        return ((std::uint64_t{rd()} << 32) ^ rd()) & ((std::uint64_t{1} << 53) - 1);
    }
} // namespace


RunResult runCircuit(const ImportedCircuit& c, const RunOptions& o)
{
    if (o.shots == 0) throw ImportError("E9001", "shots must be positive");
    RunResult r;
    r.seed = o.seed ? *o.seed : drawSeed();
    const bool stateEmit = o.emit == "statevector" || o.emit == "probabilities";

    // Clifford form of the circuit, when there is one.
    std::optional<std::vector<Operation>> clifford = std::vector<Operation>{};
    const Operation* firstNonClifford = nullptr;
    for (const Operation& op : c.ops)
    {
        if (Qputer::stabilizerSupports(op.kind))
        {
            if (clifford) clifford->push_back(op);
            continue;
        }
        if (const auto alt = cliffordForm(op))
        {
            if (clifford) clifford->insert(clifford->end(), alt->begin(), alt->end());
            continue;
        }
        if (!firstNonClifford) firstNonClifford = &op;
        clifford.reset();
    }

    const std::size_t n = c.numQubits;
    auto needStatevector = [&](const std::string& why)
    {
        if (n > Qputer::kMaxQubits)
            throw ImportError("E9004", std::format("{}, so {} qubits need the state vector, which holds at most {}", why, n, Qputer::kMaxQubits));
        r.backend = "statevector";
        r.backendReason = why;
    };

    if (stateEmit)
    {
        if (o.backend == "stabilizer") throw ImportError("E6002", std::format("--emit {} needs the state vector backend", o.emit));
        needStatevector(std::format("--emit {}", o.emit));
    }
    else if (o.backend == "stabilizer")
    {
        if (!clifford)
            throw ImportError("E6001", std::format("{} is not a Clifford operation, so the stabilizer backend cannot run it", describe(*firstNonClifford)));
        r.backend = "stabilizer";
        r.backendReason = "requested";
    }
    else if (o.backend == "statevector") needStatevector("requested");
    else if (clifford)
    {
        r.backend = "stabilizer";
        r.backendReason = "every operation is Clifford";
    }
    else needStatevector(std::format("{} is not Clifford", describe(*firstNonClifford)));

    const bool stab = r.backend == "stabilizer";
    const std::vector<Operation>& ops = stab ? *clifford : c.ops;
    const Qputer::Backend backend = stab ? Qputer::Backend::Stabilizer : Qputer::Backend::StateVector;

    if (stateEmit)
    {
        // The state just before the terminal measurements: those are dropped, and nothing else may
        // be random. A reset of a qubit no gate has touched leaves |0⟩ unchanged.
        std::vector<char> touched(n, 0), measured(n, 0);
        Qputer::QuantumStateMachine m(n, c.numClbits, r.seed, backend);
        for (std::size_t k = 0; k < ops.size(); ++k)
        {
            const Operation& op = ops[k];
            const std::string at = std::format("operation {} ({})", k, describe(op));
            if (op.condition) throw ImportError("E9003", std::format("--emit {}: {} is conditioned on a measurement", o.emit, at));
            if (isChannel(op)) throw ImportError("E9003", std::format("--emit {}: {} is a noise channel; the state is a mixture", o.emit, at));
            if (op.kind == OpKind::Measure)
            {
                measured[op.targets[0]] = 1;
                continue;
            }
            for (const Qubit q : op.controls)
                if (measured[q]) throw ImportError("E9003", std::format("--emit {}: {} acts after a measurement of qubit {}", o.emit, at, q));
            for (const Qubit q : op.targets)
                if (measured[q]) throw ImportError("E9003", std::format("--emit {}: {} acts after a measurement of qubit {}", o.emit, at, q));
            if (op.kind == OpKind::Reset)
            {
                if (touched[op.targets[0]]) throw ImportError("E9003", std::format("--emit {}: {} resets a qubit in use", o.emit, at));
                continue;
            }
            for (const Qubit q : op.controls) touched[q] = 1;
            for (const Qubit q : op.targets) touched[q] = 1;
            append(m, op, k);
        }
        r.method = "state";
        if (o.emit == "statevector") r.amplitudes = m.state_vector().vector();
        else r.probabilities = m.probabilities();
        return r;
    }

    if (!std::ranges::any_of(ops, [](const Operation& op) { return op.kind == OpKind::Measure && op.clbit.has_value(); }))
        throw ImportError("E5011", "the circuit measures nothing into a classical bit; use --emit probabilities or --emit statevector");

    r.counts = countsOf(c, ops, backend, r.seed, o.shots, o.sampleTerminal, r.method);

    if (!c.detectors.empty() || !c.observables.empty())
    {
        // Stim reports detectors and observables relative to a noiseless reference sample; a
        // well-formed detector is deterministic there, so any one noiseless shot serves.
        std::vector<Operation> noiseless;
        for (const Operation& op : ops)
            if (!isChannel(op)) noiseless.push_back(op);
        std::string unused;
        const Qputer::Counts ref = countsOf(c, noiseless, backend, r.seed ^ 0x5DEECE66DULL, 1, true, unused);
        const Outcome reference = ref.begin()->first;
        auto tally = [&](const std::vector<Outcome>& masks)
        {
            std::vector<std::size_t> out(masks.size(), 0);
            for (std::size_t d = 0; d < masks.size(); ++d)
                for (const auto& [key, count] : r.counts)
                    if (parity(key & masks[d]) != parity(reference & masks[d])) out[d] += count;
            return out;
        };
        r.detectorFires = tally(c.detectors);
        r.observableFlips = tally(c.observables);
    }
    return r;
}

} // namespace Noether::Interop
