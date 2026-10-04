#include <QuantumStateMachine.hpp>

#include "DenseGate.hpp"
#include "Parallel.hpp"
#include "ReadoutKernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <exception>
#include <initializer_list>
#include <format>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>



namespace Qputer
{

namespace
{

    using detail::Index;

    inline std::uint64_t splitmix64(std::uint64_t& x) noexcept
    {
        std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    inline Index bit(std::size_t q) noexcept { return Index{1} << q; }


    // ---- Operation signatures ----

    constexpr int kAny = -1; // variable count (targets: at least one)

    struct Spec
    {
        std::string_view name;
        int controls;
        int targets;
        int params;
    };

    constexpr std::array<Spec, 26> kSpecs{{
        {"x", 0, 1, 0},       {"y", 0, 1, 0},     {"z", 0, 1, 0},       {"h", 0, 1, 0},
        {"s", 0, 1, 0},       {"sdg", 0, 1, 0},   {"t", 0, 1, 0},       {"tdg", 0, 1, 0},
        {"sx", 0, 1, 0},      {"rx", 0, 1, 1},    {"ry", 0, 1, 1},      {"rz", 0, 1, 1},
        {"phase", 0, 1, 1},   {"u3", 0, 1, 3},
        {"cnot", 1, 1, 0},    {"cz", 1, 1, 0},    {"cphase", 1, 1, 1},  {"swap", 0, 2, 0},
        {"toffoli", 2, 1, 0}, {"fredkin", 1, 2, 0},
        {"mcx", kAny, 1, 0},  {"mcz", 0, kAny, 0}, {"mcphase", 0, kAny, 1},
        {"unitary", kAny, kAny, 0},
        {"measure", 0, 1, 0}, {"reset", 0, 1, 0},
    }};
    static_assert(static_cast<std::size_t>(OpKind::Reset) + 1 == kSpecs.size());

    const Spec& specOf(OpKind kind)
    {
        const auto k = static_cast<std::size_t>(kind);
        if (k >= kSpecs.size()) throw std::invalid_argument(std::format("unknown OpKind {}", k));
        return kSpecs[k];
    }

    // Range + distinctness over every list; returns the union mask.
    Index claimQubits(std::size_t n, std::string_view ctx, std::initializer_list<std::span<const Qubit>> lists)
    {
        Index seen = 0;
        for (const auto list : lists)
            for (const Qubit q : list)
            {
                if (q >= n)
                    throw std::out_of_range(std::format("{}: qubit {} out of range [0, {})", ctx, q, n));
                if (seen & bit(q))
                    throw std::invalid_argument(std::format("{}: qubit {} used more than once", ctx, q));
                seen |= bit(q);
            }
        return seen;
    }

    void requireReadoutQubits(const QubitList& qubits, std::size_t n, std::string_view ctx)
    {
        if (qubits.empty()) throw std::invalid_argument(std::format("{}: at least one qubit required", ctx));
        claimQubits(n, ctx, {qubits});
    }


    // ---- Execution (shared by the live system and every replayed shot) ----

    int measureQubit(QuantumStateVector& s, Qubit q, Rng& rng)
    {
        const auto [w0, w1] = detail::qubitWeights(s, q);
        const double total = w0 + w1;
        if (!(total > 0.0) || !std::isfinite(total))
            throw std::domain_error(std::format("measure: state has squared norm {}", total));

        // u * total < w0 is impossible when w0 == 0, so a zero-weight outcome is never chosen.
        const int r = rng.uniform() * total < w0 ? 0 : 1;
        detail::collapse(s, bit(q), r ? bit(q) : 0, 1.0 / std::sqrt(r ? w1 : w0));
        return r;
    }

    // Returns the measured value for measure, -1 otherwise (including skipped operations).
    // `dense` is the prepared form of op when op is a unitary, unused otherwise.
    int execute(QuantumStateVector& s, Outcome& creg, const Operation& op, const detail::DenseGate* dense, Rng& rng)
    {
        if (op.condition && (((creg >> op.condition->clbit) & 1U) != 0) != op.condition->value) return -1;

        const QubitList& c = op.controls;
        const QubitList& t = op.targets;
        const std::vector<double>& p = op.params;

        switch (op.kind)
        {
            case OpKind::X:       QuantumGate::x(s, t[0]); break;
            case OpKind::Y:       QuantumGate::y(s, t[0]); break;
            case OpKind::Z:       QuantumGate::z(s, t[0]); break;
            case OpKind::H:       QuantumGate::h(s, t[0]); break;
            case OpKind::S:       QuantumGate::s(s, t[0]); break;
            case OpKind::Sdg:     QuantumGate::sdg(s, t[0]); break;
            case OpKind::T:       QuantumGate::t(s, t[0]); break;
            case OpKind::Tdg:     QuantumGate::tdg(s, t[0]); break;
            case OpKind::SX:      QuantumGate::sx(s, t[0]); break;
            case OpKind::RX:      QuantumGate::rx(s, t[0], p[0]); break;
            case OpKind::RY:      QuantumGate::ry(s, t[0], p[0]); break;
            case OpKind::RZ:      QuantumGate::rz(s, t[0], p[0]); break;
            case OpKind::Phase:   QuantumGate::phase(s, t[0], p[0]); break;
            case OpKind::U3:      QuantumGate::u3(s, t[0], p[0], p[1], p[2]); break;
            case OpKind::CNOT:    QuantumGate::cnot(s, c[0], t[0]); break;
            case OpKind::CZ:      QuantumGate::cz(s, c[0], t[0]); break;
            case OpKind::CPhase:  QuantumGate::cphase(s, c[0], t[0], p[0]); break;
            case OpKind::Swap:    QuantumGate::swap(s, t[0], t[1]); break;
            case OpKind::Toffoli: QuantumGate::toffoli(s, c[0], c[1], t[0]); break;
            case OpKind::Fredkin: QuantumGate::fredkin(s, c[0], t[0], t[1]); break;
            case OpKind::MCX:     QuantumGate::mcx(s, c, t[0]); break;
            case OpKind::MCZ:     QuantumGate::mcz(s, t); break;
            case OpKind::MCPhase: QuantumGate::mcphase(s, t, p[0]); break;
            case OpKind::Unitary: detail::applyDense(s, *dense); break;
            case OpKind::Measure:
            {
                const int r = measureQubit(s, t[0], rng);
                if (op.clbit) creg = (creg & ~bit(*op.clbit)) | (static_cast<Outcome>(r) << *op.clbit);
                return r;
            }
            case OpKind::Reset:
                if (measureQubit(s, t[0], rng) == 1) QuantumGate::x(s, t[0]);
                break;
        }
        return -1;
    }

    // Inclusive prefix sums of the marginal over `qubits`. `last` is the highest outcome
    // with nonzero weight: rounding that pushes a draw past the total resolves there
    // instead of onto a zero-probability tail.
    struct Cdf
    {
        std::vector<double> cum;
        Outcome last = 0;
    };

    Cdf marginalCdf(const QuantumStateVector& s, const QubitList& qubits)
    {
        Cdf cdf{std::vector<double>(bit(qubits.size())), 0};
        detail::marginalWeights(s, qubits, cdf.cum.data());
        for (std::size_t o = cdf.cum.size(); o-- > 0;)
            if (cdf.cum[o] > 0.0)
            {
                cdf.last = o;
                break;
            }
        std::inclusive_scan(cdf.cum.begin(), cdf.cum.end(), cdf.cum.begin());
        if (!(cdf.cum.back() > 0.0) || !std::isfinite(cdf.cum.back()))
            throw std::domain_error(std::format("sample: state has squared norm {}", cdf.cum.back()));
        return cdf;
    }

    // Below this many shots, thread start-up outweighs sampling from a CDF.
    constexpr std::size_t kParallelSampleShots = 4096;

    // Up to 2^16 outcomes (512 KiB) the CDF stays cache-resident and per-shot binary search
    // wins. Beyond that every probe misses cache, so draws are sorted and matched against
    // the CDF in one sequential sweep: O(S log S + 2^k) instead of O(S k) cache misses.
    constexpr std::size_t kSearchMaxOutcomes = std::size_t{1} << 16;

    // out[shot] ~ cdf using stream (base, shot), so results are independent of threading.
    // upper_bound never lands on a zero-weight outcome below `last`: its cum equals its
    // predecessor's.
    void drawMany(const Cdf& cdf, std::uint64_t base, std::vector<Outcome>& out)
    {
        const std::size_t shots = out.size();
        const double total = cdf.cum.back();

        if (cdf.cum.size() <= kSearchMaxOutcomes)
        {
            const double* cum = cdf.cum.data();
            const std::size_t len = cdf.cum.size();
            QPUTER_OMP(parallel for schedule(static) if(shots >= kParallelSampleShots))
            for (std::size_t shot = 0; shot < shots; ++shot)
            {
                Rng r{base, shot};
                const auto j = static_cast<Outcome>(std::upper_bound(cum, cum + len, r.uniform() * total) - cum);
                out[shot] = std::min(j, cdf.last);
            }
            return;
        }

        std::vector<std::pair<double, std::size_t>> draws(shots);
        QPUTER_OMP(parallel for schedule(static) if(shots >= kParallelSampleShots))
        for (std::size_t shot = 0; shot < shots; ++shot)
        {
            Rng r{base, shot};
            draws[shot] = {r.uniform() * total, shot};
        }
        std::ranges::sort(draws);

        Outcome j = 0;
        for (const auto& [u, shot] : draws)
        {
            while (j < cdf.last && cdf.cum[j] <= u) ++j;
            out[shot] = j;
        }
    }

    // Sort + run-length encoding: one cache-friendly pass instead of a tree insert per shot.
    Counts tally(std::vector<Outcome>& outcomes)
    {
        std::ranges::sort(outcomes);
        Counts counts;
        for (std::size_t i = 0; i < outcomes.size();)
        {
            std::size_t j = i + 1;
            while (j < outcomes.size() && outcomes[j] == outcomes[i]) ++j;
            counts.emplace_hint(counts.end(), outcomes[i], j - i);
            i = j;
        }
        return counts;
    }

    // Operations with only the given fields set; matrix / clbit / condition stay empty.
    Operation makeOp(OpKind kind, QubitList controls, QubitList targets, std::vector<double> params = {})
    {
        Operation op;
        op.kind = kind;
        op.controls = std::move(controls);
        op.targets = std::move(targets);
        op.params = std::move(params);
        return op;
    }

} // namespace



// ---- Rng / names ----

Rng::Rng(std::uint64_t seed, std::uint64_t stream) noexcept
{
    // Odd multiplier: distinct streams under one seed give distinct SplitMix starting points.
    std::uint64_t x = seed ^ (stream * 0xd1342543de82ef95ULL);
    for (auto& word : s) word = splitmix64(x);
}

std::string_view opName(OpKind kind) { return specOf(kind).name; }

std::optional<OpKind> opKindFromName(std::string_view name)
{
    for (std::size_t k = 0; k < kSpecs.size(); ++k)
        if (kSpecs[k].name == name) return static_cast<OpKind>(k);
    return std::nullopt;
}



// ---- System ----

QuantumStateMachine::QuantumStateMachine(std::size_t numQubits, std::size_t numClbits,
                                         std::optional<std::uint64_t> seed)
    : nClbits(numClbits),
      live(numQubits),
      seedValue(seed.value_or(0)),
      rng(seedValue)
{
    if (numClbits > kMaxClbits)
        throw std::length_error(std::format("QuantumStateMachine: num_clbits={} > {}", numClbits, kMaxClbits));
    if (!seed)
    {
        std::random_device rd;
        reseed((std::uint64_t{rd()} << 32) ^ rd());
    }
}

void QuantumStateMachine::reseed(std::uint64_t seed)
{
    seedValue = seed;
    rng = Rng{seed};
}

bool QuantumStateMachine::clbit(std::size_t index) const
{
    if (index >= nClbits)
        throw std::out_of_range(std::format("QuantumStateMachine::clbit: {} out of range [0, {})", index, nClbits));
    return (creg >> index) & 1U;
}



// ---- Preparation ----

void QuantumStateMachine::clearCircuit()
{
    ops.clear();
    dense.clear();
    pending.reset();
    creg = 0;
}

void QuantumStateMachine::prepare() { prepare_basis(0); }

void QuantumStateMachine::prepare_basis(Outcome index)
{
    if (index >= live.size())
        throw std::out_of_range(std::format("QuantumStateMachine::prepare_basis: index {} out of range [0, {})",
                                            index, live.size()));
    detail::setBasis(live, index);
    prepIndex = index;
    prepAmplitudes.reset();
    clearCircuit();
}

void QuantumStateMachine::prepare_state(const Eigen::VectorXcd& amplitudes)
{
    constexpr std::string_view ctx = "QuantumStateMachine::prepare_state";
    if (static_cast<std::size_t>(amplitudes.size()) != live.size())
        throw std::invalid_argument(std::format("{}: {} amplitudes, expected {} for {} qubits",
                                                ctx, amplitudes.size(), live.size(), num_qubits()));
    if (!amplitudes.allFinite())
        throw std::invalid_argument(std::format("{}: amplitudes contain NaN or Inf", ctx));
    const double normSq = amplitudes.squaredNorm();
    if (!(std::abs(normSq - 1.0) <= kNormTolerance))
        throw std::invalid_argument(std::format("{}: squared norm {:.17g} differs from 1 by more than {:.0e}",
                                                ctx, normSq, kNormTolerance));

    live = QuantumStateVector{amplitudes};
    prepAmplitudes = live;
    clearCircuit();
}

void QuantumStateMachine::loadPreparation(QuantumStateVector& target) const
{
    if (prepAmplitudes) detail::copyAmplitudes(*prepAmplitudes, target);
    else detail::setBasis(target, prepIndex);
}



// ---- Validation / commit ----

void QuantumStateMachine::validate(const Operation& op) const
{
    const Spec& spec = specOf(op.kind);
    const std::string ctx = std::format("QuantumStateMachine::{}", spec.name);

    auto requireCount = [&](std::string_view role, std::size_t got, int want)
    {
        if (want == kAny) return;
        if (got != static_cast<std::size_t>(want))
            throw std::invalid_argument(std::format("{}: {} {} given, expected {}", ctx, got, role, want));
    };
    requireCount("controls", op.controls.size(), spec.controls);
    requireCount("targets", op.targets.size(), spec.targets);
    requireCount("params", op.params.size(), spec.params);
    if (op.targets.empty()) throw std::invalid_argument(std::format("{}: at least one target required", ctx));

    for (const double v : op.params)
        if (!std::isfinite(v)) throw std::invalid_argument(std::format("{}: non-finite parameter {}", ctx, v));

    claimQubits(num_qubits(), ctx, {op.controls, op.targets});

    if (op.kind == OpKind::Unitary)
    {
        if (op.targets.size() > QuantumGate::kMaxDenseTargets)
            throw std::invalid_argument(std::format("{}: {} targets exceeds {}", ctx, op.targets.size(),
                                                    QuantumGate::kMaxDenseTargets));
        QuantumGate::require_unitary(op.matrix, op.targets.size(), ctx);
    }

    if (op.clbit)
    {
        if (op.kind != OpKind::Measure)
            throw std::invalid_argument(std::format("{}: only measure writes a classical bit", ctx));
        if (*op.clbit >= nClbits)
            throw std::out_of_range(std::format("{}: clbit {} out of range [0, {})", ctx, *op.clbit, nClbits));
    }

    if (op.condition)
    {
        if (op.kind == OpKind::Measure)
            throw std::invalid_argument(std::format("{}: measurements cannot be conditioned", ctx));
        if (op.condition->clbit >= nClbits)
            throw std::out_of_range(std::format("{}: condition clbit {} out of range [0, {})",
                                                ctx, op.condition->clbit, nClbits));
    }
}

int QuantumStateMachine::commit(Operation op)
{
    if (pending)
    {
        if (op.condition)
            throw std::invalid_argument("QuantumStateMachine::when: operation already carries a condition");
        op.condition = pending;
    }
    validate(op);

    std::shared_ptr<const detail::DenseGate> prepared;
    if (op.kind == OpKind::Unitary)
        prepared = std::make_shared<const detail::DenseGate>(detail::prepareDense(op.controls, op.targets, op.matrix));

    const int result = execute(live, creg, op, prepared.get(), rng);
    dense.push_back(std::move(prepared));
    try
    {
        ops.push_back(std::move(op));
    }
    catch (...)
    {
        dense.pop_back(); // keep the two vectors aligned
        throw;
    }
    pending.reset();
    return result;
}

QuantumStateMachine& QuantumStateMachine::append(Operation op)
{
    commit(std::move(op));
    return *this;
}

QuantumStateMachine& QuantumStateMachine::when(std::size_t clbit, bool value)
{
    if (clbit >= nClbits)
        throw std::out_of_range(std::format("QuantumStateMachine::when: clbit {} out of range [0, {})",
                                            clbit, nClbits));
    pending = Condition{clbit, value};
    return *this;
}



// ---- Gates ----

#define QPUTER_GATE1(name, KIND)                                                \
    QuantumStateMachine& QuantumStateMachine::name(Qubit target)                \
    {                                                                           \
        return append(makeOp(OpKind::KIND, {}, {target}));    \
    }

QPUTER_GATE1(x, X)
QPUTER_GATE1(y, Y)
QPUTER_GATE1(z, Z)
QPUTER_GATE1(h, H)
QPUTER_GATE1(s, S)
QPUTER_GATE1(sdg, Sdg)
QPUTER_GATE1(t, T)
QPUTER_GATE1(tdg, Tdg)
QPUTER_GATE1(sx, SX)
QPUTER_GATE1(reset, Reset)
#undef QPUTER_GATE1

QuantumStateMachine& QuantumStateMachine::rx(Qubit target, double theta)
{
    return append(makeOp(OpKind::RX, {}, {target}, {theta}));
}

QuantumStateMachine& QuantumStateMachine::ry(Qubit target, double theta)
{
    return append(makeOp(OpKind::RY, {}, {target}, {theta}));
}

QuantumStateMachine& QuantumStateMachine::rz(Qubit target, double theta)
{
    return append(makeOp(OpKind::RZ, {}, {target}, {theta}));
}

QuantumStateMachine& QuantumStateMachine::phase(Qubit target, double lambda)
{
    return append(makeOp(OpKind::Phase, {}, {target}, {lambda}));
}

QuantumStateMachine& QuantumStateMachine::u3(Qubit target, double theta, double phi, double lambda)
{
    return append(makeOp(OpKind::U3, {}, {target}, {theta, phi, lambda}));
}

QuantumStateMachine& QuantumStateMachine::cnot(Qubit control, Qubit target)
{
    return append(makeOp(OpKind::CNOT, {control}, {target}));
}

QuantumStateMachine& QuantumStateMachine::cz(Qubit control, Qubit target)
{
    return append(makeOp(OpKind::CZ, {control}, {target}));
}

QuantumStateMachine& QuantumStateMachine::cphase(Qubit control, Qubit target, double lambda)
{
    return append(makeOp(OpKind::CPhase, {control}, {target}, {lambda}));
}

QuantumStateMachine& QuantumStateMachine::swap(Qubit a, Qubit b)
{
    return append(makeOp(OpKind::Swap, {}, {a, b}));
}

QuantumStateMachine& QuantumStateMachine::toffoli(Qubit control0, Qubit control1, Qubit target)
{
    return append(makeOp(OpKind::Toffoli, {control0, control1}, {target}));
}

QuantumStateMachine& QuantumStateMachine::fredkin(Qubit control, Qubit a, Qubit b)
{
    return append(makeOp(OpKind::Fredkin, {control}, {a, b}));
}

QuantumStateMachine& QuantumStateMachine::mcx(const QubitList& controls, Qubit target)
{
    return append(makeOp(OpKind::MCX, controls, {target}));
}

QuantumStateMachine& QuantumStateMachine::mcz(const QubitList& qubits)
{
    return append(makeOp(OpKind::MCZ, {}, qubits));
}

QuantumStateMachine& QuantumStateMachine::mcphase(const QubitList& qubits, double lambda)
{
    return append(makeOp(OpKind::MCPhase, {}, qubits, {lambda}));
}

QuantumStateMachine& QuantumStateMachine::unitary(const QubitList& targets, const Eigen::MatrixXcd& U)
{
    Operation op = makeOp(OpKind::Unitary, {}, targets);
    op.matrix = U;
    return append(std::move(op));
}

QuantumStateMachine& QuantumStateMachine::controlled_unitary(const QubitList& controls, const QubitList& targets,
                                                             const Eigen::MatrixXcd& U)
{
    Operation op = makeOp(OpKind::Unitary, controls, targets);
    op.matrix = U;
    return append(std::move(op));
}



// ---- Measurement ----

int QuantumStateMachine::measure(Qubit target, std::optional<std::size_t> clbit)
{
    if (pending)
        throw std::invalid_argument("QuantumStateMachine::measure: measurements cannot be conditioned");
    Operation op = makeOp(OpKind::Measure, {}, {target});
    op.clbit = clbit;
    return commit(std::move(op));
}

Outcome QuantumStateMachine::measure_all()
{
    constexpr std::string_view ctx = "QuantumStateMachine::measure_all";
    const std::size_t n = num_qubits();
    if (pending) throw std::invalid_argument(std::format("{}: measurements cannot be conditioned", ctx));
    if (nClbits < n)
        throw std::invalid_argument(std::format("{}: {} clbits cannot hold {} qubits", ctx, nClbits, n));

    // One joint draw over all 2^N outcomes: equivalent in distribution to N sequential
    // single-qubit measurements, at the cost of two passes instead of 2N.
    const Index index = detail::findCumulative(live, rng.uniform());
    detail::collapse(live, live.size() - 1, index, 1.0 / std::abs(live[index]));

    const Outcome mask = bit(n) - 1;
    creg = (creg & ~mask) | index;
    ops.reserve(ops.size() + n);
    for (Qubit q = 0; q < n; ++q)
    {
        ops.push_back(makeOp(OpKind::Measure, {}, {q}));
        ops.back().clbit = q;
    }
    dense.resize(ops.size());
    return index;
}



// ---- Readout ----

Eigen::VectorXd QuantumStateMachine::probabilities() const
{
    Eigen::VectorXd p(static_cast<Eigen::Index>(live.size()));
    detail::squaredMagnitudes(live, p.data());
    return p;
}

double QuantumStateMachine::probability(Outcome basisIndex) const
{
    if (basisIndex >= live.size())
        throw std::out_of_range(std::format("QuantumStateMachine::probability: index {} out of range [0, {})",
                                            basisIndex, live.size()));
    const auto a = live[basisIndex];
    return a.real() * a.real() + a.imag() * a.imag();
}

Eigen::VectorXd QuantumStateMachine::marginal_probabilities(const QubitList& qubits) const
{
    requireReadoutQubits(qubits, num_qubits(), "QuantumStateMachine::marginal_probabilities");
    Eigen::VectorXd p(static_cast<Eigen::Index>(bit(qubits.size())));
    detail::marginalWeights(live, qubits, p.data());
    return p;
}

double QuantumStateMachine::expectation(std::string_view paulis, const QubitList& qubits) const
{
    constexpr std::string_view ctx = "QuantumStateMachine::expectation";
    if (paulis.size() != qubits.size())
        throw std::invalid_argument(std::format("{}: {} Pauli letters for {} qubits", ctx, paulis.size(),
                                                qubits.size()));
    claimQubits(num_qubits(), ctx, {qubits});

    // P = i^{#Y} X^x Z^z with Y = iXZ, so P|i> = i^{#Y} (-1)^{|i & z|} |i ^ x>.
    Index xMask = 0, zMask = 0;
    unsigned numY = 0;
    for (std::size_t k = 0; k < paulis.size(); ++k)
    {
        const Index b = bit(qubits[k]);
        switch (paulis[k])
        {
            case 'I': break;
            case 'X': xMask |= b; break;
            case 'Z': zMask |= b; break;
            case 'Y': xMask |= b; zMask |= b; ++numY; break;
            default:
                throw std::invalid_argument(std::format("{}: invalid Pauli letter '{}' (expected I, X, Y, Z)",
                                                        ctx, paulis[k]));
        }
    }

    const std::complex<double> sum = detail::pauliSum(live, xMask, zMask);
    switch (numY % 4)
    {
        case 0:  return sum.real();
        case 1:  return -sum.imag();  // Re(i z)
        case 2:  return -sum.real();
        default: return sum.imag();   // Re(-i z)
    }
}

std::vector<Outcome> QuantumStateMachine::sample(const QubitList& qubits, std::size_t shots)
{
    requireReadoutQubits(qubits, num_qubits(), "QuantumStateMachine::sample");
    std::vector<Outcome> out(shots);
    drawMany(marginalCdf(live, qubits), rng(), out);
    return out;
}

Counts QuantumStateMachine::sample_counts(const QubitList& qubits, std::size_t shots)
{
    std::vector<Outcome> out = sample(qubits, shots);
    return tally(out);
}



// ---- Circuit execution ----

bool QuantumStateMachine::terminal_measurements_only() const
{
    Index measured = 0;
    for (const Operation& op : ops)
    {
        if (op.kind == OpKind::Reset || op.condition) return false;
        Index touched = 0;
        for (const Qubit q : op.controls) touched |= bit(q);
        for (const Qubit q : op.targets) touched |= bit(q);
        if (op.kind == OpKind::Measure) measured |= touched;
        else if (touched & measured) return false;
    }
    return true;
}

Counts QuantumStateMachine::run(std::size_t shots)
{
    constexpr std::string_view ctx = "QuantumStateMachine::run";
    if (shots == 0) throw std::invalid_argument(std::format("{}: shots must be positive", ctx));
    const bool writesClbit = std::ranges::any_of(ops, [](const Operation& op)
    {
        return op.kind == OpKind::Measure && op.clbit.has_value();
    });
    if (!writesClbit)
        throw std::invalid_argument(std::format("{}: circuit has no measurement into a classical bit", ctx));

    // One draw from the machine RNG per run: repeated runs differ, yet all follow from seed().
    const std::uint64_t base = rng();
    std::vector<Outcome> outcomes(shots);
    if (terminal_measurements_only()) runSampled(base, outcomes);
    else runTrajectories(base, outcomes);

    return tally(outcomes);
}

void QuantumStateMachine::runSampled(std::uint64_t base, std::vector<Outcome>& outcomes) const
{
    QuantumStateVector scratch{num_qubits()};
    loadPreparation(scratch);
    Outcome unusedCreg = 0;
    Rng unusedRng{base}; // no measure, reset or condition reaches execute() on this path
    for (std::size_t k = 0; k < ops.size(); ++k)
        if (ops[k].kind != OpKind::Measure) execute(scratch, unusedCreg, ops[k], dense[k].get(), unusedRng);

    // Joint distribution over the distinct measured qubits, then each recorded
    // measurement copies its qubit's sampled bit into its clbit, in circuit order.
    QubitList measured;
    std::vector<std::size_t> position(num_qubits(), num_qubits());
    struct Write { std::size_t from, clbit; };
    std::vector<Write> writes;
    for (const Operation& op : ops)
    {
        if (op.kind != OpKind::Measure) continue;
        const Qubit q = op.targets[0];
        if (position[q] == num_qubits())
        {
            position[q] = measured.size();
            measured.push_back(q);
        }
        if (op.clbit) writes.push_back({position[q], *op.clbit});
    }

    drawMany(marginalCdf(scratch, measured), base, outcomes);
    const std::size_t shots = outcomes.size();

    QPUTER_OMP(parallel for schedule(static) if(shots >= kParallelSampleShots))
    for (std::size_t shot = 0; shot < shots; ++shot)
    {
        const Outcome o = outcomes[shot];
        Outcome reg = 0;
        for (const auto& [from, clbit] : writes) reg = (reg & ~bit(clbit)) | (((o >> from) & 1U) << clbit);
        outcomes[shot] = reg;
    }
}

void QuantumStateMachine::runTrajectories(std::uint64_t base, std::vector<Outcome>& outcomes) const
{
    const std::size_t shots = outcomes.size();
    // Small registers leave the gate kernels serial, so parallelize across shots instead;
    // large ones run shots in sequence and let every gate use all threads.
    [[maybe_unused]] const bool shotParallel = live.size() < detail::kParallelThreshold && shots > 1;
    std::exception_ptr failure;

    QPUTER_OMP(parallel if(shotParallel))
    {
        QuantumStateVector scratch{num_qubits()};

        // Feed-forward makes trajectory cost vary, hence dynamic chunks.
        QPUTER_OMP(for schedule(dynamic, 16))
        for (std::size_t shot = 0; shot < shots; ++shot)
        {
            try
            {
                loadPreparation(scratch);
                Outcome reg = 0;
                Rng r{base, shot};
                for (std::size_t k = 0; k < ops.size(); ++k) execute(scratch, reg, ops[k], dense[k].get(), r);
                outcomes[shot] = reg;
            }
            catch (...)
            {
                QPUTER_OMP(critical)
                if (!failure) failure = std::current_exception();
            }
        }
    }
    if (failure) std::rethrow_exception(failure);
}

std::string QuantumStateMachine::bitstring(Outcome value, std::size_t width)
{
    if (width > 64) throw std::invalid_argument(std::format("bitstring: width {} > 64", width));
    std::string out(width, '0');
    for (std::size_t c = 0; c < width; ++c)
        if ((value >> c) & 1U) out[width - 1 - c] = '1';
    return out;
}

} // namespace Qputer
