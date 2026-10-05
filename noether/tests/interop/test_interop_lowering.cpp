// The gate algebra (control, inverse, power) and Pauli-product constructions, checked two ways:
// matrixOf against independent block and adjoint formulas, and the emitted core operations against
// matrixOf, which ties the algebra to the state vector's own gate kernels.

#include "InteropTestUtil.hpp"

#include "interop/Lowering.hpp"

#include <numbers>
#include <numeric>

using namespace InteropTest;

namespace
{

constexpr double kTol = 1e-10;

// matrixOf in core order: support[0] is the most significant bit, so list qubits high to low.
Eigen::MatrixXcd coreMatrix(const Seq& s, std::size_t n)
{
    QubitList support;
    for (std::size_t q = n; q-- > 0;) support.push_back(q);
    return matrixOf(s, support);
}

ImportedCircuit circuitOf(std::size_t n)
{
    ImportedCircuit c;
    c.numQubits = n;
    for (std::size_t q = 0; q < n; ++q) c.qubitLabels.push_back(std::format("q{}", q));
    return c;
}

Eigen::MatrixXcd emitted(const Seq& s, std::size_t n)
{
    ImportedCircuit c = circuitOf(n);
    CircuitBuilder(c).apply(s);
    return unitaryOf(c);
}

bool hasUncontrolledPhase(const Seq& s)
{
    return std::ranges::any_of(s, [](const Gate& g) { return g.prim == Prim::GPhase && g.controls.empty(); });
}

// A random gate on distinct qubits drawn from `pool`.
Gate randomGate(QubitList pool, std::mt19937_64& rng)
{
    std::ranges::shuffle(pool, rng);
    std::uniform_real_distribution<double> angle(-3.1, 3.1);
    const int kind = std::uniform_int_distribution<int>(0, 17)(rng);
    if (kind == 15 && pool.size() >= 2) return makeGate(Prim::Swap, {pool[0], pool[1]});
    if (kind == 16)
    {
        const std::size_t k = std::min<std::size_t>(pool.size(), 2);
        return denseGate(QubitList(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(k)), randomUnitary(k, rng));
    }
    if (kind == 17) return makeGate(Prim::GPhase, {}, {angle(rng)});
    static constexpr Prim oneQubit[] = {Prim::X, Prim::Y, Prim::Z, Prim::H, Prim::S, Prim::Sdg, Prim::T, Prim::Tdg,
                                        Prim::SX, Prim::RX, Prim::RY, Prim::RZ, Prim::Phase, Prim::U3, Prim::U3};
    const Prim p = oneQubit[static_cast<std::size_t>(kind) % std::size(oneQubit)];
    std::vector<double> params;
    if (p == Prim::RX || p == Prim::RY || p == Prim::RZ || p == Prim::Phase) params = {angle(rng)};
    if (p == Prim::U3) params = {angle(rng), angle(rng), angle(rng)};
    return makeGate(p, {pool[0]}, params);
}

Seq randomSeq(const QubitList& pool, std::size_t length, std::mt19937_64& rng)
{
    Seq s;
    for (std::size_t k = 0; k < length; ++k) s.push_back(randomGate(pool, rng));
    return s;
}

// Expected controlled matrix: the body's columns where every control matches its active value, the
// identity elsewhere. The body leaves the controls alone, so a column stays in its control sector.
Eigen::MatrixXcd controlledExpected(const Eigen::MatrixXcd& body, const QubitList& controls, const std::vector<bool>& active)
{
    Eigen::MatrixXcd e = Eigen::MatrixXcd::Identity(body.rows(), body.cols());
    for (Eigen::Index col = 0; col < body.cols(); ++col)
    {
        bool on = true;
        for (std::size_t k = 0; k < controls.size(); ++k)
            on = on && (((static_cast<std::uint64_t>(col) >> controls[k]) & 1U) == (active.empty() || active[k] ? 1U : 0U));
        if (on) e.col(col) = body.col(col);
    }
    return e;
}

// The Pauli product as a matrix in core order.
Eigen::MatrixXcd pauliMatrix(const std::vector<PauliFactor>& factors, std::size_t n)
{
    const auto dim = Eigen::Index{1} << n;
    const cd i(0, 1);
    Eigen::MatrixXcd m = Eigen::MatrixXcd::Zero(dim, dim);
    for (Eigen::Index col = 0; col < dim; ++col)
    {
        auto idx = static_cast<std::uint64_t>(col);
        cd amp = 1.0;
        for (const PauliFactor& f : factors)
        {
            const bool bit = ((idx >> f.qubit) & 1U) != 0;
            if (f.letter == 'Z') amp *= bit ? -1.0 : 1.0;
            if (f.letter == 'Y') amp *= bit ? -i : i;
            if (f.letter != 'Z') idx ^= std::uint64_t{1} << f.qubit;
        }
        m(static_cast<Eigen::Index>(idx), col) = amp;
    }
    return m;
}

std::vector<PauliFactor> randomPauli(std::size_t n, std::mt19937_64& rng)
{
    QubitList qs(n);
    std::iota(qs.begin(), qs.end(), Qubit{0});
    std::ranges::shuffle(qs, rng);
    const std::size_t k = std::uniform_int_distribution<std::size_t>(1, n)(rng);
    std::vector<PauliFactor> out;
    for (std::size_t j = 0; j < k; ++j) out.push_back({"XYZ"[std::uniform_int_distribution<int>(0, 2)(rng)], qs[j]});
    return out;
}

Eigen::MatrixXcd unitaryOfOps(const std::vector<Qputer::Operation>& ops, std::size_t n)
{
    ImportedCircuit c = circuitOf(n);
    c.ops = ops;
    return unitaryOf(c);
}

} // namespace

TEST(InteropLowering, BaseMatricesMatchTheCoreKernels)
{
    std::mt19937_64 rng(1);
    for (int trial = 0; trial < 300; ++trial)
    {
        const Seq s{randomGate({0, 1, 2}, rng)};
        SCOPED_TRACE(trial);
        const Eigen::MatrixXcd want = coreMatrix(s, 3);
        if (hasUncontrolledPhase(s)) EXPECT_LE(maxDiffUpToPhase(emitted(s, 3), want), kTol);
        else EXPECT_LE(maxDiff(emitted(s, 3), want), kTol);
    }
}

TEST(InteropLowering, ControlledIsTheBlockForm)
{
    std::mt19937_64 rng(2);
    for (int trial = 0; trial < 300; ++trial)
    {
        SCOPED_TRACE(trial);
        QubitList all{0, 1, 2, 3, 4};
        std::ranges::shuffle(all, rng);
        const std::size_t nc = std::uniform_int_distribution<std::size_t>(1, 3)(rng);
        const QubitList controls(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(nc));
        const QubitList pool(all.begin() + static_cast<std::ptrdiff_t>(nc), all.end());
        std::vector<bool> active;
        for (std::size_t k = 0; k < nc; ++k) active.push_back(std::uniform_int_distribution<int>(0, 3)(rng) != 0);
        const Seq body = randomSeq(pool, std::uniform_int_distribution<std::size_t>(1, 3)(rng), rng);

        const Seq c = controlled(body, controls, active);
        const Eigen::MatrixXcd want = controlledExpected(coreMatrix(body, 5), controls, active);
        EXPECT_LE(maxDiff(coreMatrix(c, 5), want), kTol);
        EXPECT_LE(maxDiff(emitted(c, 5), want), kTol);
    }
}

TEST(InteropLowering, InverseIsTheAdjoint)
{
    std::mt19937_64 rng(3);
    for (int trial = 0; trial < 200; ++trial)
    {
        SCOPED_TRACE(trial);
        Seq s = randomSeq({0, 1, 2}, 4, rng);
        const Seq c = controlled(randomSeq({1, 2}, 2, rng), {0}, {trial % 2 == 0});
        s.insert(s.end(), c.begin(), c.end());
        const Eigen::MatrixXcd m = coreMatrix(s, 3);
        EXPECT_LE(maxDiff(coreMatrix(inverse(s), 3), m.adjoint()), kTol);
        EXPECT_LE(maxDiffUpToPhase(emitted(inverse(s), 3), m.adjoint()), kTol);
    }
}

TEST(InteropLowering, IntegerPowersRepeat)
{
    std::mt19937_64 rng(4);
    for (const double r : {-3.0, -1.0, 0.0, 1.0, 2.0, 5.0})
    {
        SCOPED_TRACE(r);
        const Seq s = randomSeq({0, 1}, 3, rng);
        const Eigen::MatrixXcd m = coreMatrix(s, 2);
        Eigen::MatrixXcd want = Eigen::MatrixXcd::Identity(4, 4);
        for (int k = 0; k < std::abs(static_cast<int>(r)); ++k) want = (r < 0 ? m.adjoint() : m) * want;
        EXPECT_LE(maxDiff(coreMatrix(power(s, r), 2), want), kTol);
    }
}

TEST(InteropLowering, LargeIntegerPowerFallsBackToTheMatrix)
{
    std::mt19937_64 rng(5);
    const Seq s = randomSeq({0, 1}, 3, rng);
    const Eigen::MatrixXcd m = coreMatrix(s, 2);
    Eigen::MatrixXcd want = Eigen::MatrixXcd::Identity(4, 4), base = m;
    for (unsigned e = 5000; e; e >>= 1, base = base * base)
        if (e & 1U) want = want * base;
    const Seq p = power(s, 5000.0);
    EXPECT_LE(p.size(), 4096U);
    EXPECT_LE(maxDiffUpToPhase(coreMatrix(p, 2), want), 1e-8);
}

TEST(InteropLowering, RotationPowersScaleTheAngle)
{
    for (const Prim p : {Prim::RX, Prim::RY, Prim::RZ, Prim::Phase})
        for (const double r : {0.5, -0.25, 1.7})
        {
            const double t = 1.3;
            const Seq s{makeGate(p, {0}, {t})};
            EXPECT_LE(maxDiff(coreMatrix(power(s, r), 1), coreMatrix(Seq{makeGate(p, {0}, {r * t})}, 1)), kTol);
        }
    // S, T and Z are phase gates, so their powers are phase gates too: S^0.5 = T.
    EXPECT_LE(maxDiff(coreMatrix(power(Seq{makeGate(Prim::S, {0})}, 0.5), 1), coreMatrix(Seq{makeGate(Prim::T, {0})}, 1)), kTol);
}

TEST(InteropLowering, FractionalPowersAreRoots)
{
    std::mt19937_64 rng(6);
    for (int trial = 0; trial < 20; ++trial)
    {
        const Seq s{denseGate({0, 1}, randomUnitary(2, rng))};
        const Eigen::MatrixXcd m = coreMatrix(s, 2);
        const Eigen::MatrixXcd half = coreMatrix(power(s, 0.5), 2);
        const Eigen::MatrixXcd third = coreMatrix(power(s, 1.0 / 3.0), 2);
        EXPECT_LE(maxDiff(half * half, m), 1e-9);
        EXPECT_LE(maxDiff(third * third * third, m), 1e-9);
        EXPECT_LE(maxDiff(emitted(power(s, 0.5), 2), half), kTol);
    }
}

TEST(InteropLowering, PowerCommutesWithControl)
{
    // eig(C(U)) = {1} ∪ eig(U), so the principal power of C(U) is C(principal power of U).
    std::mt19937_64 rng(7);
    for (int trial = 0; trial < 20; ++trial)
    {
        const Seq body{denseGate({1, 2}, randomUnitary(2, rng))};
        const double r = std::uniform_real_distribution<double>(-2.0, 2.0)(rng);
        EXPECT_LE(maxDiff(coreMatrix(power(controlled(body, {0}), r), 3), coreMatrix(controlled(power(body, r), {0}), 3)), 1e-9);
    }
}

TEST(InteropLowering, QasmModifiersFollowTheAlgebra)
{
    // ctrl @ inv @ pow(2) @ s = controlled(S^-2) = controlled(Z); negctrl flips the active value.
    const ImportResult a = load("m.qasm", "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\nctrl @ inv @ pow(2) @ s q[0], q[1];\n");
    const ImportResult b = load("m.qasm", "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\nnegctrl @ x q[0], q[1];\n");
    const ImportResult c = load("m.qasm", "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[3] q;\nctrl(2) @ pow(0.5) @ x q[0], q[1], q[2];\n");
    ASSERT_TRUE(a.circuit && b.circuit && c.circuit);
    const Seq z{makeGate(Prim::Z, {1})};
    EXPECT_LE(maxDiff(unitaryOf(*a.circuit), coreMatrix(controlled(z, {0}), 2)), kTol);
    EXPECT_LE(maxDiff(unitaryOf(*b.circuit), coreMatrix(controlled(Seq{makeGate(Prim::X, {1})}, {0}, {false}), 2)), kTol);
    // √X with principal branch is SX; controlled twice it is a doubly controlled SX.
    EXPECT_LE(maxDiff(unitaryOf(*c.circuit), coreMatrix(controlled(Seq{makeGate(Prim::SX, {2})}, {0, 1}), 3)), kTol);
}

TEST(InteropLowering, PauliMeasurementMapsTheProductOntoZ)
{
    // Ops before the measurement form W with W P W† = ±Z on the measured qubit; the ops after undo W.
    std::mt19937_64 rng(8);
    for (int trial = 0; trial < 100; ++trial)
    {
        SCOPED_TRACE(trial);
        const auto factors = randomPauli(4, rng);
        const bool invert = trial % 3 == 0;
        ImportedCircuit c = circuitOf(4);
        c.numClbits = 1;
        CircuitBuilder(c).measurePauli(factors, 0, invert);
        const auto m = std::ranges::find_if(c.ops, [](const Qputer::Operation& o) { return o.kind == Qputer::OpKind::Measure; });
        ASSERT_NE(m, c.ops.end());
        ASSERT_EQ(std::ranges::count_if(c.ops, [](const Qputer::Operation& o) { return o.kind == Qputer::OpKind::Measure; }), 1);
        const Eigen::MatrixXcd w = unitaryOfOps({c.ops.begin(), m}, 4);
        const Eigen::MatrixXcd undo = unitaryOfOps({m + 1, c.ops.end()}, 4);
        const Eigen::MatrixXcd z = pauliMatrix({{'Z', m->targets[0]}}, 4);
        EXPECT_LE(maxDiff(w * pauliMatrix(factors, 4) * w.adjoint(), invert ? Eigen::MatrixXcd(-z) : z), kTol);
        EXPECT_LE(maxDiffUpToPhase(undo * w, Eigen::MatrixXcd::Identity(16, 16)), kTol);
    }
}

TEST(InteropLowering, PauliRotationPhasesTheMinusOneEigenspace)
{
    std::mt19937_64 rng(9);
    const cd i(0, 1);
    for (int trial = 0; trial < 60; ++trial)
    {
        SCOPED_TRACE(trial);
        const auto factors = randomPauli(3, rng);
        const bool dagger = trial % 2 == 1;
        ImportedCircuit c = circuitOf(3);
        CircuitBuilder(c).rotatePauli(factors, dagger);
        const Eigen::MatrixXcd p = pauliMatrix(factors, 3), id = Eigen::MatrixXcd::Identity(8, 8);
        const Eigen::MatrixXcd want = (id + p) / 2.0 + (dagger ? -i : i) * (id - p) / 2.0;
        EXPECT_LE(maxDiffUpToPhase(unitaryOf(c), want), kTol);
    }
}

TEST(InteropLowering, CliffordFormsAreExactUpToPhase)
{
    for (int k = -4; k <= 4; ++k)
        for (const Qputer::OpKind kind : {Qputer::OpKind::RX, Qputer::OpKind::RY, Qputer::OpKind::RZ, Qputer::OpKind::Phase})
        {
            const double angle = k * std::numbers::pi / 2;
            const auto op = makeOperation(kind, {}, {0}, {angle}, {}, {}, std::nullopt, std::nullopt);
            const auto form = cliffordForm(op);
            ASSERT_TRUE(form) << Qputer::opName(kind) << " " << k;
            EXPECT_LE(maxDiffUpToPhase(unitaryOfOps(*form, 1), unitaryOfOps({op}, 1)), kTol);
        }
    EXPECT_FALSE(cliffordForm(makeOperation(Qputer::OpKind::RZ, {}, {0}, {0.3}, {}, {}, std::nullopt, std::nullopt)));
    const auto cp = makeOperation(Qputer::OpKind::CPhase, {0}, {1}, {std::numbers::pi}, {}, {}, std::nullopt, std::nullopt);
    ASSERT_TRUE(cliffordForm(cp));
    EXPECT_LE(maxDiffUpToPhase(unitaryOfOps(*cliffordForm(cp), 2), unitaryOfOps({cp}, 2)), kTol);
}

TEST(InteropLowering, ValidationRejectsWhatTheCoreRejects)
{
    using K = Qputer::OpKind;
    auto op = [](K k, QubitList c, QubitList t, std::vector<double> p = {}) { return makeOperation(k, c, t, p, {}, {}, std::nullopt, std::nullopt); };
    EXPECT_EQ(validateOperation(op(K::H, {}, {0}), 1, 0), "");
    EXPECT_NE(validateOperation(op(K::H, {}, {1}), 1, 0), "");           // out of range
    EXPECT_NE(validateOperation(op(K::CNOT, {0}, {0}), 2, 0), "");       // repeated qubit
    EXPECT_NE(validateOperation(op(K::RX, {}, {0}), 1, 0), "");          // missing angle
    EXPECT_NE(validateOperation(op(K::RX, {}, {0}, {NAN}), 1, 0), "");   // non-finite angle
    EXPECT_NE(validateOperation(op(K::Swap, {}, {0}), 2, 0), "");        // wrong arity
    EXPECT_NE(validateOperation(makeOperation(K::Measure, {}, {0}, {}, {}, {}, 3, std::nullopt), 1, 2), ""); // clbit out of range
    EXPECT_NE(validateOperation(makeOperation(K::Unitary, {}, {0}, {}, Eigen::MatrixXcd::Ones(2, 2), {}, std::nullopt, std::nullopt), 1, 0), "");
}
