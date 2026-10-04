#include <QuantumGates.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <format>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <Eigen/Dense>



#include "DenseGate.hpp"
#include "Parallel.hpp"

#define QPUTER_OMP_PARALLEL QPUTER_OMP(parallel if(count >= kParallelThreshold))



namespace Qputer
{

namespace
{

    using cd = std::complex<double>;
    using Index = std::size_t;

    using detail::kParallelThreshold;

    // std::complex operator* carries a NaN/Inf recovery branch (__muldc3) that blocks
    // vectorization; unitary gates on finite states never need it.
    inline cd cmul(cd a, cd b) noexcept
    {
        return {a.real() * b.real() - a.imag() * b.imag(),
                a.real() * b.imag() + a.imag() * b.real()};
    }

    inline cd rmul(double a, cd b) noexcept { return {a * b.real(), a * b.imag()}; }


    // Index geometry of one gate application. Every gate touches disjoint subspaces
    // spanned by its active (control + target) qubits; the loop counter p enumerates
    // the 2^(N - |active|) assignments of the inactive qubits, and insertZeros(p)
    // spreads p's bits around the active positions to form the subspace base index.
    struct Layout
    {
        Index loopCount = 0;                       // 2^(N - |active|)
        Index ctrlMask = 0;                        // OR'd into every base index
        Index runMask = 0;                         // 2^(lowest active qubit) - 1
        std::size_t nActive = 0;
        std::array<Index, kMaxQubits> lowMasks{};  // (1 << q) - 1 for active q, ascending
    };

    inline Index insertZeros(Index p, const Layout& L) noexcept
    {
        // Ascending insertion: each lowMask refers to a final bit position, and all
        // positions below it have already been opened up.
        for (std::size_t k = 0; k < L.nActive; ++k)
        {
            const Index low = L.lowMasks[k];
            p = ((p & ~low) << 1) | (p & low);
        }
        return p;
    }

    inline Index bit(Qubit q) noexcept { return Index{1} << q; }


    Layout buildLayout(const QuantumStateVector& state, std::span<const Qubit> controls,
                       std::span<const Qubit> targets, std::string_view context)
    {
        const std::size_t n = state.num_qubits();
        if (n == 0 || n > kMaxQubits)
            throw std::length_error(std::format("{}: num_qubits={} outside [1, {}]", context, n, kMaxQubits));
        if (state.size() != (Index{1} << n))
            throw std::logic_error(std::format("{}: state size {} != 2^{}", context, state.size(), n));

        Index seen = 0;
        auto claim = [&](Qubit q, std::string_view role)
        {
            if (q >= n)
                throw std::out_of_range(std::format("{}: {} qubit {} out of range [0, {})", context, role, q, n));
            if (seen & bit(q))
                throw std::invalid_argument(std::format("{}: qubit {} used more than once", context, q));
            seen |= bit(q);
        };

        Layout L;
        for (const Qubit c : controls) { claim(c, "control"); L.ctrlMask |= bit(c); }
        for (const Qubit t : targets)  { claim(t, "target"); }

        for (Index rest = seen; rest != 0; rest &= rest - 1)
            L.lowMasks[L.nActive++] = (Index{1} << std::countr_zero(rest)) - 1;

        L.loopCount = Index{1} << (n - L.nActive);
        L.runMask = L.nActive != 0 ? L.lowMasks[0] : L.loopCount - 1;
        return L;
    }


    // ---- Traversal ----

    // Static contiguous partition of [0, count) for the calling OpenMP thread.
    inline std::pair<Index, Index> threadSlice(Index count) noexcept
    {
        const Index nt = detail::threadCount();
        const Index t = detail::threadId();
        return {count * t / nt, count * (t + 1) / nt};
    }

    // Bits of p below the lowest active qubit pass through insertZeros unchanged, so
    // consecutive p within one aligned block of 2^q_min map to consecutive amplitude
    // indices. body(base, len) therefore receives contiguous runs it can stream with
    // unit stride (SIMD + prefetch) instead of recomputing an index per element.
    template <class Body>
    inline void visitRuns(const Layout& L, Index begin, Index end, Body&& body)
    {
        for (Index p = begin; p < end;)
        {
            const Index stop = std::min(end, (p | L.runMask) + 1);
            body(insertZeros(p, L) | L.ctrlMask, stop - p);
            p = stop;
        }
    }

    template <class Body>
    void forEachRun(const Layout& L, Body&& body)
    {
        const Index count = L.loopCount;
        QPUTER_OMP_PARALLEL
        {
            const auto [begin, end] = threadSlice(count);
            visitRuns(L, begin, end, body);
        }
    }


    // ---- Streams: unit-stride inner loops over one run. Paired streams are offset by
    // a target bit >= run length, so they never overlap and __restrict is valid. ----

    inline void streamSwap(cd* __restrict lo, cd* __restrict hi, Index len) noexcept
    {
        for (Index j = 0; j < len; ++j)
        {
            const cd t = lo[j];
            lo[j] = hi[j];
            hi[j] = t;
        }
    }

    inline void streamScale(cd* __restrict x, Index len, cd ph) noexcept
    {
        for (Index j = 0; j < len; ++j) x[j] = cmul(x[j], ph);
    }

    inline void streamReal2(cd* __restrict lo, cd* __restrict hi, Index len,
                            double m00, double m01, double m10, double m11) noexcept
    {
        for (Index j = 0; j < len; ++j)
        {
            const cd a0 = lo[j];
            const cd a1 = hi[j];
            lo[j] = rmul(m00, a0) + rmul(m01, a1);
            hi[j] = rmul(m10, a0) + rmul(m11, a1);
        }
    }

    inline void stream2(cd* __restrict lo, cd* __restrict hi, Index len,
                        cd m00, cd m01, cd m10, cd m11) noexcept
    {
        for (Index j = 0; j < len; ++j)
        {
            const cd a0 = lo[j];
            const cd a1 = hi[j];
            lo[j] = cmul(m00, a0) + cmul(m01, a1);
            hi[j] = cmul(m10, a0) + cmul(m11, a1);
        }
    }


    // ---- Kernels: each visits exactly the amplitudes the gate can change ----

    // Pauli X on target within the control subspace: pure permutation, no FLOPs.
    void kernelX(cd* a, const Layout& L, Index tbit)
    {
        forEachRun(L, [=](Index base, Index len) { streamSwap(a + base, a + (base | tbit), len); });
    }

    // Exchange |..1_a..0_b..> <-> |..0_a..1_b..>; the |00>, |11> amplitudes are untouched.
    void kernelSwap(cd* a, const Layout& L, Index abit, Index bbit)
    {
        forEachRun(L, [=](Index base, Index len) { streamSwap(a + (base | abit), a + (base | bbit), len); });
    }

    // Multiply by `ph` where every bit of ctrlMask is set: diag(1,...,1,ph) on the
    // active qubits. Touches 2^(N-k) of 2^N amplitudes.
    void kernelPhase(cd* a, const Layout& L, cd ph)
    {
        forEachRun(L, [=](Index base, Index len) { streamScale(a + base, len, ph); });
    }

    // diag(d0, d1) on target.
    void kernelDiag(cd* a, const Layout& L, Index tbit, cd d0, cd d1)
    {
        forEachRun(L, [=](Index base, Index len)
        {
            streamScale(a + base, len, d0);
            streamScale(a + (base | tbit), len, d1);
        });
    }

    // Real 2x2 (H, RY): half the multiplies of the complex kernel.
    void kernelReal2(cd* a, const Layout& L, Index tbit, double m00, double m01, double m10, double m11)
    {
        forEachRun(L, [=](Index base, Index len)
        {
            streamReal2(a + base, a + (base | tbit), len, m00, m01, m10, m11);
        });
    }

    // General complex 2x2, m row-major.
    void kernel2(cd* a, const Layout& L, Index tbit, const std::array<cd, 4>& m)
    {
        const cd m00 = m[0], m01 = m[1], m10 = m[2], m11 = m[3];
        forEachRun(L, [=](Index base, Index len)
        {
            stream2(a + base, a + (base | tbit), len, m00, m01, m10, m11);
        });
    }

    // Dense 2^M x 2^M (row-major U) on the subspace {base | offsets[v]}. kDim > 0 fixes
    // the dimension at compile time so the gather / mat-vec / scatter fully unroll;
    // kDim == 0 is the runtime-sized fallback with one scratch buffer per thread.
    template <std::size_t kDim>
    void kernelDense(cd* a, const Layout& L, const std::vector<Index>& offsets,
                     const std::vector<cd>& U)
    {
        const Index count = L.loopCount;
        const std::size_t dim = kDim != 0 ? kDim : offsets.size();
        const Index* off = offsets.data();
        const cd* u = U.data();

        QPUTER_OMP_PARALLEL
        {
            std::conditional_t<kDim != 0, std::array<cd, kDim>, std::vector<cd>> in{};
            if constexpr (kDim == 0) in.resize(dim);

            const auto [begin, end] = threadSlice(count);
            visitRuns(L, begin, end, [&](Index base, Index len)
            {
                for (Index j = 0; j < len; ++j)
                {
                    cd* sub = a + base + j;
                    for (std::size_t c = 0; c < dim; ++c) in[c] = sub[off[c]];
                    for (std::size_t r = 0; r < dim; ++r)
                    {
                        const cd* row = u + r * dim;
                        cd acc{0.0, 0.0};
                        for (std::size_t c = 0; c < dim; ++c) acc += cmul(row[c], in[c]);
                        sub[off[r]] = acc;
                    }
                }
            });
        }
    }


    // ---- Dispatch helpers ----

    void applyX(QuantumStateVector& state, std::span<const Qubit> controls, Qubit target, std::string_view ctx)
    {
        const std::array<Qubit, 1> tg{target};
        const Layout L = buildLayout(state, controls, tg, ctx);
        kernelX(state.data(), L, bit(target));
    }

    void applySwap(QuantumStateVector& state, std::span<const Qubit> controls, Qubit a, Qubit b, std::string_view ctx)
    {
        const std::array<Qubit, 2> tg{a, b};
        const Layout L = buildLayout(state, controls, tg, ctx);
        kernelSwap(state.data(), L, bit(a), bit(b));
    }

    // diag(1, ..., 1, ph) on the given qubits: symmetric, so all are treated as controls.
    void applyPhase(QuantumStateVector& state, std::span<const Qubit> qubits, cd ph, std::string_view ctx)
    {
        if (qubits.empty())
            throw std::invalid_argument(std::format("{}: at least one qubit required", ctx));
        const Layout L = buildLayout(state, qubits, {}, ctx);
        kernelPhase(state.data(), L, ph);
    }

    void applyDiag(QuantumStateVector& state, Qubit target, cd d0, cd d1, std::string_view ctx)
    {
        const std::array<Qubit, 1> tg{target};
        const Layout L = buildLayout(state, {}, tg, ctx);
        kernelDiag(state.data(), L, bit(target), d0, d1);
    }

    void applyReal2(QuantumStateVector& state, Qubit target, double m00, double m01, double m10, double m11,
                    std::string_view ctx)
    {
        const std::array<Qubit, 1> tg{target};
        const Layout L = buildLayout(state, {}, tg, ctx);
        kernelReal2(state.data(), L, bit(target), m00, m01, m10, m11);
    }

    void apply2(QuantumStateVector& state, Qubit target, const std::array<cd, 4>& m, std::string_view ctx)
    {
        const std::array<Qubit, 1> tg{target};
        const Layout L = buildLayout(state, {}, tg, ctx);
        kernel2(state.data(), L, bit(target), m);
    }

    // Validating entry point for the public API; the prepared path below does the work.
    void applyDenseChecked(QuantumStateVector& state, std::span<const Qubit> controls,
                           std::span<const Qubit> targets, const Eigen::MatrixXcd& U, std::string_view ctx)
    {
        const std::size_t m = targets.size();
        if (m == 0 || m > QuantumGate::kMaxDenseTargets)
            throw std::invalid_argument(std::format("{}: {} targets outside [1, {}]", ctx, m,
                                                    QuantumGate::kMaxDenseTargets));
        buildLayout(state, controls, targets, ctx); // validates indices before the O(8^M) check
        QuantumGate::require_unitary(U, m, ctx);
        detail::applyDense(state, detail::prepareDense(controls, targets, U));
    }

    constexpr double kInvSqrt2 = std::numbers::sqrt2 / 2.0;

    inline cd expi(double phi) noexcept { return std::polar(1.0, phi); }

} // namespace



void QuantumGate::require_unitary(const Eigen::MatrixXcd& U, std::size_t m, std::string_view ctx)
{
    const auto dim = static_cast<Eigen::Index>(Index{1} << m);
    if (U.rows() != dim || U.cols() != dim)
        throw std::invalid_argument(std::format("{}: matrix is {}x{}, expected {}x{} for {} target(s)",
                                                ctx, U.rows(), U.cols(), dim, dim, m));
    const double err = (U.adjoint() * U - Eigen::MatrixXcd::Identity(dim, dim)).cwiseAbs().maxCoeff();
    if (!(err <= kUnitaryTolerance)) // negated form also rejects NaN
        throw std::invalid_argument(std::format("{}: matrix not unitary, max|U^dag U - I| = {:.3e} > {:.1e}",
                                                ctx, err, kUnitaryTolerance));
}


// ---- 1 qubit ----

void QuantumGate::x(QuantumStateVector& state, Qubit target) { applyX(state, {}, target, "QuantumGate::x"); }

void QuantumGate::y(QuantumStateVector& state, Qubit target)
{
    apply2(state, target, {cd{0, 0}, cd{0, -1}, cd{0, 1}, cd{0, 0}}, "QuantumGate::y");
}

void QuantumGate::z(QuantumStateVector& state, Qubit target)
{
    const std::array<Qubit, 1> q{target};
    applyPhase(state, q, cd{-1, 0}, "QuantumGate::z");
}

void QuantumGate::h(QuantumStateVector& state, Qubit target)
{
    applyReal2(state, target, kInvSqrt2, kInvSqrt2, kInvSqrt2, -kInvSqrt2, "QuantumGate::h");
}

void QuantumGate::s(QuantumStateVector& state, Qubit target)
{
    const std::array<Qubit, 1> q{target};
    applyPhase(state, q, cd{0, 1}, "QuantumGate::s");
}

void QuantumGate::sdg(QuantumStateVector& state, Qubit target)
{
    const std::array<Qubit, 1> q{target};
    applyPhase(state, q, cd{0, -1}, "QuantumGate::sdg");
}

void QuantumGate::t(QuantumStateVector& state, Qubit target)
{
    const std::array<Qubit, 1> q{target};
    applyPhase(state, q, cd{kInvSqrt2, kInvSqrt2}, "QuantumGate::t");
}

void QuantumGate::tdg(QuantumStateVector& state, Qubit target)
{
    const std::array<Qubit, 1> q{target};
    applyPhase(state, q, cd{kInvSqrt2, -kInvSqrt2}, "QuantumGate::tdg");
}

void QuantumGate::sx(QuantumStateVector& state, Qubit target)
{
    const cd p{0.5, 0.5}, m{0.5, -0.5};
    apply2(state, target, {p, m, m, p}, "QuantumGate::sx");
}

void QuantumGate::rx(QuantumStateVector& state, Qubit target, double theta)
{
    const double c = std::cos(theta / 2), s = std::sin(theta / 2);
    apply2(state, target, {cd{c, 0}, cd{0, -s}, cd{0, -s}, cd{c, 0}}, "QuantumGate::rx");
}

void QuantumGate::ry(QuantumStateVector& state, Qubit target, double theta)
{
    const double c = std::cos(theta / 2), s = std::sin(theta / 2);
    applyReal2(state, target, c, -s, s, c, "QuantumGate::ry");
}

void QuantumGate::rz(QuantumStateVector& state, Qubit target, double theta)
{
    applyDiag(state, target, expi(-theta / 2), expi(theta / 2), "QuantumGate::rz");
}

void QuantumGate::phase(QuantumStateVector& state, Qubit target, double lambda)
{
    const std::array<Qubit, 1> q{target};
    applyPhase(state, q, expi(lambda), "QuantumGate::phase");
}

void QuantumGate::u3(QuantumStateVector& state, Qubit target, double theta, double phi, double lambda)
{
    const double c = std::cos(theta / 2), s = std::sin(theta / 2);
    apply2(state, target,
           {cd{c, 0}, -s * expi(lambda), s * expi(phi), c * expi(phi + lambda)},
           "QuantumGate::u3");
}


// ---- 2 qubit ----

void QuantumGate::cnot(QuantumStateVector& state, Qubit control, Qubit target)
{
    const std::array<Qubit, 1> c{control};
    applyX(state, c, target, "QuantumGate::cnot");
}

void QuantumGate::cz(QuantumStateVector& state, Qubit a, Qubit b)
{
    const std::array<Qubit, 2> q{a, b};
    applyPhase(state, q, cd{-1, 0}, "QuantumGate::cz");
}

void QuantumGate::cphase(QuantumStateVector& state, Qubit a, Qubit b, double lambda)
{
    const std::array<Qubit, 2> q{a, b};
    applyPhase(state, q, expi(lambda), "QuantumGate::cphase");
}

void QuantumGate::swap(QuantumStateVector& state, Qubit a, Qubit b)
{
    applySwap(state, {}, a, b, "QuantumGate::swap");
}


// ---- 3 qubit ----

void QuantumGate::toffoli(QuantumStateVector& state, Qubit control0, Qubit control1, Qubit target)
{
    const std::array<Qubit, 2> c{control0, control1};
    applyX(state, c, target, "QuantumGate::toffoli");
}

void QuantumGate::fredkin(QuantumStateVector& state, Qubit control, Qubit a, Qubit b)
{
    const std::array<Qubit, 1> c{control};
    applySwap(state, c, a, b, "QuantumGate::fredkin");
}


// ---- N qubit ----

void QuantumGate::mcx(QuantumStateVector& state, const QubitList& controls, Qubit target)
{
    applyX(state, controls, target, "QuantumGate::mcx");
}

void QuantumGate::mcz(QuantumStateVector& state, const QubitList& qubits)
{
    applyPhase(state, qubits, cd{-1, 0}, "QuantumGate::mcz");
}

void QuantumGate::mcphase(QuantumStateVector& state, const QubitList& qubits, double lambda)
{
    applyPhase(state, qubits, expi(lambda), "QuantumGate::mcphase");
}


// ---- Arbitrary ----

void QuantumGate::apply(QuantumStateVector& state, const QubitList& targets, const Eigen::MatrixXcd& U)
{
    applyDenseChecked(state, {}, targets, U, "QuantumGate::apply");
}

void QuantumGate::controlled(QuantumStateVector& state, const QubitList& controls,
                             const QubitList& targets, const Eigen::MatrixXcd& U)
{
    applyDenseChecked(state, controls, targets, U, "QuantumGate::controlled");
}


// ---- Prepared dense gates ----

detail::DenseGate detail::prepareDense(std::span<const Qubit> controls, std::span<const Qubit> targets,
                                       const Eigen::MatrixXcd& U)
{
    const std::size_t m = targets.size();
    const std::size_t dim = Index{1} << m;
    DenseGate g{{controls.begin(), controls.end()}, {targets.begin(), targets.end()}, std::vector<Index>(dim, 0),
                std::vector<cd>(dim * dim)};

    // targets[0] <-> MSB of v.
    for (std::size_t v = 0; v < dim; ++v)
        for (std::size_t r = 0; r < m; ++r)
            if ((v >> (m - 1 - r)) & 1U) g.offsets[v] |= bit(targets[r]);

    for (std::size_t r = 0; r < dim; ++r)
        for (std::size_t c = 0; c < dim; ++c)
            g.rowMajor[r * dim + c] = U(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c));
    return g;
}

void detail::applyDense(QuantumStateVector& state, const DenseGate& g)
{
    const Layout L = buildLayout(state, g.controls, g.targets, "dense gate");
    const std::vector<cd>& u = g.rowMajor;
    switch (g.targets.size())
    {
        case 1:  kernel2(state.data(), L, bit(g.targets[0]), {u[0], u[1], u[2], u[3]}); break;
        case 2:  kernelDense<4>(state.data(), L, g.offsets, u); break;
        case 3:  kernelDense<8>(state.data(), L, g.offsets, u); break;
        default: kernelDense<0>(state.data(), L, g.offsets, u); break;
    }
}


}// namespace qstate
