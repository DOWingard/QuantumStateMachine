#include <QuantumGates.hpp>
#include "DenseGate.hpp"
#include "Parallel.hpp"

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

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif





// `work` is in single-qubit-pair units: the parallel cutoff and team size scale with it.
#define QPUTER_OMP_PARALLEL(work) QPUTER_OMP(parallel if((work) >= kParallelThreshold) num_threads(detail::teamSize(work)))



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


    // ---- Arithmetic shared by the vector kernels and their scalar tails ----
    //
    // Each helper evaluates exactly the per-lane operations of its AVX2 counterpart (same
    // fused multiply-adds, same order), so an amplitude's result never depends on whether a
    // thread slice boundary put it on the vector path or the scalar one.

#if defined(__AVX2__) && defined(__FMA__)
#define QPUTER_SIMD_AVX2 1
    inline double fmadd(double a, double b, double c) noexcept { return std::fma(a, b, c); }
#else
    inline double fmadd(double a, double b, double c) noexcept { return a * b + c; }
#endif

    // x * (pr + i pi); vector form _mm256_fmaddsub_pd(x, pr, swap(x) * pi).
    inline cd scale(cd x, double pr, double pi) noexcept
    {
        return {fmadd(x.real(), pr, -(x.imag() * pi)), fmadd(x.imag(), pr, x.real() * pi)};
    }

    // m0 x0 + m1 x1 for real m0, m1; vector form fmadd(m1, x1, m0 * x0).
    inline cd combineReal(double m0, cd x0, double m1, cd x1) noexcept
    {
        return {fmadd(m1, x1.real(), m0 * x0.real()), fmadd(m1, x1.imag(), m0 * x0.imag())};
    }

    // m0 x0 + m1 x1 for complex m0, m1: the real parts of m scale x, the imaginary parts scale
    // swap(x) = (im, re), and addsub folds the two; vector form in mix2 below.
    inline cd combine(cd m0, cd x0, cd m1, cd x1) noexcept
    {
        const double tr = fmadd(m1.real(), x1.real(), m0.real() * x0.real());
        const double ti = fmadd(m1.real(), x1.imag(), m0.real() * x0.imag());
        const double ur = fmadd(m1.imag(), x1.imag(), m0.imag() * x0.imag());
        const double ui = fmadd(m1.imag(), x1.real(), m0.imag() * x0.real());
        return {tr - ur, ti + ui};
    }

#ifdef QPUTER_SIMD_AVX2
    // Two complex amplitudes per register: [re0, im0, re1, im1].
    inline __m256d load2(const cd* p) noexcept { return _mm256_loadu_pd(reinterpret_cast<const double*>(p)); }
    inline void store2(cd* p, __m256d v) noexcept { _mm256_storeu_pd(reinterpret_cast<double*>(p), v); }
    inline __m256d swapReIm(__m256d v) noexcept { return _mm256_permute_pd(v, 0b0101); }
    inline __m256d lowHalf(__m256d v) noexcept { return _mm256_permute2f128_pd(v, v, 0x00); }  // [a0, a0]
    inline __m256d highHalf(__m256d v) noexcept { return _mm256_permute2f128_pd(v, v, 0x11); } // [a1, a1]

    inline __m256d scale2(__m256d x, __m256d pr, __m256d pi) noexcept
    {
        return _mm256_fmaddsub_pd(x, pr, _mm256_mul_pd(swapReIm(x), pi));
    }

    // Lane-wise m0 x0 + m1 x1 with (r, i) = real and imaginary parts of m, as in combine().
    inline __m256d mix2(__m256d r0, __m256d i0, __m256d x0, __m256d r1, __m256d i1, __m256d x1) noexcept
    {
        const __m256d t = _mm256_fmadd_pd(r1, x1, _mm256_mul_pd(r0, x0));
        const __m256d u = _mm256_fmadd_pd(i1, swapReIm(x1), _mm256_mul_pd(i0, swapReIm(x0)));
        return _mm256_addsub_pd(t, u);
    }

    inline __m256d broadcastRe(cd a) noexcept { return _mm256_set1_pd(a.real()); }
    inline __m256d broadcastIm(cd a) noexcept { return _mm256_set1_pd(a.imag()); }
    inline __m256d perLaneRe(cd lo, cd hi) noexcept { return _mm256_setr_pd(lo.real(), lo.real(), hi.real(), hi.real()); }
    inline __m256d perLaneIm(cd lo, cd hi) noexcept { return _mm256_setr_pd(lo.imag(), lo.imag(), hi.imag(), hi.imag()); }
#endif


    // Index geometry of one gate application. Every gate touches disjoint subspaces
    // spanned by its active (control + target) qubits; the loop counter p enumerates
    // the 2^(N - |active|) assignments of the inactive qubits, and insertZeros(p)
    // spreads p's bits around the active positions to form the subspace base index.
    struct Layout
    {
        Index loopCount = 0;                       // 2^(N - |active|)
        Index ctrlMask = 0;                        // OR'd into every base index
        Index runMask = 0;                         // 2^(lowest active qubit) - 1
        Index windowMask = 0;                      // p bits that vary within one window, see visitRuns
        Index runStride = 0;                       // amplitudes between consecutive runs of a window
        int runShift = 0;                          // log2(runMask + 1)
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
        L.runShift = std::countr_zero(L.runMask + 1);

        // The lowest g active qubits are consecutive: runs step over all of them at once, and a
        // window lasts until p reaches the next active qubit above that group.
        const auto g = L.nActive != 0 ? static_cast<std::size_t>(std::countr_one(seen >> L.runShift)) : 0;
        L.runStride = (L.runMask + 1) << g;
        L.windowMask = g < L.nActive ? L.lowMasks[g] >> g : L.loopCount - 1;
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
    // consecutive p within one aligned block of 2^q_min map to a contiguous run of
    // amplitudes. If the lowest g active qubits are consecutive, then while the p bits from the
    // next active qubit up stay fixed (a window), consecutive runs are 2^g runs apart, so one
    // insertZeros serves the whole window: body(base, len, count) receives `count` runs of `len`
    // amplitudes starting at base + L.runStride * r. Only a thread slice's first and last run
    // can be partial (count 1).
    template <class Body>
    inline void visitRuns(const Layout& L, Index begin, Index end, Body&& body)
    {
        const Index run = L.runMask + 1;
        for (Index p = begin; p < end;)
        {
            const Index base = insertZeros(p, L) | L.ctrlMask;
            if ((p & L.runMask) != 0 || end - p < run)
            {
                const Index stop = std::min(end, (p | L.runMask) + 1);
                body(base, stop - p, Index{1});
                p = stop;
                continue;
            }
            const Index count = (std::min(end, (p | L.windowMask) + 1) - p) >> L.runShift;
            body(base, run, count);
            p += count << L.runShift;
        }
    }

    template <class Body>
    void forEachRun(const Layout& L, Body&& body)
    {
        const Index count = L.loopCount;
        QPUTER_OMP_PARALLEL(count)
        {
            const auto [begin, end] = threadSlice(count);
            visitRuns(L, begin, end, body);
        }
    }

    // f(base) for each of `count` runs `stride` amplitudes apart.
    template <class F>
    inline void eachRun(Index base, Index count, Index stride, F&& f)
    {
        for (Index r = 0; r < count; ++r, base += stride) f(base);
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
        Index j = 0;
#ifdef QPUTER_SIMD_AVX2
        const __m256d pr = broadcastRe(ph), pi = broadcastIm(ph);
        for (; j + 2 <= len; j += 2) store2(x + j, scale2(load2(x + j), pr, pi));
#endif
        for (; j < len; ++j) x[j] = scale(x[j], ph.real(), ph.imag());
    }

    inline void streamReal2(cd* __restrict lo, cd* __restrict hi, Index len,
                            double m00, double m01, double m10, double m11) noexcept
    {
        Index j = 0;
#ifdef QPUTER_SIMD_AVX2
        const __m256d a = _mm256_set1_pd(m00), b = _mm256_set1_pd(m01);
        const __m256d c = _mm256_set1_pd(m10), d = _mm256_set1_pd(m11);
        for (; j + 2 <= len; j += 2)
        {
            const __m256d x0 = load2(lo + j), x1 = load2(hi + j);
            store2(lo + j, _mm256_fmadd_pd(b, x1, _mm256_mul_pd(a, x0)));
            store2(hi + j, _mm256_fmadd_pd(d, x1, _mm256_mul_pd(c, x0)));
        }
#endif
        for (; j < len; ++j)
        {
            const cd a0 = lo[j];
            const cd a1 = hi[j];
            lo[j] = combineReal(m00, a0, m01, a1);
            hi[j] = combineReal(m10, a0, m11, a1);
        }
    }

    inline void stream2(cd* __restrict lo, cd* __restrict hi, Index len,
                        cd m00, cd m01, cd m10, cd m11) noexcept
    {
        Index j = 0;
#ifdef QPUTER_SIMD_AVX2
        const __m256d r00 = broadcastRe(m00), i00 = broadcastIm(m00), r01 = broadcastRe(m01), i01 = broadcastIm(m01);
        const __m256d r10 = broadcastRe(m10), i10 = broadcastIm(m10), r11 = broadcastRe(m11), i11 = broadcastIm(m11);
        for (; j + 2 <= len; j += 2)
        {
            const __m256d x0 = load2(lo + j), x1 = load2(hi + j);
            store2(lo + j, mix2(r00, i00, x0, r01, i01, x1));
            store2(hi + j, mix2(r10, i10, x0, r11, i11, x1));
        }
#endif
        for (; j < len; ++j)
        {
            const cd a0 = lo[j];
            const cd a1 = hi[j];
            lo[j] = combine(m00, a0, m01, a1);
            hi[j] = combine(m10, a0, m11, a1);
        }
    }


    // ---- Pair streams: qubit 0 is the lowest active qubit, so runs have length 1 and each
    // run's pair (a[s r], a[s r + 1]) is adjacent; one pair fills one vector. s = stride. ----

    inline void pairsSwap(cd* a, Index count, Index s) noexcept
    {
        for (Index r = 0; r < count; ++r) std::swap(a[s * r], a[s * r + 1]);
    }

    // diag(d0, d1) on each pair; d1 alone (d0 = 1, exact) when only the odd amplitude changes.
    inline void pairsDiag(cd* a, Index count, Index s, cd d0, cd d1) noexcept
    {
#ifdef QPUTER_SIMD_AVX2
        const __m256d pr = perLaneRe(d0, d1), pi = perLaneIm(d0, d1);
        for (Index r = 0; r < count; ++r) store2(a + s * r, scale2(load2(a + s * r), pr, pi));
#else
        for (Index r = 0; r < count; ++r)
        {
            a[s * r] = scale(a[s * r], d0.real(), d0.imag());
            a[s * r + 1] = scale(a[s * r + 1], d1.real(), d1.imag());
        }
#endif
    }

    // Scales the odd amplitude of each pair by ph, leaving the even one bit-exact.
    inline void pairsScaleOdd(cd* a, Index count, Index s, cd ph) noexcept
    {
#ifdef QPUTER_SIMD_AVX2
        const __m256d pr = broadcastRe(ph), pi = broadcastIm(ph);
        for (Index r = 0; r < count; ++r)
        {
            const __m256d v = load2(a + s * r);
            store2(a + s * r, _mm256_blend_pd(v, scale2(v, pr, pi), 0b1100));
        }
#else
        for (Index r = 0; r < count; ++r) a[s * r + 1] = scale(a[s * r + 1], ph.real(), ph.imag());
#endif
    }

    inline void pairsReal2(cd* a, Index count, Index s, double m00, double m01, double m10, double m11) noexcept
    {
#ifdef QPUTER_SIMD_AVX2
        const __m256d m0 = _mm256_setr_pd(m00, m00, m10, m10), m1 = _mm256_setr_pd(m01, m01, m11, m11);
        for (Index r = 0; r < count; ++r)
        {
            const __m256d v = load2(a + s * r);
            store2(a + s * r, _mm256_fmadd_pd(m1, highHalf(v), _mm256_mul_pd(m0, lowHalf(v))));
        }
#else
        for (Index r = 0; r < count; ++r)
        {
            const cd a0 = a[s * r], a1 = a[s * r + 1];
            a[s * r] = combineReal(m00, a0, m01, a1);
            a[s * r + 1] = combineReal(m10, a0, m11, a1);
        }
#endif
    }

    inline void pairs2(cd* a, Index count, Index s, cd m00, cd m01, cd m10, cd m11) noexcept
    {
#ifdef QPUTER_SIMD_AVX2
        const __m256d r0 = perLaneRe(m00, m10), i0 = perLaneIm(m00, m10);
        const __m256d r1 = perLaneRe(m01, m11), i1 = perLaneIm(m01, m11);
        for (Index r = 0; r < count; ++r)
        {
            const __m256d v = load2(a + s * r);
            store2(a + s * r, mix2(r0, i0, lowHalf(v), r1, i1, highHalf(v)));
        }
#else
        for (Index r = 0; r < count; ++r)
        {
            const cd a0 = a[s * r], a1 = a[s * r + 1];
            a[s * r] = combine(m00, a0, m01, a1);
            a[s * r + 1] = combine(m10, a0, m11, a1);
        }
#endif
    }


    // ---- Kernels: each visits exactly the amplitudes the gate can change ----

    // Pauli X on target within the control subspace: pure permutation, no FLOPs.
    void kernelX(cd* a, const Layout& L, Index tbit)
    {
        forEachRun(L, [=, s = L.runStride](Index base, Index len, Index count)
        {
            if (tbit == 1) return pairsSwap(a + base, count, s);
            eachRun(base, count, s, [=](Index b) { streamSwap(a + b, a + (b | tbit), len); });
        });
    }

    // Exchange |..1_a..0_b..> <-> |..0_a..1_b..>; the |00>, |11> amplitudes are untouched.
    void kernelSwap(cd* a, const Layout& L, Index abit, Index bbit)
    {
        forEachRun(L, [=, s = L.runStride](Index base, Index len, Index count)
        {
            eachRun(base, count, s, [=](Index b) { streamSwap(a + (b | abit), a + (b | bbit), len); });
        });
    }

    // Multiply by `ph` where every bit of ctrlMask is set: diag(1,...,1,ph) on the
    // active qubits. Touches 2^(N-k) of 2^N amplitudes.
    void kernelPhase(cd* a, const Layout& L, cd ph)
    {
        // With qubit 0 active, each run is the odd amplitude of an adjacent pair.
        const bool oddRuns = L.nActive != 0 && L.lowMasks[0] == 0;
        forEachRun(L, [=, s = L.runStride](Index base, Index len, Index count)
        {
            if (oddRuns) return pairsScaleOdd(a + base - 1, count, s, ph);
            eachRun(base, count, s, [=](Index b) { streamScale(a + b, len, ph); });
        });
    }

    // diag(d0, d1) on target.
    void kernelDiag(cd* a, const Layout& L, Index tbit, cd d0, cd d1)
    {
        forEachRun(L, [=, s = L.runStride](Index base, Index len, Index count)
        {
            if (tbit == 1) return pairsDiag(a + base, count, s, d0, d1);
            eachRun(base, count, s, [=](Index b)
            {
                streamScale(a + b, len, d0);
                streamScale(a + (b | tbit), len, d1);
            });
        });
    }

    // Real 2x2 (H, RY): half the multiplies of the complex kernel.
    void kernelReal2(cd* a, const Layout& L, Index tbit, double m00, double m01, double m10, double m11)
    {
        forEachRun(L, [=, s = L.runStride](Index base, Index len, Index count)
        {
            if (tbit == 1) return pairsReal2(a + base, count, s, m00, m01, m10, m11);
            eachRun(base, count, s, [=](Index b) { streamReal2(a + b, a + (b | tbit), len, m00, m01, m10, m11); });
        });
    }

    // General complex 2x2, m row-major.
    void kernel2(cd* a, const Layout& L, Index tbit, const std::array<cd, 4>& m)
    {
        const cd m00 = m[0], m01 = m[1], m10 = m[2], m11 = m[3];
        forEachRun(L, [=, s = L.runStride](Index base, Index len, Index count)
        {
            if (tbit == 1) return pairs2(a + base, count, s, m00, m01, m10, m11);
            eachRun(base, count, s, [=](Index b) { stream2(a + b, a + (b | tbit), len, m00, m01, m10, m11); });
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
        // A subspace costs dim^2 complex multiply-adds against a pair's 4, so this compute-bound
        // kernel parallelizes at fewer subspaces and with more threads than the streaming ones.
        const Index work = count * dim * dim / 4;

        QPUTER_OMP_PARALLEL(work)
        {
            std::conditional_t<kDim != 0, std::array<cd, kDim>, std::vector<cd>> in{};
            if constexpr (kDim == 0) in.resize(dim);

            const auto [begin, end] = threadSlice(count);
            visitRuns(L, begin, end, [&](Index base, Index len, Index runs)
            {
                eachRun(base, runs, L.runStride, [&](Index b)
                {
                    for (Index j = 0; j < len; ++j)
                    {
                        cd* sub = a + b + j;
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
