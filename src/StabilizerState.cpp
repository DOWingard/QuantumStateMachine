#include <StabilizerState.hpp>
#include "Parallel.hpp"
#include "ReadoutKernels.hpp"

#include <algorithm>
#include <bit>
#include <complex>
#include <format>
#include <numeric>
#include <stdexcept>
#include <utility>

#if defined(__PCLMUL__)
#include <immintrin.h>
#endif



namespace Qputer
{

namespace
{

    using Word = std::uint64_t;
    using detail::Index;

    constexpr std::size_t kWordBits = 64;

    constexpr std::size_t wordsFor(std::size_t bits) noexcept { return (bits + kWordBits - 1) / kWordBits; }
    constexpr Word bitOf(std::size_t j) noexcept { return Word{1} << (j % kWordBits); }
    constexpr Word broadcast(bool b) noexcept { return b ? ~Word{0} : Word{0}; }

    // Below this many column-words (~1 ms on one core) a measurement update stays on the calling
    // thread: smaller tableaus are cache-resident, and splitting them across cores costs more
    // in fork/join and cache-to-cache traffic than it saves.
    constexpr std::size_t kParallelWords = std::size_t{1} << 22;

    void requireQubit(std::size_t n, Qubit q, std::string_view ctx)
    {
        if (q >= n) throw std::out_of_range(std::format("{}: qubit {} out of range [0, {})", ctx, q, n));
    }

    void requirePair(std::size_t n, Qubit a, Qubit b, std::string_view ctx)
    {
        requireQubit(n, a, ctx);
        requireQubit(n, b, ctx);
        if (a == b) throw std::invalid_argument(std::format("{}: qubit {} used more than once", ctx, a));
    }

    void requireDistinct(std::size_t n, std::span<const Qubit> qubits, std::string_view ctx)
    {
        for (const Qubit q : qubits) requireQubit(n, q, ctx);
        std::vector<Qubit> sorted(qubits.begin(), qubits.end());
        std::ranges::sort(sorted);
        if (const auto dup = std::ranges::adjacent_find(sorted); dup != sorted.end())
            throw std::invalid_argument(std::format("{}: qubit {} used more than once", ctx, *dup));
    }

    // Bit i of the result is the XOR of bits [0, i) of v. A carry-less product with all ones
    // is the inclusive prefix XOR in one instruction.
    inline Word exclusivePrefixXor(Word v) noexcept
    {
#if defined(__PCLMUL__)
        const __m128i p = _mm_clmulepi64_si128(_mm_cvtsi64_si128(static_cast<long long>(v)),
                                               _mm_set1_epi64x(-1), 0x00);
        return static_cast<Word>(_mm_cvtsi128_si64(p)) << 1;
#else
        v ^= v << 1;
        v ^= v << 2;
        v ^= v << 4;
        v ^= v << 8;
        v ^= v << 16;
        v ^= v << 32;
        return v << 1;
#endif
    }

    // Per-bit-lane counters mod 4 (lo, hi) of the i-exponent P1 * P2 picks up per qubit: +1 for
    // XY, YZ, ZX and -1 for XZ, YX, ZY (the g function of Aaronson-Gottesman). The pair
    // anticommutes where g != 0, and among those g = -1 exactly where x1^z1^x2^z2^(x1 z2) is set.
    // Summing lanes, lo + 2 hi, gives the total mod 4.
    inline void accumulatePhase(Word x1, Word z1, Word x2, Word z2, Word& lo, Word& hi) noexcept
    {
        const Word x1z2 = x1 & z2;
        const Word anti = x1z2 ^ (z1 & x2);
        hi ^= (lo ^ x1 ^ z1 ^ x2 ^ z2 ^ x1z2) & anti;
        lo ^= anti;
    }

    inline unsigned laneTotal(Word lo, Word hi) noexcept
    {
        return static_cast<unsigned>(std::popcount(lo) + 2 * std::popcount(hi));
    }

    // Highest set bit of a bit vector, which must be nonzero.
    std::size_t leadingBit(std::span<const Word> v) noexcept
    {
        std::size_t k = v.size();
        while (v[--k] == 0) {}
        return k * kWordBits + kWordBits - 1 - static_cast<std::size_t>(std::countl_zero(v[k]));
    }

    // Words per block of a column visited together: one 64-byte cache line.
    constexpr std::size_t kBlock = 8;

    // Column strides that are multiples of this many words (512 bytes) are padded by a line.
    constexpr std::size_t kSetStride = 64;

    // Distinct buffers per thread, reused across measurements so trajectory loops do not
    // allocate per shot.
    struct Scratch
    {
        std::vector<Word> mask;
        std::vector<std::size_t> blocks;
        std::vector<Qubit> xs, ys, zs;
        std::vector<Word> lo, hi;       // this thread's phase counters
        std::vector<Word> sumLo, sumHi; // all threads' counters, on the measuring thread
    };

    Scratch& scratch()
    {
        thread_local Scratch s;
        return s;
    }

    // psi := (psi + P psi) / 2 for P|k> = c (-1)^popcount(k & zMask) |k ^ xMask>, i.e. the
    // projector onto P's +1 eigenspace, with c = (-1)^r i^{#Y} absorbing P's sign and Y = iXZ.
    void projectOnto(QuantumStateVector& psi, Index xMask, Index zMask, std::complex<double> c)
    {
        std::complex<double>* a = psi.data();
        const Index dim = psi.size();
        auto coef = [&](Index k) { return (std::popcount(k & zMask) & 1) ? -c : c; };

        if (xMask == 0)
        {
            QPUTER_OMP(parallel for schedule(static) if(dim >= detail::kParallelThreshold))
            for (Index k = 0; k < dim; ++k) a[k] = 0.5 * (a[k] + coef(k) * a[k]);
            return;
        }

        // Pairs (k, k ^ xMask), enumerated by k with xMask's highest bit clear.
        const Index low = std::bit_floor(xMask) - 1;
        const Index half = dim / 2;
        QPUTER_OMP(parallel for schedule(static) if(half >= detail::kParallelThreshold))
        for (Index p = 0; p < half; ++p)
        {
            const Index k = ((p & ~low) << 1) | (p & low);
            const Index kp = k ^ xMask;
            const std::complex<double> ak = a[k], akp = a[kp];
            a[k] = 0.5 * (ak + coef(kp) * akp);
            a[kp] = 0.5 * (akp + coef(k) * ak);
        }
    }

    // Word-parallel bodies over one or two columns; distinct columns and the sign vector never
    // overlap, which lets the compiler vectorize the loops.
    template <class F>
    inline void forWords(Word* __restrict x, Word* __restrict z, Word* __restrict r, std::size_t len, F f)
    {
        for (std::size_t k = 0; k < len; ++k) f(x[k], z[k], r[k]);
    }

    template <class F>
    inline void forWords(Word* __restrict xa, Word* __restrict za, Word* __restrict xb, Word* __restrict zb,
                         Word* __restrict r, std::size_t len, F f)
    {
        for (std::size_t k = 0; k < len; ++k) f(xa[k], za[k], xb[k], zb[k], r[k]);
    }

} // namespace



// ---- OutcomeSupport ----

bool OutcomeSupport::contains(std::span<const std::uint64_t> outcome) const
{
    if (outcome.size() != words)
        throw std::invalid_argument(std::format("OutcomeSupport::contains: {} words given, expected {}",
                                                outcome.size(), words));
    std::vector<Word> v(words);
    for (std::size_t k = 0; k < words; ++k) v[k] = outcome[k] ^ offset[k];
    for (std::size_t m = 0; m < dimension(); ++m)
    {
        const auto r = row(m);
        const std::size_t lead = leadingBit(r);
        if (v[lead / kWordBits] & bitOf(lead))
            for (std::size_t k = 0; k < words; ++k) v[k] ^= r[k];
    }
    return std::ranges::all_of(v, [](Word x) { return x == 0; });
}



// ---- Construction ----

StabilizerState::StabilizerState(std::size_t numQubits)
    : n(numQubits),
      hw(wordsFor(numQubits)),
      cw((2 * hw + kBlock - 1) / kBlock * kBlock),
      stride(2 * cw % kSetStride == 0 ? 2 * cw + kBlock : 2 * cw)
{
    if (n == 0 || n > kMaxStabilizerQubits)
        throw std::length_error(std::format("StabilizerState: num_qubits={} outside [1, {}]", n,
                                            kMaxStabilizerQubits));
    bits.resize(n * stride);
    sign.resize(cw);
    set_basis(0);
}

void StabilizerState::set_basis(std::uint64_t index)
{
    std::ranges::fill(bits, Word{0});
    std::ranges::fill(sign, Word{0});
    for (Qubit q = 0; q < n; ++q)
    {
        xCol(q)[q / kWordBits] |= bitOf(q);      // destabilizer q: X_q
        zCol(q)[hw + q / kWordBits] |= bitOf(q); // stabilizer q: (-1)^{b_q} Z_q
    }
    sign[hw] = n < kWordBits ? index & (bitOf(n) - 1) : index;
}



// ---- Gates: conjugation rules P -> U P U^dagger on each generator, 64 per word ----

template <class F>
void StabilizerState::updateColumn(Qubit q, F update)
{
    forWords(xCol(q), zCol(q), sign.data(), cw, update);
}

template <class F>
void StabilizerState::updateColumns(Qubit a, Qubit b, F update)
{
    forWords(xCol(a), zCol(a), xCol(b), zCol(b), sign.data(), cw, update);
}

// Padding bits past n in each half stay zero: every update maps all-zero x, z, r to zero.

void StabilizerState::x(Qubit target)
{
    requireQubit(n, target, "StabilizerState::x");
    updateColumn(target, [](Word&, Word& z, Word& r) { r ^= z; }); // Z, Y -> -Z, -Y
}

void StabilizerState::y(Qubit target)
{
    requireQubit(n, target, "StabilizerState::y");
    updateColumn(target, [](Word& x, Word& z, Word& r) { r ^= x ^ z; }); // X, Z -> -X, -Z
}

void StabilizerState::z(Qubit target)
{
    requireQubit(n, target, "StabilizerState::z");
    updateColumn(target, [](Word& x, Word&, Word& r) { r ^= x; }); // X, Y -> -X, -Y
}

void StabilizerState::h(Qubit target)
{
    requireQubit(n, target, "StabilizerState::h");
    updateColumn(target, [](Word& x, Word& z, Word& r) // X <-> Z, Y -> -Y
    {
        r ^= x & z;
        std::swap(x, z);
    });
}

void StabilizerState::s(Qubit target)
{
    requireQubit(n, target, "StabilizerState::s");
    updateColumn(target, [](Word& x, Word& z, Word& r) // X -> Y, Y -> -X
    {
        r ^= x & z;
        z ^= x;
    });
}

void StabilizerState::sdg(Qubit target)
{
    requireQubit(n, target, "StabilizerState::sdg");
    updateColumn(target, [](Word& x, Word& z, Word& r) // X -> -Y, Y -> X
    {
        r ^= x & ~z;
        z ^= x;
    });
}

void StabilizerState::sx(Qubit target)
{
    requireQubit(n, target, "StabilizerState::sx");
    updateColumn(target, [](Word& x, Word& z, Word& r) // sqrt(X) = H S H: Z -> -Y, Y -> Z
    {
        r ^= z & ~x;
        x ^= z;
    });
}

void StabilizerState::cnot(Qubit control, Qubit target)
{
    requirePair(n, control, target, "StabilizerState::cnot");
    updateColumns(control, target, [](Word& xc, Word& zc, Word& xt, Word& zt, Word& r)
    {
        r ^= xc & zt & ~(xt ^ zc);
        xt ^= xc;
        zc ^= zt;
    });
}

void StabilizerState::cz(Qubit a, Qubit b)
{
    requirePair(n, a, b, "StabilizerState::cz");
    updateColumns(a, b, [](Word& xa, Word& za, Word& xb, Word& zb, Word& r) // X_a -> X_a Z_b, X_b -> Z_a X_b
    {
        r ^= xa & xb & (za ^ zb);
        za ^= xb;
        zb ^= xa;
    });
}

void StabilizerState::swap(Qubit a, Qubit b)
{
    requirePair(n, a, b, "StabilizerState::swap");
    std::swap_ranges(xCol(a), xCol(a) + 2 * cw, xCol(b)); // x and z columns are adjacent
}



// ---- Measurement ----

std::optional<std::size_t> StabilizerState::randomPivot(Qubit q) const
{
    const Word* x = xCol(q) + hw;
    for (std::size_t k = 0; k < hw; ++k)
        if (x[k] != 0) return n + k * kWordBits + static_cast<std::size_t>(std::countr_zero(x[k]));
    return std::nullopt;
}

void StabilizerState::collapse(Qubit q, std::size_t p, bool outcome)
{
    const std::size_t j = p - n;                    // pivot = stabilizer j, partner = destabilizer j
    const std::size_t dw = j / kWordBits;           // partner's word in a column
    const std::size_t pw = hw + dw;                 // pivot's word
    const Word m = bitOf(j);
    const Word pivotSign = broadcast(sign[pw] & m);

    // Every generator anticommuting with Z_q except the pivot and its partner (the only one
    // anticommuting with the pivot, overwritten below) is multiplied by the pivot.
    Scratch& sc = scratch();
    sc.mask.assign(xCol(q), xCol(q) + cw);
    sc.mask[dw] &= ~m;
    sc.mask[pw] &= ~m;
    sc.blocks.clear();
    for (std::size_t k0 = 0; k0 < cw; k0 += kBlock)
    {
        Word any = 0;
        for (std::size_t k = k0; k < k0 + kBlock; ++k) any |= sc.mask[k];
        if (any != 0) sc.blocks.push_back(k0);
    }

    // Read the pivot's Pauli string by letter, move it into the partner, and reset it to
    // (-1)^outcome Z_q. The products below read the pivot only through these lists.
    sc.xs.clear();
    sc.ys.clear();
    sc.zs.clear();
    for (Qubit c = 0; c < n; ++c)
    {
        Word* xc = xCol(c);
        Word* zc = zCol(c);
        const bool px = xc[pw] & m, pz = zc[pw] & m;
        if (px) (pz ? sc.ys : sc.xs).push_back(c);
        else if (pz) sc.zs.push_back(c);
        xc[dw] = (xc[dw] & ~m) | (px ? m : 0);
        zc[dw] = (zc[dw] & ~m) | (pz ? m : 0);
        xc[pw] &= ~m;
        zc[pw] &= ~m;
    }
    zCol(q)[pw] |= m;
    sign[dw] = (sign[dw] & ~m) | (pivotSign & m);
    sign[pw] = outcome ? sign[pw] | m : sign[pw] & ~m;

    // Row i := pivot * row i, 64 rows per word. Each column streams past once, block by block,
    // updating per-row phase counters mod 4 (lo, hi) for the blocks holding updated rows. Threads
    // split the columns and add their counters exactly, so any split gives the same sum. New
    // sign = r_i ^ r_p ^ (exponent bit 1); the low bit is zero because every updated row
    // commutes with the pivot.
    const Word* mask = sc.mask.data();
    const std::size_t* blocks = sc.blocks.data();
    const std::size_t numBlocks = sc.blocks.size();
    const std::size_t span = numBlocks * kBlock;
    const std::span<const Qubit> xs{sc.xs}, ys{sc.ys}, zs{sc.zs};
    sc.sumLo.assign(span, 0);
    sc.sumHi.assign(span, 0);
    Word* sumLo = sc.sumLo.data();
    Word* sumHi = sc.sumHi.data();
    [[maybe_unused]] const std::size_t work = (xs.size() + ys.size() + zs.size()) * span;

    // body(x, z, mask, lo, hi) updates one kBlock-word block of one column.
    auto sweep = [&](std::span<const Qubit> cols, Word* lo, Word* hi, auto body)
    {
        QPUTER_OMP(for schedule(static) nowait)
        for (std::size_t i = 0; i < cols.size(); ++i)
        {
            Word* xc = xCol(cols[i]);
            Word* zc = zCol(cols[i]);
            for (std::size_t b = 0; b < numBlocks; ++b)
                body(xc + blocks[b], zc + blocks[b], mask + blocks[b], lo + b * kBlock, hi + b * kBlock);
        }
    };

    QPUTER_OMP(parallel if(work >= kParallelWords))
    {
        Scratch& own = scratch();
        own.lo.assign(span, 0);
        own.hi.assign(span, 0);
        Word* lo = own.lo.data();
        Word* hi = own.hi.data();

        sweep(xs, lo, hi, [](Word* __restrict x, const Word* __restrict z, const Word* __restrict mk,
                             Word* __restrict l, Word* __restrict h)
        {
            for (std::size_t k = 0; k < kBlock; ++k) // pivot X: anticommutes with row Z, Y
            {
                const Word zz = z[k] & mk[k];
                h[k] ^= (l[k] ^ ~x[k]) & zz;
                l[k] ^= zz;
                x[k] ^= mk[k];
            }
        });
        sweep(ys, lo, hi, [](Word* __restrict x, Word* __restrict z, const Word* __restrict mk,
                             Word* __restrict l, Word* __restrict h)
        {
            for (std::size_t k = 0; k < kBlock; ++k) // pivot Y: anticommutes with row X, Z
            {
                const Word anti = (x[k] ^ z[k]) & mk[k];
                h[k] ^= (l[k] ^ x[k]) & anti;
                l[k] ^= anti;
                x[k] ^= mk[k];
                z[k] ^= mk[k];
            }
        });
        sweep(zs, lo, hi, [](const Word* __restrict x, Word* __restrict z, const Word* __restrict mk,
                             Word* __restrict l, Word* __restrict h)
        {
            for (std::size_t k = 0; k < kBlock; ++k) // pivot Z: anticommutes with row X, Y
            {
                const Word xx = x[k] & mk[k];
                h[k] ^= (l[k] ^ z[k]) & xx;
                l[k] ^= xx;
                z[k] ^= mk[k];
            }
        });

        QPUTER_OMP(critical)
        for (std::size_t k = 0; k < span; ++k) // (sumLo, sumHi) += (lo, hi) mod 4, bit-sliced
        {
            sumHi[k] ^= hi[k] ^ (sumLo[k] & lo[k]);
            sumLo[k] ^= lo[k];
        }
    }

    for (std::size_t b = 0; b < numBlocks; ++b)
        for (std::size_t k = 0; k < kBlock; ++k)
            sign[blocks[b] + k] ^= mask[blocks[b] + k] & (sumHi[b * kBlock + k] ^ pivotSign);
}

bool StabilizerState::productSign(const Word* generators) const
{
    // Product P = S_{i_1} ... in index order: step t picks up the exponent of S_{i_t} times the
    // product so far, whose bits at each qubit are the exclusive prefix XOR of the chosen rows'
    // bits there. Every step commutes, so the sign is the XOR of the chosen signs and bit 1 of
    // the total exponent.
    thread_local std::vector<std::size_t> words;
    words.clear();
    unsigned parity = 0;
    for (std::size_t k = 0; k < hw; ++k)
        if (generators[k] != 0)
        {
            words.push_back(k);
            parity ^= static_cast<unsigned>(std::popcount(sign[hw + k] & generators[k]));
        }
    if (words.empty()) return false;

    const std::size_t* ks = words.data();
    const std::size_t numWords = words.size();
    unsigned total = 0;
    QPUTER_OMP(parallel for schedule(static) reduction(+ : total) if(n * numWords >= kParallelWords))
    for (Qubit c = 0; c < n; ++c)
    {
        const Word* xc = xCol(c) + hw;
        const Word* zc = zCol(c) + hw;
        Word carryX = 0, carryZ = 0, lo = 0, hi = 0;
        for (std::size_t w = 0; w < numWords; ++w)
        {
            const std::size_t k = ks[w];
            const Word a = xc[k] & generators[k], b = zc[k] & generators[k];
            if ((a | b) == 0) continue;
            const Word pa = exclusivePrefixXor(a) ^ carryX;
            const Word pb = exclusivePrefixXor(b) ^ carryZ;
            accumulatePhase(a, b, pa, pb, lo, hi);
            carryX ^= broadcast((std::popcount(a) & 1) != 0);
            carryZ ^= broadcast((std::popcount(b) & 1) != 0);
        }
        total += laneTotal(lo, hi);
    }
    return ((parity & 1U) ^ ((total & 3U) >> 1)) != 0;
}

int StabilizerState::measure(Qubit target, int outcomeIfRandom)
{
    constexpr std::string_view ctx = "StabilizerState::measure";
    requireQubit(n, target, ctx);
    if (outcomeIfRandom != 0 && outcomeIfRandom != 1)
        throw std::invalid_argument(std::format("{}: outcomeIfRandom={} is not 0 or 1", ctx, outcomeIfRandom));

    if (const auto p = randomPivot(target))
    {
        collapse(target, *p, outcomeIfRandom == 1);
        return outcomeIfRandom;
    }
    return productSign(xCol(target)) ? 1 : 0;
}

OutcomeSupport StabilizerState::outcome_support(std::span<const Qubit> qubits) const
{
    constexpr std::string_view ctx = "StabilizerState::outcome_support";
    if (qubits.empty()) throw std::invalid_argument(std::format("{}: at least one qubit required", ctx));
    requireDistinct(n, qubits, ctx);

    const std::size_t k = qubits.size();
    OutcomeSupport sup;
    sup.width = k;
    sup.words = wordsFor(k);
    sup.offset.assign(sup.words, 0);

    // Measure a copy in order, fixing every random outcome to 0 and treating it as a free bit
    // (symbol). The symbol lives only in the sign of the +-Z row its measurement creates:
    // later collapses never multiply that row into others (it has no X part), so each certain
    // outcome is its sign with all symbols 0, XOR the symbols of the measurement rows in its
    // product. rows holds, per symbol, the outcome bits it flips.
    constexpr std::size_t kNoSymbol = ~std::size_t{0};
    StabilizerState t = *this;
    std::vector<std::size_t> symbol(n, kNoSymbol);
    std::vector<Word>& rows = sup.basis;
    for (std::size_t j = 0; j < k; ++j)
    {
        const Qubit q = qubits[j];
        if (const auto p = t.randomPivot(q))
        {
            t.collapse(q, *p, false);
            symbol[*p - n] = rows.size() / sup.words;
            rows.resize(rows.size() + sup.words, 0);
            rows[rows.size() - sup.words + j / kWordBits] = bitOf(j);
        }
        else
        {
            const Word* ids = t.xCol(q);
            if (t.productSign(ids)) sup.offset[j / kWordBits] |= bitOf(j);
            for (std::size_t w = 0; w < hw; ++w)
                for (Word rest = ids[w]; rest != 0; rest &= rest - 1)
                {
                    const std::size_t i = w * kWordBits + static_cast<std::size_t>(std::countr_zero(rest));
                    if (symbol[i] != kNoSymbol) rows[symbol[i] * sup.words + j / kWordBits] ^= bitOf(j);
                }
        }
    }

    // Reduced echelon form with descending leading bits, offset reduced alongside. Each symbol
    // row has its own random position set, so the rows are independent and all become pivots.
    const std::size_t d = sup.dimension();
    auto rowPtr = [&](std::size_t m) { return rows.data() + m * sup.words; };
    std::size_t rank = 0;
    for (std::size_t b = k; b-- > 0 && rank < d;)
    {
        const std::size_t wb = b / kWordBits;
        const Word mb = bitOf(b);
        std::size_t r = rank;
        while (r < d && !(rowPtr(r)[wb] & mb)) ++r;
        if (r == d) continue;
        std::swap_ranges(rowPtr(r), rowPtr(r) + sup.words, rowPtr(rank));
        const Word* pivot = rowPtr(rank);
        for (std::size_t o = 0; o < d; ++o)
            if (o != rank && (rowPtr(o)[wb] & mb))
                for (std::size_t c = 0; c < sup.words; ++c) rowPtr(o)[c] ^= pivot[c];
        if (sup.offset[wb] & mb)
            for (std::size_t c = 0; c < sup.words; ++c) sup.offset[c] ^= pivot[c];
        ++rank;
    }
    return sup;
}

int StabilizerState::expectation(std::string_view paulis, std::span<const Qubit> qubits) const
{
    constexpr std::string_view ctx = "StabilizerState::expectation";
    if (paulis.size() != qubits.size())
        throw std::invalid_argument(std::format("{}: {} Pauli letters for {} qubits", ctx, paulis.size(),
                                                qubits.size()));
    requireDistinct(n, qubits, ctx);

    // Bit i of `anti` is set where generator i anticommutes with P: P's X on qubit c meets the
    // generator's Z there, P's Z its X.
    std::vector<Word> anti(cw, 0);
    auto meet = [&](const Word* col)
    {
        for (std::size_t k = 0; k < cw; ++k) anti[k] ^= col[k];
    };
    for (std::size_t j = 0; j < paulis.size(); ++j)
    {
        const Qubit c = qubits[j];
        switch (paulis[j])
        {
            case 'I': break;
            case 'X': meet(zCol(c)); break;
            case 'Z': meet(xCol(c)); break;
            case 'Y': meet(xCol(c)); meet(zCol(c)); break;
            default:
                throw std::invalid_argument(std::format("{}: invalid Pauli letter '{}' (expected I, X, Y, Z)",
                                                        ctx, paulis[j]));
        }
    }

    // P anticommuting with a stabilizer S gives <P> = <P S> = -<S P> = -<P>, so 0.
    for (std::size_t k = hw; k < cw; ++k)
        if (anti[k] != 0) return 0;

    // Otherwise +-P lies in the (maximal) stabilizer group, as the product of the stabilizers
    // whose destabilizers anticommute with P; its sign is the eigenvalue.
    return productSign(anti.data()) ? -1 : 1;
}

std::vector<std::string> StabilizerState::stabilizers() const
{
    std::vector<std::string> out(n, std::string(n + 1, 'I'));
    for (std::size_t i = 0; i < n; ++i) out[i][0] = (sign[hw + i / kWordBits] & bitOf(i)) ? '-' : '+';
    for (Qubit q = 0; q < n; ++q)
    {
        const Word* xs = xCol(q) + hw;
        const Word* zs = zCol(q) + hw;
        for (std::size_t w = 0; w < hw; ++w)
            for (Word rest = xs[w] | zs[w]; rest != 0; rest &= rest - 1)
            {
                const auto b = static_cast<std::size_t>(std::countr_zero(rest));
                const bool x = (xs[w] >> b) & 1U, z = (zs[w] >> b) & 1U;
                out[w * kWordBits + b][q + 1] = x ? (z ? 'Y' : 'X') : 'Z';
            }
    }
    return out;
}

QuantumStateVector StabilizerState::to_state_vector() const
{
    if (n > kMaxQubits)
        throw std::length_error(std::format("StabilizerState::to_state_vector: {} qubits exceed the "
                                            "state-vector limit {}", n, kMaxQubits));

    // prod_i (I + S_i) / 2 = |psi><psi|, so projecting a basis state |b> leaves <psi|b> |psi>.
    // |b> must overlap |psi>: |0...0> does not for e.g. |1>. The smallest outcome of the
    // support does, and its amplitude <psi|b> <b|psi> > 0 fixes the global phase. Every step
    // adds unit multiples and halves, so the amplitudes stay exact until the final normalize.
    std::vector<Qubit> all(n);
    std::iota(all.begin(), all.end(), Qubit{0});
    const Index start = outcome_support(all).offset[0];

    // n <= kMaxQubits < 64: each half column is one word, and stabilizer i is bit i of word hw.
    std::vector<Index> xMask(n, 0), zMask(n, 0);
    for (Qubit q = 0; q < n; ++q)
        for (std::size_t i = 0; i < n; ++i)
        {
            xMask[i] |= ((xCol(q)[hw] >> i) & 1U) << q;
            zMask[i] |= ((zCol(q)[hw] >> i) & 1U) << q;
        }

    QuantumStateVector psi{n};
    detail::setBasis(psi, start);
    constexpr std::complex<double> iPow[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    for (std::size_t i = 0; i < n; ++i)
    {
        const bool negative = (sign[hw] >> i) & 1U;
        const std::complex<double> c = (negative ? -1.0 : 1.0) * iPow[std::popcount(xMask[i] & zMask[i]) % 4];
        projectOnto(psi, xMask[i], zMask[i], c);
    }
    psi.normalize();
    return psi;
}

} // namespace Qputer
