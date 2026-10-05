#include "ReadoutKernels.hpp"
#include "Parallel.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <complex>
#include <vector>



namespace Qputer::detail
{

namespace
{

    using cd = std::complex<double>;

    // libstdc++'s std::norm(complex<double>) routes through hypot; this is plain re^2 + im^2.
    inline double sq(cd a) noexcept { return a.real() * a.real() + a.imag() * a.imag(); }

    // 256 blocks keep 12-64 threads within a few percent of balanced under static scheduling.
    constexpr Index kBlocks = 256;

    inline Index blockCount(Index n) noexcept { return n >= kParallelThreshold ? kBlocks : 1; }

    template <class T, class Partial>
    T reduceBlocks(Index n, Partial&& partial)
    {
        const Index blocks = blockCount(n);
        std::array<T, kBlocks> parts{};
        QPUTER_OMP(parallel for schedule(static) if(blocks > 1))
        for (Index b = 0; b < blocks; ++b) parts[b] = partial(n * b / blocks, n * (b + 1) / blocks);

        T total{};
        for (Index b = 0; b < blocks; ++b) total += parts[b];
        return total;
    }

    // Inverse of gather on the measured bits: bit j of o lands on bit qubits[j].
    inline Index deposit(Index o, std::span<const Qubit> qubits) noexcept
    {
        Index i = 0;
        for (std::size_t j = 0; j < qubits.size(); ++j) i |= ((o >> j) & 1U) << qubits[j];
        return i;
    }

    // Spreads p's bits around the zero positions described by ascending lowMasks
    // ((1 << q) - 1 per measured q), enumerating the unmeasured assignments.
    inline Index insertZeros(Index p, std::span<const Index> lowMasks) noexcept
    {
        for (const Index low : lowMasks) p = ((p & ~low) << 1) | (p & low);
        return p;
    }

    // Outcome index from amplitude index (bit j of the outcome is bit qubits[j] of i), one
    // table lookup per byte of i instead of one step per measured qubit.
    struct OutcomeGather
    {
        static constexpr std::size_t kBytes = (kMaxQubits + 7) / 8;
        std::array<std::array<Index, 256>, kBytes> table{};

        explicit OutcomeGather(std::span<const Qubit> qubits) noexcept
        {
            for (std::size_t j = 0; j < qubits.size(); ++j)
                for (Index v = 0; v < 256; ++v)
                    if ((v >> (qubits[j] % 8)) & 1U) table[qubits[j] / 8][v] |= Index{1} << j;
        }

        // Outcome bits contributed by every byte of i but the lowest.
        Index high(Index i) const noexcept
        {
            Index o = 0;
            for (std::size_t b = 1; b < kBytes; ++b) o |= table[b][(i >> (8 * b)) & 0xFF];
            return o;
        }
    };

    // Per-block histograms stay cache-resident up to 2^16 outcomes (512 KiB); past that each
    // outcome is summed independently.
    constexpr Index kHistogramMaxOutcomes = Index{1} << 16;
    constexpr Index kHistogramBudget = Index{1} << 20; // doubles across all block histograms

} // namespace


void squaredMagnitudes(const QuantumStateVector& s, double* out)
{
    const Index n = s.size();
    const cd* a = s.data();
    QPUTER_OMP(parallel for schedule(static) if(n >= kParallelThreshold))
    for (Index i = 0; i < n; ++i) out[i] = sq(a[i]);
}

void marginalWeights(const QuantumStateVector& s, std::span<const Qubit> qubits, double* out)
{
    const Index n = s.size();
    const Index outcomes = Index{1} << qubits.size();
    const cd* a = s.data();

    // Whole register in index order: the marginal is the full distribution.
    bool identity = qubits.size() == s.num_qubits();
    for (std::size_t j = 0; identity && j < qubits.size(); ++j) identity = qubits[j] == j;
    if (identity)
    {
        squaredMagnitudes(s, out);
        return;
    }

    if (outcomes <= kHistogramMaxOutcomes)
    {
        const Index blocks = std::min(blockCount(n), std::max<Index>(1, kHistogramBudget / outcomes));
        std::vector<double> hist(blocks * outcomes, 0.0);
        double* h = hist.data();
        const OutcomeGather gather{qubits};
        const Index* low = gather.table[0].data();

        QPUTER_OMP(parallel for schedule(static) if(blocks > 1))
        for (Index b = 0; b < blocks; ++b)
        {
            double* local = h + b * outcomes;
            const Index end = n * (b + 1) / blocks;
            for (Index i = n * b / blocks; i < end;)
            {
                // Within one aligned 256-amplitude chunk only the lowest byte of i varies.
                const Index stop = std::min(end, (i | 0xFF) + 1);
                double* chunk = local + gather.high(i);
                for (; i < stop; ++i) chunk[low[i & 0xFF]] += sq(a[i]);
            }
        }

        // Block order per outcome, as the reductions; histogram-major so the sweep is sequential.
        std::fill_n(out, outcomes, 0.0);
        for (Index b = 0; b < blocks; ++b)
            for (Index o = 0; o < outcomes; ++o) out[o] += h[b * outcomes + o];
        return;
    }

    std::vector<Index> lowMasks;
    for (const Qubit q : qubits) lowMasks.push_back((Index{1} << q) - 1);
    std::ranges::sort(lowMasks);
    const Index rest = n >> qubits.size();
    const std::span<const Index> masks{lowMasks};

    QPUTER_OMP(parallel for schedule(static) if(n >= kParallelThreshold))
    for (Index o = 0; o < outcomes; ++o)
    {
        const Index base = deposit(o, qubits);
        double acc = 0.0;
        for (Index p = 0; p < rest; ++p) acc += sq(a[insertZeros(p, masks) | base]);
        out[o] = acc;
    }
}

namespace
{
    struct WeightPair
    {
        double zero = 0.0, one = 0.0;
        WeightPair& operator+=(const WeightPair& o) noexcept
        {
            zero += o.zero;
            one += o.one;
            return *this;
        }
    };
} // namespace

QubitWeights qubitWeights(const QuantumStateVector& s, Qubit q)
{
    const cd* a = s.data();
    const WeightPair w = reduceBlocks<WeightPair>(s.size(), [a, q](Index begin, Index end)
    {
        WeightPair acc;
        for (Index i = begin; i < end; ++i)
        {
            const double v = sq(a[i]);
            const bool set = (i >> q) & 1U;
            acc.zero += set ? 0.0 : v;
            acc.one += set ? v : 0.0;
        }
        return acc;
    });
    return {w.zero, w.one};
}

cd pauliSum(const QuantumStateVector& s, Index xMask, Index zMask)
{
    const cd* a = s.data();
    auto sign = [zMask](Index i) { return (std::popcount(i & zMask) & 1) ? -1.0 : 1.0; };

    if (xMask == 0)
        return reduceBlocks<cd>(s.size(), [a, sign](Index begin, Index end)
        {
            double re = 0.0;
            for (Index i = begin; i < end; ++i) re += sign(i) * sq(a[i]);
            return cd{re, 0.0};
        });

    // Terms i and i ^ xMask pair up: with w = conj(a[k ^ x]) a[k] and sigma = (-1)^|x & z|, the
    // pair contributes s(k) (w + sigma conj(w)), i.e. 2 s(k) Re w or 2i s(k) Im w. Enumerating
    // k with xMask's top bit clear reads every amplitude once.
    const Index top = std::bit_floor(xMask) - 1;
    const bool symmetric = (std::popcount(xMask & zMask) & 1) == 0;
    const double half = reduceBlocks<double>(s.size() / 2, [=](Index begin, Index end)
    {
        double acc = 0.0;
        for (Index p = begin; p < end; ++p)
        {
            const Index k = ((p & ~top) << 1) | (p & top);
            const cd b = a[k ^ xMask], c = a[k];
            const double w = symmetric ? b.real() * c.real() + b.imag() * c.imag()
                                       : b.real() * c.imag() - b.imag() * c.real();
            acc += sign(k) * w;
        }
        return acc;
    });
    return symmetric ? cd{2.0 * half, 0.0} : cd{0.0, 2.0 * half};
}

namespace
{
    // x conj(y), as plain real arithmetic like sq().
    inline cd mulConj(cd x, cd y) noexcept
    {
        return {x.real() * y.real() + x.imag() * y.imag(), x.imag() * y.real() - x.real() * y.imag()};
    }

    // Ascending (1 << q) - 1 for insertZeros, which then enumerates the other qubits' assignments.
    std::vector<Index> sortedLowMasks(std::span<const Qubit> qubits)
    {
        std::vector<Index> lowMasks;
        for (const Qubit q : qubits) lowMasks.push_back((Index{1} << q) - 1);
        std::ranges::sort(lowMasks);
        return lowMasks;
    }

    // One Kraus channel's branch weights over a range of subspaces: sum_r |(K_k v)_r|^2 per
    // operator. Each operator takes its own pass (the range stays cache-resident) with four
    // interleaved partial sums, p mod 4, which break the dependency chain of a single sum and
    // are combined in a fixed order. D is the operator dimension when fixed at compile time,
    // 0 otherwise.
    struct KrausBlock
    {
        const cd* a;
        std::span<const Index> masks;
        const Index* offsets;
        const cd* mats;
        Index dim, count;

        template <Index D>
        void sum(Index begin, Index end, double* out) const
        {
            const Index d = D != 0 ? D : dim;
            for (Index k = 0; k < count; ++k)
            {
                const cd* u = mats + k * d * d;
                std::array<double, 4> lane{};
                for (Index p = begin; p < end; ++p)
                {
                    const Index base = insertZeros(p, masks);
                    double w = 0.0;
                    for (Index r = 0; r < d; ++r)
                    {
                        double re = 0.0, im = 0.0;
                        for (Index c = 0; c < d; ++c)
                        {
                            const cd x = a[base | offsets[c]], m = u[r * d + c];
                            re += m.real() * x.real() - m.imag() * x.imag();
                            im += m.real() * x.imag() + m.imag() * x.real();
                        }
                        w += re * re + im * im;
                    }
                    lane[p % 4] += w;
                }
                out[k] = (lane[0] + lane[1]) + (lane[2] + lane[3]);
            }
        }
    };

    // Up to 32 x 32 entries a density matrix is accumulated per block of e and the blocks are
    // summed in order, like the reductions. Larger ones have enough columns to split instead.
    constexpr Index kDensityBlockMaxDim = 32;

    // Amplitudes gathered per pass of the column split (1 MiB), so every column re-reads them
    // from cache instead of striding through the state once per column.
    constexpr Index kDensityChunk = Index{1} << 16;
} // namespace

void krausWeights(const QuantumStateVector& s, std::span<const DenseGate> ops, double* out)
{
    const std::vector<Index>& offsets = ops[0].offsets;
    const Index dim = offsets.size();
    const Index count = ops.size();
    const Index subspaces = s.size() / dim;
    const std::vector<Index> lowMasks = sortedLowMasks(ops[0].targets);
    std::vector<cd> mats; // every K_k row-major, back to back
    for (const DenseGate& op : ops) mats.insert(mats.end(), op.rowMajor.begin(), op.rowMajor.end());
    const KrausBlock block{s.data(), lowMasks, offsets.data(), mats.data(), dim, count};

    // Fixed blocks of subspaces, one partial per operator each, combined in block order.
    const Index blocks = std::min(blockCount(s.size()), subspaces);
    std::vector<double> parts(blocks * count);
    QPUTER_OMP(parallel for schedule(static) if(blocks > 1))
    for (Index b = 0; b < blocks; ++b)
    {
        double* acc = parts.data() + b * count;
        const Index begin = subspaces * b / blocks, end = subspaces * (b + 1) / blocks;
        switch (dim)
        {
            case 2:  block.sum<2>(begin, end, acc); break;
            case 4:  block.sum<4>(begin, end, acc); break;
            default: block.sum<0>(begin, end, acc); break;
        }
    }

    std::fill_n(out, count, 0.0);
    for (Index b = 0; b < blocks; ++b)
        for (Index k = 0; k < count; ++k) out[k] += parts[b * count + k];
}

void reducedDensityMatrix(const QuantumStateVector& s, std::span<const Qubit> qubits, cd* out)
{
    const std::size_t k = qubits.size();
    const Index dim = Index{1} << k;
    const Index rest = s.size() >> k;
    const std::vector<Index> lowMasks = sortedLowMasks(qubits);
    const std::span<const Index> masks{lowMasks};
    const cd* a = s.data();

    // offset[r] = amplitude bits of local index r, qubits[0] <-> MSB of r.
    std::vector<Index> offset(dim, 0);
    for (Index r = 0; r < dim; ++r)
        for (std::size_t j = 0; j < k; ++j)
            if ((r >> (k - 1 - j)) & 1U) offset[r] |= Index{1} << qubits[j];
    const Index* off = offset.data();

    // Only the lower triangle (r >= c, contiguous in column c) is summed. Small matrices sum
    // fixed blocks of e and combine them in block order; larger ones give each entry one sum
    // in ascending e. Either way the order depends on N and k alone, never on the threads.
    std::fill_n(out, dim * dim, cd{0.0, 0.0});
    if (dim <= kDensityBlockMaxDim)
    {
        const Index blocks = std::min(blockCount(s.size()), rest);
        const Index entries = dim * dim;
        std::vector<cd> parts(blocks * entries);
        QPUTER_OMP(parallel for schedule(static) if(blocks > 1))
        for (Index b = 0; b < blocks; ++b)
        {
            std::array<cd, kDensityBlockMaxDim> v{};
            std::array<cd, kDensityBlockMaxDim * kDensityBlockMaxDim> acc{};
            for (Index e = rest * b / blocks; e < rest * (b + 1) / blocks; ++e)
            {
                const Index base = insertZeros(e, masks);
                for (Index r = 0; r < dim; ++r) v[r] = a[base | off[r]];
                for (Index c = 0; c < dim; ++c)
                    for (Index r = c; r < dim; ++r) acc[c * dim + r] += mulConj(v[r], v[c]);
            }
            std::copy_n(acc.begin(), entries, parts.begin() + static_cast<std::ptrdiff_t>(b * entries));
        }
        for (Index b = 0; b < blocks; ++b)
            for (Index c = 0; c < dim; ++c)
                for (Index r = c; r < dim; ++r) out[c * dim + r] += parts[b * entries + c * dim + r];
    }
    else
    {
        // Chunks of e in order: gather each chunk e-major, then split the columns, column c
        // owning entries (r >= c, c). Worksharing barriers keep the gather and the sums apart.
        const Index chunk = std::min(rest, std::max<Index>(1, kDensityChunk / dim));
        std::vector<cd> gathered(chunk * dim);
        cd* g = gathered.data();
        QPUTER_OMP(parallel if(s.size() >= kParallelThreshold))
        for (Index e0 = 0; e0 < rest; e0 += chunk)
        {
            const Index len = std::min(chunk, rest - e0);
            QPUTER_OMP(for schedule(static))
            for (Index el = 0; el < len; ++el)
            {
                const Index base = insertZeros(e0 + el, masks);
                for (Index r = 0; r < dim; ++r) g[el * dim + r] = a[base | off[r]];
            }
            QPUTER_OMP(for schedule(dynamic, 1)) // column c holds dim - c entries
            for (Index c = 0; c < dim; ++c)
            {
                cd* col = out + c * dim;
                for (Index el = 0; el < len; ++el)
                {
                    const cd* row = g + el * dim;
                    const cd vc = row[c];
                    for (Index r = c; r < dim; ++r) col[r] += mulConj(row[r], vc);
                }
            }
        }
    }

    for (Index c = 0; c < dim; ++c)
    {
        out[c * dim + c] = cd{out[c * dim + c].real(), 0.0};
        for (Index r = c + 1; r < dim; ++r) out[r * dim + c] = std::conj(out[c * dim + r]);
    }
}

void collapse(QuantumStateVector& s, Index mask, Index value, double scale)
{
    const Index n = s.size();
    cd* a = s.data();
    QPUTER_OMP(parallel for schedule(static) if(n >= kParallelThreshold))
    for (Index i = 0; i < n; ++i)
        a[i] = (i & mask) == value ? cd{scale * a[i].real(), scale * a[i].imag()} : cd{0.0, 0.0};
}

void scaleAmplitudes(QuantumStateVector& s, double scale)
{
    const Index n = s.size();
    cd* a = s.data();
    QPUTER_OMP(parallel for schedule(static) if(n >= kParallelThreshold))
    for (Index i = 0; i < n; ++i) a[i] = cd{scale * a[i].real(), scale * a[i].imag()};
}

Index findCumulative(const QuantumStateVector& s, double target)
{
    const Index n = s.size();
    const cd* a = s.data();
    const Index blocks = blockCount(n);

    // Pass 1 (parallel): block weights. Pass 2 (serial): locate the block, then scan it.
    std::array<double, kBlocks> parts{};
    QPUTER_OMP(parallel for schedule(static) if(blocks > 1))
    for (Index b = 0; b < blocks; ++b)
    {
        double acc = 0.0;
        for (Index i = n * b / blocks; i < n * (b + 1) / blocks; ++i) acc += sq(a[i]);
        parts[b] = acc;
    }

    double prefix = 0.0;
    for (Index b = 0; b < blocks; ++b)
    {
        if (prefix + parts[b] <= target)
        {
            prefix += parts[b];
            continue;
        }
        Index lastNonzero = n;
        for (Index i = n * b / blocks; i < n * (b + 1) / blocks; ++i)
        {
            const double v = sq(a[i]);
            if (v == 0.0) continue;
            prefix += v;
            lastNonzero = i;
            if (prefix > target) return i;
        }
        if (lastNonzero != n) return lastNonzero; // in-block summation rounded below target
    }

    for (Index i = n; i-- > 0;)
        if (sq(a[i]) != 0.0) return i;
    return n - 1; // all-zero state; callers reject that before sampling
}

void setBasis(QuantumStateVector& s, Index index)
{
    const Index n = s.size();
    cd* a = s.data();
    QPUTER_OMP(parallel for schedule(static) if(n >= kParallelThreshold))
    for (Index i = 0; i < n; ++i) a[i] = cd{0.0, 0.0};
    a[index] = cd{1.0, 0.0};
}

void copyAmplitudes(const QuantumStateVector& from, QuantumStateVector& to) { copyAmplitudes(from.data(), to); }

void copyAmplitudes(const cd* from, QuantumStateVector& to)
{
    const Index n = to.size();
    cd* dst = to.data();
    QPUTER_OMP(parallel for schedule(static) if(n >= kParallelThreshold))
    for (Index i = 0; i < n; ++i) dst[i] = from[i];
}

} // namespace Qputer::detail
