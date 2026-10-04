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

    // Outcome index from amplitude index: bit j of the result is bit qubits[j] of i.
    inline Index gather(Index i, std::span<const Qubit> qubits) noexcept
    {
        Index o = 0;
        for (std::size_t j = 0; j < qubits.size(); ++j) o |= ((i >> qubits[j]) & 1U) << j;
        return o;
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

    // Per-block histograms beat outcome-major traversal while 2^k fits in L1/L2; past
    // that, the histogram copies dominate and each outcome is summed independently.
    constexpr Index kHistogramMaxOutcomes = Index{1} << 12;
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

        QPUTER_OMP(parallel for schedule(static) if(blocks > 1))
        for (Index b = 0; b < blocks; ++b)
        {
            double* local = h + b * outcomes;
            for (Index i = n * b / blocks; i < n * (b + 1) / blocks; ++i) local[gather(i, qubits)] += sq(a[i]);
        }

        for (Index o = 0; o < outcomes; ++o)
        {
            double acc = 0.0;
            for (Index b = 0; b < blocks; ++b) acc += h[b * outcomes + o];
            out[o] = acc;
        }
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
    return reduceBlocks<cd>(s.size(), [a, xMask, zMask](Index begin, Index end)
    {
        double re = 0.0, im = 0.0;
        for (Index i = begin; i < end; ++i)
        {
            // conj(b) * c with b = a[i ^ x], c = a[i]
            const cd b = a[i ^ xMask], c = a[i];
            const double sign = (std::popcount(i & zMask) & 1) ? -1.0 : 1.0;
            re += sign * (b.real() * c.real() + b.imag() * c.imag());
            im += sign * (b.real() * c.imag() - b.imag() * c.real());
        }
        return cd{re, im};
    });
}

void collapse(QuantumStateVector& s, Index mask, Index value, double scale)
{
    const Index n = s.size();
    cd* a = s.data();
    QPUTER_OMP(parallel for schedule(static) if(n >= kParallelThreshold))
    for (Index i = 0; i < n; ++i)
        a[i] = (i & mask) == value ? cd{scale * a[i].real(), scale * a[i].imag()} : cd{0.0, 0.0};
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

void copyAmplitudes(const QuantumStateVector& from, QuantumStateVector& to)
{
    const Index n = from.size();
    const cd* src = from.data();
    cd* dst = to.data();
    QPUTER_OMP(parallel for schedule(static) if(n >= kParallelThreshold))
    for (Index i = 0; i < n; ++i) dst[i] = src[i];
}

} // namespace Qputer::detail
