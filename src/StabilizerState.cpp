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



namespace Qputer
{

namespace
{

    using Word = std::uint64_t;
    using detail::Index;

    constexpr std::size_t kWordBits = 64;

    constexpr std::size_t wordsFor(std::size_t bits) noexcept { return (bits + kWordBits - 1) / kWordBits; }
    constexpr Word bitOf(std::size_t j) noexcept { return Word{1} << (j % kWordBits); }

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

    // Sum over one word of the i-exponent that P1 * P2 picks up per qubit: +1 for XY, YZ, ZX
    // and -1 for XZ, YX, ZY (the g function of Aaronson-Gottesman).
    int productPhase(Word x1, Word z1, Word x2, Word z2) noexcept
    {
        const Word pos = (x1 & ~z1 & x2 & z2) | (x1 & z1 & ~x2 & z2) | (~x1 & z1 & x2 & ~z2);
        const Word neg = (x1 & ~z1 & ~x2 & z2) | (x1 & z1 & x2 & ~z2) | (~x1 & z1 & x2 & z2);
        return std::popcount(pos) - std::popcount(neg);
    }

    // (hx, hz, hr) := (ix, iz, ir) * (hx, hz, hr). Commuting operands give a Hermitian product,
    // i.e. an even total i-exponent, whose half is the new sign.
    void mulInto(Word* hx, Word* hz, std::uint8_t& hr, const Word* ix, const Word* iz, std::uint8_t ir,
                 std::size_t w) noexcept
    {
        int exponent = 2 * (hr + ir);
        for (std::size_t k = 0; k < w; ++k)
        {
            exponent += productPhase(ix[k], iz[k], hx[k], hz[k]);
            hx[k] ^= ix[k];
            hz[k] ^= iz[k];
        }
        hr = static_cast<std::uint8_t>((exponent & 3) >> 1);
    }

    bool anticommute(const Word* ax, const Word* az, const Word* bx, const Word* bz, std::size_t w) noexcept
    {
        Word acc = 0;
        for (std::size_t k = 0; k < w; ++k) acc ^= (ax[k] & bz[k]) ^ (az[k] & bx[k]);
        return std::popcount(acc) & 1;
    }

    // Highest set bit of a bit vector, which must be nonzero.
    std::size_t leadingBit(std::span<const Word> v) noexcept
    {
        std::size_t k = v.size();
        while (v[--k] == 0) {}
        return k * kWordBits + kWordBits - 1 - static_cast<std::size_t>(std::countl_zero(v[k]));
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

StabilizerState::StabilizerState(std::size_t numQubits) : n(numQubits), w(wordsFor(numQubits))
{
    if (n == 0 || n > kMaxStabilizerQubits)
        throw std::length_error(std::format("StabilizerState: num_qubits={} outside [1, {}]", n,
                                            kMaxStabilizerQubits));
    bits.resize(2 * n * 2 * w);
    sign.resize(2 * n);
    set_basis(0);
}

void StabilizerState::set_basis(std::uint64_t index)
{
    std::ranges::fill(bits, Word{0});
    std::ranges::fill(sign, std::uint8_t{0});
    for (Qubit q = 0; q < n; ++q)
    {
        xRow(q)[q / kWordBits] = bitOf(q);     // destabilizer q: X_q
        zRow(n + q)[q / kWordBits] = bitOf(q); // stabilizer q: (-1)^{b_q} Z_q
        sign[n + q] = static_cast<std::uint8_t>(q < kWordBits ? (index >> q) & 1U : 0U);
    }
}



// ---- Gates: conjugation rules P -> U P U^dagger on each generator ----

template <class F>
void StabilizerState::updateColumn(Qubit q, F update)
{
    const std::size_t k = q / kWordBits;
    const Word m = bitOf(q);
    const std::size_t rows = 2 * n;
    QPUTER_OMP(parallel for schedule(static) if(rows >= detail::kParallelThreshold))
    for (std::size_t i = 0; i < rows; ++i)
    {
        Word& xw = xRow(i)[k];
        Word& zw = zRow(i)[k];
        bool x = xw & m, z = zw & m, r = sign[i];
        update(x, z, r);
        xw = x ? xw | m : xw & ~m;
        zw = z ? zw | m : zw & ~m;
        sign[i] = r;
    }
}

template <class F>
void StabilizerState::updateColumns(Qubit a, Qubit b, F update)
{
    const std::size_t ka = a / kWordBits, kb = b / kWordBits;
    const Word ma = bitOf(a), mb = bitOf(b);
    const std::size_t rows = 2 * n;
    QPUTER_OMP(parallel for schedule(static) if(rows >= detail::kParallelThreshold))
    for (std::size_t i = 0; i < rows; ++i)
    {
        Word* xr = xRow(i);
        Word* zr = zRow(i);
        bool xa = xr[ka] & ma, za = zr[ka] & ma, xb = xr[kb] & mb, zb = zr[kb] & mb, r = sign[i];
        update(xa, za, xb, zb, r);
        xr[ka] = xa ? xr[ka] | ma : xr[ka] & ~ma;
        zr[ka] = za ? zr[ka] | ma : zr[ka] & ~ma;
        xr[kb] = xb ? xr[kb] | mb : xr[kb] & ~mb;
        zr[kb] = zb ? zr[kb] | mb : zr[kb] & ~mb;
        sign[i] = r;
    }
}

void StabilizerState::x(Qubit target)
{
    requireQubit(n, target, "StabilizerState::x");
    updateColumn(target, [](bool&, bool& z, bool& r) { r = r != z; }); // Z, Y -> -Z, -Y
}

void StabilizerState::y(Qubit target)
{
    requireQubit(n, target, "StabilizerState::y");
    updateColumn(target, [](bool& x, bool& z, bool& r) { r = r != (x != z); }); // X, Z -> -X, -Z
}

void StabilizerState::z(Qubit target)
{
    requireQubit(n, target, "StabilizerState::z");
    updateColumn(target, [](bool& x, bool&, bool& r) { r = r != x; }); // X, Y -> -X, -Y
}

void StabilizerState::h(Qubit target)
{
    requireQubit(n, target, "StabilizerState::h");
    updateColumn(target, [](bool& x, bool& z, bool& r) // X <-> Z, Y -> -Y
    {
        r = r != (x && z);
        std::swap(x, z);
    });
}

void StabilizerState::s(Qubit target)
{
    requireQubit(n, target, "StabilizerState::s");
    updateColumn(target, [](bool& x, bool& z, bool& r) // X -> Y, Y -> -X
    {
        r = r != (x && z);
        z = z != x;
    });
}

void StabilizerState::sdg(Qubit target)
{
    requireQubit(n, target, "StabilizerState::sdg");
    updateColumn(target, [](bool& x, bool& z, bool& r) // X -> -Y, Y -> X
    {
        r = r != (x && !z);
        z = z != x;
    });
}

void StabilizerState::sx(Qubit target)
{
    requireQubit(n, target, "StabilizerState::sx");
    updateColumn(target, [](bool& x, bool& z, bool& r) // sqrt(X) = H S H: Z -> -Y, Y -> Z
    {
        r = r != (z && !x);
        x = x != z;
    });
}

void StabilizerState::cnot(Qubit control, Qubit target)
{
    requirePair(n, control, target, "StabilizerState::cnot");
    updateColumns(control, target, [](bool& xc, bool& zc, bool& xt, bool& zt, bool& r)
    {
        r = r != (xc && zt && xt == zc);
        xt = xt != xc;
        zc = zc != zt;
    });
}

void StabilizerState::cz(Qubit a, Qubit b)
{
    requirePair(n, a, b, "StabilizerState::cz");
    updateColumns(a, b, [](bool& xa, bool& za, bool& xb, bool& zb, bool& r) // X_a -> X_a Z_b, X_b -> Z_a X_b
    {
        r = r != (xa && xb && za != zb);
        za = za != xb;
        zb = zb != xa;
    });
}

void StabilizerState::swap(Qubit a, Qubit b)
{
    requirePair(n, a, b, "StabilizerState::swap");
    updateColumns(a, b, [](bool& xa, bool& za, bool& xb, bool& zb, bool&)
    {
        std::swap(xa, xb);
        std::swap(za, zb);
    });
}



// ---- Measurement ----

void StabilizerState::rowMul(std::size_t target, std::size_t source)
{
    mulInto(xRow(target), zRow(target), sign[target], xRow(source), zRow(source), sign[source], w);
}

std::optional<std::size_t> StabilizerState::randomPivot(Qubit q) const
{
    const std::size_t k = q / kWordBits;
    const Word m = bitOf(q);
    for (std::size_t p = n; p < 2 * n; ++p)
        if (xRow(p)[k] & m) return p;
    return std::nullopt;
}

void StabilizerState::collapse(Qubit q, std::size_t p, bool outcome)
{
    const std::size_t k = q / kWordBits;
    const Word m = bitOf(q);
    const std::size_t rows = 2 * n;
    const std::size_t partner = p - n; // the only generator anticommuting with row p; overwritten below

    // Every other generator anticommuting with Z_q is multiplied by row p so it commutes;
    // rows only read row p, so they update independently.
    QPUTER_OMP(parallel for schedule(static) if(rows * w >= detail::kParallelThreshold))
    for (std::size_t i = 0; i < rows; ++i)
        if (i != p && i != partner && (xRow(i)[k] & m)) rowMul(i, p);

    // The old generator becomes the destabilizer of the new one, (-1)^outcome Z_q.
    std::copy_n(xRow(p), 2 * w, xRow(partner));
    sign[partner] = sign[p];
    std::fill_n(xRow(p), 2 * w, Word{0});
    zRow(p)[k] = m;
    sign[p] = outcome;
}

std::vector<std::size_t> StabilizerState::zComponents(Qubit q) const
{
    const std::size_t k = q / kWordBits;
    const Word m = bitOf(q);
    std::vector<std::size_t> ids;
    for (std::size_t i = 0; i < n; ++i)
        if (xRow(i)[k] & m) ids.push_back(i);
    return ids;
}

bool StabilizerState::productSign(std::span<const std::size_t> generators) const
{
    std::vector<Word> px(w), pz(w);
    std::uint8_t r = 0;
    for (const std::size_t i : generators) mulInto(px.data(), pz.data(), r, xRow(n + i), zRow(n + i), sign[n + i], w);
    return r != 0;
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
    return productSign(zComponents(target)) ? 1 : 0;
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
            const std::vector<std::size_t> ids = t.zComponents(q);
            if (t.productSign(ids)) sup.offset[j / kWordBits] |= bitOf(j);
            for (const std::size_t i : ids)
                if (symbol[i] != kNoSymbol) rows[symbol[i] * sup.words + j / kWordBits] ^= bitOf(j);
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

    std::vector<Word> px(w), pz(w);
    for (std::size_t j = 0; j < paulis.size(); ++j)
    {
        const std::size_t k = qubits[j] / kWordBits;
        const Word m = bitOf(qubits[j]);
        switch (paulis[j])
        {
            case 'I': break;
            case 'X': px[k] |= m; break;
            case 'Z': pz[k] |= m; break;
            case 'Y': px[k] |= m; pz[k] |= m; break;
            default:
                throw std::invalid_argument(std::format("{}: invalid Pauli letter '{}' (expected I, X, Y, Z)",
                                                        ctx, paulis[j]));
        }
    }

    // P anticommuting with a stabilizer S gives <P> = <P S> = -<S P> = -<P>, so 0.
    for (std::size_t i = n; i < 2 * n; ++i)
        if (anticommute(xRow(i), zRow(i), px.data(), pz.data(), w)) return 0;

    // Otherwise +-P lies in the (maximal) stabilizer group, as the product of the stabilizers
    // whose destabilizers anticommute with P; its sign is the eigenvalue.
    std::vector<std::size_t> ids;
    for (std::size_t i = 0; i < n; ++i)
        if (anticommute(xRow(i), zRow(i), px.data(), pz.data(), w)) ids.push_back(i);
    return productSign(ids) ? -1 : 1;
}

std::vector<std::string> StabilizerState::stabilizers() const
{
    std::vector<std::string> out;
    out.reserve(n);
    for (std::size_t i = n; i < 2 * n; ++i)
    {
        std::string g(n + 1, 'I');
        g[0] = sign[i] ? '-' : '+';
        for (Qubit q = 0; q < n; ++q)
        {
            const bool x = xRow(i)[q / kWordBits] & bitOf(q), z = zRow(i)[q / kWordBits] & bitOf(q);
            if (x || z) g[q + 1] = x ? (z ? 'Y' : 'X') : 'Z';
        }
        out.push_back(std::move(g));
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

    QuantumStateVector psi{n};
    detail::setBasis(psi, start);
    constexpr std::complex<double> iPow[4] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    for (std::size_t i = n; i < 2 * n; ++i)
    {
        const Index xMask = xRow(i)[0], zMask = zRow(i)[0];
        const std::complex<double> c = (sign[i] ? -1.0 : 1.0) * iPow[std::popcount(xMask & zMask) % 4];
        projectOnto(psi, xMask, zMask, c);
    }
    psi.normalize();
    return psi;
}

} // namespace Qputer
