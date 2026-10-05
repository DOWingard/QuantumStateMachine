#pragma once

#include <QuantumGates.hpp>
#include <QuantumState.hpp>

#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>



namespace Qputer
{

// The tableau holds 2N generators of 2N bits, N^2 / 2 bytes: 2 GiB at the cap.
inline constexpr std::size_t kMaxStabilizerQubits = std::size_t{1} << 16;


// Z-basis outcomes of a stabilizer state over k qubits: uniform over the affine set
// offset ^ span(basis) in GF(2)^k, so each of the 2^d outcomes (d = dimension()) has
// probability 2^-d. Bit j of an outcome is the value of the j-th queried qubit; a bit vector
// stores bit j in word j / 64.
//
// The basis is in reduced echelon form with strictly descending leading bits, and offset is
// reduced against it. Hence offset is the smallest outcome, and the outcome of rank j in
// [0, 2^d) is offset ^ (XOR of row(m) over every m with bit d-1-m of j set).
struct OutcomeSupport
{
    std::size_t width = 0;              // k
    std::size_t words = 0;              // ceil(k / 64)
    std::vector<std::uint64_t> offset;  // `words` words
    std::vector<std::uint64_t> basis;   // dimension() rows of `words` words each

    std::size_t dimension() const { return words == 0 ? 0 : basis.size() / words; }
    std::span<const std::uint64_t> row(std::size_t m) const { return {basis.data() + m * words, words}; }

    // Throws std::invalid_argument unless outcome holds `words` words.
    bool contains(std::span<const std::uint64_t> outcome) const;
};


// N-qubit stabilizer state in Aaronson-Gottesman form (Phys. Rev. A 70, 052328, 2004):
// N destabilizer and N stabilizer generators, each a signed Pauli string
// (-1)^r P_0 (x) ... (x) P_{N-1} with P = I, X, Z, Y encoded as (x, z) = (0,0), (1,0), (0,1),
// (1,1). The destabilizers make every Z measurement O(N^2 / 64) without Gaussian elimination.
//
// The tableau is stored column-major: per qubit, the x and z bits of all 2N generators are
// packed 64 per word, so a Clifford gate is O(N / 64) word operations on one or two columns.
// The tableau carries no global phase. Qubit q is bit q of a basis index, as for
// QuantumStateVector. Invalid input throws before the state changes.
class StabilizerState
{
    public:
    explicit StabilizerState(std::size_t n); // |0...0>, 1 <= n <= kMaxStabilizerQubits

    std::size_t num_qubits() const { return n; }

    void set_basis(std::uint64_t index); // |index>; qubits 64 and above start in |0>


    // ---- Clifford gates, O(N / 64); same matrices as the QuantumGate functions of the same name ----
    void x(Qubit target);
    void y(Qubit target);
    void z(Qubit target);
    void h(Qubit target);
    void s(Qubit target);
    void sdg(Qubit target);
    void sx(Qubit target);
    void cnot(Qubit control, Qubit target);
    void cz(Qubit a, Qubit b);
    void swap(Qubit a, Qubit b);


    // ---- Measurement and readout ----

    // Collapses `target` in the Z basis and returns the outcome: the certain one if the state
    // determines it, otherwise outcomeIfRandom (0 or 1; both then have probability 1/2).
    // O(N^2 / 64).
    int measure(Qubit target, int outcomeIfRandom);

    // Joint Z-basis outcome distribution over distinct, non-empty `qubits`, leaving the state
    // unchanged. O(k N^2 / 64).
    OutcomeSupport outcome_support(std::span<const Qubit> qubits) const;

    // <psi|P|psi> in {-1, 0, +1} for P = paulis[0] on qubits[0] (x) paulis[1] on qubits[1] ...,
    // letters IXYZ. O(k N / 64) to test commutation, plus O(N^2 / 64) when P is in the group.
    int expectation(std::string_view paulis, std::span<const Qubit> qubits) const;

    // Entanglement entropy in bits of the reduced state of distinct, in-range `qubits`: rank over GF(2) of the
    // stabilizer generators restricted to those qubits, minus their count (Fattal et al., 2004). O(k N^2 / 64) worst case.
    std::size_t entanglement_entropy(std::span<const Qubit> qubits) const;

    // Stabilizer generators as signed strings: "+XZI" is +X on qubit 0, Z on qubit 1, I on qubit 2.
    std::vector<std::string> stabilizers() const;

    // Amplitudes of the state, with the global phase chosen so that the lowest-index nonzero
    // amplitude is real and positive. Requires N <= kMaxQubits; O(N 2^N).
    QuantumStateVector to_state_vector() const;


    private:
    using Word = std::uint64_t;

    // Allocates on cache-line boundaries, so every 8-word block of a column is one line.
    template <class T>
    struct LineAllocator
    {
        using value_type = T;
        static constexpr std::align_val_t kAlign{64};

        LineAllocator() = default;
        template <class U> LineAllocator(const LineAllocator<U>&) noexcept {}

        T* allocate(std::size_t count) { return static_cast<T*>(::operator new(count * sizeof(T), kAlign)); }
        void deallocate(T* p, std::size_t count) noexcept { ::operator delete(p, count * sizeof(T), kAlign); }
        template <class U> bool operator==(const LineAllocator<U>&) const noexcept { return true; }
    };

    // A column holds one bit per generator in two word-aligned halves: destabilizer i is bit i
    // of the first `hw` words, stabilizer i bit i of the next `hw`, zero-padded to whole cache
    // lines. Generator i and its destabilizer partner therefore share a bit position, and masks
    // over one half index the other.
    Word* xCol(Qubit q) { return bits.data() + q * stride; }
    Word* zCol(Qubit q) { return xCol(q) + cw; }
    const Word* xCol(Qubit q) const { return bits.data() + q * stride; }
    const Word* zCol(Qubit q) const { return xCol(q) + cw; }

    // update(x, z, r) rewrites 64 generators' qubit-q bits and signs per call, one word each.
    template <class F> void updateColumn(Qubit q, F update);
    template <class F> void updateColumns(Qubit a, Qubit b, F update);

    // A stabilizer generator anticommuting with Z_q, if any: then measuring q is random.
    std::optional<std::size_t> randomPivot(Qubit q) const;

    // Random-outcome update for measuring q with pivot generator p.
    void collapse(Qubit q, std::size_t p, bool outcome);

    // Sign of the product, in index order, of the stabilizers i with bit i of `generators` set
    // (`hw` words). For a certain Z_q outcome, Z_q is the product of the stabilizers whose
    // destabilizers anticommute with it: the first half of x column q.
    bool productSign(const Word* generators) const;

    std::size_t n;
    std::size_t hw;     // words per half column: ceil(n / 64)
    std::size_t cw;     // words per column: 2 hw rounded up to whole cache lines
    std::size_t stride; // words per qubit: 2 cw, plus one line when that is a multiple of 512 bytes
                        // so that columns do not all map to the same cache sets
    std::vector<Word, LineAllocator<Word>> bits; // per qubit q: [x column | z column | padding]
    std::vector<Word, LineAllocator<Word>> sign; // cw words, one sign bit per generator, laid out as a column
};


} // namespace Qputer
