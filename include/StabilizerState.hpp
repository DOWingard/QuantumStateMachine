#pragma once

#include <QuantumGates.hpp>
#include <QuantumState.hpp>

#include <cstddef>
#include <cstdint>
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
// Clifford gates cost O(N). The tableau carries no global phase. Qubit q is bit q of a basis
// index, as for QuantumStateVector. Invalid input throws before the state changes.
class StabilizerState
{
    public:
    explicit StabilizerState(std::size_t n); // |0...0>, 1 <= n <= kMaxStabilizerQubits

    std::size_t num_qubits() const { return n; }

    void set_basis(std::uint64_t index); // |index>; qubits 64 and above start in |0>


    // ---- Clifford gates, O(N); same matrices as the QuantumGate functions of the same name ----
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
    // letters IXYZ. O(N^2 / 64).
    int expectation(std::string_view paulis, std::span<const Qubit> qubits) const;

    // Stabilizer generators as signed strings: "+XZI" is +X on qubit 0, Z on qubit 1, I on qubit 2.
    std::vector<std::string> stabilizers() const;

    // Amplitudes of the state, with the global phase chosen so that the lowest-index nonzero
    // amplitude is real and positive. Requires N <= kMaxQubits; O(N 2^N).
    QuantumStateVector to_state_vector() const;


    private:
    using Word = std::uint64_t;

    Word* xRow(std::size_t row) { return bits.data() + row * 2 * w; }
    Word* zRow(std::size_t row) { return xRow(row) + w; }
    const Word* xRow(std::size_t row) const { return bits.data() + row * 2 * w; }
    const Word* zRow(std::size_t row) const { return xRow(row) + w; }

    // update(x, z, r) rewrites the qubit-q bits and the sign of every generator.
    template <class F> void updateColumn(Qubit q, F update);
    template <class F> void updateColumns(Qubit a, Qubit b, F update);

    // Row `target` := row `source` * row `target`; the two must commute.
    void rowMul(std::size_t target, std::size_t source);

    // A stabilizer generator anticommuting with Z_q, if any: then measuring q is random.
    std::optional<std::size_t> randomPivot(Qubit q) const;

    // Random-outcome update for measuring q with pivot generator p.
    void collapse(Qubit q, std::size_t p, bool outcome);

    // Generators i whose destabilizer anticommutes with Z_q. For a certain outcome, Z_q is the
    // product of exactly those stabilizers, up to the sign productSign returns.
    std::vector<std::size_t> zComponents(Qubit q) const;
    bool productSign(std::span<const std::size_t> generators) const;

    std::size_t n;
    std::size_t w;          // words per x or z half: ceil(n / 64)
    std::vector<Word> bits; // 2n rows of [x words | z words]; rows [0, n) destabilizers, [n, 2n) stabilizers
    std::vector<std::uint8_t> sign; // r of each row
};


} // namespace Qputer
