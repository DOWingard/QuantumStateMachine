#pragma once

#include "Ast.hpp"
#include "Source.hpp"

#include <QuantumGates.hpp>

#include <complex>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include <Eigen/Core>



namespace Noether
{

using Qputer::Qubit;
using Qputer::QubitList;
using cd = std::complex<double>;

class Scope;


// ---- Exact numbers ----

// n/d in lowest terms with d > 0. Arithmetic returns nullopt on int64 overflow; callers then fall
// back to floating point, which drops exactness but never the value.
struct Rational
{
    std::int64_t n = 0;
    std::int64_t d = 1;

    static std::optional<Rational> make(std::int64_t num, std::int64_t den);
    double value() const { return static_cast<double>(n) / static_cast<double>(d); }
    bool isZero() const { return n == 0; }
    bool isInt() const { return d == 1; }
    friend bool operator==(const Rational&, const Rational&) = default;
};

std::optional<Rational> add(Rational a, Rational b);
std::optional<Rational> mul(Rational a, Rational b);
std::optional<Rational> div(Rational a, Rational b);
std::optional<Rational> parseDecimal(std::string_view text); // "0.3125" -> 5/16, "1e-3" -> 1/1000


// A real number kept exact as p·π + r (p, r rational) for as long as possible, so angles such as
// π/4 or 3π/2 stay exact for Clifford classification and T-counting. Once an operation has no
// exact result (π², sin, overflow) the value continues as a plain double.
class Real
{
    public:
    Real() = default;
    static Real integer(std::int64_t v) { return Real(Rational{v, 1}, Rational{0, 1}); }
    static Real rational(Rational r) { return Real(r, Rational{0, 1}); }
    static Real pi(Rational coef) { return Real(Rational{0, 1}, coef); }
    static Real approx(double v);

    bool exact() const { return isExact; }
    double value() const;
    const Rational& rat() const { return r; } // exact only
    const Rational& piCoef() const { return p; }
    bool isZero() const { return isExact ? r.isZero() && p.isZero() : approxValue == 0.0; }
    std::optional<std::int64_t> asInt() const; // exact integers only
    bool isExactInt() const { return asInt().has_value(); }

    friend Real operator+(const Real& a, const Real& b);
    friend Real operator-(const Real& a, const Real& b);
    friend Real operator*(const Real& a, const Real& b);
    friend Real operator/(const Real& a, const Real& b); // caller checks b != 0
    Real operator-() const;

    // Text form for messages and IR: 3π/4, -1/2, 0.70710678…
    std::string str() const;

    private:
    Real(Rational rr, Rational pp) : r(rr), p(pp) {}
    Rational r{0, 1};
    Rational p{0, 1};
    bool isExact = true;
    double approxValue = 0.0;
};


// c + Σ a_k θ_k: a real quantity that depends affinely on the program's params. Angles keep this
// form in the IR, so programs compile once and rebind, and parameter-shift gradients stay exact.
struct Affine
{
    Real c;
    std::vector<std::pair<std::uint32_t, Real>> terms; // param index -> coefficient, sorted, nonzero

    Affine() = default;
    Affine(Real constant) : c(std::move(constant)) {}
    static Affine param(std::uint32_t index);

    bool isConst() const { return terms.empty(); }
    bool isZero() const { return terms.empty() && c.isZero(); }
    double eval(const std::vector<double>& params) const;

    friend Affine operator+(const Affine& a, const Affine& b);
    friend Affine operator-(const Affine& a, const Affine& b);
    Affine operator-() const;
    Affine scaled(const Real& s) const;
    std::string str(const std::vector<std::string>& paramNames) const;
};


// Complex number with affine real and imaginary parts. `nonlinear` marks a value that depends on a
// param non-affinely: usable in prints (re-evaluated at run time), an error in an angle. `phase`,
// when set, records that the value is exactly e^{i·phase}, which keeps global phases exact.
struct Num
{
    Affine re;
    Affine im;
    bool nonlinear = false;
    std::optional<Affine> phase;

    Num() = default;
    Num(Real r) : re(std::move(r)) {}
    Num(Affine r) : re(std::move(r)) {}
    static Num complex(Real r, Real i);
    static Num fromDouble(cd z);

    bool isReal() const { return !nonlinear && im.isZero(); }
    bool isConst() const { return !nonlinear && re.isConst() && im.isConst(); }
    std::optional<std::int64_t> asInt() const;
    cd constValue() const; // requires isConst() (or ignores param terms)
    cd eval(const std::vector<double>& params) const;
};

Num operator+(const Num& a, const Num& b);
Num operator-(const Num& a, const Num& b);
Num operator*(const Num& a, const Num& b);
Num operator-(const Num& a);
std::optional<Num> divide(const Num& a, const Num& b); // nullopt when b is zero
Num conj(const Num& a);
Num expi(const Affine& angle); // e^{i·angle}


// ---- Quantum values ----

// A state as a tensor product of factors, leftmost factor on the lowest qubits; a basis-string
// factor keeps 50 000-qubit |0…0⟩ cheap. Superpositions are dense factors.
struct KetFactor
{
    enum class Kind : std::uint8_t { Basis, Plus, Minus, PlusI, MinusI, Dense };
    Kind kind = Kind::Basis;
    std::string bits;    // Basis: one char per qubit, q0 first
    Eigen::VectorXcd amp; // Dense: Kronecker order, first qubit = most significant
    std::size_t nq = 1;
};

struct KetV
{
    std::vector<KetFactor> factors;
    cd scale{1.0, 0.0};
    bool live = false; // |ψ⟩, the live state (run stage)

    std::size_t nq() const;
    bool isBasis() const;          // a single computational basis state (up to scale)
    bool isStabilizerProduct() const; // product of basis / ± / ±i factors
    std::string basisBits() const; // requires isBasis()
    Eigen::VectorXcd dense() const; // Kronecker order, q0 = MSB, scale applied; nq() <= 25
    double norm2() const;
};

struct BraV
{
    KetV ket; // ⟨ket|
};

struct MatV
{
    Eigen::MatrixXcd m;
    std::size_t nq() const;
};

// One Pauli string with a complex coefficient: coef · ⊗ P_q.
struct PTerm
{
    Num coef;
    std::vector<std::pair<Qubit, char>> ps; // sorted by qubit; letters X Y Z
};

struct Cube
{
    std::uint64_t mask = 0;
    std::uint64_t value = 0;
};

struct ChannelSpec
{
    std::string name;
    bool pauli = true;
    std::size_t arity = 1;          // qubits per application
    std::vector<double> probs;      // Pauli: 3 or 15 probabilities (core order)
    std::vector<Eigen::MatrixXcd> kraus;
};

enum class GK : std::uint8_t
{
    I, X, Y, Z, H, S, Sdg, T, Tdg, SX, SXdg, RX, RY, RZ, P, U3,
    CNOT, CZ, CP, SWAP, Toffoli, Fredkin,
    PauliRot,     // exp(-i θ/2 P) on targets, letters in `paulis`
    Matrix,       // dense matrix on targets (targets[0] = MSB)
    GPhase,       // e^{iθ}; observable only under control
    MeasureZ, MeasurePauli, Reset, Channel,
};

std::string_view gkName(GK k);

// One operation of the Noether IR. Angles stay exact and affine in params; controls added by
// C_{…}(…) keep their polarity; Pauli measurements and rotations stay single ops. The executor
// lowers each op onto QuantumStateMachine calls for the chosen backend.
struct IrOp
{
    GK kind = GK::I;
    QubitList controls;    // active on |1⟩ (added by C_{…})
    QubitList negControls; // active on |0⟩ (C_{¬…})
    QubitList targets;     // CNOT {c, t}; Toffoli {c0, c1, t}; Fredkin {c, a, b}; CZ/CP/SWAP {a, b}
    std::vector<Affine> angles;
    std::string paulis;    // PauliRot / MeasurePauli: one letter per target
    bool negate = false;   // MeasurePauli of -P
    bool unitaryMatrix = true; // Matrix: validated unitary (observables may hold Hermitian matrices)
    std::shared_ptr<const Eigen::MatrixXcd> matrix;
    std::shared_ptr<const ChannelSpec> channel;
    std::optional<std::size_t> clbit;
    std::optional<Cube> cond;
    bool clifford = false;
    bool noise = false;    // inserted by a noise: block
    Span span;

    QubitList qubits() const; // controls, negControls, targets
};

// An applied operator: a product of IR ops in time order, times e^{i·gphase}.
struct OpV
{
    std::vector<IrOp> ops;
    Affine gphase;
    bool unitary() const;    // no measurement, reset or channel and no non-unitary matrix
    bool hasChannel() const;
};

// Linear combination of operators: Pauli strings plus general operator terms.
struct LinOpV
{
    std::vector<PTerm> paulis;
    std::vector<std::pair<Num, OpV>> general;
    bool isPauli() const { return general.empty(); }
};

struct QubitsV
{
    QubitList q;
    bool single = false; // a single qubit (q.size() == 1) rather than a register or slice
};

struct BitsV
{
    std::vector<std::size_t> clbits;
    bool single = false;
    std::string reg; // register name, for messages
};

struct CountsV
{
    std::map<std::string, std::uint64_t> counts; // key as text, c0 leftmost per register
    std::vector<std::pair<std::string, std::vector<std::size_t>>> layout; // register -> clbits, in key order
};

enum class Type : std::uint8_t
{
    None, Int, Real, Complex, Bool, String, Ket, Bra, Matrix, Op, Obs, Counts, Bits, List, Gate, Qubits, Channel,
};
std::string_view typeName(Type t);

enum class Stage : std::uint8_t { Compile, Bind, Run };

// A value known only at a later stage, standing in for it during compile-time checking.
struct DeferredV
{
    Type type = Type::Real;
    Stage stage = Stage::Run;
};

enum class Builtin : std::uint8_t
{
    I, X, Y, Z, H, S, T, SX, Rx, Ry, Rz, P, U3, CNOT, CZ, CP, SWAP, Toffoli, Fredkin, C,
};

struct Value;

// A gate not yet applied to qubits: a built-in, a def/proc, a subscripted let, a channel, or ρ.
struct GateV
{
    enum class Kind : std::uint8_t { Builtin, Def, Proc, LetGate, Channel, Rho, Matrix };
    Kind kind = Kind::Builtin;
    Builtin builtin = Builtin::I;
    std::string name;
    const Stmt* def = nullptr;      // Def / Proc
    const Binding* letGate = nullptr; // LetGate
    std::shared_ptr<const Scope> scope; // LetGate: the scope it was defined in
    std::vector<std::shared_ptr<Value>> args; // classical arguments (Rx(θ), flip(p), def params)
    bool dagger = false;
    std::optional<Num> power;
    std::int64_t tensorPow = 1; // G^⊗n of a one-qubit gate: n parallel copies, kept as gates rather than a matrix
    std::shared_ptr<const ChannelSpec> channel; // Channel
    Span span;
};

struct ListV
{
    std::vector<std::shared_ptr<Value>> items;
};

// C_{ctrls} awaiting its operand: C_{0}(U_2).
struct ControlV
{
    QubitList pos, neg;
};

// A Boolean formula over clbits; `if` lowers it to disjoint cubes.
struct CondNode
{
    enum class Kind : std::uint8_t { Const, Var, Not, And, Or };
    Kind kind = Kind::Const;
    bool value = false;      // Const
    std::size_t clbit = 0;   // Var
    std::shared_ptr<const CondNode> a, b;
};

// A run-stage condition over clbits.
struct CondV
{
    std::shared_ptr<const CondNode> root;
};

// ⟨φ| A (a bra times operators) awaiting its ket.
struct BraOpV
{
    KetV bra;
    LinOpV op;
};

// ρ_A: the reduced state of qubits A (a run-stage readout).
struct RhoV
{
    QubitList q;
};


struct Value
{
    std::variant<std::monostate, Num, bool, std::string, KetV, BraV, MatV, OpV, LinOpV, QubitsV, BitsV, CountsV,
                 DeferredV, GateV, ListV, ControlV, CondV, BraOpV, RhoV>
        v;

    Value() = default;
    template <class T> Value(T x) : v(std::move(x)) {}

    template <class T> bool is() const { return std::holds_alternative<T>(v); }
    template <class T> const T& as() const { return std::get<T>(v); }
    template <class T> T& as() { return std::get<T>(v); }
    template <class T> const T* get() const { return std::get_if<T>(&v); }

    Type type() const;
    Stage stage() const;
};

using ValuePtr = std::shared_ptr<Value>;

// Matrix of a 1-qubit Pauli letter, a built-in gate with numeric parameters, or a Pauli string.
Eigen::MatrixXcd pauliMatrix(char letter);
Eigen::MatrixXcd builtinMatrix(Builtin g, const std::vector<double>& params, bool dagger);
Eigen::MatrixXcd gateMatrix(GK k, const std::vector<double>& angles); // 1- to 3-qubit gate kinds

// Pauli algebra: P·Q = phase · R, phase ∈ {1, i, -1, -i} as an exponent of i.
std::pair<int, char> pauliProduct(char a, char b);
PTerm multiply(const PTerm& a, const PTerm& b);
bool commute(const PTerm& a, const PTerm& b);
LinOpV simplify(LinOpV op); // merge equal Pauli strings, drop zero terms

} // namespace Noether
