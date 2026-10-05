#pragma once

#include "Lexer.hpp"
#include "Source.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>



namespace Noether
{

struct Expr;
using ExprPtr = std::shared_ptr<const Expr>;

enum class EK : std::uint8_t
{
    Name,      // name
    Int,       // name = digits
    Real,      // name = literal text
    String,    // name = contents
    Bool,      // name = "true" / "false"
    Ket,       // name = label
    Bra,       // name = label
    Braket,    // name = bra label, label2 = ket label
    List,      // kids = elements
    Paren,     // kids[0]
    Unary,     // op ∈ {Minus, Plus, Not, Sqrt}; kids[0]
    Binary,    // op ∈ {Plus, Minus, Cdot, Slash, Otimes, KwAnd, KwOr}; kids[0..1]
    Product,   // juxtaposition; kids = factors, left to right
    Compare,   // op ∈ {EqEq, NotEq, Less, LessEq, Greater, GreaterEq, Approx}; kids = lhs, rhs[, tol]
    Sub,       // kids[0] = base; qlist
    Call,      // kids[0] = callee, kids[1..] = args; argNames[k] names kids[k+1] ("" if positional)
    Index,     // kids[0] = base, kids[1] = index (may be a Slice)
    Slice,     // kids = lo, hi
    Dagger,    // kids[0]
    Power,     // kids = base, exponent; braced = exponent written in {…}
    TensorPow, // kids = base, exponent
    BigOp,     // op ∈ {Sum, Prod}; name = variable; kids = lo, hi, body
    Expval,    // ⟨A⟩; kids[0]
    Abs,       // |x|; kids[0]
    Set,       // {a, b, …} (spec files only); kids = elements
};

struct QItem
{
    bool negated = false;
    ExprPtr expr; // may be a Slice
    Span span;
};

// Items of a subscript list and the separator after each but the last: ',' or '→'.
struct QList
{
    std::vector<QItem> items;
    std::vector<Tok> seps;
    bool braced = false;
};

struct Expr
{
    EK kind = EK::Name;
    Span span;
    std::string name;
    std::string label2;
    Tok op = Tok::End;
    std::vector<ExprPtr> kids;
    std::vector<std::string> argNames;
    QList qlist;
    bool braced = false;   // Power/TensorPow exponent or BigOp bound written in {…}
    bool touching = false; // Product factor: no space before this factor (formatter hint only)
};


enum class SK : std::uint8_t
{
    Header, Qubits, Bits, Seed, Backend, Trajectories, Let, Param, Prepare, Measure, Run, Reset, Print,
    Assert, Import, Apply, For, If, Def, Proc, Noise, Spec,
};

struct Stmt;
using StmtPtr = std::shared_ptr<const Stmt>;
using Block = std::vector<StmtPtr>;

struct RegDecl
{
    std::string name;
    ExprPtr size;
    Span span;
};

struct QParam
{
    std::string name;
    std::string sizeName; // r[n]: "n"; r[3]: "3" (fixed size); empty for a single-qubit slot
    Span span;
};

struct SubParams
{
    std::vector<QParam> params;
    std::vector<Tok> seps; // ',' or '→' between params
    bool braced = false;
};

struct Binding
{
    std::string name;
    Span nameSpan;
    bool hasSub = false;
    SubParams sub; // `let Bell_{a,b} = …` defines a gate
    ExprPtr value;
    std::string doc;
};

struct PrintItem
{
    ExprPtr expr;
    std::string label;
    Span span;
};

struct NoiseRule
{
    bool after = true;
    std::string event; // gate1 gate2 measure reset prepare
    ExprPtr channel;
    Span span;
};

// Spec-file statements (task files) share one shape.
struct SpecStmt
{
    std::string keyword; // task candidate target metric gates coupling allow forbid readout optimize
                         // require minimize maximize goal instances holdout aggregate budget
    std::string word;    // target kind, candidate def/proc, coupling kind, readout kind, metric name, …
    std::string name;    // task name, candidate name, instances variable
    ExprPtr expr;        // target / metric / require / goal expression, instances set, coupling map
    SubParams sub;       // candidate signature
    std::vector<std::pair<std::string, ExprPtr>> options; // optimize / budget key=value pairs
    std::vector<std::pair<std::string, bool>> gates;       // gates / forbid: name, dagger
};

struct Comments
{
    std::vector<Comment> leading; // own-line comments before the statement
    std::vector<Comment> trailing; // same-line comment after it
    bool blankBefore = false;
};

struct Stmt
{
    SK kind = SK::Apply;
    Span span;
    Span head; // the statement's own line(s): up to the ':' of a compound statement
    Comments comments;
    std::string doc; // ## doc comment

    // Header: version text. Seed/Trajectories: expr. Backend: word. Import: text = path.
    std::string text;
    ExprPtr expr; // Seed, Trajectories, Prepare, Assert, Apply, Run (shots), If (cond), Measure (Pauli)

    std::vector<RegDecl> regs;      // Qubits, Bits
    std::vector<Binding> bindings;  // Let

    // Param: name, optional size, interval, default
    std::string name;
    Span nameSpan;
    ExprPtr paramSize, lo, hi, init;

    // Measure / Run assignment: target bit register (with optional index or slice)
    ExprPtr lvalue;
    bool measureSub = false; // measure_q (Z on each qubit) rather than a Pauli string
    QList qlist;             // Measure (sub form), Reset

    std::vector<PrintItem> items; // Print

    // For: variable `name`, range lo..hi [by step] or list `expr`
    ExprPtr step;

    Block body;     // For, If (then), Def, Proc
    Block orelse;   // If
    std::vector<std::string> cparams; // Def / Proc
    SubParams sub;                    // Def / Proc

    std::vector<NoiseRule> rules; // Noise
    SpecStmt spec;                // Spec
};

struct Program
{
    std::uint32_t file = 0;
    std::string version;   // header version text, e.g. "0.1"
    Span headerSpan;
    Block stmts;
    bool isSpec = false;   // first statement after the header is `task`
    std::vector<Comment> trailingComments; // comments after the last statement
};

} // namespace Noether
