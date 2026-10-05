#include "Diagnostics.hpp"

#include <algorithm>
#include <array>
#include <format>



namespace Noether
{

Diagnostic& Diagnostics::error(std::string code, Span span, std::string message)
{
    list.push_back({std::move(code), Severity::Error, std::move(message), span, {}, {}});
    return list.back();
}

Diagnostic& Diagnostics::warning(std::string code, Span span, std::string message)
{
    list.push_back({std::move(code), Severity::Warning, std::move(message), span, {}, {}});
    return list.back();
}

void Diagnostics::fatal(std::string code, Span span, std::string message)
{
    error(std::move(code), span, std::move(message));
    throw CompileAbort{};
}

bool Diagnostics::suppressed(const Diagnostic& d) const
{
    return d.severity == Severity::Warning && allowed.contains({d.span.file, d.code});
}

bool Diagnostics::hasErrors() const { return errorCount() > 0; }

std::size_t Diagnostics::errorCount() const
{
    std::size_t n = 0;
    for (const Diagnostic& d : list)
        if (!suppressed(d) && (d.severity == Severity::Error || denyWarn)) ++n;
    return n;
}

std::vector<Diagnostic> Diagnostics::sorted() const
{
    std::vector<Diagnostic> out;
    for (const Diagnostic& d : list)
        if (!suppressed(d)) out.push_back(d);
    std::stable_sort(out.begin(), out.end(), [](const Diagnostic& a, const Diagnostic& b)
    {
        if (a.span.file != b.span.file) return a.span.file < b.span.file;
        return a.span.begin < b.span.begin;
    });
    // One diagnostic per (code, span): error recovery can rediscover the same fault.
    out.erase(std::unique(out.begin(), out.end(), [](const Diagnostic& a, const Diagnostic& b)
    {
        return a.code == b.code && a.span.file == b.span.file && a.span.begin == b.span.begin &&
               a.span.end == b.span.end && a.message == b.message;
    }), out.end());
    return out;
}

Json spanJson(const SourceManager& sm, const Span& s)
{
    Json j = Json::object();
    if (s.file >= sm.size()) return j;
    const SourceFile& f = sm.file(s.file);
    const LineCol b = f.lineCol(s.begin);
    const LineCol e = f.lineCol(std::max(s.end, s.begin));
    j["file"] = f.path();
    j["line"] = b.line;
    j["col"] = b.col;
    j["endLine"] = e.line;
    j["endCol"] = e.col;
    return j;
}

Json diagnosticJson(const SourceManager& sm, const Diagnostic& d)
{
    Json j = Json::object();
    j["code"] = d.code;
    j["severity"] = d.severity == Severity::Error ? "error" : "warning";
    j["message"] = d.message;
    j["span"] = spanJson(sm, d.span);
    Json notes = Json::array();
    for (const Note& n : d.notes)
    {
        Json nj = Json::object();
        nj["span"] = spanJson(sm, n.span);
        nj["message"] = n.message;
        notes.push(std::move(nj));
    }
    j["notes"] = std::move(notes);
    Json fixes = Json::array();
    for (const Fix& f : d.fixes)
    {
        Json fj = Json::object();
        fj["title"] = f.title;
        Json edits = Json::array();
        for (const Edit& e : f.edits)
        {
            Json ej = Json::object();
            ej["span"] = spanJson(sm, e.span);
            ej["text"] = e.text;
            edits.push(std::move(ej));
        }
        fj["edits"] = std::move(edits);
        fixes.push(std::move(fj));
    }
    j["fixes"] = std::move(fixes);
    j["explain"] = std::format("noether explain {}", d.code);
    return j;
}

Json Diagnostics::toJson() const
{
    Json arr = Json::array();
    for (const Diagnostic& d : sorted())
    {
        Diagnostic copy = d;
        if (denyWarn && copy.severity == Severity::Warning) copy.severity = Severity::Error;
        arr.push(diagnosticJson(*sm, copy));
    }
    return arr;
}

std::string renderDiagnostic(const SourceManager& sm, const Diagnostic& d)
{
    std::string out;
    const char* sev = d.severity == Severity::Error ? "error" : "warning";
    if (d.span.file < sm.size())
    {
        const SourceFile& f = sm.file(d.span.file);
        const LineCol b = f.lineCol(d.span.begin);
        out += std::format("{}:{}:{}: {}[{}]: {}\n", f.path(), b.line, b.col, sev, d.code, d.message);
        const std::string_view line = f.lineText(b.line);
        if (!line.empty())
        {
            out += std::format("  {:>4} | {}\n", b.line, line);
            const LineCol e = f.lineCol(std::max(d.span.end, d.span.begin));
            std::size_t width = 1;
            if (e.line == b.line && e.col > b.col) width = e.col - b.col;
            out += std::format("       | {}{}\n", std::string(b.col - 1, ' '), std::string(width, '^'));
        }
    }
    else
    {
        out += std::format("{}[{}]: {}\n", sev, d.code, d.message);
    }
    for (const Note& n : d.notes)
    {
        if (n.span.file < sm.size())
        {
            const SourceFile& f = sm.file(n.span.file);
            const LineCol b = f.lineCol(n.span.begin);
            out += std::format("  note: {}:{}:{}: {}\n", f.path(), b.line, b.col, n.message);
        }
        else out += std::format("  note: {}\n", n.message);
    }
    for (const Fix& f : d.fixes)
    {
        out += std::format("  fix: {}", f.title);
        if (f.edits.size() == 1) out += std::format(" -> `{}`", f.edits[0].text);
        out += '\n';
    }
    return out;
}

std::string Diagnostics::render() const
{
    std::string out;
    for (const Diagnostic& d : sorted())
    {
        Diagnostic copy = d;
        if (denyWarn && copy.severity == Severity::Warning) copy.severity = Severity::Error;
        out += renderDiagnostic(*sm, copy);
    }
    return out;
}


// ---- Catalog ----

namespace
{

    constexpr std::array kCatalog{
        CodeInfo{"E1001", "invalid UTF-8",
                 "Source files are UTF-8. The byte at this position does not start a valid UTF-8 sequence.\n"
                 "Re-save the file as UTF-8, or write the symbol in ASCII or LaTeX form (\\psi, ->, \\otimes)."},
        CodeInfo{"E1002", "tab in indentation",
                 "Blocks use spaces only, so indentation means the same in every editor.\n"
                 "wrong:  for k in 0..3:\n\t    H_k\n"
                 "right:  for k in 0..3:\n            H_k        (4 spaces)"},
        CodeInfo{"E1003", "confusable or mixed-script identifier",
                 "A non-Latin letter that looks like a Latin one (Greek ο ν ρ, Cyrillic а е о р с) touches Latin\n"
                 "letters, so the name is easy to misread.\n"
                 "wrong:  let pοs = 1      (Greek omicron)\n"
                 "right:  let pos = 1"},
        CodeInfo{"E1004", "unknown LaTeX command",
                 "Only the commands in the token table are understood (`noether tokens` lists them).\n"
                 "Commands take exactly one backslash.\n"
                 "wrong:  S\\dager_0\n"
                 "right:  S\\dagger_0"},
        CodeInfo{"E2001", "unexpected token",
                 "The parser expected something else here. The message names what it expected.\n"
                 "wrong:  for k in 0..3\n            H_k\n"
                 "right:  for k in 0..3:\n            H_k"},
        CodeInfo{"E2002", "bad dedent",
                 "A line dedents to a column that no enclosing block uses.\n"
                 "wrong:  for k in 0..1:\n              H_k\n          X_k\n"
                 "right:  for k in 0..1:\n              H_k\n              X_k"},
        CodeInfo{"E2004", "± without ≈",
                 "A tolerance is only meaningful in an approximate comparison.\n"
                 "wrong:  assert ⟨Z_0⟩ == 1 ± 1e-6\n"
                 "right:  assert ⟨Z_0⟩ ≈ 1 ± 1e-6"},
        CodeInfo{"E2005", "missing header",
                 "The first non-comment line of a file must name the language version.\n"
                 "right:  noether 0.1"},
        CodeInfo{"E2006", "header newer than this binary",
                 "The file asks for a language version this noether does not implement. Upgrade noether or\n"
                 "lower the header if the program does not use newer features."},
        CodeInfo{"E2007", "nested absolute-value bars",
                 "Bars cannot be nested because `|` both opens and closes.\n"
                 "wrong:  ||x| - 1|\n"
                 "right:  abs(abs(x) - 1)"},
        CodeInfo{"E2008", "gate role syntax",
                 "Gates with a control/target role need the arrow between the roles.\n"
                 "wrong:  CNOT_{0,1}\n"
                 "right:  CNOT_{0→1}   (ASCII: CNOT_{0->1})"},
        CodeInfo{"E2009", "Qiskit-ism",
                 "Qiskit spelling is not Noether syntax, so the statement is not compiled; the fix-it gives the Noether\n"
                 "form.\n"
                 "wrong:  qc.cx(0, 1)\n"
                 "right:  CNOT_{0→1}"},
        CodeInfo{"E3001", "unknown name",
                 "The name is not bound in this scope. The message suggests similar names, and a split when\n"
                 "two bound names were written together.\n"
                 "wrong:  let g = 2\n        print gX\n"
                 "right:  print g X  (if a product was meant)"},
        CodeInfo{"E3002", "binding a reserved name",
                 "i, e, π, ψ, ρ, gate names and function names are reserved.\n"
                 "wrong:  for i in 0..3:\n"
                 "right:  for k in 0..3:\n"
                 "wrong:  let H = …     (H is the Hadamard)\n"
                 "right:  let Ham = …"},
        CodeInfo{"E3003", "rebinding in the same scope",
                 "Bindings are immutable. Choose a new name.\n"
                 "wrong:  let x = 1\n        let x = 2\n"
                 "right:  let x = 1\n        let y = 2"},
        CodeInfo{"E3004", "use before definition",
                 "There is no hoisting: a name is visible only after the statement that defines it.\n"
                 "wrong:  print a\n        let a = 1\n"
                 "right:  let a = 1\n        print a"},
        CodeInfo{"E3005", "underscore inside a name",
                 "`_` always means subscript, so names use camelCase.\n"
                 "wrong:  let my_var = 1\n"
                 "right:  let myVar = 1"},
        CodeInfo{"E4001", "type mismatch",
                 "The value has the wrong type for this position. The message names the expected and actual\n"
                 "types (Int, Real, Complex, Bool, Ket, Bra, Matrix, Op, Obs, Counts, Bits, …).\n"
                 "wrong:  Rx(|0⟩)_0\n"
                 "right:  Rx(π/2)_0"},
        CodeInfo{"E4002", "not callable",
                 "Only gates, defs, functions and channels take arguments in parentheses.\n"
                 "wrong:  let a = 2\n        a(3)\n"
                 "right:  a (3)   (a space makes it a product)"},
        CodeInfo{"E4003", "non-integer index",
                 "Qubit indices, register sizes, loop bounds and list indices must be exact integers known at\n"
                 "compile time.\n"
                 "wrong:  H_{N/2}       (N odd)\n"
                 "right:  H_{floor(N/2)}"},
        CodeInfo{"E4004", "stage violation",
                 "A value is used where it is not yet known. Qubit indices and loop bounds must be compile-time;\n"
                 "angles may depend on params but never on measurement outcomes.\n"
                 "wrong:  c[0] ← measure Z_0\n        Rx(c[0])_1\n"
                 "right:  if c[0]: X_1"},
        CodeInfo{"E4005", "non-affine param",
                 "A param may enter an angle only affinely (a·θ + b with a, b compile-time), so gradients stay\n"
                 "exact and Clifford classification decidable.\n"
                 "wrong:  Rz(sin(θ))_0\n"
                 "right:  Rz(2θ + π/4)_0"},
        CodeInfo{"E4006", "expectation of a non-Hermitian operator",
                 "⟨A⟩ needs an observable. For a general operator use the matrix element ⟨ψ|A|ψ⟩.\n"
                 "wrong:  print ⟨S_0⟩\n"
                 "right:  print ⟨ψ|S_0|ψ⟩"},
        CodeInfo{"E4007", "integer overflow",
                 "Constant folding overflowed int64.\n"
                 "wrong:  let n = 2^70\n"
                 "right:  let x = 2.0^70"},
        CodeInfo{"E4008", "unsupported √",
                 "√ of an operator is only defined for X (the √X gate).\n"
                 "wrong:  √Y_0\n"
                 "right:  √X_0  or  Ry(π/2)_0"},
        CodeInfo{"E5001", "repeated qubit",
                 "The qubits of one operation must be distinct (no-cloning), and a control cannot also be a\n"
                 "target of C_{…}(…).\n"
                 "wrong:  CNOT_{0→0}\n"
                 "right:  CNOT_{0→1}"},
        CodeInfo{"E5002", "overlapping tensor supports",
                 "The factors of ⊗ must act on disjoint qubits.\n"
                 "wrong:  X_0 ⊗ Z_0\n"
                 "right:  X_0 ⊗ Z_1"},
        CodeInfo{"E5003", "not unitary",
                 "An applied operator must be unitary (a global phase is fine), custom matrices must be unitary\n"
                 "within 1e-12·dim, and Kraus operators must be complete.\n"
                 "wrong:  2 X_0\n"
                 "right:  X_0"},
        CodeInfo{"E5004", "not Hermitian",
                 "An observable or Hamiltonian must be Hermitian.\n"
                 "wrong:  trotter(i Z_0, 1.0, steps=10)\n"
                 "right:  trotter(Z_0, 1.0, steps=10)"},
        CodeInfo{"E5005", "invalid exp",
                 "exp of an operator needs -i·r·O with O a Pauli string or a sum of mutually commuting Pauli\n"
                 "strings. Non-commuting sums need an explicit product formula.\n"
                 "wrong:  exp(-i t (X_0 + Z_0))\n"
                 "right:  trotter(X_0 + Z_0, t, steps=100, order=2)"},
        CodeInfo{"E5006", "† or C on a proc",
                 "A proc may measure, reset or branch, so it has no adjoint or controlled form, and a def (a\n"
                 "unitary) cannot call it.\n"
                 "wrong:  Round†_q\n"
                 "right:  make Round a def if it is unitary"},
        CodeInfo{"E5007", "ket not normalised",
                 "Prepared states must have norm 1 within 1e-12.\n"
                 "wrong:  prepare |00⟩ + |11⟩\n"
                 "right:  prepare (|00⟩ + |11⟩)/√2"},
        CodeInfo{"E5008", "qubit out of range",
                 "Integer subscripts index the global qubits; registers concatenate in declaration order.\n"
                 "wrong:  qubits q[2]\n        H_2\n"
                 "right:  H_1"},
        CodeInfo{"E5009", "support or dimension mismatch",
                 "An operator needs an explicit support of the right size: a matrix of dimension 2^k needs k\n"
                 "qubits, broadcast lists must have equal lengths, and def bodies reach qubits only through\n"
                 "their subscript parameters.\n"
                 "wrong:  let U = [[0,1],[1,0]]\n        U\n"
                 "right:  U_0"},
        CodeInfo{"E5010", "measuring a non-Pauli",
                 "`measure P` measures one Pauli string with coefficient ±1 and records one bit.\n"
                 "wrong:  c[0] ← measure H_0\n"
                 "right:  c[0] ← measure X_0"},
        CodeInfo{"E5011", "run with no measurement",
                 "`run` tallies classical bits, so the record since the last prepare must measure into one.\n"
                 "wrong:  H_0\n        counts ← run 100\n"
                 "right:  H_0\n        c[0] ← measure Z_0\n        counts ← run 100"},
        CodeInfo{"E5012", "measurement inside if",
                 "v0.1 has no conditional measurement. Measure unconditionally and branch on the result.\n"
                 "wrong:  if c[0]: c[1] ← measure Z_1\n"
                 "right:  c[1] ← measure Z_1"},
        CodeInfo{"E6001", "operation unsupported on backend",
                 "The stabilizer backend runs Clifford operations and Pauli channels only; the state vector holds\n"
                 "at most 25 qubits. The message names the first offending operation.\n"
                 "wrong:  backend stabilizer\n        T_0\n"
                 "right:  backend statevector   (N ≤ 25)"},
        CodeInfo{"E6002", "readout unsupported on backend",
                 "Amplitudes, reduced density matrices, overlaps with non-basis states and dense observables need\n"
                 "the state vector. Pauli expectations, basis probabilities and entropy work on both.\n"
                 "wrong:  backend stabilizer\n        print |ψ⟩\n"
                 "right:  print |⟨00|ψ⟩|^2"},
        CodeInfo{"E6003", "too many qubits for the backend",
                 "The state vector holds at most 25 qubits and the tableau 65536."},
        CodeInfo{"E6004", "estimated memory over --max-mem",
                 "`noether estimate` predicts more memory than allowed. Reduce the problem or raise --max-mem."},
        CodeInfo{"E6005", "condition too large",
                 "A condition lowers to at most 64 disjoint cubes over clbits below 64. Simplify the formula or\n"
                 "split it across several if statements."},
        CodeInfo{"E7001", "assertion failed",
                 "An assert evaluated to false on the live pass. The JSON result shows lhs, rhs and tol. Exit\n"
                 "code 3."},
        CodeInfo{"E7002", "timeout", "Execution exceeded --timeout. Exit code 4."},
        CodeInfo{"E7003", "numerical drift",
                 "The live state's norm drifted from 1 by more than 1e-8 by the end of the run. Every gate is unitary to\n"
                 "1e-12, so this signals an internal error or a matrix that is unitary only to the validation tolerance\n"
                 "applied very many times. Exit code 5."},
        CodeInfo{"E8001", "spec hash mismatch",
                 "spec.ntr no longer matches spec.lock, so the evaluator was edited. Restore spec.ntr; never edit a\n"
                 "frozen spec during a research run. Exit code 6."},
        CodeInfo{"E8002", "forbidden statement in candidate or import",
                 "Candidates and imported files may contain only the header, import, let, param, def and proc."},
        CodeInfo{"E8003", "not decomposable into the basis",
                 "An operation of the candidate has no exact decomposition into the spec's gate set (e.g. a\n"
                 "non-Clifford+T angle into {H, S, T, CNOT})."},
        CodeInfo{"E8004", "coupling violation",
                 "A two-qubit gate acts on a pair the spec's coupling map does not allow. There is no automatic\n"
                 "SWAP insertion; route explicitly."},
        CodeInfo{"E8005", "candidate signature mismatch",
                 "The candidate must define the def or proc declared by `candidate` in the spec, with the same\n"
                 "subscript parameters."},
        CodeInfo{"E8006", "metric not estimable under readout shots",
                 "Fidelity and amplitudes cannot be estimated from measurement shots."},
        CodeInfo{"E8007", "forbidden gate",
                 "The candidate uses a construct the spec forbids."},
        CodeInfo{"W0001", "1/2π ambiguity",
                 "`/` binds tighter than juxtaposition, so 1/2π means (1/2)·π.\n"
                 "wrong:  Rz(1/2π)_0     (if 1/(2π) was meant)\n"
                 "right:  Rz(1/(2π))_0"},
        CodeInfo{"W0002", "float angle near a multiple of π/4",
                 "Exact angles drive Clifford detection and T-count.\n"
                 "wrong:  Rz(0.785398)_0\n"
                 "right:  Rz(π/4)_0"},
        CodeInfo{"W0003", "shadowing", "A binding hides one from an enclosing scope."},
        CodeInfo{"W0004", "unused binding or param", "A let or param is never used."},
        CodeInfo{"W0005", "readout on a single noisy trajectory",
                 "With noise, a readout samples one trajectory. Set `trajectories K` to average over K."},
        CodeInfo{"W0006", "large amplitude output",
                 "Printing |ψ⟩ above 12 qubits produces up to 2^N entries. Print probabilities or marginals."},
        CodeInfo{"W0009", "f (x) with a space",
                 "A space before `(` makes a product, not a call.\n"
                 "wrong:  sin (x)\n"
                 "right:  sin(x)"},
    };

} // namespace

std::span<const CodeInfo> codeCatalog() { return kCatalog; }

const CodeInfo* findCode(std::string_view code)
{
    for (const CodeInfo& c : kCatalog)
        if (c.code == code) return &c;
    return nullptr;
}

} // namespace Noether
