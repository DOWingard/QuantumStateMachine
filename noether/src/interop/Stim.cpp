#include "interop/Stim.hpp"

#include "interop/Lowering.hpp"

#include <StabilizerState.hpp>

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <complex>
#include <format>
#include <map>
#include <set>



namespace Noether::Interop
{

namespace
{
    constexpr std::size_t kMaxUnrolledOps = 10'000'000;
    constexpr std::size_t kMaxNesting = 64;

    struct Fail
    {
        std::string code;
        Span span;
        std::string message;
    };

    enum class TK : std::uint8_t { Qubit, Pauli, Rec, Sweep };

    struct Target
    {
        TK kind = TK::Qubit;
        bool inverted = false;
        char pauli = 0;          // Pauli: X, Y or Z
        std::int64_t value = 0;  // Qubit/Pauli: qubit id; Rec: k of rec[-k]; Sweep: index
        bool joined = false;     // joined to the previous target by `*`
        Span span;
    };

    struct Instr
    {
        std::string name; // upper case, aliases kept
        Span span;        // the name
        std::vector<double> args;
        std::vector<Target> targets;
        std::uint64_t repeat = 0;
        std::vector<Instr> body; // REPEAT
    };

    // ---- Unitary gates: time-order recipes, equal to Stim's gate up to global phase ----

    // One-qubit recipes over h s sdg x y z sx.
    const std::map<std::string_view, std::string_view>& oneQubitRecipes()
    {
        static const std::map<std::string_view, std::string_view> m{
            {"I", ""},           {"H", "h"},          {"H_XZ", "h"},          {"S", "s"},           {"SQRT_Z", "s"},
            {"S_DAG", "sdg"},    {"SQRT_Z_DAG", "sdg"}, {"X", "x"},           {"Y", "y"},           {"Z", "z"},
            {"SQRT_X", "sx"},    {"SQRT_X_DAG", "x sx"}, {"SQRT_Y", "h x"},   {"SQRT_Y_DAG", "h z"}, {"H_XY", "s y"},
            {"H_YZ", "y sx"},    {"H_NXY", "s x"},    {"H_NXZ", "h y"},       {"H_NYZ", "z sx"},    {"C_XYZ", "sdg h"},
            {"C_ZYX", "h s"},    {"C_NXYZ", "sx sdg"}, {"C_XNYZ", "h sx"},    {"C_XYNZ", "h y sx"}, {"C_NZYX", "sdg sx"},
            {"C_ZNYX", "h sdg"}, {"C_ZYNX", "h s y"},
        };
        return m;
    }

    // Two-qubit recipes: a word is a gate name followed by operand positions, `cx01` = CNOT 0 → 1.
    const std::map<std::string_view, std::string_view>& twoQubitRecipes()
    {
        static const std::map<std::string_view, std::string_view> m{
            {"II", ""},
            {"CX", "cx01"}, {"CNOT", "cx01"}, {"ZCX", "cx01"},
            {"CY", "sdg1 cx01 s1"}, {"ZCY", "sdg1 cx01 s1"},
            {"CZ", "cz01"}, {"ZCZ", "cz01"},
            {"SWAP", "swap01"},
            {"ISWAP", "s0 s1 cz01 swap01"}, {"ISWAP_DAG", "sdg0 sdg1 cz01 swap01"},
            {"CXSWAP", "cx01 swap01"}, {"SWAPCX", "cx01 cx10"}, {"CZSWAP", "cz01 swap01"}, {"SWAPCZ", "cz01 swap01"},
            {"XCX", "h0 cx01 h0"}, {"XCY", "sx1 cx10 x1 sx1"}, {"XCZ", "cx10"},
            {"YCX", "sx0 cx01 x0 sx0"}, {"YCY", "sx0 cx01 h0 cx10 s0"}, {"YCZ", "sdg0 cx10 s0"},
        };
        return m;
    }

    // SQRT_PP is SPP on P⊗P: it phases the −1 eigenspace of P⊗P by i.
    const std::map<std::string_view, std::pair<char, bool>>& sqrtPairGates()
    {
        static const std::map<std::string_view, std::pair<char, bool>> m{
            {"SQRT_XX", {'X', false}}, {"SQRT_XX_DAG", {'X', true}}, {"SQRT_YY", {'Y', false}},
            {"SQRT_YY_DAG", {'Y', true}}, {"SQRT_ZZ", {'Z', false}}, {"SQRT_ZZ_DAG", {'Z', true}},
        };
        return m;
    }

    Prim primOf(std::string_view w)
    {
        static const std::map<std::string_view, Prim> m{{"h", Prim::H}, {"s", Prim::S}, {"sdg", Prim::Sdg}, {"x", Prim::X},
                                                        {"y", Prim::Y}, {"z", Prim::Z}, {"sx", Prim::SX}};
        return m.at(w);
    }

    Seq recipe(std::string_view text, const QubitList& q)
    {
        Seq out;
        std::size_t pos = 0;
        while (pos < text.size())
        {
            std::size_t end = text.find(' ', pos);
            if (end == std::string_view::npos) end = text.size();
            std::string_view w = text.substr(pos, end - pos);
            pos = end + 1;
            std::vector<Qubit> args;
            while (!w.empty() && std::isdigit(static_cast<unsigned char>(w.back())))
            {
                args.insert(args.begin(), q[static_cast<std::size_t>(w.back() - '0')]);
                w.remove_suffix(1);
            }
            if (args.empty()) args.push_back(q[0]);
            if (w == "cx") out.push_back(makeGate(Prim::X, {args[1]}, {}, {args[0]}));
            else if (w == "cz") out.push_back(makeGate(Prim::Z, {args[1]}, {}, {args[0]}));
            else if (w == "swap") out.push_back(makeGate(Prim::Swap, {args[0], args[1]}));
            else out.push_back(makeGate(primOf(w), {args[0]}));
        }
        return out;
    }

    // ---- Other instruction families ----

    struct MeasureSpec
    {
        char basis;
        bool measures;
        bool resets;
    };
    const std::map<std::string_view, MeasureSpec>& measureFamily()
    {
        static const std::map<std::string_view, MeasureSpec> m{
            {"M", {'Z', true, false}},  {"MZ", {'Z', true, false}}, {"MX", {'X', true, false}}, {"MY", {'Y', true, false}},
            {"MR", {'Z', true, true}},  {"MRZ", {'Z', true, true}}, {"MRX", {'X', true, true}}, {"MRY", {'Y', true, true}},
            {"R", {'Z', false, true}},  {"RZ", {'Z', false, true}}, {"RX", {'X', false, true}}, {"RY", {'Y', false, true}},
        };
        return m;
    }

    const std::set<std::string_view>& noiseNames()
    {
        static const std::set<std::string_view> s{"X_ERROR", "Y_ERROR", "Z_ERROR", "DEPOLARIZE1", "DEPOLARIZE2", "PAULI_CHANNEL_1",
                                                  "PAULI_CHANNEL_2", "I_ERROR", "II_ERROR", "E", "CORRELATED_ERROR", "ELSE_CORRELATED_ERROR"};
        return s;
    }

    const std::set<std::string_view>& otherNames()
    {
        static const std::set<std::string_view> s{"MPP", "MXX", "MYY", "MZZ", "SPP", "SPP_DAG", "DETECTOR", "OBSERVABLE_INCLUDE",
                                                  "TICK", "QUBIT_COORDS", "SHIFT_COORDS", "REPEAT", "MPAD", "HERALDED_ERASE",
                                                  "HERALDED_PAULI_CHANNEL_1"};
        return s;
    }

    bool known(std::string_view n)
    {
        return oneQubitRecipes().contains(n) || twoQubitRecipes().contains(n) || sqrtPairGates().contains(n) || measureFamily().contains(n) ||
               noiseNames().contains(n) || otherNames().contains(n);
    }

    bool producesResults(std::string_view n)
    {
        return n == "MPP" || n == "MXX" || n == "MYY" || n == "MZZ" || n == "MPAD" || n.starts_with("HERALDED") ||
               (measureFamily().contains(n) && measureFamily().at(n).measures);
    }


    // ---- Parser: one instruction per line, `REPEAT n {` … `}` blocks ----

    class Parser
    {
        public:
        Parser(const SourceFile& f, std::uint32_t id) : text(f.text()), file(id) {}

        std::vector<Instr> parse()
        {
            std::vector<Instr> root;
            std::vector<std::vector<Instr>*> stack{&root};
            std::vector<Span> opened;
            std::size_t pos = 0;
            while (pos <= text.size())
            {
                std::size_t end = text.find('\n', pos);
                if (end == std::string_view::npos) end = text.size();
                line(pos, end, stack, opened);
                pos = end + 1;
            }
            if (stack.size() > 1) throw Fail{"E9001", opened.back(), "REPEAT block is never closed with `}`"};
            return root;
        }

        private:
        Span span(std::size_t b, std::size_t e) const { return Span{file, static_cast<std::uint32_t>(b), static_cast<std::uint32_t>(e)}; }

        static bool space(char c) { return c == ' ' || c == '\t' || c == '\r'; }

        void line(std::size_t b, std::size_t e, std::vector<std::vector<Instr>*>& stack, std::vector<Span>& opened)
        {
            if (const std::size_t hash = text.find('#', b); hash < e) e = hash;
            while (b < e && space(text[b])) ++b;
            while (e > b && space(text[e - 1])) --e;
            if (b == e) return;
            if (text[b] == '}')
            {
                if (e != b + 1) throw Fail{"E9001", span(b, e), "`}` must stand alone on its line"};
                if (stack.size() == 1) throw Fail{"E9001", span(b, e), "`}` without an open REPEAT block"};
                stack.pop_back();
                opened.pop_back();
                return;
            }
            std::size_t i = b;
            while (i < e && (std::isalnum(static_cast<unsigned char>(text[i])) || text[i] == '_')) ++i;
            if (i == b) throw Fail{"E9001", span(b, e), "expected an instruction name"};
            Instr in;
            in.span = span(b, i);
            for (std::size_t k = b; k < i; ++k) in.name += static_cast<char>(std::toupper(static_cast<unsigned char>(text[k])));
            if (i < e && text[i] == '(')
            {
                const std::size_t close = text.find(')', i);
                if (close == std::string_view::npos || close >= e) throw Fail{"E9001", span(i, e), "unclosed `(`"};
                std::size_t a = i + 1;
                while (a <= close)
                {
                    std::size_t comma = text.find(',', a);
                    if (comma == std::string_view::npos || comma > close) comma = close;
                    std::size_t x = a, y = comma;
                    while (x < y && space(text[x])) ++x;
                    while (y > x && space(text[y - 1])) --y;
                    if (x == y && comma == close && in.args.empty()) break; // `()`
                    double v = 0.0;
                    const auto r = std::from_chars(text.data() + x, text.data() + y, v);
                    if (r.ec != std::errc{} || r.ptr != text.data() + y || x == y) throw Fail{"E9001", span(x, std::max(y, x + 1)), "expected a number"};
                    in.args.push_back(v);
                    a = comma + 1;
                }
                i = close + 1;
            }
            if (i < e && !space(text[i])) throw Fail{"E9001", span(i, i + 1), std::format("unexpected `{}` after the instruction name", text[i])};

            if (in.name == "REPEAT")
            {
                // REPEAT n {
                std::size_t x = i;
                while (x < e && space(text[x])) ++x;
                std::size_t y = x;
                while (y < e && std::isdigit(static_cast<unsigned char>(text[y]))) ++y;
                std::uint64_t n = 0;
                const auto r = std::from_chars(text.data() + x, text.data() + y, n);
                if (x == y || r.ec != std::errc{} || n == 0) throw Fail{"E9001", span(x, std::max(y, x + 1)), "REPEAT needs a positive count"};
                while (y < e && space(text[y])) ++y;
                if (y + 1 != e || text[y] != '{') throw Fail{"E9001", span(y, e), "expected `{` to end the REPEAT line"};
                if (stack.size() > kMaxNesting) throw Fail{"E9004", in.span, std::format("REPEAT blocks nest deeper than {}", kMaxNesting)};
                in.repeat = n;
                stack.back()->push_back(std::move(in));
                opened.push_back(stack.back()->back().span);
                stack.push_back(&stack.back()->back().body);
                return;
            }

            bool joinNext = false;
            while (i < e)
            {
                while (i < e && space(text[i])) ++i;
                if (i >= e) break;
                std::size_t j = i;
                while (j < e && !space(text[j])) ++j;
                // A whitespace-separated word may hold several `*`-joined targets.
                std::size_t p = i;
                while (p < j)
                {
                    std::size_t q = p;
                    while (q < j && text[q] != '*') ++q;
                    if (q > p)
                    {
                        Target t = target(p, q);
                        t.joined = joinNext;
                        in.targets.push_back(t);
                        joinNext = false;
                    }
                    if (q < j)
                    {
                        if (in.targets.empty() || joinNext) throw Fail{"E9001", span(q, q + 1), "`*` must join two Pauli targets"};
                        joinNext = true;
                    }
                    p = q + 1;
                }
                i = j;
            }
            if (joinNext) throw Fail{"E9001", in.span, "a trailing `*` joins nothing"};
            stack.back()->push_back(std::move(in));
        }

        Target target(std::size_t b, std::size_t e) const
        {
            Target t;
            t.span = span(b, e);
            std::string_view s = text.substr(b, e - b);
            if (s.starts_with('!'))
            {
                t.inverted = true;
                s.remove_prefix(1);
            }
            auto integer = [&](std::string_view digits) -> std::int64_t
            {
                std::int64_t v = 0;
                const auto r = std::from_chars(digits.data(), digits.data() + digits.size(), v);
                if (digits.empty() || r.ec != std::errc{} || r.ptr != digits.data() + digits.size() || v < 0)
                    throw Fail{"E9001", t.span, std::format("bad target `{}`", text.substr(b, e - b))};
                return v;
            };
            auto bracketed = [&](std::string_view prefix) -> std::optional<std::string_view>
            {
                if (s.size() < prefix.size() + 2 || !s.starts_with(prefix) || s[prefix.size()] != '[' || s.back() != ']') return std::nullopt;
                return s.substr(prefix.size() + 1, s.size() - prefix.size() - 2);
            };
            if (const auto r = bracketed("rec"))
            {
                if (!r->starts_with('-')) throw Fail{"E9001", t.span, "record targets look back: rec[-k] with k ≥ 1"};
                t.kind = TK::Rec;
                t.value = integer(r->substr(1));
                if (t.value == 0) throw Fail{"E9001", t.span, "rec[-0] is not a measurement record"};
                return t;
            }
            if (const auto r = bracketed("sweep"))
            {
                t.kind = TK::Sweep;
                t.value = integer(*r);
                return t;
            }
            if (!s.empty() && (s[0] == 'X' || s[0] == 'Y' || s[0] == 'Z' || s[0] == 'x' || s[0] == 'y' || s[0] == 'z'))
            {
                t.kind = TK::Pauli;
                t.pauli = static_cast<char>(std::toupper(static_cast<unsigned char>(s[0])));
                t.value = integer(s.substr(1));
                return t;
            }
            t.kind = TK::Qubit;
            t.value = integer(s);
            return t;
        }

        std::string_view text;
        std::uint32_t file;
    };


    // ---- Lowering ----

    std::uint64_t saturatingMul(std::uint64_t a, std::uint64_t b) { return a != 0 && b > UINT64_MAX / a ? UINT64_MAX : a * b; }
    std::uint64_t saturatingAdd(std::uint64_t a, std::uint64_t b) { return a > UINT64_MAX - b ? UINT64_MAX : a + b; }

    std::uint64_t resultsOf(const Instr& in)
    {
        if (in.name == "MPP")
        {
            std::uint64_t n = 0;
            for (const Target& t : in.targets) n += t.joined ? 0 : 1;
            return n;
        }
        if (in.name == "MXX" || in.name == "MYY" || in.name == "MZZ") return in.targets.size() / 2;
        return producesResults(in.name) ? in.targets.size() : 0;
    }

    std::uint64_t countResults(const std::vector<Instr>& block)
    {
        std::uint64_t n = 0;
        for (const Instr& in : block)
            n = saturatingAdd(n, in.name == "REPEAT" ? saturatingMul(in.repeat, countResults(in.body)) : resultsOf(in));
        return n;
    }

    // The instruction that records result number `limit` (0-based), or null.
    const Instr* resultAt(const std::vector<Instr>& block, std::uint64_t& seen, std::uint64_t limit)
    {
        for (const Instr& in : block)
        {
            if (in.name == "REPEAT")
            {
                for (std::uint64_t k = 0; k < in.repeat && seen <= limit; ++k)
                    if (const Instr* hit = resultAt(in.body, seen, limit)) return hit;
                continue;
            }
            seen += resultsOf(in);
            if (seen > limit) return &in;
        }
        return nullptr;
    }

    void collectQubits(const std::vector<Instr>& block, std::set<std::int64_t>& ids)
    {
        for (const Instr& in : block)
        {
            if (in.name == "REPEAT")
            {
                collectQubits(in.body, ids);
                continue;
            }
            if (in.name == "QUBIT_COORDS" || in.name == "DETECTOR" || in.name == "OBSERVABLE_INCLUDE" || in.name == "SHIFT_COORDS") continue;
            for (const Target& t : in.targets)
                if (t.kind == TK::Qubit || t.kind == TK::Pauli) ids.insert(t.value);
        }
    }

    class Lowerer
    {
        public:
        Lowerer(ImportedCircuit& circuit, Diagnostics& d, std::map<std::int64_t, Qubit> ids)
            : c(circuit), b(circuit), diags(d), qubitOf(std::move(ids))
        {
        }

        void run(const std::vector<Instr>& program)
        {
            block(program);
            flushChain();
            checkRecordFlips();
        }

        private:
        struct Chain
        {
            Span span;
            std::vector<std::pair<double, std::vector<PauliFactor>>> terms;
        };

        // A measurement whose reported result may flip: exact only if its qubits idle afterwards.
        struct NoisyMeasure
        {
            QubitList qubits;
            std::size_t after = 0; // ops index past the instruction
            Span span;
            std::string name;
        };

        void block(const std::vector<Instr>& body)
        {
            for (const Instr& in : body)
            {
                if (in.name != "ELSE_CORRELATED_ERROR") flushChain();
                if (++steps > kMaxUnrolledOps)
                    throw Fail{"E9004", in.span, std::format("unrolling runs more than {} instructions", kMaxUnrolledOps)};
                if (in.name == "REPEAT")
                {
                    for (std::uint64_t k = 0; k < in.repeat; ++k)
                    {
                        if (++steps > kMaxUnrolledOps)
                            throw Fail{"E9004", in.span, std::format("unrolling runs more than {} instructions", kMaxUnrolledOps)};
                        block(in.body);
                        flushChain();
                    }
                    continue;
                }
                try
                {
                    instruction(in);
                }
                catch (const ImportError& e)
                {
                    throw Fail{e.code(), in.span, e.what()};
                }
                if (b.size() > kMaxUnrolledOps)
                    throw Fail{"E9004", in.span, std::format("the unrolled circuit exceeds {} operations", kMaxUnrolledOps)};
            }
        }

        Qubit qubit(const Target& t) const { return qubitOf.at(t.value); }

        void ignored(const Instr& in, std::string_view what)
        {
            if (!warned.insert(std::string(what)).second) return;
            diags.warning("W9002", in.span, std::format("{} has no effect on the simulation and is ignored (further occurrences are not reported)", what));
        }

        void requireArgs(const Instr& in, std::size_t lo, std::size_t hi) const
        {
            if (in.args.size() < lo || in.args.size() > hi)
            {
                const std::string want = lo == hi ? std::to_string(lo) : std::format("{} to {}", lo, hi);
                throw Fail{"E9001", in.span, std::format("{} takes {} parenthesised argument{}, got {}", in.name, want, hi == 1 ? "" : "s", in.args.size())};
            }
        }

        double probability(const Instr& in, double p) const
        {
            if (!(p >= 0.0 && p <= 1.0)) throw Fail{"E9001", in.span, std::format("probability {} outside [0, 1]", p)};
            return p;
        }

        void requireQubitTargets(const Instr& in, bool allowInverted) const
        {
            for (const Target& t : in.targets)
            {
                if (t.kind != TK::Qubit) throw Fail{"E9001", t.span, std::format("{} takes qubit targets", in.name)};
                if (t.inverted && !allowInverted) throw Fail{"E9001", t.span, std::format("{} cannot invert a target", in.name)};
            }
        }

        void requirePairs(const Instr& in) const
        {
            if (in.targets.size() % 2 != 0)
                throw Fail{"E9006", in.span, std::format("{} acts on pairs of targets, got {}", in.name, in.targets.size())};
        }

        std::size_t record() { return clbits++; }

        Condition recCondition(const Target& t) const
        {
            if (static_cast<std::uint64_t>(t.value) > clbits)
                throw Fail{"E9001", t.span, std::format("rec[-{}] reaches before the first measurement ({} recorded so far)", t.value, clbits)};
            const Outcome bit = Outcome{1} << (clbits - static_cast<std::size_t>(t.value));
            return Condition{bit, bit};
        }

        Outcome recMask(const Instr& in) const
        {
            Outcome m = 0;
            for (const Target& t : in.targets)
            {
                if (t.kind != TK::Rec) throw Fail{"E9001", t.span, std::format("{} takes rec[-k] targets", in.name)};
                m ^= recCondition(t).mask;
            }
            return m;
        }

        void instruction(const Instr& in)
        {
            const std::string& n = in.name;
            if (!known(n)) throw Fail{"E9002", in.span, std::format("unknown Stim instruction `{}`", n)};
            if (n == "TICK" || n == "QUBIT_COORDS" || n == "SHIFT_COORDS") return; // layout and timing metadata
            if (n == "MPAD" || n.starts_with("HERALDED"))
                throw Fail{"E9001", in.span, std::format("{} is not supported (it needs heralded or padded records)", n)};
            if (n == "DETECTOR")
            {
                c.detectors.push_back(recMask(in));
                return;
            }
            if (n == "OBSERVABLE_INCLUDE")
            {
                requireArgs(in, 1, 1);
                const double k = in.args[0];
                if (!(k >= 0) || k != std::floor(k) || k > 1e6) throw Fail{"E9001", in.span, "OBSERVABLE_INCLUDE needs an observable index"};
                const auto idx = static_cast<std::size_t>(k);
                if (c.observables.size() <= idx) c.observables.resize(idx + 1, 0);
                c.observables[idx] ^= recMask(in);
                return;
            }
            if (const auto it = oneQubitRecipes().find(n); it != oneQubitRecipes().end())
            {
                requireArgs(in, 0, 0);
                requireQubitTargets(in, false);
                for (const Target& t : in.targets) b.apply(recipe(it->second, {qubit(t)}));
                return;
            }
            if (twoQubitRecipes().contains(n) || sqrtPairGates().contains(n))
            {
                requireArgs(in, 0, 0);
                requirePairs(in);
                for (std::size_t k = 0; k < in.targets.size(); k += 2) pair(in, in.targets[k], in.targets[k + 1]);
                return;
            }
            if (const auto it = measureFamily().find(n); it != measureFamily().end())
            {
                const MeasureSpec spec = it->second;
                requireArgs(in, 0, spec.measures ? 1 : 0);
                requireQubitTargets(in, spec.measures);
                const double flip = in.args.empty() ? 0.0 : probability(in, in.args[0]);
                for (const Target& t : in.targets)
                {
                    const Qubit q = qubit(t);
                    const std::vector<PauliFactor> f{{spec.basis, q}};
                    if (spec.measures && !spec.resets)
                    {
                        b.measurePauli(f, record(), t.inverted, flip);
                        if (flip > 0) noisy.push_back({{q}, b.size(), in.span, n});
                    }
                    else
                    {
                        // Measure (optionally) and reset in the basis: V, measure Z, reset, V†.
                        changeBasis(spec.basis, q, false);
                        if (spec.measures) b.measure(q, record(), t.inverted, flip);
                        b.reset(q);
                        changeBasis(spec.basis, q, true);
                    }
                }
                return;
            }
            if (n == "MPP" || n == "SPP" || n == "SPP_DAG")
            {
                requireArgs(in, 0, n == "MPP" ? 1 : 0);
                const double flip = in.args.empty() ? 0.0 : probability(in, in.args[0]);
                for (std::size_t k = 0; k < in.targets.size();)
                {
                    std::vector<PauliFactor> product;
                    bool inverted = false;
                    std::size_t j = k;
                    do
                    {
                        const Target& t = in.targets[j];
                        if (t.kind != TK::Pauli) throw Fail{"E9001", t.span, std::format("{} takes Pauli targets such as X0*Z1", n)};
                        inverted ^= t.inverted;
                        product.push_back({t.pauli, qubit(t)});
                        ++j;
                    } while (j < in.targets.size() && in.targets[j].joined);
                    if (n == "MPP")
                    {
                        b.measurePauli(product, record(), inverted, flip);
                        if (flip > 0) noisy.push_back({qubitsOf(product), b.size(), in.span, n});
                    }
                    else b.rotatePauli(product, (n == "SPP_DAG") != inverted); // SPP of −P is SPP_DAG of P
                    k = j;
                }
                return;
            }
            if (n == "MXX" || n == "MYY" || n == "MZZ")
            {
                requireArgs(in, 0, 1);
                requirePairs(in);
                requireQubitTargets(in, true);
                const double flip = in.args.empty() ? 0.0 : probability(in, in.args[0]);
                const char letter = n[1];
                for (std::size_t k = 0; k < in.targets.size(); k += 2)
                {
                    const std::vector<PauliFactor> product{{letter, qubit(in.targets[k])}, {letter, qubit(in.targets[k + 1])}};
                    b.measurePauli(product, record(), in.targets[k].inverted != in.targets[k + 1].inverted, flip);
                    if (flip > 0) noisy.push_back({qubitsOf(product), b.size(), in.span, n});
                }
                return;
            }
            noise(in);
        }

        static QubitList qubitsOf(const std::vector<PauliFactor>& product)
        {
            QubitList out;
            for (const PauliFactor& f : product) out.push_back(f.qubit);
            return out;
        }

        void changeBasis(char basis, Qubit q, bool undo)
        {
            // V maps the basis onto Z: X by H, Y by S† then H; V† undoes it.
            if (basis == 'X') b.apply({makeGate(Prim::H, {q})});
            else if (basis == 'Y') b.apply(undo ? Seq{makeGate(Prim::H, {q}), makeGate(Prim::S, {q})} : Seq{makeGate(Prim::Sdg, {q}), makeGate(Prim::H, {q})});
        }

        void pair(const Instr& in, const Target& a, const Target& t)
        {
            const std::string& n = in.name;
            const bool ca = a.kind == TK::Rec || a.kind == TK::Sweep, ct = t.kind == TK::Rec || t.kind == TK::Sweep;
            if (ca || ct)
            {
                // Classically controlled Pauli: the record (or sweep bit) sits on the Z-control side.
                const bool zFirst = n == "CX" || n == "CNOT" || n == "ZCX" || n == "CY" || n == "ZCY" || n == "CZ" || n == "ZCZ";
                const bool zSecond = n == "XCZ" || n == "YCZ" || n == "CZ" || n == "ZCZ";
                const Target* bitT = nullptr;
                const Target* qT = nullptr;
                char letter = 'Z';
                if (ca && !ct && zFirst)
                {
                    bitT = &a;
                    qT = &t;
                    letter = n == "CX" || n == "CNOT" || n == "ZCX" ? 'X' : n == "CY" || n == "ZCY" ? 'Y' : 'Z';
                }
                else if (ct && !ca && zSecond)
                {
                    bitT = &t;
                    qT = &a;
                    letter = n == "XCZ" ? 'X' : n == "YCZ" ? 'Y' : 'Z';
                }
                else throw Fail{"E9001", (ca ? a : t).span, std::format("{} cannot take a classical control in this position", n)};
                if (qT->kind != TK::Qubit || qT->inverted) throw Fail{"E9001", qT->span, "expected a qubit target"};
                if (bitT->kind == TK::Sweep)
                {
                    ignored(in, "a sweep[k] control (treated as 0, Stim's default)");
                    return;
                }
                const Prim p = letter == 'X' ? Prim::X : letter == 'Y' ? Prim::Y : Prim::Z;
                b.apply({makeGate(p, {qubit(*qT)})}, recCondition(*bitT));
                return;
            }
            if (a.kind != TK::Qubit || t.kind != TK::Qubit || a.inverted || t.inverted)
                throw Fail{"E9001", (a.kind != TK::Qubit || a.inverted ? a : t).span, std::format("{} takes qubit targets", n)};
            const Qubit qa = qubit(a), qt = qubit(t);
            if (qa == qt) throw Fail{"E9006", a.span, std::format("{} on qubit {} twice", n, a.value)};
            if (const auto it = sqrtPairGates().find(n); it != sqrtPairGates().end())
            {
                b.rotatePauli({{it->second.first, qa}, {it->second.first, qt}}, it->second.second);
                return;
            }
            b.apply(recipe(twoQubitRecipes().at(n), {qa, qt}));
        }

        void noise(const Instr& in)
        {
            const std::string& n = in.name;
            if (n == "I_ERROR" || n == "II_ERROR") return;
            if (n == "E" || n == "CORRELATED_ERROR" || n == "ELSE_CORRELATED_ERROR")
            {
                requireArgs(in, 1, 1);
                const double p = probability(in, in.args[0]);
                std::vector<PauliFactor> product;
                for (const Target& t : in.targets)
                {
                    if (t.kind != TK::Pauli || t.inverted || t.joined) throw Fail{"E9001", t.span, std::format("{} takes Pauli targets such as X1 Y2", n)};
                    product.push_back({t.pauli, qubit(t)});
                }
                if (n == "ELSE_CORRELATED_ERROR")
                {
                    if (!chain) throw Fail{"E9001", in.span, "ELSE_CORRELATED_ERROR must follow E or another ELSE_CORRELATED_ERROR"};
                }
                else chain = Chain{in.span, {}};
                chain->terms.emplace_back(p, std::move(product));
                return;
            }
            requireQubitTargets(in, false);
            const bool two = n == "DEPOLARIZE2" || n == "PAULI_CHANNEL_2";
            if (two) requirePairs(in);
            std::vector<double> probs;
            if (n == "PAULI_CHANNEL_1" || n == "PAULI_CHANNEL_2")
            {
                const std::size_t terms = two ? 15 : 3;
                requireArgs(in, terms, terms);
                double total = 0.0;
                for (const double p : in.args) total += probability(in, p);
                if (total > 1.0 + 1e-12) throw Fail{"E9001", in.span, std::format("{} probabilities sum to {} > 1", n, total)};
                probs = in.args;
            }
            else
            {
                requireArgs(in, 1, 1);
                const double p = probability(in, in.args[0]);
                if (n == "X_ERROR") probs = {p, 0, 0};
                else if (n == "Y_ERROR") probs = {0, p, 0};
                else if (n == "Z_ERROR") probs = {0, 0, p};
                else if (n == "DEPOLARIZE1") probs.assign(3, p / 3);
                else probs.assign(15, p / 15);
            }
            if (two)
                for (std::size_t k = 0; k < in.targets.size(); k += 2)
                {
                    if (in.targets[k].value == in.targets[k + 1].value) throw Fail{"E9006", in.targets[k].span, std::format("{} on qubit {} twice", n, in.targets[k].value)};
                    b.pauliChannel({qubit(in.targets[k]), qubit(in.targets[k + 1])}, probs);
                }
            else
                for (const Target& t : in.targets) b.pauliChannel({qubit(t)}, probs);
        }

        // One categorical draw: term j fires with q_j = p_j·∏_{i<j}(1 − p_i).
        void flushChain()
        {
            if (!chain) return;
            const Chain ch = std::move(*chain);
            chain.reset();
            try
            {
                lowerChain(ch);
            }
            catch (const ImportError& e)
            {
                throw Fail{e.code(), ch.span, e.what()};
            }
        }

        void lowerChain(const Chain& ch)
        {
            QubitList support;
            for (const auto& [p, product] : ch.terms)
                for (const PauliFactor& f : product)
                    if (std::ranges::find(support, f.qubit) == support.end()) support.push_back(f.qubit);
            if (support.empty()) return;

            std::vector<double> q;
            double remaining = 1.0;
            for (const auto& [p, product] : ch.terms)
            {
                q.push_back(p * remaining);
                remaining *= 1.0 - p;
            }
            // Letters of each term on the support: I X Y Z = 0 1 2 3, support[0] the high digit.
            auto digits = [&](const std::vector<PauliFactor>& product)
            {
                std::vector<int> d(support.size(), 0);
                for (const PauliFactor& f : product)
                {
                    const auto pos = static_cast<std::size_t>(std::ranges::find(support, f.qubit) - support.begin());
                    const int letter = f.letter == 'X' ? 1 : f.letter == 'Y' ? 2 : 3;
                    if (d[pos] != 0) throw Fail{"E9006", ch.span, std::format("a correlated error names qubit {} twice", f.qubit)};
                    d[pos] = letter;
                }
                return d;
            };

            if (support.size() <= 2)
            {
                std::vector<double> probs(support.size() == 1 ? 3 : 15, 0.0);
                for (std::size_t j = 0; j < ch.terms.size(); ++j)
                {
                    const std::vector<int> d = digits(ch.terms[j].second);
                    std::size_t index = 0;
                    for (const int x : d) index = index * 4 + static_cast<std::size_t>(x);
                    if (index > 0) probs[index - 1] += q[j];
                }
                b.pauliChannel(support, probs);
                return;
            }
            if (support.size() > Qputer::QuantumGate::kMaxDenseTargets)
                throw Fail{"E9004", ch.span, std::format("a correlated error on {} qubits needs Kraus operators on at most {}", support.size(),
                                                         Qputer::QuantumGate::kMaxDenseTargets)};
            // Exact as Kraus operators {√q_I·I, √q_j·P_j}: P_j is unitary, so branch j has probability q_j.
            using cd = std::complex<double>;
            const auto dim = Eigen::Index{1} << support.size();
            std::vector<Eigen::MatrixXcd> ops;
            double qI = 1.0;
            for (const double x : q) qI -= x;
            ops.push_back(std::sqrt(std::max(qI, 0.0)) * Eigen::MatrixXcd::Identity(dim, dim));
            for (std::size_t j = 0; j < ch.terms.size(); ++j)
            {
                Eigen::MatrixXcd P = Eigen::MatrixXcd::Ones(1, 1);
                for (const int letter : digits(ch.terms[j].second))
                {
                    Eigen::MatrixXcd s(2, 2);
                    if (letter == 0) s << 1, 0, 0, 1;
                    else if (letter == 1) s << 0, 1, 1, 0;
                    else if (letter == 2) s << 0, cd(0, -1), cd(0, 1), 0;
                    else s << 1, 0, 0, -1;
                    Eigen::MatrixXcd next(P.rows() * 2, P.cols() * 2);
                    for (Eigen::Index r = 0; r < P.rows(); ++r)
                        for (Eigen::Index k = 0; k < P.cols(); ++k) next.block(r * 2, k * 2, 2, 2) = P(r, k) * s;
                    P = next;
                }
                ops.push_back(std::sqrt(q[j]) * P);
            }
            b.kraus(support, std::move(ops));
        }

        void checkRecordFlips()
        {
            if (noisy.empty()) return;
            std::vector<std::size_t> lastUse(c.numQubits, 0);
            for (std::size_t k = 0; k < c.ops.size(); ++k)
            {
                for (const Qubit q : c.ops[k].controls) lastUse[q] = k + 1;
                for (const Qubit q : c.ops[k].targets) lastUse[q] = k + 1;
            }
            for (const NoisyMeasure& m : noisy)
                for (const Qubit q : m.qubits)
                    if (lastUse[q] > m.after)
                        throw Fail{"E9003", m.span,
                                   std::format("{} with a result-flip probability is exact only if the measured qubits stay idle afterwards, "
                                               "but {} is used again (the core has no classical noise)", m.name, c.qubitLabels[q])};
        }

        ImportedCircuit& c;
        CircuitBuilder b;
        Diagnostics& diags;
        std::map<std::int64_t, Qubit> qubitOf;
        std::size_t clbits = 0;
        std::size_t steps = 0; // instructions and REPEAT iterations run, so empty loops cannot spin
        std::optional<Chain> chain;
        std::vector<NoisyMeasure> noisy;
        std::set<std::string> warned;
    };
} // namespace


std::optional<ImportedCircuit> importStim(const SourceFile& file, std::uint32_t fileId, Diagnostics& diags)
{
    try
    {
        const std::vector<Instr> program = Parser(file, fileId).parse();
        ImportedCircuit c;
        c.format = "stim";
        c.source = Json::object();
        c.source["framework"] = "stim";

        const std::uint64_t results = countResults(program);
        if (results > Qputer::QuantumStateMachine::kMaxClbits)
        {
            std::uint64_t seen = 0;
            const Instr* at = resultAt(program, seen, Qputer::QuantumStateMachine::kMaxClbits);
            throw Fail{"E9004", at ? at->span : Span{fileId, 0, 0},
                       std::format("the circuit records {} measurement results; the classical register holds at most {} (this "
                                   "instruction records result {})",
                                   results == UINT64_MAX ? std::string("over 2^64") : std::to_string(results),
                                   Qputer::QuantumStateMachine::kMaxClbits, Qputer::QuantumStateMachine::kMaxClbits + 1)};
        }
        c.numClbits = static_cast<std::size_t>(results);
        if (results > 0) c.clbitRegisters.push_back({"rec", 0, c.numClbits});

        std::set<std::int64_t> ids;
        collectQubits(program, ids);
        if (ids.size() > Qputer::kMaxStabilizerQubits)
            throw Fail{"E9004", Span{fileId, 0, 0}, std::format("{} qubits; the simulator holds at most {}", ids.size(), Qputer::kMaxStabilizerQubits)};
        std::map<std::int64_t, Qubit> qubitOf;
        for (const std::int64_t id : ids)
        {
            qubitOf[id] = c.qubitLabels.size();
            c.qubitLabels.push_back(std::format("q{}", id));
        }
        c.numQubits = std::max<std::size_t>(ids.size(), 1);
        if (ids.empty()) c.qubitLabels.push_back("q0");

        Lowerer(c, diags, std::move(qubitOf)).run(program);
        return c;
    }
    catch (const Fail& f)
    {
        diags.error(f.code, f.span, f.message);
    }
    catch (const ImportError& e)
    {
        diags.error(e.code(), Span{fileId, 0, 0}, e.what());
    }
    return std::nullopt;
}

std::vector<std::string_view> stimGateNames()
{
    std::vector<std::string_view> out;
    for (const auto& [n, r] : oneQubitRecipes()) out.push_back(n);
    for (const auto& [n, r] : twoQubitRecipes()) out.push_back(n);
    for (const auto& [n, r] : sqrtPairGates()) out.push_back(n);
    out.insert(out.end(), {"SPP", "SPP_DAG"});
    return out;
}

} // namespace Noether::Interop
