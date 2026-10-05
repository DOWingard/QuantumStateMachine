#include "interop/CircuitJson.hpp"

#include "interop/Lowering.hpp"

#include <StabilizerState.hpp>

#include <algorithm>
#include <charconv>
#include <complex>
#include <format>
#include <functional>



namespace Noether::Interop
{

namespace
{
    using cd = std::complex<double>;

    struct PathElem
    {
        std::string key; // empty for an array index
        std::size_t index = 0;
    };
    using Path = std::vector<PathElem>;

    Path operator+(Path p, std::string key)
    {
        p.push_back({std::move(key), 0});
        return p;
    }
    Path operator+(Path p, std::size_t index)
    {
        p.push_back({"", index});
        return p;
    }

    std::string pathText(const Path& p)
    {
        std::string s;
        for (const PathElem& e : p) s += e.key.empty() ? std::format("[{}]", e.index) : (s.empty() ? "" : ".") + e.key;
        return s.empty() ? "document" : s;
    }

    // A structural fault, reported once at the value the path names.
    struct Bad
    {
        Path path;
        std::string message;
        std::string code = "E9007";
        bool atKey = false; // point at the last element's key rather than its value
    };

    // Finds the byte range of the value at a path in JSON text that already parsed. Only spans
    // are computed here; values come from Noether::Json.
    class Locator
    {
        public:
        explicit Locator(std::string_view text) : t(text) {}

        std::pair<std::uint32_t, std::uint32_t> find(const Path& path, bool atKey = false)
        {
            i = 0;
            for (const PathElem& e : path)
                if (!descend(e)) return whole();
            if (atKey && !path.empty() && !path.back().key.empty()) return {static_cast<std::uint32_t>(keyBegin), static_cast<std::uint32_t>(keyEnd)};
            ws();
            const std::size_t b = i;
            if (!skip()) return whole();
            return {static_cast<std::uint32_t>(b), static_cast<std::uint32_t>(i)};
        }

        private:
        std::pair<std::uint32_t, std::uint32_t> whole() const { return {0, static_cast<std::uint32_t>(t.size())}; }
        bool at(char c) const { return i < t.size() && t[i] == c; }

        void ws()
        {
            while (i < t.size() && (t[i] == ' ' || t[i] == '\t' || t[i] == '\n' || t[i] == '\r')) ++i;
        }

        bool skipString()
        {
            if (!at('"')) return false;
            for (++i; i < t.size() && t[i] != '"'; ++i)
                if (t[i] == '\\') ++i;
            if (i >= t.size()) return false;
            ++i;
            return true;
        }

        bool skip()
        {
            ws();
            if (i >= t.size()) return false;
            if (at('"')) return skipString();
            if (at('{') || at('['))
            {
                int depth = 0;
                while (i < t.size())
                {
                    if (at('"'))
                    {
                        if (!skipString()) return false;
                        continue;
                    }
                    if (at('{') || at('[')) ++depth;
                    else if (at('}') || at(']')) --depth;
                    ++i;
                    if (depth == 0) return true;
                }
                return false;
            }
            while (i < t.size() && t[i] != ',' && t[i] != '}' && t[i] != ']' && t[i] != ' ' && t[i] != '\n' && t[i] != '\r' && t[i] != '\t') ++i;
            return true;
        }

        bool descend(const PathElem& e)
        {
            ws();
            const bool object = !e.key.empty();
            if (!at(object ? '{' : '[')) return false;
            ++i;
            for (std::size_t k = 0;; ++k)
            {
                ws();
                if (i >= t.size() || at('}') || at(']')) return false;
                if (object)
                {
                    const std::size_t kb = i;
                    if (!skipString()) return false;
                    const std::string_view key = t.substr(kb + 1, i - kb - 2);
                    const std::size_t ke = i;
                    ws();
                    if (!at(':')) return false;
                    ++i;
                    ws();
                    if (key == e.key)
                    {
                        keyBegin = kb;
                        keyEnd = ke;
                        return true;
                    }
                }
                else if (k == e.index) return true;
                if (!skip()) return false;
                ws();
                if (at(',')) ++i;
            }
        }

        std::string_view t;
        std::size_t i = 0;
        std::size_t keyBegin = 0, keyEnd = 0;
    };

    class Reader
    {
        public:
        Reader(Diagnostics& d, std::function<Span(const Path&)> loc) : diags(d), locate(std::move(loc)) {}

        ImportedCircuit read(const Json& doc)
        {
            ImportedCircuit c;
            c.format = "circuit";
            const Path root;
            requireKeys(doc, root, {"schema", "num_qubits", "num_clbits", "qubit_labels", "clbit_registers", "source", "ops", "notes",
                                    "detectors", "observables"});
            const Json& schema = member(doc, root, "schema");
            if (!schema.isString() || schema.asString() != kCircuitSchema)
                throw Bad{root + "schema", std::format("unsupported schema; expected \"{}\"", kCircuitSchema)};

            c.numQubits = index(member(doc, root, "num_qubits"), root + "num_qubits");
            if (c.numQubits == 0) throw Bad{root + "num_qubits", "a circuit needs at least one qubit"};
            if (c.numQubits > Qputer::kMaxStabilizerQubits)
                throw Bad{root + "num_qubits", std::format("{} qubits; the simulator holds at most {}", c.numQubits, Qputer::kMaxStabilizerQubits),
                          "E9004"};
            c.numClbits = doc.contains("num_clbits") ? index(*doc.find("num_clbits"), root + "num_clbits") : 0;
            if (c.numClbits > Qputer::QuantumStateMachine::kMaxClbits)
                throw Bad{root + "num_clbits",
                          std::format("{} clbits; the classical register holds at most {}", c.numClbits, Qputer::QuantumStateMachine::kMaxClbits),
                          "E9004"};

            if (const Json* labels = doc.find("qubit_labels"))
            {
                const Path p = root + "qubit_labels";
                const Json::Array& a = array(*labels, p);
                if (a.size() != c.numQubits) throw Bad{p, std::format("{} labels for {} qubits", a.size(), c.numQubits)};
                for (std::size_t k = 0; k < a.size(); ++k) c.qubitLabels.push_back(string(a[k], p + k));
            }
            else
                for (std::size_t q = 0; q < c.numQubits; ++q) c.qubitLabels.push_back(std::format("q[{}]", q));

            if (const Json* regs = doc.find("clbit_registers"))
            {
                const Path p = root + "clbit_registers";
                const Json::Array& a = array(*regs, p);
                for (std::size_t k = 0; k < a.size(); ++k)
                {
                    const Path rp = p + k;
                    requireKeys(a[k], rp, {"name", "offset", "size"});
                    ClbitRegister r;
                    r.name = string(member(a[k], rp, "name"), rp + "name");
                    r.offset = index(member(a[k], rp, "offset"), rp + "offset");
                    r.size = index(member(a[k], rp, "size"), rp + "size");
                    if (r.offset + r.size > c.numClbits)
                        throw Bad{rp, std::format("register `{}` [{}, {}) exceeds {} clbits", r.name, r.offset, r.offset + r.size, c.numClbits)};
                    c.clbitRegisters.push_back(std::move(r));
                }
            }
            if (const Json* src = doc.find("source"))
            {
                if (!src->isObject()) throw Bad{root + "source", "expected an object"};
                c.source = *src;
            }

            const Path opsPath = root + "ops";
            const Json::Array& ops = array(member(doc, root, "ops"), opsPath);
            c.ops.reserve(ops.size());
            for (std::size_t k = 0; k < ops.size(); ++k) c.ops.push_back(operation(ops[k], opsPath + k, c));

            if (const Json* notes = doc.find("notes"))
            {
                const Path p = root + "notes";
                const Json::Array& a = array(*notes, p);
                for (std::size_t k = 0; k < a.size(); ++k)
                {
                    const Path np = p + k;
                    requireKeys(a[k], np, {"code", "message"});
                    const std::string code = string(member(a[k], np, "code"), np + "code");
                    if (!code.starts_with("W9") || !findCode(code))
                        throw Bad{np + "code", std::format("`{}` is not an interop warning code (W9xxx)", code)};
                    diags.warning(code, locate(np), string(member(a[k], np, "message"), np + "message"));
                }
            }
            c.detectors = masks(doc, root, "detectors", c);
            c.observables = masks(doc, root, "observables", c);
            return c;
        }

        private:
        static const Json& member(const Json& obj, const Path& p, std::string_view key)
        {
            const Json* v = obj.find(key);
            if (!v) throw Bad{p, std::format("missing \"{}\"", key)};
            return *v;
        }

        static void requireKeys(const Json& obj, const Path& p, std::initializer_list<std::string_view> allowed)
        {
            if (!obj.isObject()) throw Bad{p, std::format("{}: expected an object", pathText(p))};
            for (const auto& [k, v] : obj.asObject())
                if (std::ranges::find(allowed, std::string_view(k)) == allowed.end())
                    throw Bad{p + k, std::format("unknown key \"{}\" in {}", k, pathText(p)), "E9007", true};
        }

        static const Json::Array& array(const Json& v, const Path& p)
        {
            if (!v.isArray()) throw Bad{p, std::format("{}: expected an array", pathText(p))};
            return v.asArray();
        }

        static std::string string(const Json& v, const Path& p)
        {
            if (!v.isString()) throw Bad{p, std::format("{}: expected a string", pathText(p))};
            return v.asString();
        }

        static std::size_t index(const Json& v, const Path& p)
        {
            if (!v.isInt() || v.asInt() < 0) throw Bad{p, std::format("{}: expected a non-negative integer", pathText(p))};
            return static_cast<std::size_t>(v.asInt());
        }

        static double number(const Json& v, const Path& p)
        {
            if (!v.isNumber()) throw Bad{p, std::format("{}: expected a number", pathText(p))};
            return v.asDouble();
        }

        static cd complex(const Json& v, const Path& p)
        {
            if (v.isNumber()) return {v.asDouble(), 0.0};
            if (v.isArray() && v.size() == 2) return {number(v.asArray()[0], p + 0), number(v.asArray()[1], p + 1)};
            throw Bad{p, std::format("{}: expected a number or [re, im]", pathText(p))};
        }

        static Eigen::MatrixXcd matrix(const Json& v, const Path& p)
        {
            const Json::Array& rows = array(v, p);
            const auto n = static_cast<Eigen::Index>(rows.size());
            Eigen::MatrixXcd m(n, n);
            for (Eigen::Index r = 0; r < n; ++r)
            {
                const Path rp = p + static_cast<std::size_t>(r);
                const Json::Array& row = array(rows[static_cast<std::size_t>(r)], rp);
                if (static_cast<Eigen::Index>(row.size()) != n) throw Bad{rp, std::format("row has {} entries, expected {}", row.size(), n)};
                for (Eigen::Index k = 0; k < n; ++k) m(r, k) = complex(row[static_cast<std::size_t>(k)], rp + static_cast<std::size_t>(k));
            }
            return m;
        }

        static QubitList indices(const Json& v, const Path& p)
        {
            QubitList out;
            const Json::Array& a = array(v, p);
            for (std::size_t k = 0; k < a.size(); ++k) out.push_back(index(a[k], p + k));
            return out;
        }

        static std::vector<Outcome> masks(const Json& doc, const Path& root, std::string_view key, const ImportedCircuit& c)
        {
            std::vector<Outcome> out;
            const Json* v = doc.find(key);
            if (!v) return out;
            const Path p = root + std::string(key);
            const Json::Array& a = array(*v, p);
            for (std::size_t k = 0; k < a.size(); ++k)
            {
                Outcome m = 0;
                for (const std::size_t b : indices(a[k], p + k))
                {
                    if (b >= c.numClbits) throw Bad{p + k, std::format("clbit {} out of range [0, {})", b, c.numClbits)};
                    m ^= Outcome{1} << b;
                }
                out.push_back(m);
            }
            return out;
        }

        Qputer::Operation operation(const Json& v, const Path& p, const ImportedCircuit& c)
        {
            requireKeys(v, p, {"op", "controls", "targets", "params", "matrix", "kraus", "clbit", "condition"});
            const std::string name = string(member(v, p, "op"), p + "op");
            const auto kind = Qputer::opKindFromName(name);
            if (!kind) throw Bad{p + "op", std::format("unknown operation \"{}\"", name)};
            QubitList controls = v.contains("controls") ? indices(*v.find("controls"), p + "controls") : QubitList{};
            QubitList targets = v.contains("targets") ? indices(*v.find("targets"), p + "targets") : QubitList{};
            std::vector<double> params;
            if (const Json* ps = v.find("params"))
            {
                const Path pp = p + "params";
                const Json::Array& a = array(*ps, pp);
                for (std::size_t k = 0; k < a.size(); ++k) params.push_back(number(a[k], pp + k));
            }
            Eigen::MatrixXcd m;
            if (const Json* mj = v.find("matrix")) m = matrix(*mj, p + "matrix");
            std::vector<Eigen::MatrixXcd> kraus;
            if (const Json* kj = v.find("kraus"))
            {
                const Path kp = p + "kraus";
                const Json::Array& a = array(*kj, kp);
                for (std::size_t k = 0; k < a.size(); ++k) kraus.push_back(matrix(a[k], kp + k));
            }
            std::optional<std::size_t> clbit;
            if (const Json* cb = v.find("clbit")) clbit = index(*cb, p + "clbit");
            std::optional<Condition> condition;
            if (const Json* cj = v.find("condition"))
            {
                const Path cp = p + "condition";
                requireKeys(*cj, cp, {"clbits", "values"});
                const QubitList bits = indices(member(*cj, cp, "clbits"), cp + "clbits");
                const Path vp = cp + "values";
                const Json::Array& values = array(member(*cj, cp, "values"), vp);
                if (values.size() != bits.size()) throw Bad{cp, std::format("{} clbits but {} values", bits.size(), values.size())};
                Condition cond;
                for (std::size_t k = 0; k < bits.size(); ++k)
                {
                    const Json& x = values[k];
                    bool one = false;
                    if (x.isBool()) one = x.asBool();
                    else if (x.isInt() && (x.asInt() == 0 || x.asInt() == 1)) one = x.asInt() == 1;
                    else throw Bad{cp + "values" + k, "expected 0, 1, true or false"};
                    if (bits[k] >= 64) throw Bad{cp + "clbits" + k, std::format("clbit {} out of range [0, {})", bits[k], c.numClbits)};
                    const Outcome bit = Outcome{1} << bits[k];
                    if (cond.mask & bit) throw Bad{cp + "clbits" + k, std::format("clbit {} listed twice", bits[k])};
                    cond.mask |= bit;
                    if (one) cond.value |= bit;
                }
                condition = cond;
            }
            Qputer::Operation o = makeOperation(*kind, std::move(controls), std::move(targets), std::move(params), std::move(m),
                                                std::move(kraus), clbit, condition);
            if (const std::string why = validateOperation(o, c.numQubits, c.numClbits); !why.empty())
                throw Bad{p, std::format("ops[{}] ({}): {}", p.back().index, name, why)};
            return o;
        }

        Diagnostics& diags;
        std::function<Span(const Path&)> locate;
    };

    Json complexJson(cd z)
    {
        Json a = Json::array();
        a.push(z.real());
        a.push(z.imag());
        return a;
    }

    Json matrixJson(const Eigen::MatrixXcd& m)
    {
        Json rows = Json::array();
        for (Eigen::Index r = 0; r < m.rows(); ++r)
        {
            Json row = Json::array();
            for (Eigen::Index k = 0; k < m.cols(); ++k) row.push(complexJson(m(r, k)));
            rows.push(std::move(row));
        }
        return rows;
    }

    Json indexArray(const QubitList& qs)
    {
        Json a = Json::array();
        for (const Qubit q : qs) a.push(q);
        return a;
    }

    Json maskArray(Outcome mask)
    {
        Json a = Json::array();
        for (std::size_t b = 0; b < 64; ++b)
            if ((mask >> b) & 1U) a.push(b);
        return a;
    }
} // namespace


std::optional<ImportedCircuit> readCircuitJson(const SourceFile& file, std::uint32_t fileId, Diagnostics& diags)
{
    std::string error;
    const auto doc = Json::parse(file.text(), &error);
    if (!doc)
    {
        // Json::parse reports "offset N: message".
        std::uint32_t at = 0;
        if (error.starts_with("offset "))
        {
            const char* b = error.data() + 7;
            std::from_chars(b, error.data() + error.size(), at);
        }
        at = std::min<std::uint32_t>(at, static_cast<std::uint32_t>(file.text().size()));
        diags.error("E9007", Span{fileId, at, at}, std::format("malformed circuit JSON: {}", error));
        return std::nullopt;
    }
    Locator loc(file.text());
    auto locate = [&](const Path& p)
    {
        const auto [b, e] = loc.find(p);
        return Span{fileId, b, e};
    };
    try
    {
        return Reader(diags, locate).read(*doc);
    }
    catch (const Bad& b)
    {
        const auto [from, to] = loc.find(b.path, b.atKey);
        diags.error(b.code, Span{fileId, from, to}, b.message);
        return std::nullopt;
    }
}

Json circuitJson(const ImportedCircuit& c, const Diagnostics& diags)
{
    Json j = Json::object();
    j["schema"] = kCircuitSchema;
    j["num_qubits"] = c.numQubits;
    j["num_clbits"] = c.numClbits;
    Json labels = Json::array();
    for (const std::string& l : c.qubitLabels) labels.push(l);
    j["qubit_labels"] = labels;
    Json regs = Json::array();
    for (const ClbitRegister& r : c.clbitRegisters)
    {
        Json x = Json::object();
        x["name"] = r.name;
        x["offset"] = r.offset;
        x["size"] = r.size;
        regs.push(std::move(x));
    }
    j["clbit_registers"] = regs;
    j["source"] = c.source.isObject() ? c.source : Json::object();
    Json ops = Json::array();
    for (const Qputer::Operation& o : c.ops)
    {
        Json x = Json::object();
        x["op"] = Qputer::opName(o.kind);
        if (!o.controls.empty()) x["controls"] = indexArray(o.controls);
        x["targets"] = indexArray(o.targets);
        if (!o.params.empty())
        {
            Json ps = Json::array();
            for (const double v : o.params) ps.push(v);
            x["params"] = ps;
        }
        if (o.kind == Qputer::OpKind::Unitary) x["matrix"] = matrixJson(o.matrix);
        if (o.kind == Qputer::OpKind::Kraus)
        {
            Json ks = Json::array();
            for (const Eigen::MatrixXcd& K : o.kraus) ks.push(matrixJson(K));
            x["kraus"] = ks;
        }
        if (o.clbit) x["clbit"] = *o.clbit;
        if (o.condition)
        {
            Json cond = Json::object();
            Json bits = maskArray(o.condition->mask), values = Json::array();
            for (const Json& b : bits.asArray()) values.push(static_cast<int>((o.condition->value >> b.asInt()) & 1U));
            cond["clbits"] = bits;
            cond["values"] = values;
            x["condition"] = cond;
        }
        ops.push(std::move(x));
    }
    j["ops"] = ops;
    Json notes = Json::array();
    for (const Diagnostic& d : diags.sorted())
    {
        if (d.severity != Severity::Warning) continue;
        Json n = Json::object();
        n["code"] = d.code;
        n["message"] = d.message;
        notes.push(std::move(n));
    }
    j["notes"] = notes;
    if (!c.detectors.empty())
    {
        Json ds = Json::array();
        for (const Outcome m : c.detectors) ds.push(maskArray(m));
        j["detectors"] = ds;
    }
    if (!c.observables.empty())
    {
        Json os = Json::array();
        for (const Outcome m : c.observables) os.push(maskArray(m));
        j["observables"] = os;
    }
    return j;
}

} // namespace Noether::Interop
