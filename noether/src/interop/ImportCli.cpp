#include "interop/ImportCli.hpp"

#include "Cli.hpp"
#include "Executor.hpp"
#include "interop/CircuitJson.hpp"
#include "interop/Import.hpp"
#include "interop/Lowering.hpp"
#include "interop/Run.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <format>
#include <ostream>
#include <set>



namespace Noether::Interop
{

namespace
{
    using Clock = std::chrono::steady_clock;

    struct Usage
    {
        std::string message;
    };

    // Options that take a value, Noether's and the importer's, so that their values are never taken
    // for the source file.
    const std::set<std::string, std::less<>>& valuedOptions()
    {
        static const std::set<std::string, std::less<>> s{
            "--timeout", "--max-mem", "--top", "--seed", "--set", "--backend", "--deny", "--minimize", "--maximize", "--of",
            "--method", "--restarts", "--phase", "--install", "--iters", "--out", "--params", "--name", "--cost", "--dir",
            "--spec", "--candidate", "--record", "--from", "--holdout", "--shots", "--param", "--emit", "--format"};
        return s;
    }

    bool isImportCommand(const std::vector<std::string>& args)
    {
        if (args.empty()) return false;
        if (args[0] == "import") return true;
        if (args[0] != "run" && args[0] != "check") return false;
        for (std::size_t k = 1; k < args.size(); ++k)
        {
            if (args[k] == "--format") return true;
            if (valuedOptions().contains(args[k]))
            {
                ++k;
                continue;
            }
            if (!args[k].starts_with("-")) return !formatForPath(args[k]).empty();
        }
        return false;
    }

    template <class T>
    T parseNumber(const std::string& option, const std::string& v)
    {
        T x{};
        const auto r = std::from_chars(v.data(), v.data() + v.size(), x);
        if (r.ec != std::errc{} || r.ptr != v.data() + v.size()) throw Usage{std::format("bad {} {}", option, v)};
        return x;
    }

    class ImportCli
    {
        public:
        ImportCli(std::ostream& o, std::ostream& e) : out(o), err(e) {}

        int run(const std::vector<std::string>& args)
        {
            try
            {
                cmd = args[0];
                parseOptions(args);
                if (positional.size() != 1) throw Usage{std::format("`noether {}` needs one source file", cmd)};
                if (cmd == "run") return runCmd();
                if (cmd == "check") return check();
                return importCmd();
            }
            catch (const Usage& u)
            {
                if (json)
                {
                    Json j = Json::object();
                    j["schema"] = "noether.error/1";
                    j["version"] = kVersion;
                    j["ok"] = false;
                    j["error"] = u.message;
                    out << j.dump() << "\n";
                }
                else err << "noether: " << u.message << "\n";
                return 2;
            }
        }

        private:
        void parseOptions(const std::vector<std::string>& args)
        {
            for (std::size_t k = 1; k < args.size(); ++k)
            {
                const std::string& a = args[k];
                if (a == "--json") json = true;
                else if (a == "--no-timing") timing = false;
                else if (a == "--no-format") continue; // imported files are never rewritten
                else if (valuedOptions().contains(a))
                {
                    if (k + 1 >= args.size()) throw Usage{std::format("{} needs a value", a)};
                    const std::string& v = args[++k];
                    if (a == "--shots")
                    {
                        run_.shots = parseNumber<std::size_t>(a, v);
                        if (run_.shots == 0) throw Usage{"--shots must be positive"};
                    }
                    else if (a == "--seed")
                    {
                        const auto s = parseNumber<std::uint64_t>(a, v);
                        if (s >= (std::uint64_t{1} << 53)) throw Usage{std::format("bad --seed {}; expected an integer in [0, 2^53)", v)};
                        run_.seed = s;
                    }
                    else if (a == "--backend")
                    {
                        if (v != "auto" && v != "statevector" && v != "stabilizer")
                            throw Usage{std::format("unknown backend {}; expected auto, statevector or stabilizer", v)};
                        run_.backend = v;
                    }
                    else if (a == "--param")
                    {
                        const auto eq = v.find('=');
                        if (eq == std::string::npos || eq == 0) throw Usage{"--param needs name=value"};
                        imp.params[v.substr(0, eq)] = parseNumber<double>(a, v.substr(eq + 1));
                    }
                    else if (a == "--emit") emit = v;
                    else if (a == "--format")
                    {
                        if (v != "qasm" && v != "qasm2" && v != "qasm3" && v != "stim" && v != "circuit")
                            throw Usage{std::format("unknown format {}; expected qasm, qasm2, qasm3, stim or circuit", v)};
                        imp.format = v;
                    }
                    else if (a == "--top") top = parseNumber<std::size_t>(a, v);
                    else if (a == "--deny")
                    {
                        if (v != "warnings") throw Usage{"--deny takes `warnings`"};
                        denyWarnings = true;
                    }
                    else throw Usage{std::format("{} does not apply to `noether {}` on an imported circuit", a, cmd)};
                }
                else if (a.starts_with("--")) throw Usage{std::format("unknown option {}", a)};
                else positional.push_back(a);
            }
            const std::set<std::string> emits = cmd == "import" ? std::set<std::string>{"circuit"}
                                                : cmd == "run"  ? std::set<std::string>{"counts", "probabilities", "statevector"}
                                                                : std::set<std::string>{};
            if (emit.empty()) emit = cmd == "import" ? "circuit" : "counts";
            else if (!emits.contains(emit))
                throw Usage{cmd == "check" ? "`noether check` takes no --emit"
                                           : std::format("--emit {} does not apply to `noether {}`", emit, cmd)};
            run_.emit = emit;
        }

        ImportResult load(double* ms)
        {
            const auto t0 = Clock::now();
            ImportResult r = importFile(positional[0], imp);
            *ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            if (r.readFailed) throw Usage{r.readError};
            if (denyWarnings)
            {
                r.diags->denyWarnings(true);
                if (r.diags->hasErrors()) r.circuit.reset();
            }
            return r;
        }

        Json header(std::string_view kind, bool ok, const ImportResult& r) const
        {
            Json j = documentHeader(kind, ok, *r.diags);
            j["format"] = r.circuit ? Json(r.circuit->format) : Json(imp.format.empty() ? formatForPath(positional[0]) : imp.format);
            j["sourceSha256"] = r.sha256;
            return j;
        }

        int fail(std::string_view kind, const ImportResult& r)
        {
            if (json) out << header(kind, false, r).dump() << "\n";
            else err << r.diags->render();
            return 1;
        }

        void warnings(const ImportResult& r)
        {
            if (!json)
                if (const std::string d = r.diags->render(); !d.empty()) err << d;
        }

        std::string countsKey(Outcome value, std::size_t width) const
        {
            std::string key(width, '0');
            for (std::size_t c = 0; c < width; ++c)
                if ((value >> c) & 1U) key[c] = '1';
            return key;
        }

        int runCmd()
        {
            double importMs = 0.0;
            ImportResult r = load(&importMs);
            if (!r.circuit) return fail("import-run", r);
            const ImportedCircuit& c = *r.circuit;
            RunResult res;
            const auto t0 = Clock::now();
            try
            {
                res = runCircuit(c, run_);
            }
            catch (const ImportError& e)
            {
                r.diags->error(e.code(), Span{UINT32_MAX, 0, 0}, e.what());
                return fail("import-run", r);
            }
            const double runMs = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

            Json j = header("import-run", true, r);
            j["numQubits"] = c.numQubits;
            j["numClbits"] = c.numClbits;
            Json labels = Json::array();
            for (const std::string& l : c.qubitLabels) labels.push(l);
            j["qubitLabels"] = labels;
            Json regs = Json::array();
            for (const ClbitRegister& reg : c.clbitRegisters)
            {
                Json x = Json::object();
                x["name"] = reg.name;
                x["offset"] = reg.offset;
                x["size"] = reg.size;
                regs.push(std::move(x));
            }
            j["clbitRegisters"] = regs;
            j["backend"] = res.backend;
            j["backendReason"] = res.backendReason;
            j["method"] = res.method;
            j["seed"] = res.seed;
            std::string human = std::format("{}: {} qubits, {} clbits; backend {} ({}); seed {}\n", c.format, c.numQubits, c.numClbits,
                                            res.backend, res.backendReason, res.seed);

            if (emit == "statevector")
            {
                j["basisOrder"] = "q0-lsb";
                Json amps = Json::array();
                for (Eigen::Index k = 0; k < res.amplitudes.size(); ++k)
                {
                    Json z = Json::array();
                    z.push(res.amplitudes[k].real());
                    z.push(res.amplitudes[k].imag());
                    amps.push(std::move(z));
                }
                human += std::format("amplitudes (index bit q = qubit q): {}\n", amps.dump(-1));
                j["amplitudes"] = std::move(amps);
            }
            else if (emit == "probabilities")
            {
                j["basisOrder"] = "q0-lsb";
                Json ps = Json::array();
                for (Eigen::Index k = 0; k < res.probabilities.size(); ++k) ps.push(res.probabilities[k]);
                human += std::format("probabilities (index bit q = qubit q): {}\n", ps.dump(-1));
                j["probabilities"] = std::move(ps);
            }
            else
            {
                j["shots"] = run_.shots;
                j["bitOrder"] = "c0-left";
                std::vector<std::pair<Outcome, std::size_t>> rows(res.counts.begin(), res.counts.end());
                if (top > 0 && rows.size() > top)
                {
                    std::ranges::stable_sort(rows, [](const auto& a, const auto& b) { return a.second > b.second; });
                    rows.resize(top);
                }
                Json counts = Json::object();
                for (const auto& [value, n] : rows) counts[countsKey(value, c.numClbits)] = n;
                human += std::format("{} shots; counts (clbit 0 leftmost): {}\n", run_.shots, counts.dump(-1));
                j["counts"] = std::move(counts);
                auto rates = [&](const std::vector<std::size_t>& hits, std::string_view what, std::string_view field)
                {
                    Json a = Json::array();
                    for (std::size_t k = 0; k < hits.size(); ++k)
                    {
                        Json x = Json::object();
                        x["index"] = k;
                        x[field] = hits[k];
                        x["rate"] = static_cast<double>(hits[k]) / static_cast<double>(run_.shots);
                        a.push(std::move(x));
                        human += std::format("{} {}: {} of {} shots ({})\n", what, k, hits[k], run_.shots,
                                             jsonNumber(static_cast<double>(hits[k]) / static_cast<double>(run_.shots)));
                    }
                    return a;
                };
                if (!c.detectors.empty()) j["detectors"] = rates(res.detectorFires, "detector", "fires");
                if (!c.observables.empty()) j["observables"] = rates(res.observableFlips, "observable", "flips");
            }
            if (timing)
            {
                Json t = Json::object();
                t["importMs"] = importMs;
                t["runMs"] = runMs;
                j["timing"] = t;
            }
            if (json) out << j.dump() << "\n";
            else
            {
                warnings(r);
                out << human;
            }
            return 0;
        }

        int check()
        {
            double importMs = 0.0;
            ImportResult r = load(&importMs);
            if (!r.circuit) return fail("check", r);
            const ImportedCircuit& c = *r.circuit;
            Json j = header("check", true, r);
            j["numQubits"] = c.numQubits;
            j["numClbits"] = c.numClbits;
            j["ops"] = c.ops.size();
            if (json) out << j.dump() << "\n";
            else
            {
                warnings(r);
                out << std::format("ok: {}: {} qubits, {} clbits, {} operations\n", c.format, c.numQubits, c.numClbits, c.ops.size());
            }
            return 0;
        }

        int importCmd()
        {
            double importMs = 0.0;
            ImportResult r = load(&importMs);
            if (!r.circuit) return fail("import", r);
            if (!json) warnings(r);
            out << circuitJson(*r.circuit, *r.diags).dump() << "\n";
            return 0;
        }

        std::ostream& out;
        std::ostream& err;
        std::string cmd;
        std::vector<std::string> positional;
        bool json = false, timing = true, denyWarnings = false;
        std::string emit;
        std::size_t top = 0;
        ImportOptions imp;
        RunOptions run_;
    };
} // namespace

int runCli(const std::vector<std::string>& args, std::ostream& out, std::ostream& err)
{
    if (!isImportCommand(args)) return Noether::runCli(args, out, err);
    try
    {
        return ImportCli(out, err).run(args);
    }
    catch (const std::exception& e)
    {
        err << "noether: internal error: " << e.what() << "\n";
        return 5;
    }
}

} // namespace Noether::Interop
