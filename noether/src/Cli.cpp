#include "Cli.hpp"

#include "Compiler.hpp"
#include "Executor.hpp"
#include "Estimate.hpp"
#include "Formatter.hpp"
#include "Lexer.hpp"
#include "Parser.hpp"
#include "Tools.hpp"

#include <charconv>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <iostream>
#include <ostream>
#include <set>



namespace Noether
{

namespace
{
    using Clock = std::chrono::steady_clock;

    constexpr std::string_view kUsage = R"(usage: noether <command> [options] [files]

commands:
  check f.ntr                  parse and analyse; report every diagnostic
  run f.ntr                    execute on the quantum state machine
  fmt f.ntr [-w] [--check] [--ascii]
                               print the canonical form (-w rewrites the file)
  fix f.ntr [--apply]          list or apply machine-applicable fix-its
  draw f.ntr                   circuit diagram, one wire per qubit
  ir f.ntr                     lowered IR (ops, spans, Clifford tags, params)
  estimate f.ntr               resources without running
  equiv a.ntr b.ntr [--phase ignore|exact]
                               semantic equivalence of two programs' operations
  opt f.ntr --minimize LABEL [--method nelder-mead|spsa|adam] [--restarts k]
                               optimise params against a labelled print
  grad f.ntr --of LABEL        gradient of a labelled print (parameter shift)
  qasm export f.ntr            OpenQASM 3 text
  run f.qasm|f.stim|f.json [--shots n] [--emit counts|probabilities|statevector] [--param name=value]
                               import an OpenQASM 2/3, Stim or circuit JSON file and run it
  check f.qasm|f.stim|f.json   import only; report diagnostics
  import f.qasm|f.stim|f.json  print the imported circuit as noether.circuit/1 JSON
                               (--format qasm|qasm2|qasm3|stim|circuit overrides the extension)
  eval [--dir D | --spec S --candidate C] [--record "description"]
                               score a candidate against a research task spec
  research init TAG --spec S [--from F] | status [--dir D] | report [--dir D] [--holdout F]
                               autoresearch workspace commands
  check algo.ntr --spec S      candidate diagnostics against a task spec
  explain CODE                 long explanation of a diagnostic
  grammar | tokens             EBNF grammar; token and alias table
  skill [--full] [--install DIR]
                               print or install the agent skills
  repl                         interactive session
  version

options:
  --json  --no-timing  --timeout 60s  --max-mem 8G  --top k  --seed n  --set name=value
  --params params.json  --backend auto|statevector|stabilizer  --deny warnings  --no-format

Source files are rewritten to canonical Unicode before compiling (check, run, draw, ir, estimate,
opt, grad, eval) when they parse cleanly; --no-format leaves them untouched.
)";

    std::optional<double> parseDuration(std::string_view s)
    {
        if (s.empty()) return std::nullopt;
        double scale = 1.0;
        const char last = s.back();
        if (last == 's') s.remove_suffix(1);
        else if (last == 'm') scale = 60.0, s.remove_suffix(1);
        else if (last == 'h') scale = 3600.0, s.remove_suffix(1);
        double v = 0.0;
        const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
        if (r.ec != std::errc{} || r.ptr != s.data() + s.size() || v <= 0) return std::nullopt;
        return v * scale;
    }

    std::optional<std::uint64_t> parseBytes(std::string_view s)
    {
        if (s.empty()) return std::nullopt;
        std::uint64_t scale = 1;
        const char last = s.back();
        if (last == 'K' || last == 'k') scale = std::uint64_t{1} << 10, s.remove_suffix(1);
        else if (last == 'M') scale = std::uint64_t{1} << 20, s.remove_suffix(1);
        else if (last == 'G') scale = std::uint64_t{1} << 30, s.remove_suffix(1);
        else if (last == 'T') scale = std::uint64_t{1} << 40, s.remove_suffix(1);
        double v = 0.0;
        const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
        if (r.ec != std::errc{} || r.ptr != s.data() + s.size() || v <= 0) return std::nullopt;
        return static_cast<std::uint64_t>(v * static_cast<double>(scale));
    }

    struct Usage
    {
        std::string message;
    };

    std::string readOrThrow(const std::string& path)
    {
        bool ok = false;
        std::string t = readFileText(path, ok);
        if (!ok) throw Usage{std::format("cannot read {}", path)};
        return t;
    }

    bool writeAtomically(const std::string& path, const std::string& text, std::string& error)
    {
        const std::filesystem::path p(path);
        const std::filesystem::path tmp = p.string() + ".noether-tmp";
        {
            std::ofstream o(tmp, std::ios::binary | std::ios::trunc);
            if (!o)
            {
                error = std::format("cannot write {}", tmp.string());
                return false;
            }
            o << text;
            if (!o)
            {
                error = std::format("cannot write {}", tmp.string());
                return false;
            }
        }
        std::error_code ec;
        const auto perms = std::filesystem::status(p, ec).permissions();
        if (!ec) std::filesystem::permissions(tmp, perms, ec);
        std::filesystem::rename(tmp, p, ec);
        if (ec)
        {
            std::filesystem::remove(tmp, ec);
            error = std::format("cannot replace {}", path);
            return false;
        }
        return true;
    }
} // namespace


bool canonicalizeFile(const std::string& path, std::string& error)
{
    bool ok = false;
    const std::string text = readFileText(path, ok);
    if (!ok) return false;
    SourceManager sm;
    Diagnostics d(sm);
    const std::uint32_t id = sm.add(path, text);
    const Program prog = parse(sm.file(id), id, d);
    if (d.hasErrors() || prog.stmts.empty()) return false;
    const std::string canonical = formatProgram(prog, sm.file(id), false);
    if (canonical == text) return false;

    // Safety gate: the canonical text must parse cleanly to the same tree.
    SourceManager sm2;
    Diagnostics d2(sm2);
    const std::uint32_t id2 = sm2.add(path, canonical);
    const Program prog2 = parse(sm2.file(id2), id2, d2);
    if (d2.hasErrors()) return false;
    if (formatProgram(prog2, sm2.file(id2), true) != formatProgram(prog, sm.file(id), true)) return false;
    return writeAtomically(path, canonical, error);
}


namespace
{
    class Cli
    {
        public:
        Cli(std::ostream& o, std::ostream& e) : out(o), err(e) {}

        int run(const std::vector<std::string>& args)
        {
            if (args.empty())
            {
                err << kUsage;
                return 2;
            }
            try
            {
                cmd = args[0];
                parseOptions(args);
                return dispatch();
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
            static const std::set<std::string> valued{"--timeout", "--max-mem", "--top", "--seed", "--set", "--backend", "--deny",
                                                      "--minimize", "--maximize", "--of", "--method", "--restarts", "--phase",
                                                      "--install", "--iters", "--out", "--params", "--name", "--cost",
                                                      "--dir", "--spec", "--candidate", "--record", "--from", "--holdout"};
            for (std::size_t k = 1; k < args.size(); ++k)
            {
                const std::string& a = args[k];
                if (a == "--json") json = true;
                else if (a == "--no-timing") timing = false;
                else if (a == "--no-format") noFormat = true;
                else if (a == "-w" || a == "--write") write = true;
                else if (a == "--check") checkOnly = true;
                else if (a == "--ascii") ascii = true;
                else if (a == "--apply") apply = true;
                else if (a == "--full") full = true;
                else if (a == "--force") force = true;
                else if (valued.contains(a))
                {
                    if (k + 1 >= args.size()) throw Usage{std::format("{} needs a value", a)};
                    const std::string& v = args[++k];
                    if (a == "--set")
                    {
                        const auto eq = v.find('=');
                        if (eq == std::string::npos) throw Usage{"--set needs name=value"};
                        copt.sets.emplace_back(v.substr(0, eq), v.substr(eq + 1));
                    }
                    else if (a == "--timeout")
                    {
                        timeout = parseDuration(v);
                        if (!timeout) throw Usage{std::format("bad --timeout {}", v)};
                    }
                    else if (a == "--max-mem")
                    {
                        const auto b = parseBytes(v);
                        if (!b) throw Usage{std::format("bad --max-mem {}", v)};
                        maxMem = *b;
                    }
                    else if (a == "--top")
                    {
                        std::size_t t = 0;
                        const auto r = std::from_chars(v.data(), v.data() + v.size(), t);
                        if (r.ec != std::errc{}) throw Usage{std::format("bad --top {}", v)};
                        top = t;
                    }
                    else if (a == "--seed")
                    {
                        std::uint64_t s = 0;
                        const auto r = std::from_chars(v.data(), v.data() + v.size(), s);
                        if (r.ec != std::errc{} || s >= (std::uint64_t{1} << 53))
                            throw Usage{std::format("bad --seed {}; expected an integer in [0, 2^53)", v)};
                        copt.seed = s;
                    }
                    else if (a == "--backend")
                    {
                        if (v != "auto" && v != "statevector" && v != "stabilizer")
                            throw Usage{std::format("unknown backend {}; expected auto, statevector or stabilizer", v)};
                        copt.backend = v;
                    }
                    else if (a == "--deny")
                    {
                        if (v != "warnings") throw Usage{"--deny takes `warnings`"};
                        denyWarnings = true;
                    }
                    else named[a.substr(2)] = v;
                }
                else if (a.starts_with("--")) throw Usage{std::format("unknown option {}", a)};
                else positional.push_back(a);
            }
            if (const auto it = named.find("params"); it != named.end()) loadParams(it->second);
        }

        // --params FILE: a noether.params/1 document (written by `opt` and `eval`) applied like
        // --set; explicit --set options given on the command line still win.
        void loadParams(const std::string& path)
        {
            bool ok = false;
            const std::string text = readFileText(path, ok);
            if (!ok) throw Usage{std::format("cannot read {}", path)};
            std::string error;
            const auto doc = Json::parse(text, &error);
            // `opt` writes {"params": {...}}; research `eval` writes {"instances": {name: {...}}}, of
            // which a single instance (or "default") is usable here.
            const Json* ps = doc && doc->isObject() ? doc->find("params") : nullptr;
            if (!ps && doc && doc->isObject())
                if (const Json* inst = doc->find("instances"); inst && inst->isObject())
                {
                    if (const Json* d = inst->find("default")) ps = d;
                    else if (inst->size() == 1) ps = &inst->asObject().begin()->second;
                    else throw Usage{std::format("{} holds params for several instances; pick one with --set", path)};
                }
            if (!ps || !ps->isObject()) throw Usage{std::format("{} is not a params document (expected {{\"params\": {{…}}}})", path)};
            std::vector<std::pair<std::string, std::string>> loaded;
            for (const auto& [name, v] : ps->asObject())
            {
                if (v.isNumber()) loaded.emplace_back(name, jsonNumber(v.asDouble()));
                else if (v.isArray())
                {
                    std::string list = "[";
                    for (const Json& x : v.asArray())
                    {
                        if (!x.isNumber()) throw Usage{std::format("{}: `{}` holds a non-number", path, name)};
                        list += (list.size() > 1 ? ", " : "") + jsonNumber(x.asDouble());
                    }
                    loaded.emplace_back(name, list + "]");
                }
                else throw Usage{std::format("{}: `{}` is neither a number nor a list", path, name)};
            }
            copt.sets.insert(copt.sets.begin(), loaded.begin(), loaded.end());
        }

        ExecOptions execOptions() const
        {
            ExecOptions o;
            o.timing = timing;
            o.timeoutSeconds = timeout;
            o.maxMemBytes = maxMem;
            o.top = top;
            return o;
        }

        const std::string& file(std::size_t k = 0) const
        {
            if (positional.size() <= k) throw Usage{std::format("`noether {}` needs a source file", cmd)};
            return positional[k];
        }

        // Canonicalises (unless --no-format), then compiles. Read failures are usage errors.
        std::unique_ptr<Compilation> compile(const std::string& path, double* ms = nullptr)
        {
            if (!noFormat)
            {
                std::string error;
                formatted = canonicalizeFile(path, error) || formatted;
                if (!error.empty()) err << "noether: " << error << "\n";
            }
            const auto t0 = Clock::now();
            auto c = compileFile(path, copt);
            if (ms) *ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            if (c->readFailed) throw Usage{c->readError};
            if (denyWarnings) c->diags->denyWarnings(true);
            return c;
        }

        int emit(const Json& j, const std::string& human)
        {
            if (json) out << j.dump() << "\n";
            else out << human;
            return 0;
        }

        int failCompile(Compilation& c, std::string_view kind)
        {
            if (json)
            {
                Json j = documentHeader(kind, false, *c.diags);
                out << j.dump() << "\n";
            }
            else err << c.diags->render();
            return 1;
        }

        int dispatch()
        {
            if (cmd == "check") return check();
            if (cmd == "run") return runCmd();
            if (cmd == "fmt") return fmt();
            if (cmd == "fix") return fix();
            if (cmd == "estimate") return estimate();
            if (cmd == "ir") return irCmd();
            if (cmd == "draw") return draw();
            if (cmd == "equiv") return equiv();
            if (cmd == "opt") return optimize();
            if (cmd == "grad") return gradient();
            if (cmd == "qasm") return qasm();
            if (cmd == "eval") return evalCmd();
            if (cmd == "research") return research();
            if (cmd == "explain") return explain();
            if (cmd == "grammar") return grammar();
            if (cmd == "tokens") return tokens();
            if (cmd == "skill") return skill();
            if (cmd == "repl") return repl();
            if (cmd == "version" || cmd == "--version")
            {
                out << "noether " << kVersion << " (language " << kLangVersion << ")\n";
                return 0;
            }
            if (cmd == "help" || cmd == "--help" || cmd == "-h")
            {
                out << kUsage;
                return 0;
            }
            throw Usage{std::format("unknown command `{}`; run `noether help`", cmd)};
        }

        int check()
        {
            if (named.contains("spec"))
            {
                ResearchOptions ro = researchOptions();
                ro.candidate = file();
                return finishResearch(checkCandidate(ro));
            }
            auto c = compile(file());
            const Ir& ir = c->compiler->ir();
            const bool ok = !c->diags->hasErrors();
            Json j = documentHeader("check", ok, *c->diags);
            j["formatted"] = formatted;
            if (ok)
            {
                j["backend"] = ir.backend;
                j["backendReason"] = ir.backendReason;
                j["sourceSha256"] = ir.sourceSha256;
            }
            std::string human = c->diags->render();
            if (ok) human += std::format("ok: {} qubits, {} clbits, backend {} ({})\n", ir.nQubits, ir.nClbits, ir.backend, ir.backendReason);
            if (json) out << j.dump() << "\n";
            else (ok ? out : err) << human;
            return ok ? 0 : 1;
        }

        int runCmd()
        {
            double compileMs = 0.0;
            auto c = compile(file(), &compileMs);
            if (c->diags->hasErrors()) return failCompile(*c, "run");
            const ExecOptions o = execOptions();
            const ExecResult r = execute(*c->compiler, o);
            if (json)
            {
                Json j = runJson(*c->compiler, r, timing ? std::optional<double>(compileMs) : std::nullopt, o);
                j["formatted"] = formatted;
                out << j.dump() << "\n";
            }
            else
            {
                const std::string diag = c->diags->render();
                if (!diag.empty()) err << diag;
                out << runText(*c->compiler, r, o);
            }
            if (r.exitCode != 0) return r.exitCode;
            return c->diags->hasErrors() ? 1 : 0;
        }

        int fmt()
        {
            const std::string& path = file();
            const std::string text = readOrThrow(path);
            SourceManager sm;
            Diagnostics d(sm);
            const std::uint32_t id = sm.add(path, text);
            const Program prog = parse(sm.file(id), id, d);
            if (d.hasErrors())
            {
                if (json) out << documentHeader("fmt", false, d).dump() << "\n";
                else err << d.render();
                return 1;
            }
            const std::string canonical = formatProgram(prog, sm.file(id), ascii);
            const bool changed = canonical != text;
            if (checkOnly)
            {
                if (json)
                {
                    Json j = documentHeader("fmt", !changed, d);
                    j["changed"] = changed;
                    out << j.dump() << "\n";
                }
                else if (changed) err << path << " is not canonically formatted\n";
                return changed ? 1 : 0;
            }
            if (write)
            {
                std::string error;
                if (changed && !writeAtomically(path, canonical, error)) throw Usage{error};
                if (json)
                {
                    Json j = documentHeader("fmt", true, d);
                    j["changed"] = changed;
                    out << j.dump() << "\n";
                }
                return 0;
            }
            if (json)
            {
                Json j = documentHeader("fmt", true, d);
                j["changed"] = changed;
                j["text"] = canonical;
                out << j.dump() << "\n";
            }
            else out << canonical;
            return 0;
        }

        int fix()
        {
            const std::string& path = file();
            auto c = compileFile(path, copt);
            if (c->readFailed) throw Usage{c->readError};
            std::vector<Edit> edits;
            for (const Diagnostic& diag : c->diags->sorted())
                if (!diag.fixes.empty() && diag.span.file == c->mainFile)
                    for (const Edit& e : diag.fixes.front().edits) edits.push_back(e);
            if (!apply)
            {
                Json j = documentHeader("fix", true, *c->diags);
                j["fixable"] = edits.size();
                return emit(j, std::format("{}{} machine-applicable fix{}; rerun with --apply\n", c->diags->render(), edits.size(),
                                           edits.size() == 1 ? "" : "es"));
            }
            // Apply from the end so earlier offsets stay valid; overlapping edits are skipped.
            std::ranges::sort(edits, [](const Edit& a, const Edit& b) { return a.span.begin > b.span.begin; });
            std::string text = std::string(c->sources->file(c->mainFile).text());
            std::uint32_t limit = UINT32_MAX;
            std::size_t applied = 0;
            for (const Edit& e : edits)
            {
                if (e.span.end > limit) continue;
                text.replace(e.span.begin, e.span.end - e.span.begin, e.text);
                limit = e.span.begin;
                ++applied;
            }
            std::string error;
            if (applied && !writeAtomically(path, text, error)) throw Usage{error};
            Json j = documentHeader("fix", true, *c->diags);
            j["applied"] = applied;
            return emit(j, std::format("applied {} fix{}\n", applied, applied == 1 ? "" : "es"));
        }

        int estimate()
        {
            auto c = compile(file());
            if (c->diags->hasErrors()) return failCompile(*c, "estimate");
            const Ir& ir = c->compiler->ir();
            Json body = estimateJson(ir, *c->sources);
            Json j = documentHeader("estimate", true, *c->diags);
            for (auto& [k, v] : body.asObject()) j[k] = v;
            const Resources r = resources(ir);
            std::string human = std::format(
                "backend     {} ({})\nqubits      {}\nclbits      {}\nops         {}\ndepth       {} (2-qubit depth {})\n"
                "2-qubit ops {}\nT-count     {} (T-depth {})\nclifford    {}\nparams      {}\nmemory      {} bytes\n",
                ir.backend, ir.backendReason, r.qubits, r.clbits, r.ops, r.depth, r.depth2q, r.count2q, r.tcount, r.tdepth,
                r.clifford ? "yes" : "no", r.nparams, r.peakBytes);
            return emit(j, c->diags->render() + human);
        }

        int irCmd()
        {
            auto c = compile(file());
            if (c->diags->hasErrors()) return failCompile(*c, "ir");
            Json j = documentHeader("ir", true, *c->diags);
            const Json body = irJson(*c->compiler);
            for (const auto& [k, v] : body.asObject()) j[k] = v;
            return emit(j, irText(*c->compiler));
        }

        int draw()
        {
            auto c = compile(file());
            if (c->diags->hasErrors()) return failCompile(*c, "draw");
            Json j = documentHeader("draw", true, *c->diags);
            j["layers"] = drawJson(c->compiler->ir());
            return emit(j, drawText(c->compiler->ir(), ascii));
        }

        int equiv()
        {
            if (positional.size() != 2) throw Usage{"equiv needs two files"};
            auto a = compile(positional[0]);
            auto b = compile(positional[1]);
            if (a->diags->hasErrors()) return failCompile(*a, "equiv");
            if (b->diags->hasErrors()) return failCompile(*b, "equiv");
            const std::string phase = named.contains("phase") ? named["phase"] : "ignore";
            if (phase != "ignore" && phase != "exact") throw Usage{"--phase takes ignore or exact"};
            const EquivResult r = equivalent(a->compiler->ir(), b->compiler->ir(), phase == "ignore");
            Json j = documentHeader("equiv", r.error.empty(), *a->diags);
            j["equivalent"] = r.equivalent;
            j["method"] = r.method;
            j["maxDeviation"] = r.maxDeviation;
            if (!r.counterexample.empty()) j["counterexample"] = r.counterexample;
            if (!r.error.empty()) j["error"] = r.error;
            if (!r.error.empty())
            {
                if (json) out << j.dump() << "\n";
                else err << "noether: " << r.error << "\n";
                return 2;
            }
            return emit(j, std::format("{} (method {}, max deviation {})\n", r.equivalent ? "equivalent" : "NOT equivalent",
                                       r.method, jsonNumber(r.maxDeviation)) +
                               (r.counterexample.empty() ? "" : std::format("counterexample input: {}\n", r.counterexample)));
        }

        int optimize()
        {
            const bool maximize = named.contains("maximize");
            const std::string label = maximize ? named["maximize"] : (named.contains("minimize") ? named["minimize"] : "");
            if (label.empty()) throw Usage{"opt needs --minimize LABEL or --maximize LABEL"};
            auto c = compile(file());
            if (c->diags->hasErrors()) return failCompile(*c, "opt");
            OptOptions oo;
            oo.label = label;
            oo.maximize = maximize;
            if (named.contains("method")) oo.method = named["method"];
            if (named.contains("restarts")) oo.restarts = static_cast<std::size_t>(std::stoul(named["restarts"]));
            if (named.contains("iters")) oo.iterations = static_cast<std::size_t>(std::stoul(named["iters"]));
            oo.exec = execOptions();
            const OptResult r = optimizeParams(*c->compiler, oo);
            if (!r.error.empty()) throw Usage{r.error};
            Json j = documentHeader("opt", true, *c->diags);
            j["label"] = label;
            j["method"] = oo.method;
            j["best"] = r.best;
            j["evaluations"] = r.evaluations;
            j["params"] = r.paramsJson;
            const std::string outPath = named.contains("out") ? named["out"] : "params.json";
            std::string error;
            Json pj = Json::object();
            pj["schema"] = "noether.params/1";
            pj["params"] = r.paramsJson;
            pj["label"] = label;
            pj["value"] = r.best;
            if (!writeAtomically(outPath, pj.dump() + "\n", error)) throw Usage{error};
            j["written"] = outPath;
            return emit(j, std::format("{} {} = {} after {} evaluations\nparams: {}\nwritten to {}\n", maximize ? "max" : "min", label,
                                       jsonNumber(r.best), r.evaluations, r.paramsJson.dump(-1), outPath));
        }

        int gradient()
        {
            if (!named.contains("of")) throw Usage{"grad needs --of LABEL"};
            auto c = compile(file());
            if (c->diags->hasErrors()) return failCompile(*c, "grad");
            const GradResult r = gradientOf(*c->compiler, named["of"], execOptions());
            if (!r.error.empty()) throw Usage{r.error};
            Json j = documentHeader("grad", true, *c->diags);
            j["label"] = named["of"];
            j["value"] = r.value;
            j["gradient"] = r.gradientJson;
            j["method"] = r.method;
            return emit(j, std::format("{} = {}\ngradient ({}): {}\n", named["of"], jsonNumber(r.value), r.method, r.gradientJson.dump(-1)));
        }

        int qasm()
        {
            if (positional.size() != 2 || positional[0] != "export") throw Usage{"usage: noether qasm export f.ntr"};
            auto c = compile(positional[1]);
            if (c->diags->hasErrors()) return failCompile(*c, "qasm");
            std::string error;
            const std::string text = qasmExport(c->compiler->ir(), error);
            if (!error.empty()) throw Usage{error};
            Json j = documentHeader("qasm", true, *c->diags);
            j["text"] = text;
            return emit(j, text);
        }

        ResearchOptions researchOptions() const
        {
            ResearchOptions ro;
            ro.exec = execOptions();
            ro.noFormat = noFormat;
            auto get = [&](const char* k) { const auto it = named.find(k); return it == named.end() ? std::string() : it->second; };
            ro.dir = get("dir");
            ro.spec = get("spec");
            ro.candidate = get("candidate");
            ro.from = get("from");
            ro.holdout = get("holdout");
            if (named.contains("record")) ro.record = named.at("record");
            return ro;
        }

        int finishResearch(const ResearchOutcome& r)
        {
            if (json) out << r.json.dump() << "\n";
            else (r.exitCode == 0 ? out : err) << r.text;
            return r.exitCode;
        }

        int evalCmd()
        {
            ResearchOptions ro = researchOptions();
            if (ro.dir.empty() && ro.spec.empty())
            {
                if (positional.size() == 2)
                {
                    ro.spec = positional[0];
                    ro.candidate = positional[1];
                }
                else ro.dir = ".";
            }
            if (!ro.spec.empty() && ro.candidate.empty()) throw Usage{"eval --spec S needs --candidate C"};
            return finishResearch(evaluateCommand(ro));
        }

        int research()
        {
            if (positional.empty()) throw Usage{"usage: noether research init TAG --spec S [--from F] | status [--dir D] | report [--dir D] [--holdout F]"};
            ResearchOptions ro = researchOptions();
            if (positional[0] == "init")
            {
                if (positional.size() < 2) throw Usage{"research init needs a tag"};
                if (ro.spec.empty()) throw Usage{"research init needs --spec S"};
                return finishResearch(researchInit(positional[1], ro));
            }
            if (ro.dir.empty()) ro.dir = positional.size() > 1 ? positional[1] : ".";
            if (positional[0] == "status") return finishResearch(researchStatus(ro));
            if (positional[0] == "report") return finishResearch(researchReport(ro));
            throw Usage{std::format("unknown research command `{}`", positional[0])};
        }

        int explain()
        {
            if (positional.empty()) throw Usage{"usage: noether explain CODE"};
            const CodeInfo* info = findCode(positional[0]);
            if (!info) throw Usage{std::format("unknown diagnostic code {}", positional[0])};
            Json j = Json::object();
            j["schema"] = "noether.explain/1";
            j["version"] = kVersion;
            j["ok"] = true;
            j["code"] = info->code;
            j["title"] = info->title;
            j["explanation"] = info->explanation;
            return emit(j, std::format("{}: {}\n\n{}\n", info->code, info->title, info->explanation));
        }

        int grammar()
        {
            Json j = Json::object();
            j["schema"] = "noether.grammar/1";
            j["version"] = kVersion;
            j["ok"] = true;
            j["ebnf"] = grammarText();
            return emit(j, std::string(grammarText()));
        }

        int tokens()
        {
            Json rows = Json::array();
            std::string human = "canonical | latex | ascii input | ascii canonical | meaning\n";
            for (const AliasRow& r : aliasTable())
            {
                Json x = Json::object();
                x["canonical"] = r.canonical;
                x["latex"] = r.latex;
                x["ascii"] = r.ascii;
                x["asciiCanonical"] = r.asciiCanonical;
                x["meaning"] = r.meaning;
                rows.push(std::move(x));
                human += std::format("{} | {} | {} | {} | {}\n", r.canonical, r.latex, r.ascii, r.asciiCanonical, r.meaning);
            }
            Json j = Json::object();
            j["schema"] = "noether.tokens/1";
            j["version"] = kVersion;
            j["ok"] = true;
            j["tokens"] = rows;
            return emit(j, human);
        }

        int skill()
        {
            if (named.contains("install"))
            {
                std::string error;
                const std::size_t n = installSkills(named["install"], error);
                if (!error.empty()) throw Usage{error};
                Json j = Json::object();
                j["schema"] = "noether.skill/1";
                j["version"] = kVersion;
                j["ok"] = true;
                j["installed"] = n;
                j["dir"] = named["install"];
                return emit(j, std::format("installed {} skill files into {}\n", n, named["install"]));
            }
            const std::string text = skillText(full);
            Json j = Json::object();
            j["schema"] = "noether.skill/1";
            j["version"] = kVersion;
            j["ok"] = true;
            j["text"] = text;
            return emit(j, text);
        }

        int repl() { return runRepl(std::cin, out, err, copt); }

        std::ostream& out;
        std::ostream& err;
        std::string cmd;
        bool json = false, timing = true, noFormat = false, write = false, checkOnly = false, ascii = false, apply = false,
             full = false, force = false, denyWarnings = false, formatted = false;
        std::optional<double> timeout;
        std::uint64_t maxMem = std::uint64_t{8} << 30;
        std::size_t top = 0;
        CompileOptions copt;
        std::vector<std::string> positional;
        std::map<std::string, std::string> named;
    };
} // namespace

int runCli(const std::vector<std::string>& args, std::ostream& out, std::ostream& err)
{
    try
    {
        return Cli(out, err).run(args);
    }
    catch (const std::exception& e)
    {
        err << "noether: internal error: " << e.what() << "\n";
        return 5;
    }
}

} // namespace Noether
