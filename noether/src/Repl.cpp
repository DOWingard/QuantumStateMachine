#include "Tools.hpp"

#include "Estimate.hpp"
#include "Formatter.hpp"

#include <format>

#include <istream>
#include <ostream>



namespace Noether
{

// The REPL keeps the session's source and re-runs it after every accepted statement, so each line
// sees the effects of all earlier ones; a statement that fails to compile is dropped. The backend
// follows the size rule (state vector up to 25 qubits) because the session cannot see ahead.
int runRepl(std::istream& in, std::ostream& out, std::ostream& err, const CompileOptions& options)
{
    std::string session = "noether 0.1\n";
    bool jsonMode = false;
    out << "noether " << kVersion << " REPL. :help for commands, :quit to leave.\n";
    std::string pending;
    std::string line;
    while (true)
    {
        out << (pending.empty() ? ">>> " : "... ") << std::flush;
        if (!std::getline(in, line)) break;
        if (pending.empty() && line.starts_with(":"))
        {
            const std::string cmd = line.substr(0, line.find(' '));
            const std::string arg = line.find(' ') == std::string::npos ? "" : line.substr(line.find(' ') + 1);
            if (cmd == ":quit" || cmd == ":q") break;
            if (cmd == ":help")
            {
                out << ":draw :state :estimate :ir :load FILE :reset :json on|off :source :quit\n";
                continue;
            }
            if (cmd == ":reset")
            {
                session = "noether 0.1\n";
                continue;
            }
            if (cmd == ":json")
            {
                jsonMode = arg == "on";
                continue;
            }
            if (cmd == ":source")
            {
                out << session;
                continue;
            }
            if (cmd == ":load")
            {
                bool ok = false;
                std::string text = readFileText(arg, ok);
                if (!ok)
                {
                    err << "cannot read " << arg << "\n";
                    continue;
                }
                const auto nl = text.find('\n');
                session = "noether 0.1\n" + (nl == std::string::npos ? std::string() : text.substr(nl + 1));
                if (!session.ends_with('\n')) session += '\n';
            }
            auto c = compileText("<repl>", session, options);
            if (c->diags->hasErrors())
            {
                err << c->diags->render();
                continue;
            }
            Ir& ir = c->compiler->ir();
            if (cmd == ":draw") out << drawText(ir, false);
            else if (cmd == ":ir") out << irText(*c->compiler);
            else if (cmd == ":estimate") out << estimateJson(ir, *c->sources).dump() << "\n";
            else if (cmd == ":state" || cmd == ":load")
            {
                ExecOptions o;
                o.timing = false;
                const ExecResult r = execute(*c->compiler, o);
                if (cmd == ":state" && ir.backend == "statevector" && ir.nQubits > 0 && ir.nQubits <= 12)
                {
                    // Print the amplitudes through a synthetic `print |ψ⟩` evaluation.
                    auto src = session + "print |ψ⟩\n";
                    auto c2 = compileText("<repl>", src, options);
                    if (!c2->diags->hasErrors())
                    {
                        const ExecResult r2 = execute(*c2->compiler, o);
                        if (!r2.prints.empty()) out << r2.prints.back().value.dump(-1) << "\n";
                    }
                }
                else if (cmd == ":state") out << std::format("{} qubits on the {} backend\n", ir.nQubits, ir.backend);
                (void)r;
            }
            else err << "unknown command " << cmd << "\n";
            continue;
        }

        // Compound statements continue until a blank line.
        pending += line + "\n";
        const bool opensBlock = !line.empty() && line.back() == ':';
        if ((opensBlock || line.starts_with(" ")) && !line.empty()) continue;
        const std::string candidate = session + pending;
        pending.clear();
        auto c = compileText("<repl>", candidate, options);
        if (c->diags->hasErrors())
        {
            err << c->diags->render();
            continue;
        }
        session = candidate;
        ExecOptions o;
        o.timing = false;
        const ExecResult r = execute(*c->compiler, o);
        // Show only what the newest statement printed.
        static std::size_t shownPrints = 0, shownRuns = 0;
        if (r.prints.size() < shownPrints) shownPrints = 0;
        if (r.runs.size() < shownRuns) shownRuns = 0;
        if (jsonMode) out << runJson(*c->compiler, r, std::nullopt, o).dump(-1) << "\n";
        else
        {
            for (std::size_t k = shownPrints; k < r.prints.size(); ++k)
                out << (r.prints[k].label.empty() ? r.prints[k].text : r.prints[k].label) << " = " << r.prints[k].value.dump(-1) << "\n";
            for (std::size_t k = shownRuns; k < r.runs.size(); ++k)
            {
                Json counts = Json::object();
                for (const auto& [key, n] : r.runs[k].counts) counts[key] = n;
                out << r.runs[k].name << " = " << counts.dump(-1) << "\n";
            }
        }
        shownPrints = r.prints.size();
        shownRuns = r.runs.size();
        if (r.exitCode != 0) err << c->diags->render();
    }
    return 0;
}

} // namespace Noether
