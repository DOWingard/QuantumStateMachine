#include "interop/Import.hpp"

#include "Compiler.hpp"
#include "Sha256.hpp"
#include "interop/CircuitJson.hpp"
#include "interop/Qasm.hpp"
#include "interop/Stim.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <format>



namespace Noether::Interop
{

std::string formatForPath(const std::string& path)
{
    std::string ext = std::filesystem::path(path).extension().string();
    std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (ext == ".qasm" || ext == ".qasm2" || ext == ".qasm3") return "qasm";
    if (ext == ".stim") return "stim";
    if (ext == ".json") return "circuit";
    return {};
}

ImportResult importText(const std::string& path, std::string text, const ImportOptions& options)
{
    ImportResult r;
    r.sha256 = sha256Hex(text);
    r.mainFile = r.sources->add(path, std::move(text));
    const SourceFile& file = r.sources->file(r.mainFile);
    const std::string format = options.format.empty() ? formatForPath(path) : options.format;
    if (format == "stim") r.circuit = importStim(file, r.mainFile, *r.diags);
    else if (format == "circuit") r.circuit = readCircuitJson(file, r.mainFile, *r.diags);
    else if (format == "qasm" || format == "qasm2" || format == "qasm3")
    {
        QasmOptions qo;
        qo.version = format == "qasm2" ? 2 : format == "qasm3" ? 3 : 0;
        qo.params = options.params;
        r.circuit = importQasm(*r.sources, r.mainFile, *r.diags, qo);
    }
    else
        r.diags->error("E9001", Span{r.mainFile, 0, 0},
                       std::format("cannot tell the format of {}; use --format qasm2|qasm3|stim|circuit", path));
    if (r.diags->hasErrors()) r.circuit.reset();
    return r;
}

ImportResult importFile(const std::string& path, const ImportOptions& options)
{
    bool ok = false;
    std::string text = readFileText(path, ok);
    if (!ok)
    {
        ImportResult r;
        r.readFailed = true;
        r.readError = std::format("cannot read {}", path);
        return r;
    }
    return importText(path, std::move(text), options);
}

} // namespace Noether::Interop
