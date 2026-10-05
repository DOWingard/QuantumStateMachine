#include "StdLib.hpp"

#include "Embedded.hpp"



namespace Noether
{

std::optional<std::string_view> stdSource(std::string_view importPath)
{
    if (!importPath.starts_with("std/")) return std::nullopt;
    const std::string_view rel = importPath.substr(4);
    for (const EmbeddedFile& f : embeddedStdLib())
        if (f.path == rel) return f.text;
    return std::nullopt;
}

std::string stdModuleList()
{
    std::string out;
    for (const EmbeddedFile& f : embeddedStdLib())
    {
        if (!out.empty()) out += ", ";
        out += "std/";
        out += f.path;
    }
    return out;
}

} // namespace Noether
