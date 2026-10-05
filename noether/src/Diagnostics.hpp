#pragma once

#include "Json.hpp"
#include "Source.hpp"

#include <cstdint>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>



namespace Noether
{

enum class Severity : std::uint8_t { Error, Warning };

struct Edit
{
    Span span;
    std::string text;
};

struct Fix
{
    std::string title;
    std::vector<Edit> edits;
};

struct Note
{
    Span span;
    std::string message;
};

struct Diagnostic
{
    std::string code;
    Severity severity = Severity::Error;
    std::string message;
    Span span;
    std::vector<Note> notes;
    std::vector<Fix> fixes;

    Diagnostic& note(Span s, std::string m)
    {
        notes.push_back({s, std::move(m)});
        return *this;
    }
    Diagnostic& fix(std::string title, Span s, std::string text)
    {
        fixes.push_back({std::move(title), {{s, std::move(text)}}});
        return *this;
    }
};

// Thrown after an error has been reported to abandon the current statement; the caller resumes at
// the next one so that one `check` reports every independent error.
struct CompileAbort
{
};

class Diagnostics
{
    public:
    explicit Diagnostics(const SourceManager& sources) : sm(&sources) {}

    Diagnostic& error(std::string code, Span span, std::string message);
    Diagnostic& warning(std::string code, Span span, std::string message);
    [[noreturn]] void fatal(std::string code, Span span, std::string message);

    // `# noether: allow W0003` in a file suppresses that lint for spans in that file.
    void allow(std::uint32_t file, std::string code) { allowed.emplace(file, std::move(code)); }
    void denyWarnings(bool deny) { denyWarn = deny; }

    bool hasErrors() const;
    std::size_t errorCount() const;
    const std::vector<Diagnostic>& all() const { return list; }
    std::vector<Diagnostic> sorted() const; // by file, then position; suppressed lints removed
    const SourceManager& sources() const { return *sm; }

    Json toJson() const;
    std::string render() const; // human form with source excerpts

    private:
    bool suppressed(const Diagnostic& d) const;

    const SourceManager* sm;
    std::vector<Diagnostic> list;
    std::set<std::pair<std::uint32_t, std::string>> allowed;
    bool denyWarn = false;
};

Json spanJson(const SourceManager& sm, const Span& s);
Json diagnosticJson(const SourceManager& sm, const Diagnostic& d);
std::string renderDiagnostic(const SourceManager& sm, const Diagnostic& d);

// ---- Catalog: every code with a title and a long explanation (wrong and right example) ----
struct CodeInfo
{
    std::string_view code;
    std::string_view title;
    std::string_view explanation;
};

std::span<const CodeInfo> codeCatalog();
const CodeInfo* findCode(std::string_view code);

} // namespace Noether
