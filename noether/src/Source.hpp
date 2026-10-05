#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>



namespace Noether
{

// Half-open byte range [begin, end) in one source file.
struct Span
{
    std::uint32_t file = 0;
    std::uint32_t begin = 0;
    std::uint32_t end = 0;

    static Span join(const Span& a, const Span& b)
    {
        return {a.file, a.begin < b.begin ? a.begin : b.begin, a.end > b.end ? a.end : b.end};
    }
};

struct LineCol
{
    std::uint32_t line = 1; // 1-based
    std::uint32_t col = 1;  // 1-based, counted in code points
};

class SourceFile
{
    public:
    SourceFile(std::string path, std::string text);

    const std::string& path() const { return filePath; }
    const std::string& text() const { return content; }
    std::string_view slice(const Span& s) const;

    LineCol lineCol(std::uint32_t offset) const;
    std::string_view lineText(std::uint32_t line) const; // without the newline
    std::uint32_t lineCount() const { return static_cast<std::uint32_t>(lineStarts.size()); }
    std::uint32_t lineStart(std::uint32_t line) const { return lineStarts[line - 1]; }

    private:
    std::string filePath;
    std::string content;
    std::vector<std::uint32_t> lineStarts;
};

// Owns every file loaded in one compilation (the main file, imports, a spec); spans refer to
// files by index.
class SourceManager
{
    public:
    std::uint32_t add(std::string path, std::string text);
    const SourceFile& file(std::uint32_t id) const { return *files.at(id); }
    std::size_t size() const { return files.size(); }

    private:
    std::vector<std::unique_ptr<SourceFile>> files;
};

// Decodes one UTF-8 code point at text[pos]; returns 0 bytes consumed for invalid input.
struct CodePoint
{
    char32_t value = 0;
    std::uint32_t length = 0;
};
CodePoint decodeUtf8(std::string_view text, std::size_t pos);
std::string encodeUtf8(char32_t cp);
std::size_t codePointCount(std::string_view text);

} // namespace Noether
