#include "Source.hpp"

#include <algorithm>
#include <utility>



namespace Noether
{

SourceFile::SourceFile(std::string path, std::string text) : filePath(std::move(path)), content(std::move(text))
{
    lineStarts.push_back(0);
    for (std::size_t i = 0; i < content.size(); ++i)
        if (content[i] == '\n') lineStarts.push_back(static_cast<std::uint32_t>(i + 1));
}

std::string_view SourceFile::slice(const Span& s) const
{
    const std::size_t b = std::min<std::size_t>(s.begin, content.size());
    const std::size_t e = std::min<std::size_t>(std::max(s.end, s.begin), content.size());
    return std::string_view(content).substr(b, e - b);
}

LineCol SourceFile::lineCol(std::uint32_t offset) const
{
    const auto it = std::upper_bound(lineStarts.begin(), lineStarts.end(), offset);
    const auto line = static_cast<std::uint32_t>(it - lineStarts.begin());
    const std::uint32_t start = lineStarts[line - 1];
    const std::size_t end = std::min<std::size_t>(offset, content.size());
    const std::uint32_t col =
        static_cast<std::uint32_t>(codePointCount(std::string_view(content).substr(start, end - start))) + 1;
    return {line, col};
}

std::string_view SourceFile::lineText(std::uint32_t line) const
{
    if (line == 0 || line > lineStarts.size()) return {};
    const std::size_t b = lineStarts[line - 1];
    std::size_t e = line < lineStarts.size() ? lineStarts[line] : content.size();
    while (e > b && (content[e - 1] == '\n' || content[e - 1] == '\r')) --e;
    return std::string_view(content).substr(b, e - b);
}

std::uint32_t SourceManager::add(std::string path, std::string text)
{
    files.push_back(std::make_unique<SourceFile>(std::move(path), std::move(text)));
    return static_cast<std::uint32_t>(files.size() - 1);
}


CodePoint decodeUtf8(std::string_view text, std::size_t pos)
{
    if (pos >= text.size()) return {};
    const auto b0 = static_cast<unsigned char>(text[pos]);
    if (b0 < 0x80) return {b0, 1};

    std::uint32_t len = 0;
    char32_t cp = 0;
    if ((b0 & 0xE0) == 0xC0) { len = 2; cp = b0 & 0x1F; }
    else if ((b0 & 0xF0) == 0xE0) { len = 3; cp = b0 & 0x0F; }
    else if ((b0 & 0xF8) == 0xF0) { len = 4; cp = b0 & 0x07; }
    else return {};
    if (pos + len > text.size()) return {};

    for (std::uint32_t k = 1; k < len; ++k)
    {
        const auto b = static_cast<unsigned char>(text[pos + k]);
        if ((b & 0xC0) != 0x80) return {};
        cp = (cp << 6) | (b & 0x3F);
    }
    // Reject overlong encodings, surrogates and values past U+10FFFF.
    if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000)) return {};
    if ((cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF) return {};
    return {cp, len};
}

std::string encodeUtf8(char32_t cp)
{
    std::string out;
    if (cp < 0x80) out += static_cast<char>(cp);
    else if (cp < 0x800)
    {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    else if (cp < 0x10000)
    {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    else
    {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return out;
}

std::size_t codePointCount(std::string_view text)
{
    std::size_t n = 0;
    for (const char c : text)
        if ((static_cast<unsigned char>(c) & 0xC0) != 0x80) ++n;
    return n;
}

} // namespace Noether
