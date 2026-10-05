#include "Json.hpp"

#include <cctype>
#include <charconv>
#include <cmath>
#include <format>



namespace Noether
{

Json& Json::operator[](std::string_view key)
{
    if (isNull()) v = Object{};
    auto& o = std::get<Object>(v);
    auto it = o.find(key);
    if (it == o.end()) it = o.emplace(std::string(key), Json{}).first;
    return it->second;
}

const Json* Json::find(std::string_view key) const
{
    if (!isObject()) return nullptr;
    const auto& o = std::get<Object>(v);
    const auto it = o.find(key);
    return it == o.end() ? nullptr : &it->second;
}

void Json::push(Json item)
{
    if (isNull()) v = Array{};
    std::get<Array>(v).push_back(std::move(item));
}

std::size_t Json::size() const
{
    if (isArray()) return asArray().size();
    if (isObject()) return asObject().size();
    return 0;
}

std::string jsonNumber(double d)
{
    if (!std::isfinite(d)) return "null";
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof buf, d);
    std::string s(buf, res.ptr);
    if (s.find_first_of(".eE") == std::string::npos) s += ".0";
    return s;
}

std::string jsonQuote(std::string_view s)
{
    std::string out;
    out.reserve(s.size() + 2);
    out += '"';
    for (const char c : s)
    {
        switch (c)
        {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) out += std::format("\\u{:04x}", static_cast<unsigned>(c));
                else out += c;
        }
    }
    out += '"';
    return out;
}

void Json::write(std::string& out, int indent, int depth) const
{
    auto newline = [&](int d)
    {
        if (indent < 0) return;
        out += '\n';
        out.append(static_cast<std::size_t>(indent * d), ' ');
    };

    if (isNull()) out += "null";
    else if (isBool()) out += asBool() ? "true" : "false";
    else if (isInt()) out += std::to_string(std::get<std::int64_t>(v));
    else if (isDouble()) out += jsonNumber(std::get<double>(v));
    else if (isString()) out += jsonQuote(asString());
    else if (isArray())
    {
        const auto& a = asArray();
        if (a.empty())
        {
            out += "[]";
            return;
        }
        // Arrays of scalars stay on one line: amplitudes, keys and matrices remain readable.
        bool flat = true;
        for (const Json& e : a)
            if (e.isObject() || (e.isArray() && !e.asArray().empty() && (e.asArray()[0].isObject() || e.asArray()[0].isArray()))) flat = false;
        out += '[';
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            if (i) out += flat || indent < 0 ? ", " : ",";
            if (!flat) newline(depth + 1);
            a[i].write(out, flat ? -1 : indent, depth + 1);
        }
        if (!flat) newline(depth);
        out += ']';
    }
    else
    {
        const auto& o = asObject();
        if (o.empty())
        {
            out += "{}";
            return;
        }
        out += '{';
        bool first = true;
        for (const auto& [k, val] : o)
        {
            if (!first) out += indent < 0 ? ", " : ",";
            first = false;
            newline(depth + 1);
            out += jsonQuote(k);
            out += ": ";
            val.write(out, indent, depth + 1);
        }
        newline(depth);
        out += '}';
    }
}

std::string Json::dump(int indent) const
{
    std::string out;
    write(out, indent, 0);
    return out;
}


namespace
{

    struct Parser
    {
        std::string_view s;
        std::size_t i = 0;
        std::string err;

        void ws()
        {
            while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
        }

        bool fail(std::string m)
        {
            if (err.empty()) err = std::format("offset {}: {}", i, m);
            return false;
        }

        bool literal(std::string_view word)
        {
            if (s.substr(i, word.size()) != word) return fail(std::format("expected '{}'", word));
            i += word.size();
            return true;
        }

        bool string(std::string& out)
        {
            if (i >= s.size() || s[i] != '"') return fail("expected string");
            ++i;
            while (i < s.size() && s[i] != '"')
            {
                if (s[i] == '\\')
                {
                    if (++i >= s.size()) return fail("bad escape");
                    switch (s[i])
                    {
                        case 'n': out += '\n'; break;
                        case 't': out += '\t'; break;
                        case 'r': out += '\r'; break;
                        case 'b': out += '\b'; break;
                        case 'f': out += '\f'; break;
                        case 'u':
                        {
                            if (i + 4 >= s.size()) return fail("bad \\u escape");
                            unsigned cp = 0;
                            const auto r = std::from_chars(s.data() + i + 1, s.data() + i + 5, cp, 16);
                            if (r.ec != std::errc{}) return fail("bad \\u escape");
                            i += 4;
                            if (cp < 0x80) out += static_cast<char>(cp);
                            else if (cp < 0x800)
                            {
                                out += static_cast<char>(0xC0 | (cp >> 6));
                                out += static_cast<char>(0x80 | (cp & 0x3F));
                            }
                            else
                            {
                                out += static_cast<char>(0xE0 | (cp >> 12));
                                out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                out += static_cast<char>(0x80 | (cp & 0x3F));
                            }
                            break;
                        }
                        default: out += s[i];
                    }
                    ++i;
                }
                else out += s[i++];
            }
            if (i >= s.size()) return fail("unterminated string");
            ++i;
            return true;
        }

        bool value(Json& out, int depth)
        {
            if (depth > 512) return fail("nesting too deep");
            ws();
            if (i >= s.size()) return fail("unexpected end");
            const char c = s[i];
            if (c == '{')
            {
                ++i;
                Json::Object o;
                ws();
                if (i < s.size() && s[i] == '}')
                {
                    ++i;
                    out = Json(std::move(o));
                    return true;
                }
                while (true)
                {
                    ws();
                    std::string key;
                    if (!string(key)) return false;
                    ws();
                    if (i >= s.size() || s[i] != ':') return fail("expected ':'");
                    ++i;
                    Json val;
                    if (!value(val, depth + 1)) return false;
                    o[std::move(key)] = std::move(val);
                    ws();
                    if (i < s.size() && s[i] == ',') { ++i; continue; }
                    if (i < s.size() && s[i] == '}') { ++i; break; }
                    return fail("expected ',' or '}'");
                }
                out = Json(std::move(o));
                return true;
            }
            if (c == '[')
            {
                ++i;
                Json::Array a;
                ws();
                if (i < s.size() && s[i] == ']')
                {
                    ++i;
                    out = Json(std::move(a));
                    return true;
                }
                while (true)
                {
                    Json val;
                    if (!value(val, depth + 1)) return false;
                    a.push_back(std::move(val));
                    ws();
                    if (i < s.size() && s[i] == ',') { ++i; continue; }
                    if (i < s.size() && s[i] == ']') { ++i; break; }
                    return fail("expected ',' or ']'");
                }
                out = Json(std::move(a));
                return true;
            }
            if (c == '"')
            {
                std::string str;
                if (!string(str)) return false;
                out = Json(std::move(str));
                return true;
            }
            if (c == 't') { if (!literal("true")) return false; out = Json(true); return true; }
            if (c == 'f') { if (!literal("false")) return false; out = Json(false); return true; }
            if (c == 'n') { if (!literal("null")) return false; out = Json(); return true; }

            const std::size_t start = i;
            if (s[i] == '-') ++i;
            bool isFloat = false;
            while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.' || s[i] == 'e' ||
                                    s[i] == 'E' || s[i] == '+' || s[i] == '-'))
            {
                if (s[i] == '.' || s[i] == 'e' || s[i] == 'E') isFloat = true;
                ++i;
            }
            if (start == i) return fail("unexpected character");
            const std::string_view num = s.substr(start, i - start);
            if (!isFloat)
            {
                std::int64_t n = 0;
                const auto r = std::from_chars(num.data(), num.data() + num.size(), n);
                if (r.ec == std::errc{} && r.ptr == num.data() + num.size())
                {
                    out = Json(n);
                    return true;
                }
            }
            double d = 0;
            const auto r = std::from_chars(num.data(), num.data() + num.size(), d);
            if (r.ec != std::errc{}) return fail("bad number");
            out = Json(d);
            return true;
        }
    };

} // namespace

std::optional<Json> Json::parse(std::string_view text, std::string* error)
{
    Parser p{text, 0, {}};
    Json out;
    if (!p.value(out, 0))
    {
        if (error) *error = p.err;
        return std::nullopt;
    }
    p.ws();
    if (p.i != text.size())
    {
        if (error) *error = std::format("offset {}: trailing characters", p.i);
        return std::nullopt;
    }
    return out;
}

} // namespace Noether
