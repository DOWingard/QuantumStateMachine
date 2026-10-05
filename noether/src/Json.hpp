#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>



namespace Noether
{

// JSON value with sorted object keys. Doubles are written in the shortest form that parses back
// to the same bits, so every document is reproducible byte for byte; NaN and infinities become
// null.
class Json
{
    public:
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json, std::less<>>;

    Json() = default;
    Json(std::nullptr_t) {}
    Json(bool b) : v(b) {}
    Json(int i) : v(static_cast<std::int64_t>(i)) {}
    Json(long i) : v(static_cast<std::int64_t>(i)) {}
    Json(long long i) : v(static_cast<std::int64_t>(i)) {}
    Json(unsigned i) : v(static_cast<std::int64_t>(i)) {}
    Json(unsigned long i) : v(static_cast<std::int64_t>(i)) {}
    Json(unsigned long long i) : v(static_cast<std::int64_t>(i)) {}
    Json(double d) : v(d) {}
    Json(std::string s) : v(std::move(s)) {}
    Json(std::string_view s) : v(std::string(s)) {}
    Json(const char* s) : v(std::string(s)) {}
    Json(Array a) : v(std::move(a)) {}
    Json(Object o) : v(std::move(o)) {}

    static Json array() { return Json(Array{}); }
    static Json object() { return Json(Object{}); }

    bool isNull() const { return std::holds_alternative<std::monostate>(v); }
    bool isBool() const { return std::holds_alternative<bool>(v); }
    bool isInt() const { return std::holds_alternative<std::int64_t>(v); }
    bool isDouble() const { return std::holds_alternative<double>(v); }
    bool isNumber() const { return isInt() || isDouble(); }
    bool isString() const { return std::holds_alternative<std::string>(v); }
    bool isArray() const { return std::holds_alternative<Array>(v); }
    bool isObject() const { return std::holds_alternative<Object>(v); }

    bool asBool() const { return std::get<bool>(v); }
    std::int64_t asInt() const { return isInt() ? std::get<std::int64_t>(v) : static_cast<std::int64_t>(std::get<double>(v)); }
    double asDouble() const { return isInt() ? static_cast<double>(std::get<std::int64_t>(v)) : std::get<double>(v); }
    const std::string& asString() const { return std::get<std::string>(v); }
    const Array& asArray() const { return std::get<Array>(v); }
    Array& asArray() { return std::get<Array>(v); }
    const Object& asObject() const { return std::get<Object>(v); }
    Object& asObject() { return std::get<Object>(v); }

    // Object access; operator[] converts null to an empty object first.
    Json& operator[](std::string_view key);
    const Json* find(std::string_view key) const;
    bool contains(std::string_view key) const { return find(key) != nullptr; }

    void push(Json item); // converts null to an empty array first
    std::size_t size() const;

    // indent < 0: one line. Otherwise pretty-printed with that many spaces per level.
    std::string dump(int indent = 2) const;

    static std::optional<Json> parse(std::string_view text, std::string* error = nullptr);

    friend bool operator==(const Json& a, const Json& b) { return a.v == b.v; }

    private:
    void write(std::string& out, int indent, int depth) const;

    std::variant<std::monostate, bool, std::int64_t, double, std::string, Array, Object> v;
};

std::string jsonNumber(double d);
std::string jsonQuote(std::string_view s);

} // namespace Noether
