#include "Values.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <numbers>
#include <stdexcept>



namespace Noether
{

namespace
{
    constexpr double kPi = std::numbers::pi;

    std::int64_t gcd64(std::int64_t a, std::int64_t b)
    {
        if (a < 0) a = -a;
        if (b < 0) b = -b;
        while (b != 0)
        {
            const std::int64_t t = a % b;
            a = b;
            b = t;
        }
        return a == 0 ? 1 : a;
    }
} // namespace


// ---- Rational ----

std::optional<Rational> Rational::make(std::int64_t num, std::int64_t den)
{
    if (den == 0) return std::nullopt;
    if (den < 0)
    {
        if (num == INT64_MIN || den == INT64_MIN) return std::nullopt;
        num = -num;
        den = -den;
    }
    const std::int64_t g = gcd64(num, den);
    return Rational{num / g, den / g};
}

std::optional<Rational> add(Rational a, Rational b)
{
    std::int64_t x = 0, y = 0, den = 0, num = 0;
    if (__builtin_mul_overflow(a.n, b.d, &x) || __builtin_mul_overflow(b.n, a.d, &y) ||
        __builtin_add_overflow(x, y, &num) || __builtin_mul_overflow(a.d, b.d, &den))
        return std::nullopt;
    return Rational::make(num, den);
}

std::optional<Rational> mul(Rational a, Rational b)
{
    // Cross-cancel first to keep intermediates small.
    const std::int64_t g1 = gcd64(a.n, b.d), g2 = gcd64(b.n, a.d);
    std::int64_t num = 0, den = 0;
    if (__builtin_mul_overflow(a.n / g1, b.n / g2, &num) || __builtin_mul_overflow(a.d / g2, b.d / g1, &den))
        return std::nullopt;
    return Rational::make(num, den);
}

std::optional<Rational> div(Rational a, Rational b)
{
    if (b.n == 0) return std::nullopt;
    return mul(a, Rational{b.d, b.n});
}

std::optional<Rational> parseDecimal(std::string_view text)
{
    // mantissa digits . fraction digits [e exp]
    std::string digits;
    std::int64_t scale = 0;
    std::size_t i = 0;
    for (; i < text.size() && std::isdigit(static_cast<unsigned char>(text[i])); ++i) digits += text[i];
    if (i < text.size() && text[i] == '.')
        for (++i; i < text.size() && std::isdigit(static_cast<unsigned char>(text[i])); ++i)
        {
            digits += text[i];
            --scale;
        }
    if (i < text.size() && (text[i] == 'e' || text[i] == 'E'))
    {
        int e = 0;
        ++i;
        const auto r = std::from_chars(text.data() + i, text.data() + text.size(), e);
        if (r.ec != std::errc{}) return std::nullopt;
        scale += e;
    }
    while (digits.size() > 1 && digits.front() == '0') digits.erase(digits.begin());
    if (digits.size() > 18) return std::nullopt;
    std::int64_t m = 0;
    std::from_chars(digits.data(), digits.data() + digits.size(), m);
    std::int64_t num = m, den = 1;
    for (; scale > 0; --scale)
        if (__builtin_mul_overflow(num, std::int64_t{10}, &num)) return std::nullopt;
    for (; scale < 0; ++scale)
        if (__builtin_mul_overflow(den, std::int64_t{10}, &den)) return std::nullopt;
    return Rational::make(num, den);
}


// ---- Real ----

Real Real::approx(double v)
{
    Real x;
    x.isExact = false;
    x.approxValue = v;
    return x;
}

double Real::value() const { return isExact ? r.value() + p.value() * kPi : approxValue; }

std::optional<std::int64_t> Real::asInt() const
{
    if (!isExact || !p.isZero() || !r.isInt()) return std::nullopt;
    return r.n;
}

Real operator+(const Real& a, const Real& b)
{
    if (a.isExact && b.isExact)
    {
        const auto r = add(a.r, b.r);
        const auto p = add(a.p, b.p);
        if (r && p) return Real(*r, *p);
    }
    return Real::approx(a.value() + b.value());
}

Real operator-(const Real& a, const Real& b) { return a + (-b); }

Real Real::operator-() const
{
    if (isExact && r.n != INT64_MIN && p.n != INT64_MIN) return Real(Rational{-r.n, r.d}, Rational{-p.n, p.d});
    return approx(-value());
}

Real operator*(const Real& a, const Real& b)
{
    if (a.isExact && b.isExact && (a.p.isZero() || b.p.isZero()))
    {
        const auto r = mul(a.r, b.r);
        const auto p1 = mul(a.r, b.p);
        const auto p2 = mul(a.p, b.r);
        if (r && p1 && p2)
            if (const auto p = add(*p1, *p2)) return Real(*r, *p);
    }
    return Real::approx(a.value() * b.value());
}

Real operator/(const Real& a, const Real& b)
{
    if (a.isExact && b.isExact)
    {
        if (b.p.isZero() && !b.r.isZero())
        {
            const auto r = div(a.r, b.r);
            const auto p = div(a.p, b.r);
            if (r && p) return Real(*r, *p);
        }
        else if (b.r.isZero() && !b.p.isZero() && a.r.isZero())
        {
            if (const auto q = div(a.p, b.p)) return Real::rational(*q);
        }
    }
    return Real::approx(a.value() / b.value());
}

std::string Real::str() const
{
    if (!isExact)
    {
        char buf[64];
        const auto res = std::to_chars(buf, buf + sizeof buf, approxValue);
        return std::string(buf, res.ptr);
    }
    auto ratStr = [](const Rational& q) { return q.d == 1 ? std::to_string(q.n) : std::format("{}/{}", q.n, q.d); };
    std::string piPart;
    if (!p.isZero())
    {
        const std::string sign = p.n < 0 ? "-" : "";
        const std::int64_t an = p.n < 0 ? -p.n : p.n;
        piPart = sign + (an == 1 ? "" : std::to_string(an)) + "π" + (p.d == 1 ? "" : "/" + std::to_string(p.d));
    }
    if (r.isZero()) return piPart.empty() ? "0" : piPart;
    if (piPart.empty()) return ratStr(r);
    return std::format("{} {} {}", piPart, r.n < 0 ? "-" : "+", ratStr(Rational{r.n < 0 ? -r.n : r.n, r.d}));
}


// ---- Affine ----

Affine Affine::param(std::uint32_t index)
{
    Affine a;
    a.terms.emplace_back(index, Real::integer(1));
    return a;
}

double Affine::eval(const std::vector<double>& params) const
{
    double v = c.value();
    for (const auto& [k, coef] : terms) v += coef.value() * params.at(k);
    return v;
}

Affine operator+(const Affine& a, const Affine& b)
{
    Affine out(a.c + b.c);
    std::size_t i = 0, j = 0;
    while (i < a.terms.size() || j < b.terms.size())
    {
        if (j >= b.terms.size() || (i < a.terms.size() && a.terms[i].first < b.terms[j].first))
            out.terms.push_back(a.terms[i++]);
        else if (i >= a.terms.size() || b.terms[j].first < a.terms[i].first)
            out.terms.push_back(b.terms[j++]);
        else
        {
            Real s = a.terms[i].second + b.terms[j].second;
            if (!s.isZero()) out.terms.emplace_back(a.terms[i].first, s);
            ++i;
            ++j;
        }
    }
    return out;
}

Affine Affine::operator-() const
{
    Affine out(-c);
    for (const auto& [k, coef] : terms) out.terms.emplace_back(k, -coef);
    return out;
}

Affine operator-(const Affine& a, const Affine& b) { return a + (-b); }

Affine Affine::scaled(const Real& s) const
{
    Affine out(c * s);
    if (s.isZero()) return out;
    for (const auto& [k, coef] : terms) out.terms.emplace_back(k, coef * s);
    return out;
}

std::string Affine::str(const std::vector<std::string>& paramNames) const
{
    std::string out;
    for (const auto& [k, coef] : terms)
    {
        const std::string name = k < paramNames.size() ? paramNames[k] : std::format("p{}", k);
        const std::string cs = coef.str();
        if (!out.empty()) out += " + ";
        out += cs == "1" ? name : (cs == "-1" ? "-" + name : cs + "·" + name);
    }
    if (!c.isZero() || out.empty()) out += (out.empty() ? "" : " + ") + c.str();
    return out;
}


// ---- Num ----

Num Num::complex(Real r, Real i)
{
    Num n(std::move(r));
    n.im = Affine(std::move(i));
    return n;
}

Num Num::fromDouble(cd z)
{
    return complex(Real::approx(z.real()), z.imag() == 0.0 ? Real::integer(0) : Real::approx(z.imag()));
}

std::optional<std::int64_t> Num::asInt() const
{
    if (!isReal() || !re.isConst()) return std::nullopt;
    return re.c.asInt();
}

cd Num::constValue() const { return {re.c.value(), im.c.value()}; }

cd Num::eval(const std::vector<double>& params) const { return {re.eval(params), im.eval(params)}; }

namespace
{
    // a·b for affine a, b; nullopt when both depend on params (not affine).
    std::optional<Affine> affMul(const Affine& a, const Affine& b)
    {
        if (a.isConst()) return b.scaled(a.c);
        if (b.isConst()) return a.scaled(b.c);
        return std::nullopt;
    }

    bool isExactOne(const Num& n)
    {
        return n.isConst() && n.im.isZero() && n.re.c.asInt() == 1;
    }
} // namespace

Num operator+(const Num& a, const Num& b)
{
    Num out;
    out.re = a.re + b.re;
    out.im = a.im + b.im;
    out.nonlinear = a.nonlinear || b.nonlinear;
    return out;
}

Num operator-(const Num& a)
{
    Num out;
    out.re = -a.re;
    out.im = -a.im;
    out.nonlinear = a.nonlinear;
    if (a.phase) out.phase = *a.phase + Affine(Real::pi(Rational{1, 1}));
    return out;
}

Num operator-(const Num& a, const Num& b) { return a + (-b); }

Num operator*(const Num& a, const Num& b)
{
    Num out;
    const auto rr = affMul(a.re, b.re), ii = affMul(a.im, b.im), ri = affMul(a.re, b.im), ir = affMul(a.im, b.re);
    out.nonlinear = a.nonlinear || b.nonlinear || !rr || !ii || !ri || !ir;
    if (!out.nonlinear)
    {
        out.re = *rr - *ii;
        out.im = *ri + *ir;
    }
    else
    {
        // Keep the constant part's value so compile-time checks still see a number.
        const cd v = a.constValue() * b.constValue();
        out.re = Affine(Real::approx(v.real()));
        out.im = Affine(Real::approx(v.imag()));
    }
    if (a.phase && b.phase) out.phase = *a.phase + *b.phase;
    else if (a.phase && isExactOne(b)) out.phase = a.phase;
    else if (b.phase && isExactOne(a)) out.phase = b.phase;
    return out;
}

std::optional<Num> divide(const Num& a, const Num& b)
{
    if (b.nonlinear || !b.re.isConst() || !b.im.isConst())
    {
        Num out;
        out.nonlinear = true;
        const cd v = a.constValue() / b.constValue();
        out.re = Affine(Real::approx(v.real()));
        out.im = Affine(Real::approx(v.imag()));
        return out;
    }
    const Real& c = b.re.c;
    const Real& d = b.im.c;
    const Real den = c * c + d * d;
    if (den.isZero()) return std::nullopt;
    Num inv = Num::complex(c / den, (-d) / den);
    Num out = a * inv;
    if (a.phase && b.phase) out.phase = *a.phase - *b.phase;
    return out;
}

Num conj(const Num& a)
{
    Num out = a;
    out.im = -a.im;
    if (a.phase) out.phase = -*a.phase;
    return out;
}

Num expi(const Affine& angle)
{
    Num out;
    if (angle.isConst())
    {
        const Real& t = angle.c;
        // Multiples of π/2 have exact sines and cosines.
        if (t.exact() && t.rat().isZero())
        {
            if (const auto q = mul(t.piCoef(), Rational{2, 1}); q && q->isInt())
            {
                const std::int64_t k = ((q->n % 4) + 4) % 4;
                const std::int64_t cs[4] = {1, 0, -1, 0}, sn[4] = {0, 1, 0, -1};
                out = Num::complex(Real::integer(cs[k]), Real::integer(sn[k]));
                out.phase = angle;
                return out;
            }
        }
        const double v = t.value();
        out = Num::complex(Real::approx(std::cos(v)), Real::approx(std::sin(v)));
    }
    else
    {
        out.nonlinear = true;
        out.re = Affine(Real::approx(std::cos(angle.c.value())));
        out.im = Affine(Real::approx(std::sin(angle.c.value())));
    }
    out.phase = angle;
    return out;
}


// ---- Kets ----

std::size_t KetV::nq() const
{
    std::size_t n = 0;
    for (const KetFactor& f : factors) n += f.nq;
    return n;
}

bool KetV::isBasis() const
{
    return std::ranges::all_of(factors, [](const KetFactor& f) { return f.kind == KetFactor::Kind::Basis; });
}

bool KetV::isStabilizerProduct() const
{
    return std::ranges::none_of(factors, [](const KetFactor& f) { return f.kind == KetFactor::Kind::Dense; });
}

std::string KetV::basisBits() const
{
    std::string bits;
    for (const KetFactor& f : factors) bits += f.bits;
    return bits;
}

Eigen::VectorXcd KetV::dense() const
{
    const double h = 1.0 / std::sqrt(2.0);
    Eigen::VectorXcd out = Eigen::VectorXcd::Ones(1);
    for (const KetFactor& f : factors)
    {
        Eigen::VectorXcd v;
        switch (f.kind)
        {
            case KetFactor::Kind::Basis:
            {
                v = Eigen::VectorXcd::Zero(Eigen::Index{1} << f.bits.size());
                std::size_t idx = 0;
                for (const char c : f.bits) idx = (idx << 1) | (c == '1' ? 1U : 0U);
                v[static_cast<Eigen::Index>(idx)] = 1.0;
                break;
            }
            case KetFactor::Kind::Plus: v = Eigen::Vector2cd(h, h); break;
            case KetFactor::Kind::Minus: v = Eigen::Vector2cd(h, -h); break;
            case KetFactor::Kind::PlusI: v = Eigen::Vector2cd(h, cd(0, h)); break;
            case KetFactor::Kind::MinusI: v = Eigen::Vector2cd(h, cd(0, -h)); break;
            case KetFactor::Kind::Dense: v = f.amp; break;
        }
        Eigen::VectorXcd k(out.size() * v.size());
        for (Eigen::Index a = 0; a < out.size(); ++a)
            for (Eigen::Index b = 0; b < v.size(); ++b) k[a * v.size() + b] = out[a] * v[b];
        out = std::move(k);
    }
    return out * scale;
}

double KetV::norm2() const
{
    double n = std::norm(scale);
    for (const KetFactor& f : factors)
        if (f.kind == KetFactor::Kind::Dense) n *= f.amp.squaredNorm();
    return n;
}

std::size_t MatV::nq() const
{
    std::size_t n = 0;
    while ((Eigen::Index{1} << n) < m.rows()) ++n;
    return n;
}


// ---- Ops ----

std::string_view gkName(GK k)
{
    static constexpr std::string_view names[] = {
        "I", "X", "Y", "Z", "H", "S", "S†", "T", "T†", "√X", "√X†", "Rx", "Ry", "Rz", "P", "U3",
        "CNOT", "CZ", "CP", "SWAP", "Toffoli", "Fredkin", "exp", "U", "phase", "measure", "measure",
        "reset", "channel"};
    return names[static_cast<std::size_t>(k)];
}

QubitList IrOp::qubits() const
{
    QubitList q = controls;
    q.insert(q.end(), negControls.begin(), negControls.end());
    q.insert(q.end(), targets.begin(), targets.end());
    return q;
}

bool OpV::unitary() const
{
    return std::ranges::all_of(ops, [](const IrOp& o)
    {
        return o.kind != GK::MeasureZ && o.kind != GK::MeasurePauli && o.kind != GK::Reset && o.kind != GK::Channel &&
               !(o.kind == GK::Matrix && !o.unitaryMatrix);
    });
}

bool OpV::hasChannel() const
{
    return std::ranges::any_of(ops, [](const IrOp& o) { return o.kind == GK::Channel; });
}


// ---- Types ----

std::string_view typeName(Type t)
{
    switch (t)
    {
        case Type::None: return "None";
        case Type::Int: return "Int";
        case Type::Real: return "Real";
        case Type::Complex: return "Complex";
        case Type::Bool: return "Bool";
        case Type::String: return "String";
        case Type::Ket: return "Ket";
        case Type::Bra: return "Bra";
        case Type::Matrix: return "Matrix";
        case Type::Op: return "Op";
        case Type::Obs: return "Obs";
        case Type::Counts: return "Counts";
        case Type::Bits: return "Bits";
        case Type::List: return "List";
        case Type::Gate: return "Gate";
        case Type::Qubits: return "Qubits";
        case Type::Channel: return "Channel";
    }
    return "?";
}

Type Value::type() const
{
    if (const auto* n = get<Num>())
    {
        if (n->asInt()) return Type::Int;
        return n->nonlinear ? (n->im.isZero() ? Type::Real : Type::Complex) : (n->im.isZero() ? Type::Real : Type::Complex);
    }
    if (is<bool>() || is<CondV>()) return Type::Bool;
    if (is<std::string>()) return Type::String;
    if (is<KetV>()) return Type::Ket;
    if (is<BraV>() || is<BraOpV>()) return Type::Bra;
    if (is<MatV>() || is<RhoV>()) return Type::Matrix;
    if (is<OpV>()) return Type::Op;
    if (const auto* l = get<LinOpV>())
    {
        for (const PTerm& t : l->paulis)
            if (!t.coef.im.isZero()) return Type::Op;
        return l->isPauli() ? Type::Obs : Type::Op;
    }
    if (is<QubitsV>()) return Type::Qubits;
    if (is<BitsV>()) return Type::Bits;
    if (is<CountsV>()) return Type::Counts;
    if (const auto* d = get<DeferredV>()) return d->type;
    if (const auto* g = get<GateV>()) return g->kind == GateV::Kind::Channel ? Type::Channel : Type::Gate;
    if (is<ControlV>()) return Type::Gate;
    if (is<ListV>()) return Type::List;
    return Type::None;
}

Stage Value::stage() const
{
    if (const auto* n = get<Num>()) return n->isConst() ? Stage::Compile : Stage::Bind;
    if (const auto* d = get<DeferredV>()) return d->stage;
    if (is<CondV>() || is<CountsV>() || is<RhoV>()) return Stage::Run;
    if (const auto* k = get<KetV>()) return k->live ? Stage::Run : Stage::Compile;
    return Stage::Compile;
}


// ---- Matrices ----

Eigen::MatrixXcd pauliMatrix(char letter)
{
    Eigen::Matrix2cd m;
    switch (letter)
    {
        case 'X': m << 0, 1, 1, 0; break;
        case 'Y': m << 0, cd(0, -1), cd(0, 1), 0; break;
        case 'Z': m << 1, 0, 0, -1; break;
        default: m << 1, 0, 0, 1; break;
    }
    return m;
}

Eigen::MatrixXcd gateMatrix(GK k, const std::vector<double>& a)
{
    const double h = 1.0 / std::sqrt(2.0);
    const cd i(0, 1);
    auto e = [](double t) { return std::polar(1.0, t); };
    Eigen::MatrixXcd m;
    switch (k)
    {
        case GK::I: return Eigen::Matrix2cd::Identity();
        case GK::X: return pauliMatrix('X');
        case GK::Y: return pauliMatrix('Y');
        case GK::Z: return pauliMatrix('Z');
        case GK::H: m = Eigen::Matrix2cd(); m << h, h, h, -h; return m;
        case GK::S: m = Eigen::Matrix2cd(); m << 1, 0, 0, i; return m;
        case GK::Sdg: m = Eigen::Matrix2cd(); m << 1, 0, 0, -i; return m;
        case GK::T: m = Eigen::Matrix2cd(); m << 1, 0, 0, e(std::numbers::pi / 4); return m;
        case GK::Tdg: m = Eigen::Matrix2cd(); m << 1, 0, 0, e(-std::numbers::pi / 4); return m;
        case GK::SX: m = Eigen::Matrix2cd(); m << cd(0.5, 0.5), cd(0.5, -0.5), cd(0.5, -0.5), cd(0.5, 0.5); return m;
        case GK::SXdg: m = Eigen::Matrix2cd(); m << cd(0.5, -0.5), cd(0.5, 0.5), cd(0.5, 0.5), cd(0.5, -0.5); return m;
        case GK::RX:
        {
            const double c = std::cos(a[0] / 2), s = std::sin(a[0] / 2);
            m = Eigen::Matrix2cd();
            m << c, -i * s, -i * s, c;
            return m;
        }
        case GK::RY:
        {
            const double c = std::cos(a[0] / 2), s = std::sin(a[0] / 2);
            m = Eigen::Matrix2cd();
            m << c, -s, s, c;
            return m;
        }
        case GK::RZ: m = Eigen::Matrix2cd(); m << e(-a[0] / 2), 0, 0, e(a[0] / 2); return m;
        case GK::P: m = Eigen::Matrix2cd(); m << 1, 0, 0, e(a[0]); return m;
        case GK::U3:
        {
            const double c = std::cos(a[0] / 2), s = std::sin(a[0] / 2);
            m = Eigen::Matrix2cd();
            m << c, -s * e(a[2]), s * e(a[1]), c * e(a[1] + a[2]);
            return m;
        }
        case GK::CNOT:
            m = Eigen::MatrixXcd::Identity(4, 4);
            m(2, 2) = 0; m(3, 3) = 0; m(2, 3) = 1; m(3, 2) = 1;
            return m;
        case GK::CZ: m = Eigen::MatrixXcd::Identity(4, 4); m(3, 3) = -1; return m;
        case GK::CP: m = Eigen::MatrixXcd::Identity(4, 4); m(3, 3) = e(a[0]); return m;
        case GK::SWAP:
            m = Eigen::MatrixXcd::Identity(4, 4);
            m(1, 1) = 0; m(2, 2) = 0; m(1, 2) = 1; m(2, 1) = 1;
            return m;
        case GK::Toffoli:
            m = Eigen::MatrixXcd::Identity(8, 8);
            m(6, 6) = 0; m(7, 7) = 0; m(6, 7) = 1; m(7, 6) = 1;
            return m;
        case GK::Fredkin:
            m = Eigen::MatrixXcd::Identity(8, 8);
            m(5, 5) = 0; m(6, 6) = 0; m(5, 6) = 1; m(6, 5) = 1;
            return m;
        case GK::GPhase: m = Eigen::MatrixXcd::Identity(1, 1); m(0, 0) = e(a[0]); return m;
        default: break;
    }
    throw std::logic_error(std::format("gateMatrix: no fixed matrix for {}", gkName(k)));
}

Eigen::MatrixXcd builtinMatrix(Builtin g, const std::vector<double>& p, bool dagger)
{
    Eigen::MatrixXcd m;
    switch (g)
    {
        case Builtin::I: m = gateMatrix(GK::I, p); break;
        case Builtin::X: m = gateMatrix(GK::X, p); break;
        case Builtin::Y: m = gateMatrix(GK::Y, p); break;
        case Builtin::Z: m = gateMatrix(GK::Z, p); break;
        case Builtin::H: m = gateMatrix(GK::H, p); break;
        case Builtin::S: m = gateMatrix(GK::S, p); break;
        case Builtin::T: m = gateMatrix(GK::T, p); break;
        case Builtin::SX: m = gateMatrix(GK::SX, p); break;
        case Builtin::Rx: m = gateMatrix(GK::RX, p); break;
        case Builtin::Ry: m = gateMatrix(GK::RY, p); break;
        case Builtin::Rz: m = gateMatrix(GK::RZ, p); break;
        case Builtin::P: m = gateMatrix(GK::P, p); break;
        case Builtin::U3: m = gateMatrix(GK::U3, p); break;
        case Builtin::CNOT: m = gateMatrix(GK::CNOT, p); break;
        case Builtin::CZ: m = gateMatrix(GK::CZ, p); break;
        case Builtin::CP: m = gateMatrix(GK::CP, p); break;
        case Builtin::SWAP: m = gateMatrix(GK::SWAP, p); break;
        case Builtin::Toffoli: m = gateMatrix(GK::Toffoli, p); break;
        case Builtin::Fredkin: m = gateMatrix(GK::Fredkin, p); break;
        case Builtin::C: throw std::logic_error("builtinMatrix: C has no fixed matrix");
    }
    if (dagger) m.adjointInPlace();
    return m;
}


// ---- Pauli algebra ----

std::pair<int, char> pauliProduct(char a, char b)
{
    if (a == 'I') return {0, b};
    if (b == 'I') return {0, a};
    if (a == b) return {0, 'I'};
    // XY = iZ, YZ = iX, ZX = iY; reversed order gives -i.
    auto cyclic = [](char x, char y) { return (x == 'X' && y == 'Y') || (x == 'Y' && y == 'Z') || (x == 'Z' && y == 'X'); };
    char r = 'I';
    for (const char c : {'X', 'Y', 'Z'})
        if (c != a && c != b) r = c;
    return {cyclic(a, b) ? 1 : 3, r};
}

PTerm multiply(const PTerm& a, const PTerm& b)
{
    PTerm out;
    int phase = 0;
    std::size_t i = 0, j = 0;
    while (i < a.ps.size() || j < b.ps.size())
    {
        if (j >= b.ps.size() || (i < a.ps.size() && a.ps[i].first < b.ps[j].first)) out.ps.push_back(a.ps[i++]);
        else if (i >= a.ps.size() || b.ps[j].first < a.ps[i].first) out.ps.push_back(b.ps[j++]);
        else
        {
            const auto [ph, letter] = pauliProduct(a.ps[i].second, b.ps[j].second);
            phase += ph;
            if (letter != 'I') out.ps.emplace_back(a.ps[i].first, letter);
            ++i;
            ++j;
        }
    }
    static const Num kPhases[4] = {Num(Real::integer(1)), Num::complex(Real::integer(0), Real::integer(1)),
                                   Num(Real::integer(-1)), Num::complex(Real::integer(0), Real::integer(-1))};
    out.coef = a.coef * b.coef * kPhases[phase % 4];
    return out;
}

bool commute(const PTerm& a, const PTerm& b)
{
    int anti = 0;
    std::size_t i = 0, j = 0;
    while (i < a.ps.size() && j < b.ps.size())
    {
        if (a.ps[i].first < b.ps[j].first) ++i;
        else if (b.ps[j].first < a.ps[i].first) ++j;
        else
        {
            if (a.ps[i].second != b.ps[j].second) ++anti;
            ++i;
            ++j;
        }
    }
    return anti % 2 == 0;
}

LinOpV simplify(LinOpV op)
{
    std::map<std::vector<std::pair<Qubit, char>>, Num> merged;
    std::vector<std::vector<std::pair<Qubit, char>>> order;
    for (PTerm& t : op.paulis)
    {
        auto it = merged.find(t.ps);
        if (it == merged.end())
        {
            order.push_back(t.ps);
            merged.emplace(t.ps, t.coef);
        }
        else it->second = it->second + t.coef;
    }
    LinOpV out;
    for (const auto& key : order)
    {
        const Num& c = merged.at(key);
        if (c.isConst() && c.re.isZero() && c.im.isZero()) continue;
        out.paulis.push_back({c, key});
    }
    out.general = std::move(op.general);
    return out;
}

} // namespace Noether
