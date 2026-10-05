#include "Decompose.hpp"

#include <Eigen/Eigenvalues>

#include <array>
#include <cmath>
#include <format>
#include <numbers>
#include <stdexcept>



namespace Noether
{

namespace
{
    // Alternatives per gate type, in preference order; each lists the types its rule emits. A
    // type's rule is the first alternative whose components were expressible before it.
    const std::vector<std::pair<std::string, std::vector<std::vector<std::string>>>>& rules()
    {
        static const std::vector<std::pair<std::string, std::vector<std::vector<std::string>>>> r{
            {"X", {{"H", "Z"}, {"√X"}, {"Rx"}, {"U3"}}},
            {"Y", {{"X", "Z"}, {"Ry"}, {"U3"}}},
            {"Z", {{"S"}, {"T"}, {"Rz"}, {"P"}, {"S†"}}},
            {"S", {{"T"}, {"P"}, {"Rz"}, {"S†"}}},
            {"S†", {{"S"}, {"T†"}, {"T"}, {"P"}, {"Rz"}}},
            {"T", {{"P"}, {"Rz"}, {"U3"}, {"S", "T†"}, {"T†"}}},
            {"T†", {{"S†", "T"}, {"T"}, {"P"}, {"Rz"}, {"U3"}}},
            {"H", {{"S", "√X"}, {"Rz", "Rx"}, {"Ry", "Z"}, {"U3"}}},
            {"√X", {{"H", "S"}, {"Rx"}, {"U3"}, {"√X†"}}},
            {"√X†", {{"H", "S†"}, {"√X", "X"}, {"Rx"}, {"U3"}}},
            {"Rx", {{"H", "Rz"}, {"U3"}}},
            {"Ry", {{"S", "S†", "Rx"}, {"U3"}}},
            {"Rz", {{"P"}, {"H", "Rx"}, {"U3"}}},
            {"P", {{"Rz"}, {"U3"}}},
            {"U3", {{"Rz", "Ry"}}},
            {"CNOT", {{"H", "CZ"}, {"H", "CP"}}},
            {"CZ", {{"H", "CNOT"}, {"CP"}}},
            {"CP", {{"P", "CNOT"}, {"Rz", "CNOT"}}},
            {"SWAP", {{"CNOT"}}},
            {"Toffoli", {{"H", "T", "T†", "CNOT"}, {"H", "CP", "CNOT"}}},
            {"Fredkin", {{"CNOT", "Toffoli"}}},
        };
        return r;
    }

    std::optional<GK> kindOfName(std::string_view n)
    {
        static const std::map<std::string_view, GK> m{
            {"I", GK::I}, {"X", GK::X}, {"Y", GK::Y}, {"Z", GK::Z}, {"H", GK::H}, {"S", GK::S}, {"S†", GK::Sdg},
            {"T", GK::T}, {"T†", GK::Tdg}, {"√X", GK::SX}, {"√X†", GK::SXdg}, {"Rx", GK::RX}, {"Ry", GK::RY},
            {"Rz", GK::RZ}, {"P", GK::P}, {"U3", GK::U3}, {"CNOT", GK::CNOT}, {"CZ", GK::CZ}, {"CP", GK::CP},
            {"SWAP", GK::SWAP}, {"Toffoli", GK::Toffoli}, {"Fredkin", GK::Fredkin}};
        const auto it = m.find(n);
        return it == m.end() ? std::nullopt : std::optional<GK>(it->second);
    }

    IrOp gate(GK k, QubitList targets, Span at, std::vector<Affine> angles = {})
    {
        IrOp o;
        o.kind = k;
        o.targets = std::move(targets);
        o.angles = std::move(angles);
        o.span = at;
        return o;
    }

    Affine piFrac(std::int64_t n, std::int64_t d) { return Affine(Real::pi(Rational::make(n, d).value_or(Rational{n, d}))); }

    // k with angle = kπ/4 exactly, if any.
    std::optional<std::int64_t> eighthTurns(const Affine& a)
    {
        if (!a.isConst() || !a.c.exact() || !a.c.rat().isZero()) return std::nullopt;
        const Rational p = a.c.piCoef();
        if ((p.n * 4) % p.d != 0) return std::nullopt;
        const std::int64_t k = (p.n * 4) / p.d;
        return ((k % 8) + 8) % 8;
    }

    class Lowerer
    {
        public:
        Lowerer(const BasisLowering& b, std::vector<IrOp>& o) : bl(b), out(o) {}

        bool has(std::string_view t) const { return bl.expressible(t); }

        [[noreturn]] void fail(const IrOp& op, std::string why) const
        {
            throw std::runtime_error(std::format("`{}` cannot be lowered exactly into the gate set: {}", basisName(op), why));
        }

        void emit(const IrOp& op, int depth = 0)
        {
            if (depth > 64) fail(op, "the decomposition does not terminate");
            switch (op.kind)
            {
                case GK::MeasureZ: case GK::MeasurePauli: case GK::Reset: case GK::Channel:
                    out.push_back(op);
                    return;
                default: break;
            }
            if (!op.negControls.empty())
            {
                IrOp inner = op;
                inner.controls.insert(inner.controls.end(), op.negControls.begin(), op.negControls.end());
                inner.negControls.clear();
                for (const Qubit q : op.negControls) emit(withCond(gate(GK::X, {q}, op.span), op), depth + 1);
                emit(inner, depth + 1);
                for (const Qubit q : op.negControls) emit(withCond(gate(GK::X, {q}, op.span), op), depth + 1);
                return;
            }
            if (!op.controls.empty()) return controlled(op, depth);
            plain(op, depth);
        }

        private:
        static IrOp withCond(IrOp o, const IrOp& like)
        {
            o.cond = like.cond;
            return o;
        }

        void push(const IrOp& like, GK k, QubitList t, std::vector<Affine> a, int depth)
        {
            emit(withCond(gate(k, std::move(t), like.span, std::move(a)), like), depth + 1);
        }

        // Z-axis rotation by kπ/4 as T/S/Z gates (up to global phase).
        bool zEighths(const IrOp& like, Qubit q, std::int64_t k, int depth)
        {
            if (k == 0) return true;
            const bool t = has("T") || has("T†");
            if (k % 2 != 0 && !t) return false;
            if (k % 2 == 0 && !has("S") && !has("Z") && !has("S†")) return false;
            switch (k)
            {
                case 1: push(like, GK::T, {q}, {}, depth); break;
                case 2: push(like, GK::S, {q}, {}, depth); break;
                case 3: push(like, GK::S, {q}, {}, depth); push(like, GK::T, {q}, {}, depth); break;
                case 4: push(like, GK::Z, {q}, {}, depth); break;
                case 5: push(like, GK::Z, {q}, {}, depth); push(like, GK::T, {q}, {}, depth); break;
                case 6: push(like, GK::Sdg, {q}, {}, depth); break;
                case 7: push(like, GK::Tdg, {q}, {}, depth); break;
                default: break;
            }
            return true;
        }

        void plain(const IrOp& op, int depth)
        {
            const std::string name = basisName(op);
            if (op.kind == GK::I || op.kind == GK::GPhase) return;
            if (bl.expressible(name) && bl.chosenRule(name) == SIZE_MAX)
            {
                out.push_back(op);
                return;
            }
            const QubitList& t = op.targets;
            const Span at = op.span;
            (void)at;
            // Exact angles take the Clifford+T route before the arbitrary-angle rules.
            if (op.kind == GK::RZ || op.kind == GK::P)
                if (const auto k = eighthTurns(op.angles[0]); k && zEighths(op, t[0], *k, depth)) return;
            if (op.kind == GK::RX)
                if (const auto k = eighthTurns(op.angles[0]); k && has("H") && (has("T") || *k % 2 == 0) && (has("S") || has("T")))
                {
                    push(op, GK::H, {t[0]}, {}, depth);
                    zEighths(op, t[0], *k, depth);
                    push(op, GK::H, {t[0]}, {}, depth);
                    return;
                }
            if (op.kind == GK::RY)
                if (const auto k = eighthTurns(op.angles[0]); k && has("H") && has("S") && (has("T") || *k % 2 == 0))
                {
                    // Ry = S·Rx·S†
                    push(op, GK::Sdg, {t[0]}, {}, depth);
                    push(op, GK::RX, {t[0]}, op.angles, depth);
                    push(op, GK::S, {t[0]}, {}, depth);
                    return;
                }
            if (op.kind == GK::CP)
            {
                const auto k = eighthTurns(op.angles[0]);
                if (k && *k == 4 && has("CZ"))
                {
                    push(op, GK::CZ, t, {}, depth);
                    return;
                }
                // CP(λ) = P(λ/2)_a · CNOT · P(−λ/2)_b · CNOT · P(λ/2)_b; for λ ∈ (π/2)ℤ the halves are
                // multiples of π/4, which the Clifford+T route expresses exactly.
                if (k && *k % 2 == 0 && bl.expressible("CNOT") && (has("T") || has("T†") || *k % 4 == 0))
                {
                    const Affine half = op.angles[0].scaled(Real::rational({1, 2}));
                    push(op, GK::P, {t[1]}, {half}, depth);
                    push(op, GK::CNOT, t, {}, depth);
                    push(op, GK::P, {t[1]}, {-half}, depth);
                    push(op, GK::CNOT, t, {}, depth);
                    push(op, GK::P, {t[0]}, {half}, depth);
                    return;
                }
            }
            if (op.kind == GK::PauliRot)
            {
                for (std::size_t j = 0; j < t.size(); ++j) basisIn(op, t[j], op.paulis[j], depth);
                for (std::size_t j = 0; j + 1 < t.size(); ++j) push(op, GK::CNOT, {t[j], t.back()}, {}, depth);
                push(op, GK::RZ, {t.back()}, op.angles, depth);
                for (std::size_t j = t.size() - 1; j-- > 0;) push(op, GK::CNOT, {t[j], t.back()}, {}, depth);
                for (std::size_t j = 0; j < t.size(); ++j) basisOut(op, t[j], op.paulis[j], depth);
                return;
            }
            if (op.kind == GK::Matrix)
            {
                if (t.size() != 1) fail(op, "dense matrices on more than one qubit have no synthesis rule");
                const auto [alpha, angles] = zyz(*op.matrix);
                (void)alpha;
                push(op, GK::U3, t, {Affine(Real::approx(angles[0])), Affine(Real::approx(angles[1])), Affine(Real::approx(angles[2]))},
                     depth);
                return;
            }
            const std::size_t rule = bl.chosenRule(name);
            if (rule == SIZE_MAX) fail(op, "no rule reaches it");
            apply(op, name, rule, depth);
        }

        void basisIn(const IrOp& like, Qubit q, char p, int depth)
        {
            if (p == 'X') push(like, GK::H, {q}, {}, depth);
            else if (p == 'Y')
            {
                push(like, GK::Sdg, {q}, {}, depth);
                push(like, GK::H, {q}, {}, depth);
            }
        }

        void basisOut(const IrOp& like, Qubit q, char p, int depth)
        {
            if (p == 'X') push(like, GK::H, {q}, {}, depth);
            else if (p == 'Y')
            {
                push(like, GK::H, {q}, {}, depth);
                push(like, GK::S, {q}, {}, depth);
            }
        }

        // The rule `rule` of type `name` (indices follow rules()).
        void apply(const IrOp& op, const std::string& name, std::size_t rule, int depth)
        {
            const QubitList& t = op.targets;
            const std::vector<Affine>& a = op.angles;
            auto g = [&](GK k, QubitList q, std::vector<Affine> ang = {}) { push(op, k, std::move(q), std::move(ang), depth); };
            const Affine halfPi = piFrac(1, 2), negHalfPi = piFrac(-1, 2);
            if (name == "X")
            {
                if (rule == 0) { g(GK::H, t); g(GK::Z, t); g(GK::H, t); }
                else if (rule == 1) { g(GK::SX, t); g(GK::SX, t); }
                else if (rule == 2) g(GK::RX, t, {piFrac(1, 1)});
                else g(GK::U3, t, {piFrac(1, 1), Affine(Real::integer(0)), piFrac(1, 1)});
            }
            else if (name == "Y")
            {
                if (rule == 0) { g(GK::Z, t); g(GK::X, t); }
                else if (rule == 1) g(GK::RY, t, {piFrac(1, 1)});
                else g(GK::U3, t, {piFrac(1, 1), halfPi, halfPi});
            }
            else if (name == "Z")
            {
                if (rule == 0) { g(GK::S, t); g(GK::S, t); }
                else if (rule == 1) for (int k = 0; k < 4; ++k) g(GK::T, t);
                else if (rule == 2) g(GK::RZ, t, {piFrac(1, 1)});
                else if (rule == 3) g(GK::P, t, {piFrac(1, 1)});
                else { g(GK::Sdg, t); g(GK::Sdg, t); }
            }
            else if (name == "S")
            {
                if (rule == 0) { g(GK::T, t); g(GK::T, t); }
                else if (rule == 1) g(GK::P, t, {halfPi});
                else if (rule == 2) g(GK::RZ, t, {halfPi});
                else for (int k = 0; k < 3; ++k) g(GK::Sdg, t);
            }
            else if (name == "S†")
            {
                if (rule == 0) for (int k = 0; k < 3; ++k) g(GK::S, t);
                else if (rule == 1) { g(GK::Tdg, t); g(GK::Tdg, t); }
                else if (rule == 2) for (int k = 0; k < 6; ++k) g(GK::T, t);
                else if (rule == 3) g(GK::P, t, {negHalfPi});
                else g(GK::RZ, t, {negHalfPi});
            }
            else if (name == "T")
            {
                if (rule == 0) g(GK::P, t, {piFrac(1, 4)});
                else if (rule == 1) g(GK::RZ, t, {piFrac(1, 4)});
                else if (rule == 2) g(GK::U3, t, {Affine(Real::integer(0)), Affine(Real::integer(0)), piFrac(1, 4)});
                else if (rule == 3) { g(GK::Tdg, t); g(GK::S, t); } // T = S·T†: one T-type gate
                else for (int k = 0; k < 7; ++k) g(GK::Tdg, t);
            }
            else if (name == "T†")
            {
                if (rule == 0) { g(GK::T, t); g(GK::Sdg, t); } // T† = S†·T: one T gate, not seven
                else if (rule == 1) for (int k = 0; k < 7; ++k) g(GK::T, t);
                else if (rule == 2) g(GK::P, t, {piFrac(-1, 4)});
                else if (rule == 3) g(GK::RZ, t, {piFrac(-1, 4)});
                else g(GK::U3, t, {Affine(Real::integer(0)), Affine(Real::integer(0)), piFrac(-1, 4)});
            }
            else if (name == "H")
            {
                if (rule == 0) { g(GK::S, t); g(GK::SX, t); g(GK::S, t); }
                else if (rule == 1) { g(GK::RZ, t, {halfPi}); g(GK::RX, t, {halfPi}); g(GK::RZ, t, {halfPi}); }
                else if (rule == 2) { g(GK::Z, t); g(GK::RY, t, {halfPi}); }
                else g(GK::U3, t, {halfPi, Affine(Real::integer(0)), piFrac(1, 1)});
            }
            else if (name == "√X")
            {
                if (rule == 0) { g(GK::H, t); g(GK::S, t); g(GK::H, t); }
                else if (rule == 1) g(GK::RX, t, {halfPi});
                else if (rule == 2) g(GK::U3, t, {halfPi, negHalfPi, halfPi});
                else for (int k = 0; k < 3; ++k) g(GK::SXdg, t);
            }
            else if (name == "√X†")
            {
                if (rule == 0) { g(GK::H, t); g(GK::Sdg, t); g(GK::H, t); }
                else if (rule == 1) { g(GK::SX, t); g(GK::X, t); }
                else if (rule == 2) g(GK::RX, t, {negHalfPi});
                else g(GK::U3, t, {negHalfPi, negHalfPi, halfPi});
            }
            else if (name == "Rx")
            {
                if (rule == 0) { g(GK::H, t); g(GK::RZ, t, a); g(GK::H, t); }
                else g(GK::U3, t, {a[0], negHalfPi, halfPi});
            }
            else if (name == "Ry")
            {
                if (rule == 0) { g(GK::Sdg, t); g(GK::RX, t, a); g(GK::S, t); }
                else g(GK::U3, t, {a[0], Affine(Real::integer(0)), Affine(Real::integer(0))});
            }
            else if (name == "Rz")
            {
                if (rule == 0) g(GK::P, t, a);
                else if (rule == 1) { g(GK::H, t); g(GK::RX, t, a); g(GK::H, t); }
                else g(GK::U3, t, {Affine(Real::integer(0)), Affine(Real::integer(0)), a[0]});
            }
            else if (name == "P")
            {
                if (rule == 0) g(GK::RZ, t, a);
                else g(GK::U3, t, {Affine(Real::integer(0)), Affine(Real::integer(0)), a[0]});
            }
            else if (name == "U3")
            {
                // U3(θ,φ,λ) = Rz(φ) Ry(θ) Rz(λ) up to global phase.
                g(GK::RZ, t, {a[2]});
                g(GK::RY, t, {a[0]});
                g(GK::RZ, t, {a[1]});
            }
            else if (name == "CNOT")
            {
                if (rule == 0) { g(GK::H, {t[1]}); g(GK::CZ, t); g(GK::H, {t[1]}); }
                else { g(GK::H, {t[1]}); g(GK::CP, t, {piFrac(1, 1)}); g(GK::H, {t[1]}); }
            }
            else if (name == "CZ")
            {
                if (rule == 0) { g(GK::H, {t[1]}); g(GK::CNOT, t); g(GK::H, {t[1]}); }
                else g(GK::CP, t, {piFrac(1, 1)});
            }
            else if (name == "CP")
            {
                // CP(λ) = P(λ/2)_a · CNOT · P(−λ/2)_b · CNOT · P(λ/2)_b, exact.
                const Affine half = a[0].scaled(Real::rational({1, 2}));
                const GK pk = rule == 0 ? GK::P : GK::RZ;
                g(pk, {t[1]}, {half});
                g(GK::CNOT, t);
                g(pk, {t[1]}, {-half});
                g(GK::CNOT, t);
                g(pk, {t[0]}, {half});
            }
            else if (name == "SWAP")
            {
                g(GK::CNOT, {t[0], t[1]});
                g(GK::CNOT, {t[1], t[0]});
                g(GK::CNOT, {t[0], t[1]});
            }
            else if (name == "Toffoli")
            {
                const Qubit c0 = t[0], c1 = t[1], x = t[2];
                if (rule == 0)
                {
                    // Nielsen & Chuang Fig. 4.9: 6 CNOT, 7 T-type gates.
                    g(GK::H, {x});
                    g(GK::CNOT, {c1, x});
                    g(GK::Tdg, {x});
                    g(GK::CNOT, {c0, x});
                    g(GK::T, {x});
                    g(GK::CNOT, {c1, x});
                    g(GK::Tdg, {x});
                    g(GK::CNOT, {c0, x});
                    g(GK::T, {c1});
                    g(GK::T, {x});
                    g(GK::H, {x});
                    g(GK::CNOT, {c0, c1});
                    g(GK::T, {c0});
                    g(GK::Tdg, {c1});
                    g(GK::CNOT, {c0, c1});
                }
                else
                {
                    // C²(X) = C_{c1}(V) CNOT_{c0→c1} C_{c1}(V†) CNOT_{c0→c1} C_{c0}(V), V = √X = H S H.
                    auto cv = [&](Qubit c, bool dag)
                    {
                        g(GK::H, {x});
                        g(GK::CP, {c, x}, {dag ? negHalfPi : halfPi});
                        g(GK::H, {x});
                    };
                    cv(c0, false);
                    g(GK::CNOT, {c0, c1});
                    cv(c1, true);
                    g(GK::CNOT, {c0, c1});
                    cv(c1, false);
                }
            }
            else if (name == "Fredkin")
            {
                g(GK::CNOT, {t[2], t[1]});
                g(GK::Toffoli, {t[0], t[1], t[2]});
                g(GK::CNOT, {t[2], t[1]});
            }
            else fail(op, "no rule");
        }

        // U = e^{iα} U3(θ, φ, λ) for a 2×2 unitary.
        static std::pair<double, std::array<double, 3>> zyz(const Eigen::MatrixXcd& u)
        {
            const cd det = u(0, 0) * u(1, 1) - u(0, 1) * u(1, 0);
            const double alpha = std::arg(det) / 2.0;
            const Eigen::Matrix2cd v = u * std::polar(1.0, -alpha); // det v = 1
            const double theta = 2.0 * std::atan2(std::abs(v(1, 0)), std::abs(v(0, 0)));
            const double sum = 2.0 * std::arg(v(1, 1));  // φ + λ
            const double diff = 2.0 * std::arg(v(1, 0)); // φ − λ
            const double phi = (sum + diff) / 2.0, lambda = (sum - diff) / 2.0;
            // U3 carries e^{i(φ+λ)/2} relative to the SU(2) form.
            return {alpha - (phi + lambda) / 2.0, {theta, phi, lambda}};
        }

        // √U for a single-target op, as ops with the same targets (V² = U exactly, phases included).
        std::vector<IrOp> root(const IrOp& op, bool dagger) const
        {
            const Real h = Real::rational({1, 2});
            const Real s = dagger ? Real::rational({-1, 2}) : h;
            IrOp v = op;
            v.controls.clear();
            v.negControls.clear();
            switch (op.kind)
            {
                case GK::RX: case GK::RY: case GK::RZ: case GK::P: case GK::PauliRot: case GK::GPhase: case GK::CP:
                    for (Affine& x : v.angles) x = x.scaled(s);
                    return {v};
                case GK::Z: v.kind = GK::P; v.angles = {piFrac(dagger ? -1 : 1, 2)}; return {v};
                case GK::S: v.kind = GK::P; v.angles = {piFrac(dagger ? -1 : 1, 4)}; return {v};
                case GK::Sdg: v.kind = GK::P; v.angles = {piFrac(dagger ? 1 : -1, 4)}; return {v};
                case GK::T: v.kind = GK::P; v.angles = {piFrac(dagger ? -1 : 1, 8)}; return {v};
                case GK::Tdg: v.kind = GK::P; v.angles = {piFrac(dagger ? 1 : -1, 8)}; return {v};
                case GK::X: v.kind = dagger ? GK::SXdg : GK::SX; return {v};
                default: break;
            }
            // Numeric square root through the eigendecomposition.
            Eigen::MatrixXcd u;
            if (op.kind == GK::Matrix) u = *op.matrix;
            else
            {
                std::vector<double> ang;
                for (const Affine& x : op.angles)
                {
                    if (!x.isConst()) fail(op, "a param-dependent angle inside a multi-controlled gate");
                    ang.push_back(x.c.value());
                }
                u = gateMatrix(op.kind, ang);
            }
            Eigen::ComplexEigenSolver<Eigen::MatrixXcd> es(u);
            Eigen::VectorXcd d = es.eigenvalues();
            for (Eigen::Index k = 0; k < d.size(); ++k) d[k] = std::sqrt(d[k]);
            Eigen::MatrixXcd r = es.eigenvectors() * d.asDiagonal() * es.eigenvectors().inverse();
            if (dagger) r = r.adjoint().eval();
            v.kind = GK::Matrix;
            v.angles.clear();
            v.matrix = std::make_shared<const Eigen::MatrixXcd>(r);
            return {v};
        }

        void controlled(const IrOp& op, int depth)
        {
            const QubitList& c = op.controls;
            const QubitList& t = op.targets;
            auto plainOp = [&](GK k, QubitList q, std::vector<Affine> a = {})
            {
                emit(withCond(gate(k, std::move(q), op.span, std::move(a)), op), depth + 1);
            };
            auto ctrl = [&](IrOp inner, QubitList controls)
            {
                inner.controls = std::move(controls);
                inner.negControls.clear();
                inner.cond = op.cond;
                inner.span = op.span;
                emit(inner, depth + 1);
            };
            // Multi-target kinds become one target plus controls.
            auto withExtra = [&](GK k, QubitList extra, QubitList targets, std::vector<Affine> a = {})
            {
                IrOp inner = gate(k, std::move(targets), op.span, std::move(a));
                QubitList cs = c;
                cs.insert(cs.end(), extra.begin(), extra.end());
                ctrl(inner, cs);
            };
            switch (op.kind)
            {
                case GK::CNOT: return withExtra(GK::X, {t[0]}, {t[1]});
                case GK::Toffoli: return withExtra(GK::X, {t[0], t[1]}, {t[2]});
                case GK::CZ: return withExtra(GK::Z, {t[0]}, {t[1]});
                case GK::CP: return withExtra(GK::P, {t[0]}, {t[1]}, op.angles);
                case GK::SWAP:
                    plainOp(GK::CNOT, {t[1], t[0]});
                    withExtra(GK::X, {t[0]}, {t[1]});
                    plainOp(GK::CNOT, {t[1], t[0]});
                    return;
                case GK::Fredkin:
                    plainOp(GK::CNOT, {t[2], t[1]});
                    withExtra(GK::X, {t[0], t[1]}, {t[2]});
                    plainOp(GK::CNOT, {t[2], t[1]});
                    return;
                case GK::PauliRot:
                {
                    for (std::size_t j = 0; j < t.size(); ++j) basisIn(op, t[j], op.paulis[j], depth);
                    for (std::size_t j = 0; j + 1 < t.size(); ++j) plainOp(GK::CNOT, {t[j], t.back()});
                    ctrl(gate(GK::RZ, {t.back()}, op.span, op.angles), c);
                    for (std::size_t j = t.size() - 1; j-- > 0;) plainOp(GK::CNOT, {t[j], t.back()});
                    for (std::size_t j = 0; j < t.size(); ++j) basisOut(op, t[j], op.paulis[j], depth);
                    return;
                }
                case GK::GPhase:
                    // A controlled global phase is a phase gate on the controls.
                    if (c.size() == 1) return plainOp(GK::P, {c[0]}, op.angles);
                    return ctrl(gate(GK::P, {c.back()}, op.span, op.angles), QubitList(c.begin(), c.end() - 1));
                case GK::I: return;
                default: break;
            }
            if (t.size() != 1) fail(op, "a controlled multi-qubit matrix has no synthesis rule");
            const Qubit x = t[0];
            if (c.size() == 1)
            {
                const Qubit k = c[0];
                switch (op.kind)
                {
                    case GK::X: return plainOp(GK::CNOT, {k, x});
                    case GK::Z: return plainOp(GK::CZ, {k, x});
                    case GK::Y:
                        plainOp(GK::Sdg, {x});
                        plainOp(GK::CNOT, {k, x});
                        plainOp(GK::S, {x});
                        return;
                    case GK::P: return plainOp(GK::CP, {k, x}, op.angles);
                    case GK::S: return plainOp(GK::CP, {k, x}, {piFrac(1, 2)});
                    case GK::Sdg: return plainOp(GK::CP, {k, x}, {piFrac(-1, 2)});
                    case GK::T: return plainOp(GK::CP, {k, x}, {piFrac(1, 4)});
                    case GK::Tdg: return plainOp(GK::CP, {k, x}, {piFrac(-1, 4)});
                    case GK::SX:
                    case GK::SXdg:
                        plainOp(GK::H, {x});
                        plainOp(GK::CP, {k, x}, {piFrac(op.kind == GK::SX ? 1 : -1, 2)});
                        plainOp(GK::H, {x});
                        return;
                    case GK::H:
                        // H = Ry(π/4) Z Ry(−π/4)
                        plainOp(GK::RY, {x}, {piFrac(-1, 4)});
                        plainOp(GK::CZ, {k, x});
                        plainOp(GK::RY, {x}, {piFrac(1, 4)});
                        return;
                    case GK::RZ:
                    case GK::RY:
                    {
                        const Affine half = op.angles[0].scaled(Real::rational({1, 2}));
                        plainOp(op.kind, {x}, {half});
                        plainOp(GK::CNOT, {k, x});
                        plainOp(op.kind, {x}, {-half});
                        plainOp(GK::CNOT, {k, x});
                        return;
                    }
                    case GK::RX:
                        plainOp(GK::H, {x});
                        ctrl(gate(GK::RZ, {x}, op.span, op.angles), {k});
                        plainOp(GK::H, {x});
                        return;
                    case GK::U3:
                    {
                        // U3 = e^{i(φ+λ)/2} Rz(φ) Ry(θ) Rz(λ)
                        ctrl(gate(GK::RZ, {x}, op.span, {op.angles[2]}), {k});
                        ctrl(gate(GK::RY, {x}, op.span, {op.angles[0]}), {k});
                        ctrl(gate(GK::RZ, {x}, op.span, {op.angles[1]}), {k});
                        plainOp(GK::P, {k}, {(op.angles[1] + op.angles[2]).scaled(Real::rational({1, 2}))});
                        return;
                    }
                    case GK::Matrix:
                    {
                        const auto [alpha, ang] = zyz(*op.matrix);
                        const double ph = alpha + (ang[1] + ang[2]) / 2.0;
                        ctrl(gate(GK::RZ, {x}, op.span, {Affine(Real::approx(ang[2]))}), {k});
                        ctrl(gate(GK::RY, {x}, op.span, {Affine(Real::approx(ang[0]))}), {k});
                        ctrl(gate(GK::RZ, {x}, op.span, {Affine(Real::approx(ang[1]))}), {k});
                        plainOp(GK::P, {k}, {Affine(Real::approx(ph))});
                        return;
                    }
                    default: fail(op, "no controlled form");
                }
            }
            if (c.size() == 2 && op.kind == GK::X) return plainOp(GK::Toffoli, {c[0], c[1], x});
            if (c.size() == 2 && op.kind == GK::Z)
            {
                plainOp(GK::H, {x});
                plainOp(GK::Toffoli, {c[0], c[1], x});
                plainOp(GK::H, {x});
                return;
            }
            // Barenco et al. (1995), Lemma 7.5: C^k(U) = C_{ck}(V) C^{k-1}(X)_{→ck} C_{ck}(V†) C^{k-1}(X)_{→ck} C^{k-1}(V).
            const QubitList rest(c.begin(), c.end() - 1);
            const Qubit last = c.back();
            for (const IrOp& v : root(op, false)) ctrl(v, rest);
            ctrl(gate(GK::X, {last}, op.span), rest);
            for (const IrOp& v : root(op, true)) ctrl(v, {last});
            ctrl(gate(GK::X, {last}, op.span), rest);
            for (const IrOp& v : root(op, false)) ctrl(v, {last});
        }

        const BasisLowering& bl;
        std::vector<IrOp>& out;
    };
} // namespace

std::string basisName(const IrOp& op)
{
    if (op.kind == GK::PauliRot) return "exp(-iθ/2 " + op.paulis + ")";
    std::string n(gkName(op.kind));
    if (!op.controls.empty() || !op.negControls.empty()) n = "C(" + n + ")";
    return n;
}

BasisLowering::BasisLowering(const std::vector<std::string>& gates, std::vector<std::string>& unknown)
{
    for (const std::string& g : gates)
    {
        if (kindOfName(g)) basis.insert(g);
        else unknown.push_back(g);
    }
    // Fixpoint: a type becomes expressible through the first alternative whose components already are.
    bool changed = true;
    while (changed)
    {
        changed = false;
        for (const auto& [type, alts] : rules())
        {
            if (basis.contains(type) || chosen.contains(type)) continue;
            for (std::size_t k = 0; k < alts.size(); ++k)
            {
                bool ok = true;
                for (const std::string& comp : alts[k])
                    if (comp != type && !basis.contains(comp) && !chosen.contains(comp)) ok = false;
                bool self = false;
                for (const std::string& comp : alts[k]) self = self || comp == type;
                if (ok && !self)
                {
                    chosen[type] = k;
                    changed = true;
                    break;
                }
            }
        }
    }
}

bool BasisLowering::expressible(std::string_view type) const { return basis.contains(type) || chosen.contains(type); }

std::size_t BasisLowering::chosenRule(std::string_view type) const
{
    const auto it = chosen.find(type);
    return it == chosen.end() ? SIZE_MAX : it->second;
}

bool BasisLowering::lower(const IrOp& op, std::vector<IrOp>& out, std::string& error) const
{
    std::vector<IrOp> tmp;
    try
    {
        Lowerer(*this, tmp).emit(op);
    }
    catch (const std::runtime_error& e)
    {
        error = e.what();
        return false;
    }
    // Rotations by arbitrary angles reach here only when the set holds them; exact-angle checks
    // already happened. Anything left outside the set is an error.
    for (const IrOp& o : tmp)
    {
        const bool gateOp = o.kind != GK::MeasureZ && o.kind != GK::MeasurePauli && o.kind != GK::Reset && o.kind != GK::Channel;
        if (gateOp && !basis.contains(std::string(gkName(o.kind))))
        {
            error = std::format("`{}` cannot be lowered exactly into the gate set", basisName(op));
            return false;
        }
    }
    out.insert(out.end(), tmp.begin(), tmp.end());
    return true;
}

} // namespace Noether
