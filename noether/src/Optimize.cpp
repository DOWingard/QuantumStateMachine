#include "Tools.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <numbers>
#include <random>
#include <stdexcept>



namespace Noether
{

std::optional<double> labelledValue(Compiler& compiler, const std::string& label, const ExecOptions& exec, std::string& error)
{
    ExecOptions o = exec;
    o.stopOnAssert = false;
    o.timing = false;
    const ExecResult r = execute(compiler, o);
    if (r.exitCode != 0 && r.exitCode != 3)
    {
        error = r.internalError.empty() ? std::format("execution failed (exit {})", r.exitCode) : r.internalError;
        return std::nullopt;
    }
    for (const PrintRecord& p : r.prints)
        if (p.label == label)
        {
            if (!p.number) error = std::format("the print labelled `{}` is not a real number", label);
            return p.number;
        }
    error = std::format("no print is labelled `{}` (write `print ⟨…⟩ as {}`)", label, label);
    return std::nullopt;
}

namespace
{
    Json paramsJson(const Ir& ir, const std::vector<double>& x)
    {
        Json j = Json::object();
        for (const ParamInfo& p : ir.params)
        {
            if (!p.isVector) j[p.name] = x[p.base];
            else
            {
                Json a = Json::array();
                for (std::size_t k = 0; k < p.size; ++k) a.push(x[p.base + k]);
                j[p.name] = a;
            }
        }
        return j;
    }

    struct Bounds
    {
        std::vector<double> lo, hi;
        void clamp(std::vector<double>& x) const
        {
            for (std::size_t k = 0; k < x.size(); ++k) x[k] = std::clamp(x[k], lo[k], hi[k]);
        }
    };

    Bounds boundsOf(const Ir& ir)
    {
        Bounds b;
        b.lo.resize(ir.paramNames.size());
        b.hi.resize(ir.paramNames.size());
        for (const ParamInfo& p : ir.params)
            for (std::size_t k = 0; k < p.size; ++k)
            {
                b.lo[p.base + k] = p.lo;
                b.hi[p.base + k] = p.hi;
            }
        return b;
    }

    // Whether the two-term shift rule f'(φ) = [f(φ+π/2) − f(φ−π/2)]/2 is exact for angle `k` of
    // `op`: its generator has two eigenvalues one unit apart (up to a global offset).
    bool shiftRuleExact(const IrOp& op)
    {
        const bool controlled = !op.controls.empty() || !op.negControls.empty();
        switch (op.kind)
        {
            case GK::RX: case GK::RY: case GK::RZ: case GK::PauliRot: case GK::U3: return !controlled;
            case GK::P: case GK::CP: case GK::GPhase: return true; // phases: generator is a projector
            default: return false;
        }
    }
} // namespace

GradResult gradientOf(Compiler& compiler, const std::string& label, const ExecOptions& exec)
{
    GradResult g;
    const Ir& ir = compiler.ir();
    const std::vector<double> x = exec.params ? *exec.params : ir.paramValues;
    std::string error;
    ExecOptions base = exec;
    base.params = x;
    const auto v0 = labelledValue(compiler, label, base, error);
    if (!v0)
    {
        g.error = error;
        return g;
    }
    g.value = *v0;
    g.gradient.assign(x.size(), 0.0);
    bool anyFd = false;
    for (std::size_t e = 0; e < ir.events.size(); ++e)
    {
        const Event& ev = ir.events[e];
        if (ev.kind != EvK::Op) continue;
        for (std::size_t a = 0; a < ev.op.angles.size(); ++a)
        {
            const Affine& ang = ev.op.angles[a];
            if (ang.isConst()) continue;
            const bool exact = shiftRuleExact(ev.op);
            const double s = exact ? std::numbers::pi / 2 : 1e-5;
            ExecOptions plus = base, minus = base;
            plus.shift = ExecOptions::Shift{e, a, s};
            minus.shift = ExecOptions::Shift{e, a, -s};
            const auto fp = labelledValue(compiler, label, plus, error);
            const auto fm = labelledValue(compiler, label, minus, error);
            if (!fp || !fm)
            {
                g.error = error;
                return g;
            }
            const double d = exact ? (*fp - *fm) / 2.0 : (*fp - *fm) / (2.0 * s);
            anyFd = anyFd || !exact;
            for (const auto& [idx, coef] : ang.terms) g.gradient[idx] += coef.value() * d;
        }
    }
    g.method = anyFd ? "parameter-shift+finite-difference" : "parameter-shift";
    g.gradientJson = paramsJson(ir, g.gradient);
    return g;
}

namespace
{
    struct StopNow
    {
    };
} // namespace

MinimizeResult minimizeBlackBox(const std::function<double(const std::vector<double>&)>& f0, const std::vector<double>& x0,
                                const std::vector<double>& lo, const std::vector<double>& hi, const std::string& method,
                                std::size_t iterations, std::size_t restarts, std::uint64_t seed, const std::function<bool()>& stop)
{
    const std::size_t n = x0.size();
    MinimizeResult res;
    res.x = x0;
    res.value = std::numeric_limits<double>::infinity();
    auto clamp = [&](std::vector<double>& x)
    {
        for (std::size_t k = 0; k < n; ++k) x[k] = std::clamp(x[k], lo[k], hi[k]);
    };
    // Every evaluation updates the best point, so a stop keeps the best seen so far.
    auto f = [&](std::vector<double> x)
    {
        if (stop && stop()) throw StopNow{};
        clamp(x);
        const double v = f0(x);
        ++res.evaluations;
        if (v < res.value)
        {
            res.value = v;
            res.x = x;
        }
        return v;
    };
    auto gradient = [&](const std::vector<double>& x, double fx)
    {
        (void)fx;
        std::vector<double> g(n, 0.0);
        for (std::size_t d = 0; d < n; ++d)
        {
            const double h = 1e-6 * std::max(1.0, std::abs(x[d]));
            std::vector<double> xp = x, xm = x;
            xp[d] += h;
            xm[d] -= h;
            g[d] = (f(xp) - f(xm)) / (2.0 * h);
        }
        return g;
    };
    std::mt19937_64 rng(seed ^ 0x6e6f65746865ULL);

    try
    {
        for (std::size_t restart = 0; restart < std::max<std::size_t>(1, restarts); ++restart)
        {
            std::vector<double> x = x0;
            if (restart > 0)
                for (std::size_t k = 0; k < n; ++k) x[k] = std::uniform_real_distribution<double>(lo[k], hi[k])(rng);
            clamp(x);
            if (method == "nelder-mead")
            {
                std::vector<std::vector<double>> simplex{x};
                for (std::size_t k = 0; k < n; ++k)
                {
                    std::vector<double> p = x;
                    const double span = hi[k] - lo[k];
                    const double step = span > 0 ? 0.1 * span : 0.1;
                    p[k] = p[k] + step <= hi[k] ? p[k] + step : p[k] - step;
                    simplex.push_back(p);
                }
                std::vector<double> fs;
                for (const auto& p : simplex) fs.push_back(f(p));
                for (std::size_t it = 0; it < iterations; ++it)
                {
                    std::vector<std::size_t> idx(simplex.size());
                    for (std::size_t k = 0; k < idx.size(); ++k) idx[k] = k;
                    std::ranges::sort(idx, [&](std::size_t a, std::size_t b) { return fs[a] < fs[b]; });
                    if (std::abs(fs[idx.back()] - fs[idx.front()]) < 1e-12) break;
                    std::vector<double> centroid(n, 0.0);
                    for (std::size_t k = 0; k + 1 < idx.size(); ++k)
                        for (std::size_t d = 0; d < n; ++d) centroid[d] += simplex[idx[k]][d] / static_cast<double>(n);
                    const std::vector<double> worst = simplex[idx.back()];
                    auto along = [&](double t)
                    {
                        std::vector<double> p(n);
                        for (std::size_t d = 0; d < n; ++d) p[d] = centroid[d] + t * (worst[d] - centroid[d]);
                        clamp(p);
                        return p;
                    };
                    const std::vector<double> xr = along(-1.0);
                    const double fr = f(xr);
                    if (fr < fs[idx.front()])
                    {
                        const std::vector<double> xe = along(-2.0);
                        const double fe = f(xe);
                        simplex[idx.back()] = fe < fr ? xe : xr;
                        fs[idx.back()] = std::min(fe, fr);
                    }
                    else if (fr < fs[idx[idx.size() - 2]])
                    {
                        simplex[idx.back()] = xr;
                        fs[idx.back()] = fr;
                    }
                    else
                    {
                        const std::vector<double> xc = along(0.5);
                        const double fc = f(xc);
                        if (fc < fs[idx.back()])
                        {
                            simplex[idx.back()] = xc;
                            fs[idx.back()] = fc;
                        }
                        else
                            for (std::size_t k = 1; k < idx.size(); ++k)
                            {
                                for (std::size_t d = 0; d < n; ++d)
                                    simplex[idx[k]][d] = simplex[idx[0]][d] + 0.5 * (simplex[idx[k]][d] - simplex[idx[0]][d]);
                                fs[idx[k]] = f(simplex[idx[k]]);
                            }
                    }
                }
            }
            else if (method == "spsa")
            {
                f(x);
                for (std::size_t k = 1; k <= iterations; ++k)
                {
                    const double ak = 0.2 / std::pow(static_cast<double>(k) + 10.0, 0.602);
                    const double ck = 0.1 / std::pow(static_cast<double>(k), 0.101);
                    std::vector<double> delta(n), xp = x, xm = x;
                    for (std::size_t d = 0; d < n; ++d)
                    {
                        delta[d] = (rng() & 1U) ? 1.0 : -1.0;
                        xp[d] += ck * delta[d];
                        xm[d] -= ck * delta[d];
                    }
                    const double diff = f(xp) - f(xm);
                    for (std::size_t d = 0; d < n; ++d) x[d] -= ak * diff / (2.0 * ck * delta[d]);
                    clamp(x);
                }
                f(x);
            }
            else if (method == "adam")
            {
                std::vector<double> m(n, 0.0), v(n, 0.0);
                const double lr = 0.05, b1 = 0.9, b2 = 0.999;
                for (std::size_t k = 1; k <= iterations; ++k)
                {
                    const std::vector<double> g = gradient(x, 0.0);
                    double norm = 0.0;
                    for (std::size_t d = 0; d < n; ++d)
                    {
                        norm += g[d] * g[d];
                        m[d] = b1 * m[d] + (1 - b1) * g[d];
                        v[d] = b2 * v[d] + (1 - b2) * g[d] * g[d];
                        const double mh = m[d] / (1 - std::pow(b1, static_cast<double>(k)));
                        const double vh = v[d] / (1 - std::pow(b2, static_cast<double>(k)));
                        x[d] -= lr * mh / (std::sqrt(vh) + 1e-12);
                    }
                    clamp(x);
                    if (std::sqrt(norm) < 1e-9) break;
                }
                f(x);
            }
            else if (method == "lbfgs")
            {
                // Projected L-BFGS (memory 8) with Armijo backtracking.
                std::vector<std::vector<double>> sHist, yHist;
                double fx = f(x);
                std::vector<double> g = gradient(x, fx);
                for (std::size_t it = 0; it < iterations; ++it)
                {
                    double gnorm = 0.0;
                    for (const double gd : g) gnorm += gd * gd;
                    if (std::sqrt(gnorm) < 1e-9) break;
                    std::vector<double> q = g;
                    std::vector<double> alpha(sHist.size());
                    for (std::size_t j = sHist.size(); j-- > 0;)
                    {
                        double sy = 0.0, sq = 0.0;
                        for (std::size_t d = 0; d < n; ++d) sy += sHist[j][d] * yHist[j][d], sq += sHist[j][d] * q[d];
                        alpha[j] = sq / sy;
                        for (std::size_t d = 0; d < n; ++d) q[d] -= alpha[j] * yHist[j][d];
                    }
                    if (!sHist.empty())
                    {
                        double sy = 0.0, yy = 0.0;
                        for (std::size_t d = 0; d < n; ++d) sy += sHist.back()[d] * yHist.back()[d], yy += yHist.back()[d] * yHist.back()[d];
                        for (double& qd : q) qd *= sy / yy;
                    }
                    for (std::size_t j = 0; j < sHist.size(); ++j)
                    {
                        double yr = 0.0, sy = 0.0;
                        for (std::size_t d = 0; d < n; ++d) yr += yHist[j][d] * q[d], sy += sHist[j][d] * yHist[j][d];
                        const double beta = yr / sy;
                        for (std::size_t d = 0; d < n; ++d) q[d] += sHist[j][d] * (alpha[j] - beta);
                    }
                    double slope = 0.0;
                    for (std::size_t d = 0; d < n; ++d) slope -= g[d] * q[d];
                    if (slope >= 0)
                    {
                        q = g;
                        slope = -gnorm;
                        sHist.clear();
                        yHist.clear();
                    }
                    double step = 1.0;
                    std::vector<double> xn;
                    double fn = fx;
                    bool accepted = false;
                    for (int ls = 0; ls < 30; ++ls)
                    {
                        xn = x;
                        for (std::size_t d = 0; d < n; ++d) xn[d] -= step * q[d];
                        clamp(xn);
                        fn = f(xn);
                        if (fn <= fx + 1e-4 * step * slope)
                        {
                            accepted = true;
                            break;
                        }
                        step *= 0.5;
                    }
                    if (!accepted) break;
                    const std::vector<double> gn = gradient(xn, fn);
                    std::vector<double> sv(n), yv(n);
                    double sy = 0.0;
                    for (std::size_t d = 0; d < n; ++d)
                    {
                        sv[d] = xn[d] - x[d];
                        yv[d] = gn[d] - g[d];
                        sy += sv[d] * yv[d];
                    }
                    if (sy > 1e-16)
                    {
                        sHist.push_back(sv);
                        yHist.push_back(yv);
                        if (sHist.size() > 8)
                        {
                            sHist.erase(sHist.begin());
                            yHist.erase(yHist.begin());
                        }
                    }
                    if (std::abs(fx - fn) < 1e-14 * std::max(1.0, std::abs(fx))) break;
                    x = xn;
                    fx = fn;
                    g = gn;
                }
            }
            else throw std::invalid_argument(std::format("unknown method `{}`; expected nelder-mead, spsa, adam or lbfgs", method));
        }
    }
    catch (const StopNow&)
    {
    }
    return res;
}

OptResult optimizeParams(Compiler& compiler, const OptOptions& options)
{
    OptResult res;
    const Ir& ir = compiler.ir();
    if (ir.paramNames.empty())
    {
        res.error = "the program has no params to optimise";
        return res;
    }
    const Bounds bounds = boundsOf(ir);
    const double sign = options.maximize ? -1.0 : 1.0;
    std::string error;
    auto f = [&](const std::vector<double>& x) -> double
    {
        ExecOptions o = options.exec;
        o.params = x;
        const auto v = labelledValue(compiler, options.label, o, error);
        if (!v) throw std::runtime_error(error);
        return sign * *v;
    };
    try
    {
        const MinimizeResult m = minimizeBlackBox(f, ir.paramValues, bounds.lo, bounds.hi, options.method, options.iterations,
                                                  options.restarts, ir.seed.value_or(0));
        res.best = sign * m.value;
        res.params = m.x;
        res.evaluations = m.evaluations;
        res.paramsJson = paramsJson(ir, m.x);
    }
    catch (const std::exception& e)
    {
        res.error = e.what();
    }
    return res;
}

} // namespace Noether
