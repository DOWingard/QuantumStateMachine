#pragma once

#include "../unit/TestUtil.hpp"

#include "interop/CircuitJson.hpp"
#include "interop/Import.hpp"
#include "interop/Lowering.hpp"
#include "interop/Run.hpp"

#include <QuantumStateMachine.hpp>

#include <gtest/gtest.h>

#include <complex>
#include <random>
#include <string>



namespace InteropTest
{

using namespace Noether::Interop;
using cd = std::complex<double>;

inline const std::filesystem::path kInteropDir = std::filesystem::path(NOETHER_TEST_DIR) / "interop";

// Imports text in the given format; fails the test on diagnostics unless `allowErrors`.
inline ImportResult load(const std::string& path, std::string text, const std::string& format = {}, bool allowErrors = false,
                         std::map<std::string, double> params = {})
{
    ImportOptions o;
    o.format = format;
    o.params = std::move(params);
    ImportResult r = importText(path, std::move(text), o);
    if (!allowErrors)
    {
        EXPECT_TRUE(r.circuit) << path << "\n" << r.diags->render();
    }
    return r;
}

// Codes of the diagnostics an import reported.
inline std::vector<std::string> codes(const ImportResult& r)
{
    std::vector<std::string> out;
    for (const auto& d : r.diags->sorted()) out.push_back(d.code);
    return out;
}

// Unitary of the circuit's operations in core index order (index bit q = qubit q), built column by
// column on the state vector.
inline Eigen::MatrixXcd unitaryOf(const ImportedCircuit& c)
{
    const auto dim = Eigen::Index{1} << c.numQubits;
    Eigen::MatrixXcd u(dim, dim);
    for (Eigen::Index col = 0; col < dim; ++col)
    {
        Qputer::QuantumStateMachine m(c.numQubits, c.numClbits, 1, Qputer::Backend::StateVector);
        m.prepare_basis(static_cast<Qputer::Outcome>(col));
        for (const auto& op : c.ops) m.append(op);
        u.col(col) = m.state_vector().vector();
    }
    return u;
}

inline Eigen::MatrixXcd matrixFromJson(const Noether::Json& j)
{
    const auto n = static_cast<Eigen::Index>(j.size());
    Eigen::MatrixXcd m(n, n);
    for (Eigen::Index r = 0; r < n; ++r)
        for (Eigen::Index k = 0; k < n; ++k)
        {
            const Noether::Json& z = j.asArray()[static_cast<std::size_t>(r)].asArray()[static_cast<std::size_t>(k)];
            m(r, k) = cd(z.asArray()[0].asDouble(), z.asArray()[1].asDouble());
        }
    return m;
}

inline double maxDiff(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) return INFINITY;
    return (a - b).cwiseAbs().maxCoeff();
}

// max |a·e^{iφ} − b| with φ aligning the largest entry of a with b.
inline double maxDiffUpToPhase(const Eigen::MatrixXcd& a, const Eigen::MatrixXcd& b)
{
    if (a.rows() != b.rows() || a.cols() != b.cols()) return INFINITY;
    Eigen::Index r = 0, c = 0;
    a.cwiseAbs().maxCoeff(&r, &c);
    if (std::abs(b(r, c)) < 1e-12) return INFINITY;
    const cd phase = b(r, c) / a(r, c);
    return (a * (phase / std::abs(phase)) - b).cwiseAbs().maxCoeff();
}

inline Eigen::MatrixXcd randomUnitary(std::size_t n, std::mt19937_64& rng)
{
    std::normal_distribution<double> g;
    const auto dim = Eigen::Index{1} << n;
    Eigen::MatrixXcd a(dim, dim);
    for (Eigen::Index r = 0; r < dim; ++r)
        for (Eigen::Index k = 0; k < dim; ++k) a(r, k) = cd(g(rng), g(rng));
    const Eigen::HouseholderQR<Eigen::MatrixXcd> qr(a);
    return qr.householderQ();
}

inline Eigen::VectorXcd randomState(std::size_t n, std::mt19937_64& rng)
{
    std::normal_distribution<double> g;
    Eigen::VectorXcd v(Eigen::Index{1} << n);
    for (Eigen::Index k = 0; k < v.size(); ++k) v[k] = cd(g(rng), g(rng));
    return v.normalized();
}

// Total variation distance between two count tables over the same number of shots.
inline double tvd(const Qputer::Counts& a, const Qputer::Counts& b, std::size_t shots)
{
    std::map<Qputer::Outcome, double> d;
    for (const auto& [k, n] : a) d[k] += static_cast<double>(n);
    for (const auto& [k, n] : b) d[k] -= static_cast<double>(n);
    double s = 0.0;
    for (const auto& [k, v] : d) s += std::abs(v);
    return s / (2.0 * static_cast<double>(shots));
}

} // namespace InteropTest
