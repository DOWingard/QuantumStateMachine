#pragma once

#include <cstddef>

#ifdef _OPENMP
#include <omp.h>
// _Pragma keeps builds without OpenMP free of -Wunknown-pragmas noise; variadic so
// clauses containing commas (reduction(+:a, b)) survive macro expansion.
#define QPUTER_PRAGMA(...) _Pragma(#__VA_ARGS__)
#define QPUTER_OMP(...) QPUTER_PRAGMA(omp __VA_ARGS__)
#else
#define QPUTER_OMP(...)
#endif



namespace Qputer::detail
{

// Below ~2^14 iterations the OpenMP fork/join cost (~us) outweighs the work.
inline constexpr std::size_t kParallelThreshold = std::size_t{1} << 14;

inline std::size_t threadCount() noexcept
{
#ifdef _OPENMP
    return static_cast<std::size_t>(omp_get_num_threads());
#else
    return 1;
#endif
}

inline std::size_t threadId() noexcept
{
#ifdef _OPENMP
    return static_cast<std::size_t>(omp_get_thread_num());
#else
    return 0;
#endif
}

inline std::size_t maxThreads() noexcept
{
#ifdef _OPENMP
    return static_cast<std::size_t>(omp_get_max_threads());
#else
    return 1;
#endif
}

// Iterations each thread should get at least: a smaller team for mid-sized loops means fewer
// threads to fork and wait for at every barrier, which matters most when other processes
// occupy some of the cores.
inline constexpr std::size_t kThreadGrain = std::size_t{1} << 13;

inline int teamSize(std::size_t iterations) noexcept
{
    const std::size_t wanted = iterations / kThreadGrain;
    const std::size_t limit = maxThreads();
    return static_cast<int>(wanted < 1 ? 1 : (wanted < limit ? wanted : limit));
}

} // namespace Qputer::detail
