// ============================================================================================== //
//                                                                                                //
//  This file is part of the Simple Fourier Transform Library.                                    //
//                                                                                                //
//  Author:                                                                                       //
//  Marcel Hasler <marcel.hasler@h-brs.de>                                                        //
//                                                                                                //
//  Copyright (c) 2020 - 2022                                                                     //
//  Bonn-Rhein-Sieg University of Applied Sciences                                                //
//                                                                                                //
//  This library is free software: you can redistribute it and/or modify it under the terms of    //
//  the GNU Lesser General Public License as published by the Free Software Foundation, either    //
//  version 3 of the License, or (at your option) any later version.                              //
//                                                                                                //
//  This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;     //
//  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.     //
//  See the GNU Lesser General Public License for more details.                                   //
//                                                                                                //
//  You should have received a copy of the GNU Lesser General Public License along with this      //
//  library. If not, see <https://www.gnu.org/licenses/>.                                         //
//                                                                                                //
// ============================================================================================== //
//                                                                                                //
//  References:                                                                                   //
//  NXP AN3680      "Software Optimization of DFTs and IDFTs Using the StarCore SC3850 DSP Core"  //
//  TI SPRA291      "Implementing Fast Fourier Transform Algorithms of Real-Valued Sequences"     //
//  Sorensen et al. "Real-Valued Fast Fourier Transform Algorithms", IEEE, 1987                   //
//  Julius O. Smith "Mathematics of the Discrete Fourier Transform", 2nd Edition, 2007            //
//  Chu & George    "Inside the FFT Black Box", CRC Press, 2000                                   //
//                                                                                                //
// ============================================================================================== //

#include <sft.h>
using namespace sft;

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <mutex>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------------------------- //

#ifdef _OPENMP
#include <omp.h>

#if defined(__linux__)
#include <fstream>
#include <set>
#elif defined(_WIN32)
#include <windows.h>
#endif

#define DO_PRAGMA_(x) _Pragma(#x)
#define DO_PRAGMA(x) DO_PRAGMA_(x)

#define _SFT_OMP_PARALLEL_PRAGMA(size)          DO_PRAGMA(omp parallel for \
                                                          if (size >= MinimumParallelSize))

#define _SFT_OMP_PARALLEL_PRAGMA_COLLAPSE(size) DO_PRAGMA(omp parallel for collapse(2) \
                                                          if (size >= MinimumParallelSize))
#else
#define _SFT_OMP_PARALLEL_PRAGMA(size)
#define _SFT_OMP_PARALLEL_PRAGMA_COLLAPSE(size)
#endif // _OPENMP

// ---------------------------------------------------------------------------------------------- //

namespace {
    constexpr auto Pi = static_cast<Real>(3.14159265358979323846);
    constexpr auto TwoPi = static_cast<Real>(2) * Pi;

    constexpr Complex I = { static_cast<Real>(0), static_cast<Real>(1) };

#ifdef _OPENMP
    constexpr size_t MinimumParallelSize = 512;
#endif
}

// ---------------------------------------------------------------------------------------------- //

static inline
auto operator/(const Complex& lhs, size_t rhs) -> Complex
{
    return lhs / static_cast<Real>(rhs);
}

// ---------------------------------------------------------------------------------------------- //

static inline
auto isMultipleOf(size_t n, size_t d) -> bool
{
    return (n >= d) && (n % d == 0);
}

// ---------------------------------------------------------------------------------------------- //

static inline
void logError(const char* msg)
{
    std::cerr << "[SFT] Error: " << msg << std::endl;
}

// ---------------------------------------------------------------------------------------------- //

#ifdef _OPENMP
static
auto getPhysicalCoreCount() -> unsigned int
{
    const unsigned int threads = omp_get_max_threads();

#if defined(__linux__)
    try {
        std::ifstream file("/proc/cpuinfo");
        std::string line;

        std::set<std::string> ids;

        while (std::getline(file, line))
        {
            if (line.find("core id") != std::string::npos)
                ids.insert(line);
        }

        unsigned int count = ids.size();
        return (count == 0) ? threads : std::min(threads, count);
    }
    catch (const std::exception&) {
        return threads;
    }
#elif defined(_WIN32)
    DWORD size = 0;
    GetLogicalProcessorInformation(NULL, &size);

    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER)
        return threads;

    const size_t elements = size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(elements);

    if (GetLogicalProcessorInformation(buffer.data(), &size) == FALSE)
        return threads;

    unsigned int count = 0;

    for (const auto& info : buffer)
    {
        if (info.Relationship == RelationProcessorCore)
            ++count;
    }

    return (count == 0) ? threads : std::min(threads, count);
#else
    return threads;
#endif
}
#endif // _OPENMP

// ============================================================================================== //

class Stage
{
public:
    virtual ~Stage() = default;
    virtual void transform(Span<Complex> data) const = 0;
};

// ---------------------------------------------------------------------------------------------- //

template <size_t Radix>
class RadixStage : public Stage
{
public:
    RadixStage(size_t size);
    void transform(Span<Complex> data) const override;

private:
    const size_t m_size;
    const size_t m_subSize;

    std::array<std::vector<Complex>, Radix> m_factors;
};

// ---------------------------------------------------------------------------------------------- //

using Radix2Stage = RadixStage<2>;
using Radix3Stage = RadixStage<3>;
using Radix4Stage = RadixStage<4>;
using Radix5Stage = RadixStage<5>;
using Radix7Stage = RadixStage<7>;

// ---------------------------------------------------------------------------------------------- //

template <size_t Radix>
RadixStage<Radix>::RadixStage(size_t size)
    : m_size(size),
      m_subSize(size / Radix)
{
    assert(size % Radix == 0);

    for (size_t i = 1; i < Radix; ++i) // m_factors[0] is never used
    {
        m_factors[i].resize(m_subSize);

        _SFT_OMP_PARALLEL_PRAGMA(m_subSize)
        for (size_t j = 0; j < m_subSize; ++j)
            m_factors[i][j] = std::exp(-TwoPi * i * j * I / m_size);
    }
}

// ---------------------------------------------------------------------------------------------- //

template <>
void RadixStage<2>::transform(Span<Complex> data) const
{
    const size_t size = data.size();

    _SFT_OMP_PARALLEL_PRAGMA_COLLAPSE(size)
    for (size_t i = 0; i < size; i += m_size)
    {
        for (size_t j = 0; j < m_subSize; ++j)
        {
            const size_t base = i + j;

            const size_t index0 = base;
            const size_t index1 = base + m_subSize;

            const Complex term0 =                   data[index0];
            const Complex term1 = m_factors[1][j] * data[index1];

            data[index0] = term0 + term1;
            data[index1] = term0 - term1;
        }
    }
}

// ---------------------------------------------------------------------------------------------- //

template <>
void RadixStage<3>::transform(Span<Complex> data) const
{
    static constexpr Real A = 0.5;
    static constexpr Real B = 0.86602540378443864676;

    static constexpr Complex Factor1 = { -A, -B }; // = exp(-2*Pi*i * 1/3)
    static constexpr Complex Factor2 = { -A, +B }; // = exp(-2*Pi*i * 2/3)

    const size_t offset0 = 0 * m_subSize;
    const size_t offset1 = 1 * m_subSize;
    const size_t offset2 = 2 * m_subSize;

    const size_t size = data.size();

    _SFT_OMP_PARALLEL_PRAGMA_COLLAPSE(size)
    for (size_t i = 0; i < size; i += m_size)
    {
        for (size_t j = 0; j < m_subSize; ++j)
        {
            const size_t base = i + j;

            const size_t index0 = base + offset0;
            const size_t index1 = base + offset1;
            const size_t index2 = base + offset2;

            const Complex term0 =                   data[index0];
            const Complex term1 = m_factors[1][j] * data[index1];
            const Complex term2 = m_factors[2][j] * data[index2];

            data[index0] = term0 +           term1 +           term2;
            data[index1] = term0 + Factor1 * term1 + Factor2 * term2;
            data[index2] = term0 + Factor2 * term1 + Factor1 * term2;
        }
    }
}

// ---------------------------------------------------------------------------------------------- //

template <>
void RadixStage<4>::transform(Span<Complex> data) const
{
    const size_t offset0 = 0 * m_subSize;
    const size_t offset1 = 1 * m_subSize;
    const size_t offset2 = 2 * m_subSize;
    const size_t offset3 = 3 * m_subSize;

    const size_t size = data.size();

    _SFT_OMP_PARALLEL_PRAGMA_COLLAPSE(size)
    for (size_t i = 0; i < size; i += m_size)
    {
        for (size_t j = 0; j < m_subSize; ++j)
        {
            const size_t base = i + j;

            const size_t index0 = base + offset0;
            const size_t index1 = base + offset1;
            const size_t index2 = base + offset2;
            const size_t index3 = base + offset3;

            const Complex term0 =                   data[index0];
            const Complex term1 = m_factors[1][j] * data[index1];
            const Complex term2 = m_factors[2][j] * data[index2];
            const Complex term3 = m_factors[3][j] * data[index3];

            data[index0] = term0 +     term1 + term2 +     term3;
            data[index1] = term0 - I * term1 - term2 + I * term3;
            data[index2] = term0 -     term1 + term2 -     term3;
            data[index3] = term0 + I * term1 - term2 - I * term3;
        }
    }
}

// ---------------------------------------------------------------------------------------------- //

template <>
void RadixStage<5>::transform(Span<Complex> data) const
{
    static constexpr Real A = 0.30901699437494742410;
    static constexpr Real B = 0.95105651629515357212;
    static constexpr Real C = 0.80901699437494742410;
    static constexpr Real D = 0.58778525229247312918;

    static constexpr Complex Factor1 = { +A, -B }; // = exp(-2*Pi*i * 1/5)
    static constexpr Complex Factor2 = { -C, -D }; // = exp(-2*Pi*i * 2/5)
    static constexpr Complex Factor3 = { -C, +D }; // = exp(-2*Pi*i * 3/5)
    static constexpr Complex Factor4 = { +A, +B }; // = exp(-2*Pi*i * 4/5)

    const size_t offset0 = 0 * m_subSize;
    const size_t offset1 = 1 * m_subSize;
    const size_t offset2 = 2 * m_subSize;
    const size_t offset3 = 3 * m_subSize;
    const size_t offset4 = 4 * m_subSize;

    const size_t size = data.size();

    _SFT_OMP_PARALLEL_PRAGMA_COLLAPSE(size)
    for (size_t i = 0; i < size; i += m_size)
    {
        for (size_t j = 0; j < m_subSize; ++j)
        {
            const size_t base = i + j;

            const size_t index0 = base + offset0;
            const size_t index1 = base + offset1;
            const size_t index2 = base + offset2;
            const size_t index3 = base + offset3;
            const size_t index4 = base + offset4;

            const Complex term0 =                   data[index0];
            const Complex term1 = m_factors[1][j] * data[index1];
            const Complex term2 = m_factors[2][j] * data[index2];
            const Complex term3 = m_factors[3][j] * data[index3];
            const Complex term4 = m_factors[4][j] * data[index4];

            data[index0] =           term0 +
                                     term1 +
                                     term2 +
                                     term3 +
                                     term4;

            data[index1] =           term0 +
                           Factor1 * term1 +
                           Factor2 * term2 +
                           Factor3 * term3 +
                           Factor4 * term4;

            data[index2] =           term0 +
                           Factor2 * term1 +
                           Factor4 * term2 +
                           Factor1 * term3 +
                           Factor3 * term4;

            data[index3] =           term0 +
                           Factor3 * term1 +
                           Factor1 * term2 +
                           Factor4 * term3 +
                           Factor2 * term4;

            data[index4] =           term0 +
                           Factor4 * term1 +
                           Factor3 * term2 +
                           Factor2 * term3 +
                           Factor1 * term4;
        }
    }
}

// ---------------------------------------------------------------------------------------------- //

template <>
void RadixStage<7>::transform(Span<Complex> data) const
{
    static constexpr Real A = 0.62348980185873353053;
    static constexpr Real B = 0.78183148246802980871;
    static constexpr Real C = 0.22252093395631440429;
    static constexpr Real D = 0.97492791218182360702;
    static constexpr Real E = 0.90096886790241912623;
    static constexpr Real F = 0.43388373911755812048;

    static constexpr Complex Factor1 = { +A, -B }; // = exp(-2*Pi*i * 1/7)
    static constexpr Complex Factor2 = { -C, -D }; // = exp(-2*Pi*i * 2/7)
    static constexpr Complex Factor3 = { -E, -F }; // = exp(-2*Pi*i * 3/7)
    static constexpr Complex Factor4 = { -E, +F }; // = exp(-2*Pi*i * 4/7)
    static constexpr Complex Factor5 = { -C, +D }; // = exp(-2*Pi*i * 5/7)
    static constexpr Complex Factor6 = { +A, +B }; // = exp(-2*Pi*i * 6/7)

    const size_t offset0 = 0 * m_subSize;
    const size_t offset1 = 1 * m_subSize;
    const size_t offset2 = 2 * m_subSize;
    const size_t offset3 = 3 * m_subSize;
    const size_t offset4 = 4 * m_subSize;
    const size_t offset5 = 5 * m_subSize;
    const size_t offset6 = 6 * m_subSize;

    const size_t size = data.size();

    _SFT_OMP_PARALLEL_PRAGMA_COLLAPSE(size)
    for (size_t i = 0; i < size; i += m_size)
    {
        for (size_t j = 0; j < m_subSize; ++j)
        {
            const size_t base = i + j;

            const size_t index0 = base + offset0;
            const size_t index1 = base + offset1;
            const size_t index2 = base + offset2;
            const size_t index3 = base + offset3;
            const size_t index4 = base + offset4;
            const size_t index5 = base + offset5;
            const size_t index6 = base + offset6;

            const Complex term0 =                   data[index0];
            const Complex term1 = m_factors[1][j] * data[index1];
            const Complex term2 = m_factors[2][j] * data[index2];
            const Complex term3 = m_factors[3][j] * data[index3];
            const Complex term4 = m_factors[4][j] * data[index4];
            const Complex term5 = m_factors[5][j] * data[index5];
            const Complex term6 = m_factors[6][j] * data[index6];

            data[index0] =           term0 +
                                     term1 +
                                     term2 +
                                     term3 +
                                     term4 +
                                     term5 +
                                     term6;

            data[index1] =           term0 +
                           Factor1 * term1 +
                           Factor2 * term2 +
                           Factor3 * term3 +
                           Factor4 * term4 +
                           Factor5 * term5 +
                           Factor6 * term6;

            data[index2] =           term0 +
                           Factor2 * term1 +
                           Factor4 * term2 +
                           Factor6 * term3 +
                           Factor1 * term4 +
                           Factor3 * term5 +
                           Factor5 * term6;

            data[index3] =           term0 +
                           Factor3 * term1 +
                           Factor6 * term2 +
                           Factor2 * term3 +
                           Factor5 * term4 +
                           Factor1 * term5 +
                           Factor4 * term6;

            data[index4] =           term0 +
                           Factor4 * term1 +
                           Factor1 * term2 +
                           Factor5 * term3 +
                           Factor2 * term4 +
                           Factor6 * term5 +
                           Factor3 * term6;

            data[index5] =           term0 +
                           Factor5 * term1 +
                           Factor3 * term2 +
                           Factor1 * term3 +
                           Factor6 * term4 +
                           Factor4 * term5 +
                           Factor2 * term6;

            data[index6] =           term0 +
                           Factor6 * term1 +
                           Factor5 * term2 +
                           Factor4 * term3 +
                           Factor3 * term4 +
                           Factor2 * term5 +
                           Factor1 * term6;
        }
    }
}


// ---------------------------------------------------------------------------------------------- //
//  Public C++ API                                                                                //
// ---------------------------------------------------------------------------------------------- //

class ContextBase
{
public:
    auto size() const -> size_t { return m_size; }

protected:
    using IndexList = std::vector<size_t>;
    using StageList = std::vector<std::unique_ptr<Stage>>;

protected:
    ContextBase(size_t size);
    ~ContextBase() = default;

    auto reverseIndices() const -> const IndexList& { return m_reverseIndices; }
    auto stages() const -> const StageList& { return m_stages; }

    void transform(Span<Complex> data) const;

private:
    void computeReverseIndices(const std::vector<size_t>& radices);
    void createStages(const std::vector<size_t>& radices);

    static auto computeRadices(size_t size) -> std::vector<size_t>;

private:
    const size_t m_size;

    IndexList m_reverseIndices;
    StageList m_stages;
};

// ---------------------------------------------------------------------------------------------- //

template <>
class Context<Real>::Private : public ContextBase
{
public:
    Private(size_t size);

    void transform(Span<const Real> input, Span<Complex> output) const;
    void transformInverse(Span<const Complex> input, Span<Real> output) const;

    static auto getContextSize(size_t size) -> size_t;

private:
    void checkSize(size_t realSize, size_t complexSize) const;
    void computeFactors();

private:
    std::vector<Complex> m_aFactors;
    std::vector<Complex> m_bFactors;

    mutable std::mutex m_mutex;
    mutable std::vector<Complex> m_data;
};

// ---------------------------------------------------------------------------------------------- //

template <>
class Context<Complex>::Private : public ContextBase
{
public:
    Private(size_t size);

    void transform(Span<const Complex> input, Span<Complex> output) const;
    void transformInverse(Span<const Complex> input, Span<Complex> output) const;

private:
    void checkArgs(Span<const Complex> input, Span<Complex> output) const;
};

// ============================================================================================== //

ContextBase::ContextBase(size_t size)
    : m_size(size),
      m_reverseIndices(m_size)
{
#ifdef _OPENMP
    static std::once_flag flag;
    std::call_once(flag, []{ omp_set_num_threads(getPhysicalCoreCount()); });
#endif

    const std::vector<size_t> radices = computeRadices(m_size);

    computeReverseIndices(radices);
    createStages(radices);
}

// ---------------------------------------------------------------------------------------------- //

void ContextBase::transform(Span<Complex> data) const
{
    for (auto it = m_stages.rbegin(); it != m_stages.rend(); ++it)
        (*it)->transform(data);
}

// ---------------------------------------------------------------------------------------------- //

void ContextBase::computeReverseIndices(const std::vector<size_t>& radices)
{
    const auto multiply = [&](size_t start)
    {
        return std::accumulate(radices.begin() + start, radices.end(),
                               static_cast<size_t>(1), std::multiplies<size_t>());
    };

    assert(multiply(0) == m_size);

    const size_t size = radices.size();

    std::vector<size_t> counters(size);
    std::vector<size_t> factors(size);

    for (size_t i = 0; i < size; ++i)
        factors[i] = multiply(i + 1);

    for (size_t& index : m_reverseIndices)
    {
        index = 0;

        for (size_t i = 0; i < size; ++i)
            index += counters[i] * factors[i];

        for (size_t i = 0; i < size; ++i)
        {
            if (++counters[i] >= radices[i])
                counters[i] = 0;
            else
                break;
        }
    }
}

// ---------------------------------------------------------------------------------------------- //

void ContextBase::createStages(const std::vector<size_t>& radices)
{
    size_t stageSize = m_size;

    for (size_t radix : radices)
    {
        assert(radix >= 2 && radix <= 7);

        if (radix == 2)
            m_stages.push_back(std::make_unique<Radix2Stage>(stageSize));
        else if (radix == 3)
            m_stages.push_back(std::make_unique<Radix3Stage>(stageSize));
        else if (radix == 4)
            m_stages.push_back(std::make_unique<Radix4Stage>(stageSize));
        else if (radix == 5)
            m_stages.push_back(std::make_unique<Radix5Stage>(stageSize));
        else if (radix == 7)
            m_stages.push_back(std::make_unique<Radix7Stage>(stageSize));

        stageSize /= radix;
    }

    assert(stageSize == 1);
}

// ---------------------------------------------------------------------------------------------- //

auto ContextBase::computeRadices(size_t size) -> std::vector<size_t>
{
    static constexpr std::array SupportedRadices = { 7, 5, 4, 3, 2 };

    static const auto nextRadix = [](size_t size)
    {
        for (size_t radix : SupportedRadices)
        {
            if (isMultipleOf(size, radix))
                return radix;
        }

        throw Error("Input size must be divisible into factors of 2, 3, 5 and 7.");
    };

    std::vector<size_t> radices;

    while (size > 1)
    {
        const size_t radix = nextRadix(size);

        radices.push_back(radix);
        size /= radix;
    }

    return radices;
}

// ============================================================================================== //

Context<Real>::Private::Private(size_t size)
    : ContextBase(size),
      m_aFactors(size),
      m_bFactors(size),
      m_data(size)
{
    computeFactors();
}

// ---------------------------------------------------------------------------------------------- //

void Context<Real>::Private::transform(Span<const Real> input, Span<Complex> output) const
{
    checkSize(input.size(), output.size());

    const size_t size = ContextBase::size();
    const IndexList& reverseIndices = ContextBase::reverseIndices();

    std::lock_guard guard(m_mutex);

    _SFT_OMP_PARALLEL_PRAGMA(size)
    for (size_t i = 0; i < size; ++i)
        m_data[reverseIndices[i]] = { input[2*i], input[2*i + 1] };

    ContextBase::transform(m_data);

    _SFT_OMP_PARALLEL_PRAGMA(size)
    for (size_t i = 0; i < size; ++i)
    {
        const size_t index0 = i;
        const size_t index1 = (i == 0) ? 0 : (size - i);

        output[i] =             m_data[index0]  * m_aFactors[index0]
                    + std::conj(m_data[index1]) * m_bFactors[index0];
    }

    output[size] = std::real(m_data[0]) - std::imag(m_data[0]);
}

// ---------------------------------------------------------------------------------------------- //

void Context<Real>::Private::transformInverse(Span<const Complex> input, Span<Real> output) const
{
    checkSize(output.size(), input.size());

    const size_t size = ContextBase::size();
    const IndexList& reverseIndices = ContextBase::reverseIndices();

    std::lock_guard guard(m_mutex);

    _SFT_OMP_PARALLEL_PRAGMA(size)
    for (size_t i = 0; i < size; ++i)
    {
        const size_t index0 = i;
        const size_t index1 = size - i;

        const Complex value =             input[index0]  * std::conj(m_aFactors[index0])
                              + std::conj(input[index1]) * std::conj(m_bFactors[index0]);

        m_data[reverseIndices[i]] = std::conj(value);
    }

    ContextBase::transform(m_data);

    const auto scale = static_cast<Real>(1) / size;

    _SFT_OMP_PARALLEL_PRAGMA(size)
    for (size_t i = 0; i < size; ++i)
    {
        const Complex value = scale * std::conj(m_data[i]);

        output[2*i    ] = std::real(value);
        output[2*i + 1] = std::imag(value);
    }
}

// ---------------------------------------------------------------------------------------------- //

auto Context<Real>::Private::getContextSize(size_t size) -> size_t
{
    if (!isMultipleOf(size, 2))
        throw Error("Context size for real-valued transforms must be even.");

    return size / 2;
}

// ---------------------------------------------------------------------------------------------- //

void Context<Real>::Private::computeFactors()
{
    static constexpr auto One = static_cast<Real>(1);
    static constexpr auto Two = static_cast<Real>(2);

    const size_t size = ContextBase::size();

    _SFT_OMP_PARALLEL_PRAGMA(size)
    for (size_t i = 0; i < size; ++i)
    {
        const Complex factor = I * std::exp(-Pi * i * I / size);

        m_aFactors[i] = (One - factor) / Two;
        m_bFactors[i] = (One + factor) / Two;
    }
}

// ---------------------------------------------------------------------------------------------- //

void Context<Real>::Private::checkSize(size_t realSize, size_t complexSize) const
{
    const size_t size = ContextBase::size();

    if (realSize != 2*size)
        throw Error("Number of real values doesn't match context size.");

    if (complexSize != size+1)
        throw Error("Number of complex values must be equal to N/2 + 1 for context size N.");
}

// ============================================================================================== //

Context<Complex>::Private::Private(size_t size)
    : ContextBase(size) {}

// ---------------------------------------------------------------------------------------------- //

void Context<Complex>::Private::transform(Span<const Complex> input, Span<Complex> output) const
{
    checkArgs(input, output);

    const size_t size = ContextBase::size();
    const IndexList& reverseIndices = ContextBase::reverseIndices();

    _SFT_OMP_PARALLEL_PRAGMA(size)
    for (size_t i = 0; i < size; ++i)
        output[reverseIndices[i]] = input[i];

    ContextBase::transform(output);
}

// ---------------------------------------------------------------------------------------------- //

void Context<Complex>::Private::transformInverse(Span<const Complex> input,
                                                 Span<Complex> output) const
{
    checkArgs(input, output);

    const size_t size = ContextBase::size();
    const IndexList& reverseIndices = ContextBase::reverseIndices();

    _SFT_OMP_PARALLEL_PRAGMA(size)
    for (size_t i = 0; i < size; ++i)
        output[reverseIndices[i]] = std::conj(input[i]);

    ContextBase::transform(output);

    const auto scale = static_cast<Real>(1) / size;

    _SFT_OMP_PARALLEL_PRAGMA(size)
    for (size_t i = 0; i < size; ++i)
        output[i] = scale * std::conj(output[i]);
}

// ---------------------------------------------------------------------------------------------- //

void Context<Complex>::Private::checkArgs(Span<const Complex> input, Span<Complex> output) const
{
    if (input.size() != ContextBase::size())
        throw Error("Number of input values doesn't match context size.");

    if (output.size() != input.size())
        throw Error("Number of output values must be equal to number of input values.");

    if (output.data() == input.data())
        throw Error("Input and output buffers must be distinct.");
}

// ============================================================================================== //

template <> SFT_EXPORT
Context<Real>::Context(size_t size)
    : d(std::make_unique<Private>(Private::getContextSize(size))) {}

// ---------------------------------------------------------------------------------------------- //

template <> SFT_EXPORT
Context<Real>::~Context() = default;

// ---------------------------------------------------------------------------------------------- //

template <> SFT_EXPORT
auto Context<Real>::size() const -> size_t
{
    return 2 * d->size();
}

// ---------------------------------------------------------------------------------------------- //

template <> SFT_EXPORT
void Context<Real>::transform(Span<const Real> input, Span<Complex> output) const
{
    d->transform(input, output);
}

// ---------------------------------------------------------------------------------------------- //

template <> SFT_EXPORT
void Context<Real>::transformInverse(Span<const Complex> input, Span<Real> output) const
{
    d->transformInverse(input, output);
}

// ============================================================================================== //

template <> SFT_EXPORT
Context<Complex>::Context(size_t size)
    : d(std::make_unique<Private>(size)) {}

// ---------------------------------------------------------------------------------------------- //

template <> SFT_EXPORT
Context<Complex>::~Context() = default;

// ---------------------------------------------------------------------------------------------- //

template <> SFT_EXPORT
auto Context<Complex>::size() const -> size_t
{
    return d->size();
}

// ---------------------------------------------------------------------------------------------- //

template <> SFT_EXPORT
void Context<Complex>::transform(Span<const Complex> input, Span<Complex> output) const
{
    d->transform(input, output);
}

// ---------------------------------------------------------------------------------------------- //

template <> SFT_EXPORT
void Context<Complex>::transformInverse(Span<const Complex> input, Span<Complex> output) const
{
    d->transformInverse(input, output);
}


// ---------------------------------------------------------------------------------------------- //
//  Public C API (Real)                                                                           //
// ---------------------------------------------------------------------------------------------- //

struct _sft_real_context
{
    Context<Real> context;
};

// ---------------------------------------------------------------------------------------------- //

sft_result sft_create_real_context(size_t size, sft_real_context** context)
{
    try {
        *context = new sft_real_context {{ size }};
        return SFT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        logError(e.what());
        *context = nullptr;
        return SFT_INVALID_SIZE;
    }
}

// ---------------------------------------------------------------------------------------------- //

void sft_free_real_context(sft_real_context* context)
{
    delete context;
}

// ---------------------------------------------------------------------------------------------- //

size_t sft_get_real_size(sft_real_context* context)
{
    return context->context.size();
}

// ---------------------------------------------------------------------------------------------- //

void sft_transform_real(const sft_real_context* context,
                        const sft_real* input, sft_complex* output)
{
    const size_t realSize = context->context.size();
    const size_t complexSize = realSize/2 + 1;

    // Compatible layout guaranteed by C++ standard
    auto out = reinterpret_cast<Complex*>(output);

    try {
        context->context.transform(Span<const Real>(input, realSize),
                                   Span<Complex>(out, complexSize));
    }
    catch (const std::exception& e) {
        logError(e.what());
    }
}

// ---------------------------------------------------------------------------------------------- //

void sft_transform_inverse_real(const sft_real_context* context,
                                const sft_complex* input, sft_real* output)
{
    const size_t realSize = context->context.size();
    const size_t complexSize = realSize/2 + 1;

    // Compatible layout guaranteed by C++ standard
    auto in = reinterpret_cast<const Complex*>(input);

    try {
        context->context.transformInverse(Span<const Complex>(in, complexSize),
                                          Span<Real>(output, realSize));
    }
    catch (const std::exception& e) {
        logError(e.what());
    }
}


// ---------------------------------------------------------------------------------------------- //
//  Public C API (Complex)                                                                        //
// ---------------------------------------------------------------------------------------------- //

struct _sft_complex_context
{
    Context<Complex> context;
};

// ---------------------------------------------------------------------------------------------- //

sft_result sft_create_complex_context(size_t size, sft_complex_context** context)
{
    try {
        *context = new sft_complex_context {{ size }};
        return SFT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        logError(e.what());
        *context = nullptr;
        return SFT_INVALID_SIZE;
    }
}

// ---------------------------------------------------------------------------------------------- //

void sft_free_complex_context(sft_complex_context* context)
{
    delete context;
}

// ---------------------------------------------------------------------------------------------- //

size_t sft_get_complex_size(sft_complex_context* context)
{
    return context->context.size();
}

// ---------------------------------------------------------------------------------------------- //

void sft_transform_complex(const sft_complex_context* context,
                           const sft_complex* input, sft_complex* output)
{
    const size_t size = context->context.size();

    // Compatible layout guaranteed by C++ standard
    auto in = reinterpret_cast<const Complex*>(input);
    auto out = reinterpret_cast<Complex*>(output);

    try {
        context->context.transform(Span<const Complex>(in, size),
                                   Span<Complex>(out, size));
    }
    catch (const std::exception& e) {
        logError(e.what());
    }
}

// ---------------------------------------------------------------------------------------------- //

void sft_transform_inverse_complex(const sft_complex_context* context,
                                   const sft_complex* input, sft_complex* output)
{
    const size_t size = context->context.size();

    // Compatible layout guaranteed by C++ standard
    auto in = reinterpret_cast<const Complex*>(input);
    auto out = reinterpret_cast<Complex*>(output);

    try {
        context->context.transformInverse(Span<const Complex>(in, size),
                                          Span<Complex>(out, size));
    }
    catch (const std::exception& e) {
        logError(e.what());
    }
}

// ============================================================================================== //
