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

#ifndef LIBSFT_FIXED_H
#define LIBSFT_FIXED_H

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <numeric>

#if __cplusplus >= 202002L
#include <span>
#else
#include <gsl/span>
#endif

// ---------------------------------------------------------------------------------------------- //

namespace sft {

// ---------------------------------------------------------------------------------------------- //

#ifdef SFT_SINGLE_PRECISION
using Real = float;
#else
using Real = double;
#endif

using Complex = std::complex<Real>;

#if __cplusplus >= 202002L
template <typename T, size_t N = std::dynamic_extent>
using Span = std::span<T,N>;
#else
template <typename T, size_t N = gsl::dynamic_extent>
using Span = gsl::span<T,N>;
#endif

// ---------------------------------------------------------------------------------------------- //

template <size_t... Radices>
class Transform {};

// ---------------------------------------------------------------------------------------------- //

template <size_t... Radices>
class FixedContextBase
{
protected:
    static constexpr size_t ComplexSize = (... * Radices);
    using IndexArray = std::array<size_t, ComplexSize>;

protected:
    FixedContextBase();

    auto reverseIndices() const -> const IndexArray& { return m_reverseIndices; }
    void transform(Span<Complex> data) const;

private:
    void computeReverseIndices();

private:
    IndexArray m_reverseIndices;
    Transform<Radices...> m_transform;
};

// ---------------------------------------------------------------------------------------------- //

template <typename T, size_t... Radices>
class FixedContext {};

// ---------------------------------------------------------------------------------------------- //

template <size_t... Radices>
class FixedContext<Real, Radices...> : public FixedContextBase<Radices...>
{
public:
    static constexpr size_t Size = 2 * (... * Radices);

public:
    FixedContext();

    auto size() const -> size_t { return Size; }

    void transform(Span<const Real> input, Span<Complex> output) const;
    void transformInverse(Span<const Complex> input, Span<Real> output) const;

private:
    void computeFactors();

private:
    static constexpr size_t ComplexSize = FixedContextBase<Radices...>::ComplexSize;
    using IndexArray = typename FixedContextBase<Radices...>::IndexArray;

    mutable std::array<Complex, ComplexSize> m_data;

    std::array<Complex, ComplexSize> m_aFactors;
    std::array<Complex, ComplexSize> m_bFactors;
};

// ---------------------------------------------------------------------------------------------- //

template <size_t... Radices>
class FixedContext<Complex, Radices...> : public FixedContextBase<Radices...>
{
public:
    static constexpr size_t Size = (... * Radices);

public:
    FixedContext() = default;

    auto size() const -> size_t { return Size; }

    void transform(Span<const Complex> input, Span<Complex> output) const;
    void transformInverse(Span<const Complex> input, Span<Complex> output) const;

private:
    using IndexArray = typename FixedContextBase<Radices...>::IndexArray;
};


// ============================================================================================== //
//  Implementation details                                                                        //
// ============================================================================================== //

static constexpr auto Pi = static_cast<Real>(3.14159265358979323846);
static constexpr auto TwoPi = static_cast<Real>(2) * Pi;
static constexpr Complex I = { static_cast<Real>(0), static_cast<Real>(1) };

// ---------------------------------------------------------------------------------------------- //

inline
auto operator/(const Complex& lhs, unsigned int rhs) -> Complex
{
    return lhs / static_cast<Real>(rhs);
}

// ============================================================================================== //

template <size_t Radix, size_t Size>
class Stage {};

// ---------------------------------------------------------------------------------------------- //

template <size_t Size>
class Stage<2, Size>
{
public:
    static constexpr size_t Radix = 2;
    static constexpr size_t SubSize = Size / Radix;

public:
    Stage()
    {
        for (size_t i = 0; i < SubSize; ++i)
            m_factors1[i] = std::exp(-TwoPi * 1 * i * I / Size);
    }

    void transform(Span<Complex> data) const
    {
        const size_t size = data.size();

        for (size_t i = 0; i < size; i += Size)
        {
            for (size_t j = 0; j < SubSize; ++j)
            {
                const size_t base = i + j;

                const size_t index0 = base;
                const size_t index1 = base + SubSize;

                const Complex term0 =                 data[index0];
                const Complex term1 = m_factors1[j] * data[index1];

                data[index0] = term0 + term1;
                data[index1] = term0 - term1;
            }
        }
    }

private:
    std::array<Complex, SubSize> m_factors1;
};

// ---------------------------------------------------------------------------------------------- //

template <size_t Size>
class Stage<3, Size>
{
public:
    static constexpr size_t Radix = 3;
    static constexpr size_t SubSize = Size / Radix;

public:
    Stage()
    {
        for (size_t i = 0; i < SubSize; ++i)
        {
            m_factors1[i] = std::exp(-TwoPi * 1 * i * I / Size);
            m_factors2[i] = std::exp(-TwoPi * 2 * i * I / Size);
        }
    }

    void transform(Span<Complex> data) const
    {
        static constexpr Real A = 0.5;
        static constexpr Real B = 0.86602540378443864676;

        static constexpr Complex Factor1 = { -A, -B }; // = exp(-2*Pi*i * 1/3)
        static constexpr Complex Factor2 = { -A, +B }; // = exp(-2*Pi*i * 2/3)

        static constexpr size_t Offset0 = 0 * SubSize;
        static constexpr size_t Offset1 = 1 * SubSize;
        static constexpr size_t Offset2 = 2 * SubSize;

        const size_t size = data.size();

        for (size_t i = 0; i < size; i += Size)
        {
            for (size_t j = 0; j < SubSize; ++j)
            {
                const size_t base = i + j;

                const size_t index0 = base + Offset0;
                const size_t index1 = base + Offset1;
                const size_t index2 = base + Offset2;

                const Complex term0 =                 data[index0];
                const Complex term1 = m_factors1[j] * data[index1];
                const Complex term2 = m_factors2[j] * data[index2];

                data[index0] = term0 +           term1 +           term2;
                data[index1] = term0 + Factor1 * term1 + Factor2 * term2;
                data[index2] = term0 + Factor2 * term1 + Factor1 * term2;
            }
        }
    }

private:
    std::array<Complex, SubSize> m_factors1;
    std::array<Complex, SubSize> m_factors2;
};

// ---------------------------------------------------------------------------------------------- //

template <size_t Size>
class Stage<4, Size>
{
public:
    static constexpr size_t Radix = 4;
    static constexpr size_t SubSize = Size / Radix;

public:
    Stage()
    {
        for (size_t i = 0; i < SubSize; ++i)
        {
            m_factors1[i] = std::exp(-TwoPi * 1 * i * I / Size);
            m_factors2[i] = std::exp(-TwoPi * 2 * i * I / Size);
            m_factors3[i] = std::exp(-TwoPi * 3 * i * I / Size);
        }
    }

    void transform(Span<Complex> data) const
    {
        static constexpr size_t Offset0 = 0 * SubSize;
        static constexpr size_t Offset1 = 1 * SubSize;
        static constexpr size_t Offset2 = 2 * SubSize;
        static constexpr size_t Offset3 = 3 * SubSize;

        const size_t size = data.size();

        for (size_t i = 0; i < size; i += Size)
        {
            for (size_t j = 0; j < SubSize; ++j)
            {
                const size_t base = i + j;

                const size_t index0 = base + Offset0;
                const size_t index1 = base + Offset1;
                const size_t index2 = base + Offset2;
                const size_t index3 = base + Offset3;

                const Complex term0 =                 data[index0];
                const Complex term1 = m_factors1[j] * data[index1];
                const Complex term2 = m_factors2[j] * data[index2];
                const Complex term3 = m_factors3[j] * data[index3];

                data[index0] = term0 +     term1 + term2 +     term3;
                data[index1] = term0 - I * term1 - term2 + I * term3;
                data[index2] = term0 -     term1 + term2 -     term3;
                data[index3] = term0 + I * term1 - term2 - I * term3;
            }
        }
    }

private:
    std::array<Complex, SubSize> m_factors1;
    std::array<Complex, SubSize> m_factors2;
    std::array<Complex, SubSize> m_factors3;
};

// ---------------------------------------------------------------------------------------------- //

template <size_t Size>
class Stage<5, Size>
{
public:
    static constexpr size_t Radix = 5;
    static constexpr size_t SubSize = Size / Radix;

public:
    Stage()
    {
        for (size_t i = 0; i < SubSize; ++i)
        {
            m_factors1[i] = std::exp(-TwoPi * 1 * i * I / Size);
            m_factors2[i] = std::exp(-TwoPi * 2 * i * I / Size);
            m_factors3[i] = std::exp(-TwoPi * 3 * i * I / Size);
            m_factors4[i] = std::exp(-TwoPi * 4 * i * I / Size);
        }
    }

    void transform(Span<Complex> data) const
    {
        static constexpr Real A = 0.30901699437494742410;
        static constexpr Real B = 0.95105651629515357212;
        static constexpr Real C = 0.80901699437494742410;
        static constexpr Real D = 0.58778525229247312918;

        static constexpr Complex Factor1 = { +A, -B }; // = exp(-2*Pi*i * 1/5)
        static constexpr Complex Factor2 = { -C, -D }; // = exp(-2*Pi*i * 2/5)
        static constexpr Complex Factor3 = { -C, +D }; // = exp(-2*Pi*i * 3/5)
        static constexpr Complex Factor4 = { +A, +B }; // = exp(-2*Pi*i * 4/5)

        static constexpr size_t Offset0 = 0 * SubSize;
        static constexpr size_t Offset1 = 1 * SubSize;
        static constexpr size_t Offset2 = 2 * SubSize;
        static constexpr size_t Offset3 = 3 * SubSize;
        static constexpr size_t Offset4 = 4 * SubSize;

        const size_t size = data.size();

        for (size_t i = 0; i < size; i += Size)
        {
            for (size_t j = 0; j < SubSize; ++j)
            {
                const size_t base = i + j;

                const size_t index0 = base + Offset0;
                const size_t index1 = base + Offset1;
                const size_t index2 = base + Offset2;
                const size_t index3 = base + Offset3;
                const size_t index4 = base + Offset4;

                const Complex term0 =                 data[index0];
                const Complex term1 = m_factors1[j] * data[index1];
                const Complex term2 = m_factors2[j] * data[index2];
                const Complex term3 = m_factors3[j] * data[index3];
                const Complex term4 = m_factors4[j] * data[index4];

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

private:
    std::array<Complex, SubSize> m_factors1;
    std::array<Complex, SubSize> m_factors2;
    std::array<Complex, SubSize> m_factors3;
    std::array<Complex, SubSize> m_factors4;
};

// ---------------------------------------------------------------------------------------------- //

template <size_t Size>
class Stage<7, Size>
{
public:
    static constexpr size_t Radix = 7;
    static constexpr size_t SubSize = Size / Radix;

public:
    Stage()
    {
        for (size_t i = 0; i < SubSize; ++i)
        {
            m_factors1[i] = std::exp(-TwoPi * 1 * i * I / Size);
            m_factors2[i] = std::exp(-TwoPi * 2 * i * I / Size);
            m_factors3[i] = std::exp(-TwoPi * 3 * i * I / Size);
            m_factors4[i] = std::exp(-TwoPi * 4 * i * I / Size);
            m_factors5[i] = std::exp(-TwoPi * 5 * i * I / Size);
            m_factors6[i] = std::exp(-TwoPi * 6 * i * I / Size);
        }
    }

    void transform(Span<Complex> data) const
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

        static constexpr size_t Offset0 = 0 * SubSize;
        static constexpr size_t Offset1 = 1 * SubSize;
        static constexpr size_t Offset2 = 2 * SubSize;
        static constexpr size_t Offset3 = 3 * SubSize;
        static constexpr size_t Offset4 = 4 * SubSize;
        static constexpr size_t Offset5 = 5 * SubSize;
        static constexpr size_t Offset6 = 6 * SubSize;

        const size_t size = data.size();

        for (size_t i = 0; i < size; i += Size)
        {
            for (size_t j = 0; j < SubSize; ++j)
            {
                const size_t base = i + j;

                const size_t index0 = base + Offset0;
                const size_t index1 = base + Offset1;
                const size_t index2 = base + Offset2;
                const size_t index3 = base + Offset3;
                const size_t index4 = base + Offset4;
                const size_t index5 = base + Offset5;
                const size_t index6 = base + Offset6;

                const Complex term0 =                 data[index0];
                const Complex term1 = m_factors1[j] * data[index1];
                const Complex term2 = m_factors2[j] * data[index2];
                const Complex term3 = m_factors3[j] * data[index3];
                const Complex term4 = m_factors4[j] * data[index4];
                const Complex term5 = m_factors5[j] * data[index5];
                const Complex term6 = m_factors6[j] * data[index6];

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

private:
    std::array<Complex, SubSize> m_factors1;
    std::array<Complex, SubSize> m_factors2;
    std::array<Complex, SubSize> m_factors3;
    std::array<Complex, SubSize> m_factors4;
    std::array<Complex, SubSize> m_factors5;
    std::array<Complex, SubSize> m_factors6;
};

// ============================================================================================== //

template <size_t Radix, size_t... Radices>
class Transform<Radix, Radices...> : Transform<Radices...>
{
public:
    static constexpr size_t Size = (Radix * ... * Radices);

public:
    void transform(Span<Complex> data) const
    {
        if constexpr (sizeof...(Radices) > 0)
            Transform<Radices...>::transform(data);

        m_stage.transform(data);
    }

private:
    Stage<Radix, Size> m_stage;
};

// ============================================================================================== //

template <size_t... Radices>
FixedContextBase<Radices...>::FixedContextBase()
{
    computeReverseIndices();
}

// ---------------------------------------------------------------------------------------------- //

template <size_t... Radices>
void FixedContextBase<Radices...>::transform(Span<Complex> data) const
{
    m_transform.transform(data);
}

// ---------------------------------------------------------------------------------------------- //

template <size_t... Radices>
void FixedContextBase<Radices...>::computeReverseIndices()
{
    static constexpr size_t RadixCount = sizeof...(Radices);
    static const std::array<size_t, RadixCount> radices = { Radices... };

    const auto multiply = [&](size_t start)
    {
        return std::accumulate(radices.begin() + start, radices.end(),
                               static_cast<size_t>(1), std::multiplies<size_t>());
    };

    std::array<size_t, RadixCount> counters = {};
    std::array<size_t, RadixCount> factors;

    for (size_t i = 0; i < RadixCount; ++i)
        factors[i] = multiply(i + 1);

    for (size_t& index : m_reverseIndices)
    {
        index = 0;

        for (size_t i = 0; i < RadixCount; ++i)
            index += counters[i] * factors[i];

        for (size_t i = 0; i < RadixCount; ++i)
        {
            if (++counters[i] >= radices[i])
                counters[i] = 0;
            else
                break;
        }
    }
}

// ============================================================================================== //

template <size_t... Radices>
FixedContext<Real, Radices...>::FixedContext()
{
    computeFactors();
}

// ---------------------------------------------------------------------------------------------- //

template <size_t... Radices>
void FixedContext<Real, Radices...>::transform(Span<const Real> input,
                                               Span<Complex> output) const
{
    assert(input.size() == Size && output.size() == ComplexSize + 1);

    const IndexArray& reverseIndices = FixedContextBase<Radices...>::reverseIndices();

    for (size_t i = 0; i < ComplexSize; ++i)
        m_data[reverseIndices[i]] = { input[2*i], input[2*i + 1] };

    FixedContextBase<Radices...>::transform(m_data);

    for (size_t i = 0; i < ComplexSize; ++i)
    {
        const size_t index0 = i;
        const size_t index1 = (i == 0) ? 0 : (ComplexSize - i);

        output[i] =             m_data[index0]  * m_aFactors[index0]
                    + std::conj(m_data[index1]) * m_bFactors[index0];
    }

    output[ComplexSize] = std::real(m_data[0]) - std::imag(m_data[0]);
}

// ---------------------------------------------------------------------------------------------- //

template <size_t... Radices>
void FixedContext<Real, Radices...>::transformInverse(Span<const Complex> input,
                                                      Span<Real> output) const
{
    assert(output.size() == Size && input.size() == ComplexSize + 1);

    const IndexArray& reverseIndices = FixedContextBase<Radices...>::reverseIndices();

    for (size_t i = 0; i < ComplexSize; ++i)
    {
        const size_t index0 = i;
        const size_t index1 = ComplexSize - i;

        const Complex value =             input[index0]  * std::conj(m_aFactors[index0])
                              + std::conj(input[index1]) * std::conj(m_bFactors[index0]);

        m_data[reverseIndices[i]] = std::conj(value);
    }

    FixedContextBase<Radices...>::transform(m_data);

    const auto scale = static_cast<Real>(1) / ComplexSize;

    for (size_t i = 0; i < ComplexSize; ++i)
    {
        const Complex value = scale * std::conj(m_data[i]);

        output[2*i    ] = std::real(value);
        output[2*i + 1] = std::imag(value);
    }
}

// ---------------------------------------------------------------------------------------------- //

template <size_t... Radices>
void FixedContext<Real, Radices...>::computeFactors()
{
    static constexpr auto One = static_cast<Real>(1);
    static constexpr auto Two = static_cast<Real>(2);

    for (size_t i = 0; i < ComplexSize; ++i)
    {
        const Complex factor = I * std::exp(-Pi * i * I / ComplexSize);

        m_aFactors[i] = (One - factor) / Two;
        m_bFactors[i] = (One + factor) / Two;
    }
}

// ============================================================================================== //

template <size_t... Radices>
void FixedContext<Complex, Radices...>::transform(Span<const Complex> input,
                                                  Span<Complex> output) const
{
    assert(input.size() == Size && output.size() == Size);

    const IndexArray& reverseIndices = FixedContextBase<Radices...>::reverseIndices();

    for (size_t i = 0; i < Size; ++i)
        output[reverseIndices[i]] = input[i];

    FixedContextBase<Radices...>::transform(output);
}

// ---------------------------------------------------------------------------------------------- //

template <size_t... Radices>
void FixedContext<Complex, Radices...>::transformInverse(Span<const Complex> input,
                                                         Span<Complex> output) const
{
    assert(output.size() == Size && input.size() == Size);

    const IndexArray& reverseIndices = FixedContextBase<Radices...>::reverseIndices();

    for (size_t i = 0; i < Size; ++i)
        output[reverseIndices[i]] = std::conj(input[i]);

    FixedContextBase<Radices...>::transform(output);

    const auto scale = static_cast<Real>(1) / Size;

    for (size_t i = 0; i < Size; ++i)
        output[i] = scale * std::conj(output[i]);
}

// ---------------------------------------------------------------------------------------------- //

} // End of namespace sft

// ---------------------------------------------------------------------------------------------- //

#endif // LIBSFT_FIXED_H
