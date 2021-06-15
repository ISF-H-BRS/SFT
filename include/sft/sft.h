// ============================================================================================== //
//                                                                                                //
//  This file is part of the Simple Fourier Transform Library.                                    //
//                                                                                                //
//  Author:                                                                                       //
//  Marcel Hasler <marcel.hasler@h-brs.de>                                                        //
//                                                                                                //
//  Copyright (c) 2020 - 2021                                                                     //
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

#ifndef LIBSFT_H
#define LIBSFT_H

#ifdef _WIN32
  #ifdef SFT_BUILD_PROCESS
    #define SFT_EXPORT __declspec(dllexport)
  #else
    #define SFT_EXPORT __declspec(dllimport)
  #endif
#else
  #define SFT_EXPORT __attribute__((visibility("default")))
#endif

// ---------------------------------------------------------------------------------------------- //
//  C++ API                                                                                       //
// ---------------------------------------------------------------------------------------------- //

#ifdef __cplusplus

#include <complex>
#include <memory>
#include <stdexcept>

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

using Error = std::runtime_error;

// ---------------------------------------------------------------------------------------------- //

template <typename T>
class Context
{
    static_assert(std::is_same_v<T, Real> || std::is_same_v<T, Complex>);

public:
    Context(size_t size);
    ~Context();

    auto size() const -> size_t;

    void transform(Span<const T> input, Span<Complex> output) const;
    void transformInverse(Span<const Complex> input, Span<T> output) const;

private:
    class Private;
    std::unique_ptr<Private> d;
};

// ---------------------------------------------------------------------------------------------- //

} // End of namespace sft

// ---------------------------------------------------------------------------------------------- //

#endif // __cplusplus


// ---------------------------------------------------------------------------------------------- //
//  C API                                                                                         //
// ---------------------------------------------------------------------------------------------- //

#ifdef __cplusplus
extern "C" {
#else
#include <stddef.h>
#endif

// ---------------------------------------------------------------------------------------------- //

#ifdef SFT_SINGLE_PRECISION
typedef float sft_real;
#else
typedef double sft_real;
#endif

typedef struct {
    sft_real real;
    sft_real imag;
} sft_complex;

typedef enum {
    SFT_SUCCESS,
    SFT_INVALID_SIZE
} sft_result;

// ---------------------------------------------------------------------------------------------- //

typedef struct _sft_real_context sft_real_context;

SFT_EXPORT sft_result sft_create_real_context(size_t size, sft_real_context** context);
SFT_EXPORT void sft_free_real_context(sft_real_context* context);

SFT_EXPORT size_t sft_get_real_size(sft_real_context* context);

SFT_EXPORT void sft_transform_real(const sft_real_context* context,
                                   const sft_real* input, sft_complex* output);

SFT_EXPORT void sft_transform_inverse_real(const sft_real_context* context,
                                           const sft_complex* input, sft_real* output);

// ---------------------------------------------------------------------------------------------- //

typedef struct _sft_complex_context sft_complex_context;

SFT_EXPORT sft_result sft_create_complex_context(size_t size, sft_complex_context** context);
SFT_EXPORT void sft_free_complex_context(sft_complex_context* context);

SFT_EXPORT size_t sft_get_complex_size(sft_complex_context* context);

SFT_EXPORT void sft_transform_complex(const sft_complex_context* context,
                                      const sft_complex* input, sft_complex* output);

SFT_EXPORT void sft_transform_inverse_complex(const sft_complex_context* context,
                                              const sft_complex* input, sft_complex* output);

// ---------------------------------------------------------------------------------------------- //

#ifdef __cplusplus
} // extern "C"
#endif

// ---------------------------------------------------------------------------------------------- //

#endif // LIBSFT_H
