# SFT: The Simple Fourier Transform Library

## Introduction

SFT is a software library providing an efficient implementation of the Fast Fourier Transform (FFT) for real- and complex-valued input data.

It is written in modern C++ and offers a simple-to-use C++ interface as well as a traditional C interface, making it useful in a wide range of applications. Both APIs are provided by the header file `<sft.h>`.

The library was developed both out of practical necessity as well as for self-education purposes. It doesn't aim to compete with other popular FFT implementations in terms of speed, but rather to provide an efficient yet straightforward reference implementation that is easy to follow by students and researchers alike.

Both forward and inverse FFTs are implemented using iterative mixed-radix Cooley-Tukey algorithms for radices of 2, 3, 4, 5 and 7. Therefore the number of input samples must be expressible in terms of these factors. For real-valued transforms the number of input samples additionally must be even.

SFT follows the convention shared by most FFT implementations of scaling only the result of the inverse transform but leaving the result of the forward transform unscaled.

All transform functions are reentrant. For real-valued transforms, however, an intermediary buffer is used internally which is protected by a mutex. If multiple real-valued FFTs are to be performed concurrently it is therefore recommended to use one execution context per thread.

The build system will produce four distinct library versions:

- **sft**: double-precision, single-threaded
- **sftf**: single-precision, single-threaded
- **sft_parallel**: double-precision, multi-threaded
- **sftf_parallel**: single-precision, multi-threaded

The multi-threaded versions make use of the compiler's OpenMP support.

What version to use depends on your hardware architecture and execution environment. On recent 64-bit processors the speed gained from using single-precision arithmetic is usually negligible, whereas 32-bit architectures may profit significantly.

Unless multiple FFTs are to be computed simultaneously, the parallel versions will typically perform significantly better on multi-core or multi-processor systems than the single-threaded versions. This is especially true for large data sets.

## License

SFT is released under the comparatively permissive GNU Lesser General Public License (LGPL). See [COPYING](COPYING) along with [COPYING.LESSER](COPYING.LESSER) for details.

## Using the C++ Interface

The C++ interface uses two aliases to represent real- and complex-valued numbers.

`sft::Real` by default represents a `double`. If the global symbol `SFT_SINGLE_PRECISION` is set, `sft::Real` will be a `float` instead. Be sure to link against the single-precision library **sftf** rather than the default double-precision library **sft** in this case.

`sft::Complex` is an alias for `std::complex<stf::Real>` and will always represent the correct single- or double-precision complex type.

In order to allow arbitrary types of data buffers to be passed to the transform functions, SFT uses `std::span` as parameter type. Since this is a C++20 feature which isn't currently supported by all compilers, a partial copy of the [C++ Core Guidelines Support Library](https://github.com/microsoft/GSL) (GSL) is included that provides the equivalent type `gsl::span`.

### Transforming Real-valued Data

First create a new context for a given number of input samples. This will allocate any memory required to perform the transforms. In order to avoid repeated allocations and deallocations and thus maximize performance it is strongly recommended to reuse the same context whenever possible.

```cpp
sft::Context<sft::Real> context(InputSize);
```

If the context creation fails, the library will throw an exception of type `sft::Error`.

Allocate input and output buffers. Any type of buffer can be used, provided that it is supported by `std::span`. This includes traditional C arrays as well as `std::array` and `std::vector`. For real-valued transforms the number of output samples must be equal to N/2 + 1 for an input buffer of size N.

```cpp
std::vector<sft::Real> input(InputSize);
std::vector<sft::Complex> output(InputSize/2 + 1);
```

Forward-transform the sample buffer:

```cpp
context.transform(input, output);
```

Or inverse-transform a buffer:

```cpp
std::vector<sft::Real> inverse(InputSize);
context.transformInverse(output, inverse);
```

### Transforming Complex-valued Data

The process for transforming complex-valued data is almost identical, except that the number of input and output samples must match.

```cpp
sft::Context<sft::Complex> context(InputSize);

std::vector<sft::Complex> input(InputSize);
std::vector<sft::Complex> output(InputSize);
std::vector<sft::Complex> inverse(InputSize);

context.transform(input, output);
context.transformInverse(output, inverse);
```

## Using the C Interface

In line with the C++ interface the C interface uses the typedef `sft_real` to represent either a single- or double-precision sample value. The type `sft_complex` contains the real and imaginary components of a complex value.

Unlike their C++ counterparts the corresponding C functions take raw pointers to the input/output buffers. It is up to the programmer to ensure that the data buffers meet the size requirements.

### Transforming Real-valued Data

First create a new context for a given number of input samples. The value of the result type `sft_result` can be either `SFT_SUCCESS` or `SFT_INVALID_SIZE`. If the context creation fails, `context` will be set to `NULL`.

```c
sft_real_context *context;
sft_result result = sft_create_real_context(INPUT_SIZE, &context);

if (result != SFT_SUCCESS) {
    /* Handle error */
}
```

Allocate input/output buffers and transform the data:

```c
sft_real input[INPUT_SIZE];
sft_real inverse[INPUT_SIZE];

sft_complex output[INPUT_SIZE/2 + 1];

sft_transform_real(context, input, output);
sft_transform_inverse_real(context, output, inverse);
```

Remember to free the context when it is no longer needed:

```c
sft_free_real_context(context);
```

### Transforming Complex-valued Data

Again the process is almost identical, except for the matching number of input and output samples.

```c
sft_complex_context *context;
sft_result result = sft_create_complex_context(INPUT_SIZE, &context);

if (result != SFT_SUCCESS) {
    /* Handle error */
}

sft_complex input[INPUT_SIZE];
sft_complex output[INPUT_SIZE];
sft_complex inverse[INPUT_SIZE];

sft_transform_complex(context, input, output);
sft_transform_inverse_complex(context, output, inverse);

sft_free_complex_context(context);
```

## Using the Fixed-size Template Library

In addition to the dynamically linked libraries, SFT also includes a template library that provides an alternative implementation for compile-time constant context sizes. This version doesn't require any type of dynamic memory allocation and also omits any use of exceptions, making it an implementation suitable for use on embedded systems where these may be undesirable or even unavailable.

Unlike the dynamic library, the fixed-sized version requires the programmer to explicitly specify the radices to use as template arguments, the product of which determines the context size. As with the dynamic library, both real- and complex-valued transforms are supported.

The fixed-size library is provided by the header file `<sft_fixed.h>`.

### Transforming Real-valued Data

First create a new context, specifying the radices to use. In the following example a context for an input size of 20'000 is created.

**Note:** For real-valued transforms the product of radices is implicitly multiplied by two.

```cpp
using Context = sft::FixedContext<sft::Real,5,5,5,5,4,4>;
static_assert(Context::Size == InputSize);

Context context;
```

Allocate input/output buffers and transform the data:

```cpp
std::array<sft::Real, InputSize> input;
std::array<sft::Real, InputSize> inverse;

std::array<sft::Complex, InputSize/2 + 1> output;

context.transform(input, output);
context.transformInverse(output, inverse);
```

### Transforming Complex-valued Data

The process for transforming complex-valued data is once again almost identical. Note however, that unlike for real-valued contexts, there is no implicit multiplication by two.

```cpp
using Context = sft::FixedContext<sft::Complex,5,5,5,5,4,4,2>;
static_assert(Context::Size == InputSize);

Context context;

std::array<sft::Complex, InputSize> input;
std::array<sft::Complex, InputSize> output;
std::array<sft::Complex, InputSize> inverse;

context.transform(input, output);
context.transformInverse(output, inverse);
```

## Installation

SFT uses [CMake](https://cmake.org/) to build. There are two options that can be specified when building:

- **SFT_BUILD_EXAMPLES**: Builds various example programs to demonstrate the usage of the library
- **SFT_NO_AVX**: Disables AVX instructions for x86_64 targets that don't support them

On Unix systems the usual sequence can be run from the command line to build and install the library:

```sh
mkdir build && cd build
cmake -DSFT_BUILD_EXAMPLES=1 /path/to/sources
make
sudo make install
```

The example programs will not be installed but can be run locally from the build directory. For more information on building for other platforms, see the [CMake User Interaction Guide](https://cmake.org/cmake/help/latest/guide/user-interaction/index.html).

## Supported Compilers

SFT has so far been tested and confirmed to work using the following compilers:

- Clang 11.0.1 (FreeBSD, x86_64)
- Clang 11.1.0 (Linux, aarch64)
- Clang 11.1.0 (Linux, armv7l)
- Clang 11.1.0 (Linux, x86_64)
- Clang 12.0.0 (Windows, i686)
- Clang 12.0.0 (Windows, x86_64)
- GCC 9.3.1 (Bare-metal, cortex-m4, fixed-size only)
- GCC 10.2.0 (Linux, aarch64)
- GCC 10.2.0 (Linux, armv7l)
- GCC 11.1.0 (Linux, x86_64)
- GCC 10.3.0 (Windows, i686)
- GCC 10.3.0 (Windows, x86_64)
- MSVC 16.9 (Windows, x86_64)

Other recent compilers and target platforms are likely to work as well, but may require adjustments to the CMake build file.

## Contributing

While we consider the library to be feature-complete and no further development is planned at this point, patches to fix bugs or to add support for additional compilers or targets are always welcome and can be submitted by [email](mailto:marcel.hasler@h-brs.de).
