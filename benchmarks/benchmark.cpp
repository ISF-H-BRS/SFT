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

#ifdef BENCHMARK_FIXED
#include <sft_fixed.h>
#else
#include <sft.h>
#endif

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

// ---------------------------------------------------------------------------------------------- //

namespace {
    constexpr size_t WarmupIterations = 10;
    constexpr size_t IterationBase = 1<<24;

    using TimeUnit = std::chrono::nanoseconds;
}

// ---------------------------------------------------------------------------------------------- //

static inline
auto getRandom() -> sft::Real
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dist(-1000, 1000);

    return static_cast<sft::Real>(dist(gen) * 0.001);
}

// ---------------------------------------------------------------------------------------------- //

static inline
void makeSignal(sft::Span<sft::Real> signal)
{
    for (size_t i = 0; i < signal.size(); ++i)
        signal[i] = getRandom();
}

// ---------------------------------------------------------------------------------------------- //

static inline
void makeSignal(sft::Span<sft::Complex> signal)
{
    for (size_t i = 0; i < signal.size(); ++i)
    {
        signal[i].real(getRandom());
        signal[i].imag(getRandom());
    }
}

// ---------------------------------------------------------------------------------------------- //

static inline
auto getIterationCount(size_t inputSize) -> size_t
{
    return std::max<size_t>(IterationBase / inputSize, 1);
}

// ---------------------------------------------------------------------------------------------- //

template <typename Func>
auto measureTime(Func func, size_t iterationCount) -> TimeUnit::rep
{
    using Clock = std::chrono::steady_clock;

    // Make sure CPU is running at full frequency
    for (size_t i = 0; i < WarmupIterations; ++i)
        func();

    auto tp1 = Clock::now();

    for (size_t i = 0; i < iterationCount; ++i)
        func();

    auto tp2 = Clock::now();

    return std::chrono::duration_cast<TimeUnit>(tp2 - tp1).count();
}

// ---------------------------------------------------------------------------------------------- //

static inline
auto adjustUnit(double& time, std::string& unit)
{
    unit = "ns";

    if (time > 1.0e9)
    {
        time /= 1.0e9;
        unit = " s";
    }
    else if (time > 1.0e6)
    {
        time /= 1.0e6;
        unit = "ms";
    }
    else if (time > 1.0e3)
    {
        time /= 1.0e3;
        unit = "us";
    }
}

// ---------------------------------------------------------------------------------------------- //

static inline
void printLine(size_t inputSize,
               const std::string& forwardReal, const std::string& inverseReal,
               const std::string& forwardComplex, const std::string& inverseComplex)
{
    std::cout << "| " << std::setw(10) << std::right << inputSize << " "
              << "| " << std::setw(12) << std::right << forwardReal << " "
              << "| " << std::setw(12) << std::right << inverseReal << " "
              << "| " << std::setw(15) << std::right << forwardComplex << " "
              << "| " << std::setw(15) << std::right << inverseComplex << " "
              << "|"  << std::endl;
}

// ---------------------------------------------------------------------------------------------- //

template <typename ContextType, typename InputType, bool Inverse>
auto run(const ContextType& context) -> std::string
{
    const size_t inputSize = context.size();
    const size_t outputSize = std::is_same_v<InputType, sft::Real> ? (inputSize/2 + 1) : inputSize;

    const size_t iterationCount = getIterationCount(inputSize);

    std::vector<InputType> timeSignal(inputSize);
    std::vector<sft::Complex> freqSignal(outputSize);

    makeSignal(timeSignal);

    std::function<void()> func;

    if constexpr (Inverse)
        func = [&]{ context.transformInverse(freqSignal, timeSignal); };
    else
        func = [&]{ context.transform(timeSignal, freqSignal); };

    auto time = static_cast<double>(measureTime(func, iterationCount)) / iterationCount;

    std::string unit;
    adjustUnit(time, unit);

    std::ostringstream out;
    out << std::left << std::showpoint << std::setprecision(4) << std::setw(5) << std::setfill('0')
        << time << " " << unit;

    return out.str();
}

// ---------------------------------------------------------------------------------------------- //

#ifdef BENCHMARK_FIXED

// ---------------------------------------------------------------------------------------------- //

template <typename InputType, bool Inverse, size_t... Radices>
auto run() -> std::string
{
    using Context = sft::FixedContext<InputType, Radices...>;

    auto context = std::make_unique<Context>();
    return run<Context, InputType, Inverse>(*context);
}

// ---------------------------------------------------------------------------------------------- //

template <size_t... Radices>
void runAll()
{
    const size_t inputSize = sft::FixedContext<sft::Real, Radices...>::Size;

    const std::string forwardReal    = run<sft::Real,    false, Radices...   >();
    const std::string inverseReal    = run<sft::Real,    true , Radices...   >();
    const std::string forwardComplex = run<sft::Complex, false, Radices..., 2>();
    const std::string inverseComplex = run<sft::Complex, true , Radices..., 2>();

    printLine(inputSize, forwardReal, inverseReal, forwardComplex, inverseComplex);
}

// ---------------------------------------------------------------------------------------------- //

#else // !BENCHMARK_FIXED

// ---------------------------------------------------------------------------------------------- //

template <typename InputType, bool Inverse>
auto run(size_t inputSize) -> std::string
{
    using Context = sft::Context<InputType>;

    Context context(inputSize);
    return run<sft::Context<InputType>, InputType, Inverse>(context);
}

// ---------------------------------------------------------------------------------------------- //

void runAll(size_t inputSize)
{
    const std::string forwardReal    = run<sft::Real,    false>(inputSize);
    const std::string inverseReal    = run<sft::Real,    true >(inputSize);
    const std::string forwardComplex = run<sft::Complex, false>(inputSize);
    const std::string inverseComplex = run<sft::Complex, true >(inputSize);

    printLine(inputSize, forwardReal, inverseReal, forwardComplex, inverseComplex);
}

// ---------------------------------------------------------------------------------------------- //

#endif // BENCHMARK_FIXED

// ---------------------------------------------------------------------------------------------- //

auto main() -> int
{
#ifdef SFT_SINGLE_PRECISION
    std::clog << "SFT_SINGLE_PRECISION defined." << std::endl;
#else
    std::clog << "SFT_SINGLE_PRECISION *not* defined." << std::endl;
#endif

    std::cout
        << "\n"
        << "| Input Size | Forward Real | Inverse Real | Forward Complex | Inverse Complex |\n"
        << "| ---------: | -----------: | -----------: | --------------: | --------------: |\n";

#ifdef BENCHMARK_FIXED
    runAll<5,5,5,4>();
    runAll<4,4,4,4,2>();
    runAll<5,5,5,5,5,4,4>();
    runAll<7,5,5,4,4,3,3,2>();
    runAll<5,5,5,5,5,5,4,4,2>();
    runAll<4,4,4,4,4,4,4,4,4,2>();
    runAll<5,5,5,5,5,5,5,5,4,4,4,2>();
    runAll<7,7,5,5,5,4,4,4,4,4,3,3>();
#else
    const std::vector<size_t> inputSizes = {
        1000, 1024, 100'000, 100'800, 1'000'000, 1'048'576, 100'000'000, 112'896'000
    };

    for (auto size : inputSizes)
        runAll(size);
#endif

    std::cout << std::endl;
    return 0;
}

// ---------------------------------------------------------------------------------------------- //
