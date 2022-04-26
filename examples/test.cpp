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

#include <sft.h>

#include <chrono>
#include <iostream>
#include <vector>

// ---------------------------------------------------------------------------------------------- //

namespace {
    constexpr size_t InputSize = 100'800;
    constexpr size_t SampleRate = 1'000'000;
    constexpr double Frequency = 10'000.0;
    constexpr double TwoPi = 2.0 * 3.14159265358979323846;

    using TimeUnit = std::chrono::microseconds;
}

// ---------------------------------------------------------------------------------------------- //

void makeSignal(sft::Span<sft::Real> signal)
{
    for (size_t i = 0; i < signal.size(); ++i)
        signal[i] = 0.5 * std::sin(TwoPi * i * (1.0 / SampleRate) * Frequency) + 0.5;
}

// ---------------------------------------------------------------------------------------------- //

void makeSignal(sft::Span<sft::Complex> signal)
{
    for (size_t i = 0; i < signal.size(); ++i)
    {
        signal[i].real(0.5 * std::sin(TwoPi * i * (1.0 / SampleRate) * Frequency) + 0.5);
        signal[i].imag(0.5 * std::cos(TwoPi * i * (1.0 / SampleRate) * Frequency) + 0.5);
    }
}

// ---------------------------------------------------------------------------------------------- //

template <typename F>
auto measureTime(F func) -> TimeUnit::rep
{
    using Clock = std::chrono::steady_clock;

    auto tp1 = Clock::now();
    func();
    auto tp2 = Clock::now();

    return std::chrono::duration_cast<TimeUnit>(tp2 - tp1).count();
}

// ---------------------------------------------------------------------------------------------- //

#ifdef TEST_COMPLEX
void test()
{
    sft::Context<sft::Complex> context(InputSize);

    std::vector<sft::Complex> input(InputSize);
    std::vector<sft::Complex> output(InputSize);
    std::vector<sft::Complex> inverse(InputSize);

    makeSignal(input);

    std::clog << "Forward elapsed time: "
              << measureTime([&]{ context.transform(input, output); })
              << " µs\n";

    std::clog << "Inverse elapsed time: "
              << measureTime([&]{ context.transformInverse(output, inverse); })
              << " µs\n";

    std::cout << "Input Real;Input Imag;Inverse Real;Inverse Imag;;"
              << "Output Real;Output Imag;Output Abs;Output Arg\n";

    for (size_t i = 0; i < InputSize; ++i)
    {
        std::cout << std::real(input[i])   << ";"
                  << std::imag(input[i])   << ";"
                  << std::real(inverse[i]) << ";"
                  << std::imag(inverse[i]) << ";;"
                  << std::real(output[i])  << ";"
                  << std::imag(output[i])  << ";"
                  << std::abs(output[i])   << ";"
                  << std::arg(output[i]);

        std::cout << "\n";
    }
}
#endif // TEST_COMPLEX

// ---------------------------------------------------------------------------------------------- //

#ifndef TEST_COMPLEX
void test()
{
    sft::Context<sft::Real> context(InputSize);

    std::vector<sft::Real> input(InputSize);
    std::vector<sft::Real> inverse(InputSize);

    std::vector<sft::Complex> output(InputSize/2 + 1);

    makeSignal(input);

    std::clog << "Forward elapsed time: "
              << measureTime([&]{ context.transform(input, output); })
              << " µs\n";

    std::clog << "Inverse elapsed time: "
              << measureTime([&]{ context.transformInverse(output, inverse); })
              << " µs\n";

    std::cout << "Input;Inverse;;"
              << "Output Real;Output Imag;Output Abs;Output Arg\n";

    for (size_t i = 0; i < InputSize; ++i)
    {
        std::cout << input[i] << ";" << inverse[i] << ";;";

        if (i < output.size())
        {
            std::cout << std::real(output[i]) << ";"
                      << std::imag(output[i]) << ";"
                      << std::abs(output[i])  << ";"
                      << std::arg(output[i]);
        }

        std::cout << "\n";
    }
}
#endif // !TEST_COMPLEX

// ---------------------------------------------------------------------------------------------- //

auto main() -> int
{
#ifdef SFT_SINGLE_PRECISION
    std::clog << "SFT_SINGLE_PRECISION defined." << std::endl;
#else
    std::clog << "SFT_SINGLE_PRECISION *not* defined." << std::endl;
#endif

    try {
        test();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return -1;
    }
}

// ---------------------------------------------------------------------------------------------- //
