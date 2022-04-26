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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------------------------- //

#define INPUT_SIZE 100800
#define SAMPLE_RATE 1000000
#define FREQUENCY 10000.0
#define TWO_PI (2.0 * 3.14159265358979323846)

// ---------------------------------------------------------------------------------------------- //

#ifdef TEST_COMPLEX
int test()
{
    sft_complex_context *context = NULL;

    if (sft_create_complex_context(INPUT_SIZE, &context) != SFT_SUCCESS)
    {
        fprintf(stderr, "Unable to create complex-valued FFT context.\n");
        return -1;
    }

    sft_complex *input = malloc(INPUT_SIZE * sizeof(sft_complex));
    sft_complex *output = malloc(INPUT_SIZE * sizeof(sft_complex));
    sft_complex *inverse = malloc(INPUT_SIZE * sizeof(sft_complex));

    for (size_t i = 0; i < INPUT_SIZE; ++i)
    {
        input[i].real = 0.5 * sin(TWO_PI * i * (1.0 / SAMPLE_RATE) * FREQUENCY) + 0.5;
        input[i].imag = 0.5 * cos(TWO_PI * i * (1.0 / SAMPLE_RATE) * FREQUENCY) + 0.5;
    }

    sft_transform_complex(context, input, output);
    sft_transform_inverse_complex(context, output, inverse);

    printf("Input Real;Input Imag;Inverse Real;Inverse Imag;;"
           "Output Real;Output Imag;Output Abs;Output Arg\n");

    for (size_t i = 0; i < INPUT_SIZE; ++i)
    {
        sft_real in_real = input[i].real;
        sft_real in_imag = input[i].imag;
        sft_real inv_real = inverse[i].real;
        sft_real inv_imag = inverse[i].imag;
        sft_real out_real = output[i].real;
        sft_real out_imag = output[i].imag;
        sft_real out_abs = sqrt(out_real*out_real + out_imag*out_imag);
        sft_real out_arg = atan2(out_imag, out_real);

        printf("%f;%f;%f;%f;;%f;%f;%f;%f\n", in_real, in_imag, inv_real, inv_imag,
                                             out_real, out_imag, out_abs, out_arg);
    }

    free(input);
    free(output);
    free(inverse);

    sft_free_complex_context(context);

    return 0;
}
#endif // TEST_COMPLEX

// ---------------------------------------------------------------------------------------------- //

#ifndef TEST_COMPLEX
int test()
{
    sft_real_context *context = NULL;

    if (sft_create_real_context(INPUT_SIZE, &context) != SFT_SUCCESS)
    {
        fprintf(stderr, "Unable to create real-valued FFT context.\n");
        return -1;
    }

    sft_real *input = malloc(INPUT_SIZE * sizeof(sft_real));
    sft_real *inverse = malloc(INPUT_SIZE * sizeof(sft_real));

    sft_complex *output = malloc((INPUT_SIZE/2 + 1) * sizeof(sft_complex));

    for (size_t i = 0; i < INPUT_SIZE; ++i)
        input[i] = 0.5 * sin(TWO_PI * i * (1.0 / SAMPLE_RATE) * FREQUENCY) + 0.5;

    sft_transform_real(context, input, output);
    sft_transform_inverse_real(context, output, inverse);

    printf("Input;Inverse;;"
           "Output Real;Output Imag;Output Abs;Output Arg\n");

    for (size_t i = 0; i < INPUT_SIZE; ++i)
    {
        if (i < INPUT_SIZE/2 + 1)
        {
            sft_real real = output[i].real;
            sft_real imag = output[i].imag;
            sft_real abs = sqrt(real*real + imag*imag);
            sft_real arg = atan2(imag, real);

            printf("%f;%f;;%f;%f;%f;%f\n", input[i], inverse[i], real, imag, abs, arg);
        }
        else
            printf("%f;%f\n", input[i], inverse[i]);
    }

    free(input);
    free(output);
    free(inverse);

    sft_free_real_context(context);

    return 0;
}
#endif // !TEST_COMPLEX

// ---------------------------------------------------------------------------------------------- //

int main(void)
{
#ifdef SFT_SINGLE_PRECISION
    fprintf(stderr, "SFT_SINGLE_PRECISION defined.\n");
#else
    fprintf(stderr, "SFT_SINGLE_PRECISION *not* defined.\n");
#endif

    return test();
}

// ---------------------------------------------------------------------------------------------- //
