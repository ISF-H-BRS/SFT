####################################################################################################
#                                                                                                  #
#   This file is part of the Simple Fourier Transform Library.                                     #
#                                                                                                  #
#   Author:                                                                                        #
#   Marcel Hasler <marcel.hasler@h-brs.de>                                                         #
#                                                                                                  #
#   Copyright (c) 2020 - 2021                                                                      #
#   Bonn-Rhein-Sieg University of Applied Sciences                                                 #
#                                                                                                  #
#   This library is free software: you can redistribute it and/or modify it under the terms of     #
#   the GNU Lesser General Public License as published by the Free Software Foundation, either     #
#   version 3 of the License, or (at your option) any later version.                               #
#                                                                                                  #
#   This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;      #
#   without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.      #
#   See the GNU Lesser General Public License for more details.                                    #
#                                                                                                  #
#   You should have received a copy of the GNU Lesser General Public License along with this       #
#   library. If not, see <https://www.gnu.org/licenses/>.                                          #
#                                                                                                  #
####################################################################################################

message("Building examples.")

##  Real  ##########################################################################################

add_executable(test_real test.cpp)
target_link_libraries(test_real sft)

add_executable(test_real_parallel test.cpp)
target_link_libraries(test_real_parallel sft_parallel)

add_executable(test_real_float test.cpp)
target_link_libraries(test_real_float sftf)

add_executable(test_real_float_parallel test.cpp)
target_link_libraries(test_real_float_parallel sftf_parallel)

##  Complex  #######################################################################################

add_executable(test_complex test.cpp)
target_compile_definitions(test_complex PUBLIC TEST_COMPLEX)
target_link_libraries(test_complex sft)

add_executable(test_complex_parallel test.cpp)
target_compile_definitions(test_complex_parallel PUBLIC TEST_COMPLEX)
target_link_libraries(test_complex_parallel sft_parallel)

add_executable(test_complex_float test.cpp)
target_compile_definitions(test_complex_float PUBLIC TEST_COMPLEX)
target_link_libraries(test_complex_float sftf)

add_executable(test_complex_float_parallel test.cpp)
target_compile_definitions(test_complex_float_parallel PUBLIC TEST_COMPLEX)
target_link_libraries(test_complex_float_parallel sftf_parallel)

##  C API  #########################################################################################

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(SFT_LIBM "")
else()
    set(SFT_LIBM m)
endif()

add_executable(ctest_real test.c)
target_link_libraries(ctest_real sft ${SFT_LIBM})

add_executable(ctest_complex test.c)
target_compile_definitions(ctest_complex PUBLIC TEST_COMPLEX)
target_link_libraries(ctest_complex sft ${SFT_LIBM})

##  Fixed  #########################################################################################

add_executable(fixed_real fixed.cpp)
target_link_libraries(fixed_real sft_fixed)

add_executable(fixed_complex fixed.cpp)
target_compile_definitions(fixed_complex PUBLIC TEST_COMPLEX)
target_link_libraries(fixed_complex sft_fixed)

####################################################################################################
