####################################################################################################
#                                                                                                  #
#   This file is part of the Simple Fourier Transform Library.                                     #
#                                                                                                  #
#   Author:                                                                                        #
#   Marcel Hasler <marcel.hasler@h-brs.de>                                                         #
#                                                                                                  #
#   Copyright (c) 2020 - 2022                                                                      #
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

message("Building benchmarks.")

##  Single-threaded  ###############################################################################

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark sft)

add_executable(benchmark_float benchmark.cpp)
target_link_libraries(benchmark_float sftf)

##  Multi-threaded  ################################################################################

add_executable(benchmark_parallel benchmark.cpp)
target_link_libraries(benchmark_parallel sft_parallel)

add_executable(benchmark_float_parallel benchmark.cpp)
target_link_libraries(benchmark_float_parallel sftf_parallel)

##  Fixed  #########################################################################################

add_executable(benchmark_fixed benchmark.cpp)
target_compile_definitions(benchmark_fixed PRIVATE BENCHMARK_FIXED)
target_link_libraries(benchmark_fixed sft_fixed)

add_executable(benchmark_float_fixed benchmark.cpp)
target_compile_definitions(benchmark_float_fixed PRIVATE BENCHMARK_FIXED)
target_link_libraries(benchmark_float_fixed sftf_fixed)

####################################################################################################
