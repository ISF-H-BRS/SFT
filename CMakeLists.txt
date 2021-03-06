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

project(SFT)
cmake_minimum_required(VERSION 3.13)

set(SFT_VERSION_MAJOR 1)
set(SFT_VERSION_MINOR 0)
set(SFT_VERSION_PATCH 2)
set(SFT_VERSION ${SFT_VERSION_MAJOR}.${SFT_VERSION_MINOR}.${SFT_VERSION_PATCH} )

set(SFT_BUILD_EXAMPLES OFF CACHE BOOL "Build example programs")
set(SFT_BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmark programs")
set(SFT_NO_AVX OFF CACHE BOOL "Disable AVX support on x86_64")

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)

if (NOT CMAKE_CXX_COMPILER_ID MATCHES "MSVC|GNU|Clang")
    message(FATAL_ERROR "No supported C++ compiler found.")
endif()

##  Compiler flags  ################################################################################

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(CMAKE_CXX_FLAGS "/O2 /fp:fast /EHsc")
    set(SFT_OMP_FLAG /openmp::llvm)
else()
    set(CMAKE_CXX_FLAGS "-O3 -ffast-math -fvisibility=hidden -Wall")
    set(SFT_OMP_FLAG -fopenmp)
endif()

##  Disable ABI warning on ARM platforms  ##########################################################

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-psabi")
endif()

##  OpenMP library  ################################################################################

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(SFT_LIBOMP -lgomp)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(SFT_LIBOMP -lomp)
endif()

##  AVX support  ###################################################################################

if (SFT_NO_AVX)
    message("AVX support disabled.")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND CMAKE_SIZEOF_VOID_P EQUAL "8")
    message("AVX support enabled.")

    if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
    endif()
endif()

##  Headers  #######################################################################################

set(SFT_INCLUDE_DIR
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/sft>
    $<INSTALL_INTERFACE:include/sft>)

##  Double-precision, single-threaded  #############################################################

add_library(sft SHARED sft.cpp)
target_include_directories(sft PUBLIC ${SFT_INCLUDE_DIR})

##  Single-precision, single-threaded  #############################################################

add_library(sftf SHARED sft.cpp)
target_include_directories(sftf PUBLIC ${SFT_INCLUDE_DIR})
target_compile_definitions(sftf PUBLIC SFT_SINGLE_PRECISION)

##  Double-precision, multi-threaded  ##############################################################

add_library(sft_parallel SHARED sft.cpp)
target_include_directories(sft_parallel PUBLIC ${SFT_INCLUDE_DIR})
target_compile_options(sft_parallel PUBLIC ${SFT_OMP_FLAG})
target_link_libraries(sft_parallel ${SFT_LIBOMP})

##  Single-precision, multi-threaded  ##############################################################

add_library(sftf_parallel SHARED sft.cpp)
target_include_directories(sftf_parallel PUBLIC ${SFT_INCLUDE_DIR})
target_compile_definitions(sftf_parallel PUBLIC SFT_SINGLE_PRECISION)
target_compile_options(sftf_parallel PUBLIC ${SFT_OMP_FLAG})
target_link_libraries(sftf_parallel ${SFT_LIBOMP})

##  Double-precision, fixed  #######################################################################

add_library(sft_fixed INTERFACE)
target_include_directories(sft_fixed INTERFACE ${SFT_INCLUDE_DIR})

##  Single-precision, fixed  #######################################################################

add_library(sftf_fixed INTERFACE)
target_include_directories(sftf_fixed INTERFACE ${SFT_INCLUDE_DIR})
target_compile_definitions(sftf_fixed INTERFACE SFT_SINGLE_PRECISION)

##  Library version  ###############################################################################

set_target_properties(sft sftf sft_parallel sftf_parallel
                      PROPERTIES VERSION   ${SFT_VERSION}
                                 SOVERSION ${SFT_VERSION_MAJOR})

##  For Windows DLLs  ##############################################################################

if (WIN32)
    set_property(TARGET sft sftf sft_parallel sftf_parallel
                 APPEND PROPERTY COMPILE_DEFINITIONS SFT_BUILD_PROCESS)
    set_property(TARGET sft sftf sft_parallel sftf_parallel PROPERTY PREFIX "")
endif()

##  Config files  ##################################################################################

include(CMakePackageConfigHelpers)

set(SFT_CMAKE_CONFIG_PATH lib/cmake/sft)
set(INCLUDE_INSTALL_DIR include/sft)

configure_package_config_file(sft-config.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/sft-config.cmake
                              INSTALL_DESTINATION ${SFT_CMAKE_CONFIG_PATH}
                              PATH_VARS INCLUDE_INSTALL_DIR)

write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/sft-config-version.cmake
                                 VERSION ${SFT_VERSION}
                                 COMPATIBILITY SameMajorVersion)

##  Install  #######################################################################################

install(TARGETS sft sftf sft_parallel sftf_parallel sft_fixed sftf_fixed
        EXPORT sft-targets DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)
install(EXPORT sft-targets DESTINATION ${SFT_CMAKE_CONFIG_PATH})

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sft-config.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/sft-config-version.cmake
        DESTINATION ${SFT_CMAKE_CONFIG_PATH})

##  Examples  ######################################################################################

if (SFT_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

##  Benchmarks  ####################################################################################

if (SFT_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

####################################################################################################
