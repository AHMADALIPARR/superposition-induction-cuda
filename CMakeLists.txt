# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-SnapKitty-Opaque-1.0
# Copyright Ahmad Ali Parr
cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING "CUDA architectures (sm_XX numbers)")
endif()

project(superposition_induction_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -Xcompiler=-Wall")

# ---------------------------------------------------------------- CPU only --
add_library(sk_common_cpu STATIC src/common/tensor_file.cpp)
target_include_directories(sk_common_cpu PUBLIC include)

add_library(sid_cpu STATIC src/sid/model.cpp src/sid/cpu_ref.cpp)
target_link_libraries(sid_cpu PUBLIC sk_common_cpu)

add_library(amm_cpu STATIC src/alexnet_mmep/model.cpp src/alexnet_mmep/cpu_ref.cpp)
target_link_libraries(amm_cpu PUBLIC sk_common_cpu)

# -------------------------------------------------------------------- CUDA --
add_library(sk_common_cuda STATIC src/common/cuda_util.cu)
target_include_directories(sk_common_cuda PUBLIC include)
target_link_libraries(sk_common_cuda PUBLIC CUDA::cudart CUDA::cublas)

add_library(sid_cuda STATIC src/sid/kernels.cu src/sid/cuda_model.cu)
target_link_libraries(sid_cuda PUBLIC sid_cpu sk_common_cuda)

add_library(amm_cuda STATIC src/alexnet_mmep/kernels.cu src/alexnet_mmep/cuda_model.cu)
target_link_libraries(amm_cuda PUBLIC amm_cpu sk_common_cuda)

# ------------------------------------------------------------------- tests --
enable_testing()

function(sk_cpu_test name)
  add_executable(${name} tests/${name}.cpp)
  target_include_directories(${name} PRIVATE tests)
  target_link_libraries(${name} PRIVATE ${ARGN})
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES LABELS cpu)
endfunction()

function(sk_gpu_test name)
  add_executable(${name} tests/${name}.cu)
  target_include_directories(${name} PRIVATE tests)
  target_link_libraries(${name} PRIVATE ${ARGN})
  add_test(NAME ${name} COMMAND ${name})
  # exit code 77 = no CUDA device on this machine
  set_tests_properties(${name} PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
endfunction()

sk_cpu_test(weights_io sid_cpu amm_cpu)
sk_cpu_test(sid_cpu_gradcheck sid_cpu)
sk_cpu_test(amm_cpu_gradcheck amm_cpu)
sk_gpu_test(sid_cuda_forward sid_cuda)
sk_gpu_test(sid_cuda_backward sid_cuda)
sk_gpu_test(amm_cuda_forward amm_cuda)
sk_gpu_test(amm_cuda_backward amm_cuda)

# ------------------------------------------------------------ bench / apps --
add_executable(cpu_ref_bench bench/cpu_ref_bench.cpp)
target_link_libraries(cpu_ref_bench PRIVATE sid_cpu amm_cpu)
add_executable(sid_bench bench/sid_bench.cu)
target_link_libraries(sid_bench PRIVATE sid_cuda)
add_executable(amm_bench bench/amm_bench.cu)
target_link_libraries(amm_bench PRIVATE amm_cuda)

add_executable(sid_init apps/sid_init.cpp)
target_link_libraries(sid_init PRIVATE sid_cpu)
add_executable(sid_run apps/sid_run.cu)
target_link_libraries(sid_run PRIVATE sid_cuda)
add_executable(amm_init apps/amm_init.cpp)
target_link_libraries(amm_init PRIVATE amm_cpu)
add_executable(amm_run apps/amm_run.cu)
target_link_libraries(amm_run PRIVATE amm_cuda)
