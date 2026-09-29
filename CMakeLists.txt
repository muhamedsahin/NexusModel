cmake_minimum_required(VERSION 3.20)

project(NexusModel
  VERSION 1.0.0
  DESCRIPTION "Neural network layers, forward and explicit backward, for the Nexus stack"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NEXUS_MODEL_WITH_CUDA "Build fused CUDA kernels" OFF)
option(NEXUS_MODEL_WITH_MATRIXFLASH "Route optional GEMM through MatrixFlash Pro" OFF)
option(NEXUS_MODEL_BUILD_TESTS "Build correctness tests" ON)
option(NEXUS_MODEL_BUILD_BENCHMARKS "Build benchmarks" ON)
option(NEXUS_MODEL_BUILD_EXAMPLES "Build the ecosystem example" OFF)
option(NEXUS_MODEL_ENABLE_AVX512 "Compile AVX-512 kernels" ON)
option(NEXUS_MODEL_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

if(NEXUS_MODEL_WITH_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

set(NEXUS_MODEL_SOURCES
  src/core/scalar_kernels.cpp
  src/core/convolution.cpp
  src/core/normalization.cpp
  src/core/pooling.cpp
  src/core/embedding.cpp
  src/core/dropout.cpp
  src/simd/simd_dispatch.cpp
  src/simd/avx2_kernels.cpp
  src/simd/avx2/linear.cpp
  src/simd/avx2/normalization.cpp
)

if(NEXUS_MODEL_ENABLE_AVX512)
  list(APPEND NEXUS_MODEL_SOURCES src/simd/avx512_kernels.cpp)
endif()

if(NEXUS_MODEL_WITH_CUDA)
  list(APPEND NEXUS_MODEL_SOURCES
    src/cuda/fused_layernorm.cu
    src/cuda/fused_softmax.cu
    src/cuda/activation_kernels.cu)
endif()

add_library(nexus_model STATIC ${NEXUS_MODEL_SOURCES})
add_library(nexus_model::nexus_model ALIAS nexus_model)

target_include_directories(nexus_model PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(nexus_model PUBLIC cxx_std_20)

if(NEXUS_MODEL_ENABLE_AVX512)
  target_compile_definitions(nexus_model PUBLIC NEXUS_MODEL_ENABLE_AVX512=1)
endif()
if(NEXUS_MODEL_WITH_CUDA)
  target_compile_definitions(nexus_model PUBLIC NEXUS_MODEL_WITH_CUDA=1)
  set_target_properties(nexus_model PROPERTIES CUDA_STANDARD 17)
  target_link_libraries(nexus_model PUBLIC CUDA::cudart)
endif()
if(NEXUS_MODEL_WITH_MATRIXFLASH)
  target_compile_definitions(nexus_model PUBLIC NEXUS_MODEL_WITH_MATRIXFLASH=1)
  set(NEXUS_MATRIXFLASH_INCLUDE "" CACHE PATH "Path to MatrixFlash-Pro include/")
  if(NEXUS_MATRIXFLASH_INCLUDE)
    target_include_directories(nexus_model PUBLIC "${NEXUS_MATRIXFLASH_INCLUDE}")
  else()
    target_include_directories(nexus_model PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/../matrixflash-pro/include")
  endif()
endif()

if(MSVC)
  target_compile_options(nexus_model PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /EHsc /utf-8>)
  if(NEXUS_MODEL_WARNINGS_AS_ERRORS)
    target_compile_options(nexus_model PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/WX>)
  endif()
  set_source_files_properties(src/simd/avx2_kernels.cpp src/simd/avx2/linear.cpp src/simd/avx2/normalization.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  if(NEXUS_MODEL_ENABLE_AVX512)
    set_source_files_properties(src/simd/avx512_kernels.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
  endif()
  target_compile_options(nexus_model PRIVATE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2>)
else()
  target_compile_options(nexus_model PRIVATE -Wall -Wextra -Wpedantic)
  if(NEXUS_MODEL_WARNINGS_AS_ERRORS)
    target_compile_options(nexus_model PRIVATE -Werror)
  endif()
  set_source_files_properties(src/simd/avx2_kernels.cpp src/simd/avx2/linear.cpp src/simd/avx2/normalization.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  if(NEXUS_MODEL_ENABLE_AVX512)
    set_source_files_properties(src/simd/avx512_kernels.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mfma")
  endif()
  target_compile_options(nexus_model PRIVATE $<$<CONFIG:Release>:-O3>)
endif()

function(nexus_model_test_flags target)
  if(MSVC)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /EHsc /utf-8>)
    if(NEXUS_MODEL_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/WX>)
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    if(NEXUS_MODEL_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()

include(CTest)

if(NEXUS_MODEL_BUILD_TESTS)
  add_executable(nexus_model_kernel_tests tests/test_kernels.cpp)
  target_link_libraries(nexus_model_kernel_tests PRIVATE nexus_model)
  nexus_model_test_flags(nexus_model_kernel_tests)
  add_test(NAME nexus_model_kernel_tests COMMAND nexus_model_kernel_tests)
  add_executable(nexus_model_tests tests/test_model.cpp)
  target_link_libraries(nexus_model_tests PRIVATE nexus_model)
  nexus_model_test_flags(nexus_model_tests)
  add_test(NAME nexus_model_tests COMMAND nexus_model_tests)
endif()

if(NEXUS_MODEL_BUILD_BENCHMARKS)
  add_executable(nexus_model_bench_kernels benchmarks/bench_kernels.cpp)
  target_link_libraries(nexus_model_bench_kernels PRIVATE nexus_model)
  nexus_model_test_flags(nexus_model_bench_kernels)
  add_executable(nexus_model_bench benchmarks/bench_suite.cpp)
  target_link_libraries(nexus_model_bench PRIVATE nexus_model)
  nexus_model_test_flags(nexus_model_bench)
  if(NEXUS_MODEL_WITH_CUDA)
    add_executable(nexus_model_bench_gpu benchmarks/bench_gpu.cu)
    target_link_libraries(nexus_model_bench_gpu PRIVATE nexus_model CUDA::cublas)
    set_target_properties(nexus_model_bench_gpu PROPERTIES CUDA_STANDARD 17)
    nexus_model_test_flags(nexus_model_bench_gpu)
  endif()
endif()

if(NEXUS_MODEL_BUILD_EXAMPLES)
  set(NEXUSLOSS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(NEXUSLOSS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(NEXUS_OPTIM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(NEXUS_OPTIM_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(NEXUS_OPTIM_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(NEXUS_OPTIM_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../NexusLoss ${CMAKE_BINARY_DIR}/_deps/nexusloss)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../NexusOptim ${CMAKE_BINARY_DIR}/_deps/nexusoptim)
  set(NEXUSDATA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(NEXUSDATA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(NEXUSDATA_BUILD_BENCH OFF CACHE BOOL "" FORCE)
  set(NEXUSDATA_BUILD_FUZZ OFF CACHE BOOL "" FORCE)
  set(NEXUSDATA_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
  set(NEXUSDATA_FETCH_DEPS OFF CACHE BOOL "" FORCE)
  set(NEXUSDATA_WITH_WEBP OFF CACHE BOOL "" FORCE)
  set(NEXUSDATA_WITH_ZLIB OFF CACHE BOOL "" FORCE)
  set(NEXUSDATA_WITH_ZSTD OFF CACHE BOOL "" FORCE)
  set(NEXUSDATA_WITH_LZ4 OFF CACHE BOOL "" FORCE)
  set(NEXUSDATA_WITH_BZIP2 OFF CACHE BOOL "" FORCE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../NexusData ${CMAKE_BINARY_DIR}/_deps/nexusdata)
  add_executable(train_end_to_end examples/train_end_to_end_example.cpp)
  target_link_libraries(train_end_to_end PRIVATE nexus_model nexusloss nexus_optim nexusdata)
  nexus_model_test_flags(train_end_to_end)
endif()

install(TARGETS nexus_model ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
