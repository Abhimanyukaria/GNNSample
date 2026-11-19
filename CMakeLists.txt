cmake_minimum_required(VERSION 3.22)
project(gnnsample LANGUAGES CXX)

# Configure with the DPC++ compiler:  cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Ahead-of-time device targets avoid the multi-minute JIT on first run. Default: SPIR-V
# (JIT, portable). Example for the laptop iGPU: -DGNNSAMPLE_SYCL_TARGETS=spir64_gen
# -DGNNSAMPLE_AOT_DEVICE=<arch>; for NVIDIA: nvptx64-nvidia-cuda.
set(GNNSAMPLE_SYCL_TARGETS "spir64" CACHE STRING "SYCL -fsycl-targets")
set(GNNSAMPLE_AOT_DEVICE "" CACHE STRING "Device for spir64_gen AOT (e.g. ptl-h)")
set(SYCL_FLAGS -fsycl -fsycl-targets=${GNNSAMPLE_SYCL_TARGETS})
if(GNNSAMPLE_AOT_DEVICE)
  list(APPEND SYCL_FLAGS "SHELL:-Xsycl-target-backend \"-device ${GNNSAMPLE_AOT_DEVICE}\"")
endif()

add_executable(device_info cpp/tools/device_info.cpp)
target_compile_options(device_info PRIVATE ${SYCL_FLAGS})
target_link_options(device_info PRIVATE ${SYCL_FLAGS})

find_package(Python 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND ${Python_EXECUTABLE} -m nanobind --cmake_dir
                OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_C cpp/src/module.cpp)
target_include_directories(_C PRIVATE cpp/include)
target_compile_options(_C PRIVATE ${SYCL_FLAGS})
target_link_options(_C PRIVATE ${SYCL_FLAGS})
set_target_properties(_C PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/gnnsample)

add_executable(launch_overhead cpp/tools/launch_overhead.cpp)
target_compile_options(launch_overhead PRIVATE ${SYCL_FLAGS})
target_link_options(launch_overhead PRIVATE ${SYCL_FLAGS})
