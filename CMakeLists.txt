cmake_minimum_required(VERSION 3.14)
project(ggml-loader LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(GGML_LOADER_VULKAN "Build ggml with the Vulkan backend (baseline)" ON)
option(GGML_LOADER_OPENCL "Build ggml with the OpenCL backend" OFF)

set(GGML_VULKAN ${GGML_LOADER_VULKAN} CACHE BOOL "" FORCE)
set(GGML_OPENCL ${GGML_LOADER_OPENCL} CACHE BOOL "" FORCE)

add_subdirectory(third_party/ggml)

add_executable(ggml-loader
    src/main.cpp
    src/model.cpp
    src/cmd_inspect.cpp
    src/cmd_run.cpp
    src/cmd_chat.cpp
    src/cmd_bench.cpp
)
target_link_libraries(ggml-loader PRIVATE ggml)
target_include_directories(ggml-loader PRIVATE src)
