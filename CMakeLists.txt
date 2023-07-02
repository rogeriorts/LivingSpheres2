cmake_minimum_required(VERSION 3.17)


project(LivingSpheres LANGUAGES CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES 80 86)
find_package(CUDAToolkit)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

include(FetchContent)

add_subdirectory(apps)

enable_testing()

add_subdirectory(tests)

