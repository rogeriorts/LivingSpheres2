cmake_minimum_required(VERSION 3.17)

set(CMAKE_CUDA_ARCHITECTURES 80 86)
project(LivingSpheres LANGUAGES CUDA CXX)

find_package(CUDAToolkit)
find_package(X11 REQUIRED)
find_package(OpenMP)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)

if(OPENMP_FOUND)
 set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
 set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}") 
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
endif()
foreach(flag IN LISTS cxx_warning_flags)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--compiler-options ${flag}>")
endforeach()

include(FetchContent)

add_subdirectory(apps)

enable_testing()

add_subdirectory(tests)


