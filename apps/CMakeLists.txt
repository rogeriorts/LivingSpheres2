


include_directories(${CMAKE_SOURCE_DIR}/include/LivingSpheres)
add_executable(app 
    app.cpp 
    ../include/LivingSpheres/simulator.cu 
    ../include/LivingSpheres/simulator.h
    ../include/LivingSpheres/user_interface.h

    )

include_directories(${X11_INCLUDE_DIR})

target_link_libraries(app PRIVATE X11 CUDA::cudart CUDA::cuda_driver OpenMP::OpenMP_CXX)
target_compile_options(app PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)

