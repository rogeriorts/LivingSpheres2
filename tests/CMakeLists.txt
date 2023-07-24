# Testing library
FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.10)
FetchContent_MakeAvailable(catch2)
# Adds Catch2::Catch2

include_directories(${CMAKE_SOURCE_DIR}/include/LivingSpheres)

# Tests need to be added as executables first
add_executable(
  test_contact_forces 
  test_contact_forces.cpp
  ../include/LivingSpheres/simulator.cu
  )

  add_executable(
      test_grid
      test_grid.cpp
      ../include/LivingSpheres/simulator.cu
      )

  add_executable(
    test_move
    test_move.cpp
        ../include/LivingSpheres/simulator.cu
        ../include/LivingSpheres/simulator.h
        )
message(STATUS "Looking for X11...")
find_package(X11 REQUIRED)
include_directories(${X11_INCLUDE_DIR})

# I'm using C++17 in the test
#target_compile_features(testlib PRIVATE cxx_std_17)

# Should be linked to the main library, as well as the Catch2 testing library



target_link_libraries(test_contact_forces PRIVATE  Catch2::Catch2 X11 CUDA::cudart CUDA::cuda_driver)
target_link_libraries(test_grid PRIVATE  Catch2::Catch2 X11 CUDA::cudart CUDA::cuda_driver) 
target_link_libraries(test_move PRIVATE  Catch2::Catch2 X11 CUDA::cudart CUDA::cuda_driver) 

# If you register a test, then ctest and make test will run it. You can also run
# examples and check the output, as well.
add_test(NAME test_contact_forces COMMAND $<TARGET_FILE:test_contact_forces>) # Command can be a target
add_test(NAME test_grid COMMAND $<TARGET_FILE:test_contact_forces>) # Command can be a target
add_test(NAME test_move COMMAND $<TARGET_FILE:test_contact_forces>) # Command can be a target

