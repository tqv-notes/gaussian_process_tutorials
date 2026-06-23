cmake_minimum_required(VERSION 3.15)
project(gaussianprocess CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure with: -DCMAKE_TOOLCHAIN_FILE=C:/Users/quangvu197/vcpkg/scripts/buildsystems/vcpkg.cmake
find_package(Armadillo CONFIG REQUIRED)

add_executable(gp_demo   gp.cpp)
add_executable(spgp_demo spgp.cpp)

foreach(t
    gp_demo
    spgp_demo)
  target_link_libraries(${t} PRIVATE armadillo)
endforeach()