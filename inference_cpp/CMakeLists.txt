cmake_minimum_required(VERSION 3.10)
project(interference_cpp)
add_compile_options(-Wall -Wextra)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_library(DARKHELP darkhelp)
find_library(DARKNET darknet)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(binary_executable src/main.cpp src/camera.cpp)

target_include_directories(binary_executable PRIVATE headers/*)
target_link_libraries(binary_executable Threads::Threads ${OpenCV_LIBS} ${DARKHELP} ${DARKNET})
