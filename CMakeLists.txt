

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

PROJECT(ncnn_nvm_main)
set(Target ncnn_nvm_main)

cmake_minimum_required(VERSION 3.5)


if(TARGET_COMPILER STREQUAL "arm")
    message(STATUS "TARGET_COMPILE STREQUAL arm")
    SET(CMAKE_CXX_COMPILER arm-none-linux-gnueabihf-g++)
    SET(CMAKE_C_COMPILER arm-none-linux-gnueabihf-gcc)
    SET(NCNN_INSTALL_DIR /opt/external/ncnn/install/arm/)
    SET(OpenCV_DIR /opt/external/opencv/install/arm/lib/cmake/opencv4/)
else()
    message(STATUS "TARGET_COMPILE STREQUAL X64")
    SET(NCNN_INSTALL_DIR /opt/external/ncnn/install/x64/)
    SET(OpenCV_DIR /opt/external/opencv/install/x64/lib/cmake/opencv4/)
endif()


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)



find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} )
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${NCNN_INSTALL_DIR}/include/ncnn)

add_executable(${Target} ${Target}.cpp)
link_directories(${OpenCV_LIBRARY_DIRS})

target_link_libraries(${Target} ${OpenCV_LIBRARIES})
target_link_libraries (${Target} ${NCNN_INSTALL_DIR}/lib/libncnn.a -fopenmp)





