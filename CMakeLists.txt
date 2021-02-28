project(thesis-master)
cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

set(SRC src/GibbsSamplerND.cpp
        src/NDSampler.cpp
        src/ImageAsND.cpp
        src/NDImageDrawer.cpp
        src/main.cpp)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV COMPONENTS core highgui imgcodecs imgproc REQUIRED)

add_executable(gibbs_sampler ${SRC})
target_include_directories(gibbs_sampler PRIVATE src ${OpenCV_INCLUDE_DIRS})
target_link_directories(gibbs_sampler PRIVATE ${OpenCV_LIBRARY_DIR})
target_link_libraries(gibbs_sampler pthread tbb Eigen3::Eigen ${OpenCV_LIBRARIES})

