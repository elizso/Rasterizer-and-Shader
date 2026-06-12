cmake_minimum_required(VERSION 3.10)
project(Rasterizer)

# ==========================================
# Eigen3 Include Directories for various OS
# ==========================================
include_directories(/usr/include/eigen3)          # Standard Linux (Ubuntu, etc.)
include_directories(/usr/local/include/eigen3)    # Intel Mac
include_directories(/opt/homebrew/include/eigen3) # Apple Silicon Mac (M1/M2/M3)

# For opencv on Mac (Silicon)
set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

# include_directories(/usr/local/include ./include)
add_executable(task1 task1.cpp rasterizer.hpp rasterizer.cpp global.hpp mesh.cpp Triangle.hpp Triangle.cpp)
target_link_libraries(task1 ${OpenCV_LIBRARIES})

add_executable(task2 task2.cpp rasterizer.hpp rasterizer.cpp global.hpp Triangle.hpp Triangle.cpp Texture.hpp Texture.cpp Shader.hpp OBJ_Loader.h
        task2.h)
target_link_libraries(task2 ${OpenCV_LIBRARIES})

add_executable(task3 task3.cpp rasterizer.hpp rasterizer.cpp global.hpp Triangle.hpp Triangle.cpp Texture.hpp Texture.cpp Shader.hpp OBJ_Loader.h
        task3_utils.cpp
        task3_utils.h)
target_link_libraries(task3 ${OpenCV_LIBRARIES})

add_executable(task4 task4.cpp rasterizer.hpp rasterizer.cpp global.hpp Triangle.hpp Triangle.cpp Texture.hpp Texture.cpp Shader.hpp OBJ_Loader.h
        task3_utils.cpp task3_utils.h)
target_link_libraries(task4 ${OpenCV_LIBRARIES})

add_executable(conversion conversion.cpp)