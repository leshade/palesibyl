cmake_minimum_required (VERSION 3.8)

project(Palesibyl LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(Palesibyl)

add_subdirectory(simple_color_filter)
add_subdirectory(simple_upsampler)
add_subdirectory(simple_classifier)
add_subdirectory(simple_gan)
add_subdirectory(simple_vae)
add_subdirectory(simple_wave_filter)
add_subdirectory(sample_resnet)
add_subdirectory(sample_unet)

