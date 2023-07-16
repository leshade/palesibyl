cmake_minimum_required (VERSION 3.8)

project(PalesibylLib LANGUAGES CUDA CXX)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../../Palesibyl/library)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../../Palesibyl/library)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	../../Palesibyl/include
)

file(GLOB LIB_CPP_SOURCES "../../Palesibyl/source/*.cpp")
file(GLOB LIB_CUDA_SOURCES "../../Palesibyl/source/*.cu")

add_library(PalesibylLib STATIC
	${LIB_CPP_SOURCES}
	${LIB_CUDA_SOURCES}
)

target_compile_features(PalesibylLib PRIVATE cxx_std_17 cuda_std_14)

target_link_libraries(PalesibylLib
    ${OpenCV_LIBRARIES}
#	cudart
)
