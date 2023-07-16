cmake_minimum_required (VERSION 3.8)

project(sample_unet LANGUAGES CUDA CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../samples/sample_unet)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	../common/include
	../../Palesibyl/include
)

add_executable(sample_unet
	sample_unet.cpp
	../common/source/sample_basic_app.cpp
	../common/source/sample_app_stub.cpp
)

target_link_libraries(sample_unet
	PalesibylLib
)

