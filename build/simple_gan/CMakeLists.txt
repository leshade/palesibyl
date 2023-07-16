cmake_minimum_required (VERSION 3.8)

project(simple_gan LANGUAGES CUDA CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../samples/simple_gan)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	../common/include
	../../Palesibyl/include
)

add_executable(simple_gan
	simple_gan.cpp
	../common/source/sample_basic_app.cpp
	../common/source/sample_app_gan_stub.cpp
)

target_link_libraries(simple_gan
	PalesibylLib
)

