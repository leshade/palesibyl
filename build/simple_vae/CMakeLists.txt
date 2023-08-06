cmake_minimum_required (VERSION 3.8)

project(simple_vae LANGUAGES CUDA CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../samples/simple_vae)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	../common/include
	../../Palesibyl/include
)

add_executable(simple_vae
	simple_vae.cpp
	../common/source/sample_basic_app.cpp
)

target_link_libraries(simple_vae
	PalesibylLib
)

