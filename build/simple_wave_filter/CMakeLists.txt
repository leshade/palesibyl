cmake_minimum_required (VERSION 3.8)

project(simple_wave_filter LANGUAGES CUDA CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../samples/simple_wave_filter)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	../common/include
	../../Palesibyl/include
)

add_executable(simple_wave_filter
	simple_wave_filter.cpp
	../common/source/sample_basic_app.cpp
	../common/source/sample_app_stub.cpp
)

target_link_libraries(simple_wave_filter
	PalesibylLib
)

