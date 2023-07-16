cmake_minimum_required (VERSION 3.8)

project(simple_color_filter LANGUAGES CUDA CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../../samples/simple_color_filter)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	../common/include
	../../Palesibyl/include
)

add_executable(simple_color_filter
	simple_color_filter.cpp
	../common/source/sample_basic_app.cpp
	../common/source/sample_app_stub.cpp
)

target_link_libraries(simple_color_filter
	PalesibylLib
)

