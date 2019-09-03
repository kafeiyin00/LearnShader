cmake_minimum_required(VERSION 3.0)

project(LearnShader)

find_package(glfw3 REQUIRED)
find_package(glad CONFIG REQUIRED)

find_package(Pangolin CONFIG REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})


#-- test1 raw glsw and glad
add_executable(test1 test/test1.cpp)
target_link_libraries( test1 
	PRIVATE
	glfw
	glad::glad)

#-- test2 Pangolin some bugs with shader
add_executable(test2 test/test2.cpp)
target_link_libraries( test2 
	PRIVATE
	pangolin)