cmake_minimum_required(VERSION 3.0)

project(LearnShader)

find_package(glfw3 REQUIRED)
find_package(glad CONFIG REQUIRED)

add_executable(test1 test/test1.cpp)
target_link_libraries( test1 
	PRIVATE
	glfw
	glad::glad)