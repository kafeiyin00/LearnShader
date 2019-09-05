cmake_minimum_required(VERSION 3.0)

project(LearnShader)

find_package(glfw3 REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)

find_package( OpenCV REQUIRED )

include_directories("D:\\vcpkg\\installed\\x64-windows\\include")
message("glew include : ${GLEW_LIBRARIES}")

find_package(Pangolin CONFIG REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

set(SHADER_FOLDER_PATH "\"${PROJECT_SOURCE_DIR}/shaders/\"")

configure_file (
  "${PROJECT_SOURCE_DIR}/Config.h.in"
  "${PROJECT_SOURCE_DIR}/Config.h"
)


include_directories(".")
include_directories(src)

add_library(libShader 
	src/vsShaderLib.cpp
)


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


#-- test3 render to texture
add_executable(test3 test/test3.cpp)
target_link_libraries( test3 
	glfw
	${OPENGL_LIBRARIES}
	${GLEW_LIBRARIES}
	${OpenCV_LIBS}
)

#-- test4 glsl test
add_executable(test4 test/test_glsl_helloWorld.cpp)
target_link_libraries( test4 
	glfw
	${OPENGL_LIBRARIES}
	${GLEW_LIBRARIES}
	libShader
)
