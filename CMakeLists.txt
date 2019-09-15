cmake_minimum_required(VERSION 3.0)

project(LearnShader)

find_package(glfw3 REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(eigen3 REQUIRED)

find_package( OpenCV REQUIRED )

FIND_PACKAGE(CUDA REQUIRED)
message("cuda include : ${CUDA_INCLUDE_DIRS} : ${CUDA_LIBRARIES}")
include_directories(${CUDA_INCLUDE_DIRS})
cuda_include_directories(${EIGEN_INCLUDE_DIRS})
include_directories("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/include")

include_directories("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.0\\include")

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

#-- test5 texture test
#add_executable(test5 test/test_texture.cpp)
#target_link_libraries( test5 
#	glfw
#	${OPENGL_LIBRARIES}
#	${GLEW_LIBRARIES}
#	libShader
#	${OpenCV_LIBS}
#)

#-- test6 cuda test
cuda_add_executable(test6 test/test_cuda.cu)
target_link_libraries( test6 
	${CUDA_LIBRARIES}
)

#-- test7 projection test
cuda_add_library( libcuprojection test/projection.cu)
target_link_libraries( 
libcuprojection 
${CUDA_LIBRARIES}
)

add_executable(test7 test/test_projection.cpp)
target_link_libraries( test7
		libcuprojection
		${OpenCV_LIBS}
		${CUDA_LIBRARIES}
)

#-- test8 normal estimation test
cuda_add_library( libcufuncs SHARED src/cudaFunction.cu)
target_link_libraries(
		libcufuncs
		${CUDA_LIBRARIES}
)

add_executable(test8 test/test_cudaEstimation.cpp)
target_link_libraries(
		test8
		libcufuncs
		${CUDA_LIBRARIES}
		${OpenCV_LIBS}
)

#-- test9 aerial laser scanning to depthmap
add_executable(test9 test/test_arialFrames.cpp)
target_link_libraries(
		test9
		libcufuncs
		${CUDA_LIBRARIES}
		${OpenCV_LIBS}
)
