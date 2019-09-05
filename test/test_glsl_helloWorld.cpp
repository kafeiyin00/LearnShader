#include <iostream>
#include <GL/glew.h>

#include <glfw/glfw3.h>
#include "VSShaderlib.h"
#include "Config.h"

#include "./test_glsl_helloWorld.h"

GLuint vao;
VSShaderLib shader;

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

GLuint setupShaders() {

	
	// Shader for models
	shader.init();
	shader.loadShader(VSShaderLib::VERTEX_SHADER, SHADER_FOLDER_PATH"helloWorld.vert");
	shader.loadShader(VSShaderLib::FRAGMENT_SHADER, SHADER_FOLDER_PATH"helloWorld.frag");

	// set semantics for the shader variables
	shader.setProgramOutput(0, "outputF");
	shader.setVertexAttribName(VSShaderLib::VERTEX_COORD_ATTRIB, "position");

	shader.prepareProgram();
	printf("InfoLog for Hello World Shader\n%s\n\n", shader.getAllInfoLogs().c_str());

	return(shader.isProgramValid());
}

void inititializeGLEW() {
	GLenum err = glewInit();
	if (err != GLEW_OK) {
		std::cerr << glewGetErrorString(err) << std::endl;
		throw std::runtime_error("Failed to initialize GLEW.");
	}
	std::cout << "GLEW initialized." << std::endl;
		

	int32_t version[2];
	glGetIntegerv(GL_MAJOR_VERSION, &version[0]);
	glGetIntegerv(GL_MINOR_VERSION, &version[1]);
	std::cout << "OpenGL context version: " << version[0] << "." << version[1] << std::endl;
	std::cout << "OpenGL vendor string  : " << glGetString(GL_VENDOR) << std::endl;
	std::cout << "OpenGL renderer string: " << glGetString(GL_RENDERER) << std::endl;

	// consume here any OpenGL error and reset to NO_GL_ERROR:
	glGetError();
	
}

void initOpenGL()
{
	// set the camera position based on its spherical coordinates
	/*camX = r * sin(alpha * 3.14f / 180.0f) * cos(beta * 3.14f / 180.0f);
	camZ = r * cos(alpha * 3.14f / 180.0f) * cos(beta * 3.14f / 180.0f);
	camY = r * sin(beta * 3.14f / 180.0f);*/

	// some GL settings
	//glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_MULTISAMPLE);
	//glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

	// create the VAO
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	// create buffers for our vertex data
	GLuint buffers[4];
	glGenBuffers(4, buffers);

	//vertex coordinates buffer
	glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(VSShaderLib::VERTEX_COORD_ATTRIB);
	glVertexAttribPointer(VSShaderLib::VERTEX_COORD_ATTRIB, 4, GL_FLOAT, 0, 0, 0);

	//texture coordinates buffer
	glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texCoords), texCoords, GL_STATIC_DRAW);
	glEnableVertexAttribArray(VSShaderLib::TEXTURE_COORD_ATTRIB);
	glVertexAttribPointer(VSShaderLib::TEXTURE_COORD_ATTRIB, 2, GL_FLOAT, 0, 0, 0);

	//normals buffer
	glBindBuffer(GL_ARRAY_BUFFER, buffers[2]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(normals), normals, GL_STATIC_DRAW);
	glEnableVertexAttribArray(VSShaderLib::NORMAL_ATTRIB);
	glVertexAttribPointer(VSShaderLib::NORMAL_ATTRIB, 3, GL_FLOAT, 0, 0, 0);

	//index buffer
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[3]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(faceIndex), faceIndex, GL_STATIC_DRAW);

	// unbind the VAO
	glBindVertexArray(0);
}


int main() {
	glfwInit();
	GLFWwindow* window = glfwCreateWindow(1024, 768, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	inititializeGLEW();

	setupShaders();
	int pProgram = shader.getProgramIndex();
		
	glUseProgram(pProgram);
	initOpenGL();

	GLuint bindingPoint = 1, buffer, blockIndex;
	float mvp[16] = {
		1, 0.0, 0.0, 0.0,
		0.0, 1, 0.0, 0.0,
		0.0, 0.0, 1, 0.0,
		0.0, 0.0, 0.0, 1.0
	};

	blockIndex = glGetUniformBlockIndex(pProgram, "Matrices");
	glUniformBlockBinding(pProgram, blockIndex, bindingPoint);

	glGenBuffers(1, &buffer);
	glBindBuffer(GL_UNIFORM_BUFFER, buffer);

	glBufferData(GL_UNIFORM_BUFFER, sizeof(mvp), mvp, GL_DYNAMIC_DRAW);
	glBindBufferBase(GL_UNIFORM_BUFFER, bindingPoint, buffer);

	float points[] = {
			-0.9f, -0.5f, 0.0f,  // left 
		-0.0f, -0.5f, 0.0f,  // right
		-0.45f, 0.5f, 0.0f,  // top
	};

	GLuint vbo = 0;
	glGenBuffers(1, &vbo);
	GLuint vao1 = 0;
	glGenVertexArrays(1, &vao1);
	glBindVertexArray(vao1);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);	// Vertex attributes stay the same
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);

	while (!glfwWindowShouldClose(window)) {
		// input
		//processInput(window);
		
		glBindBuffer(GL_UNIFORM_BUFFER, buffer);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(mvp), mvp);

		//rendering commands ...
		glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		// render VAO
		
		glBindVertexArray(vao);
		//glDrawArrays(GL_TRIANGLES, 0, 3);
		glDrawElements(GL_TRIANGLES, faceCount*3, GL_UNSIGNED_INT, 0);
		//glUseProgram(0);
		glBindVertexArray(0);
		glUseProgram(pProgram);
		glBindVertexArray(vao1);
		glDrawArrays(GL_TRIANGLES, 0, 3);
		glBindVertexArray(0);
		//glViewport(0, 0, 1024, 768);
		// check and call events and swap the buffers
		glfwPollEvents();
		glfwSwapBuffers(window);
	}

	return 1;

}