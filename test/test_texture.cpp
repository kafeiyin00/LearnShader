#include <iostream>
#include <GL/glew.h>
#include <glfw/glfw3.h>
#include <opencv2/opencv.hpp>
#include <Config.h>

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

GLuint LoadTexture(const char* filename,
	GLboolean generateMips)
{
	cv::Mat textureMat = cv::imread(filename, cv::IMREAD_COLOR);

	cv::namedWindow("texture cv");
	cv::imshow("texture cv",textureMat);
	cv::waitKey(0);

	GLuint texture;
	glCreateTextures(GL_TEXTURE_2D, 1, &texture);
	glTextureStorage2D(texture,
		0,
		GL_RGB8,
		textureMat.cols, textureMat.rows);
	glTextureSubImage2D(texture,
		0,
		0,0,
		textureMat.cols, textureMat.rows,
		GL_RGB8, GL_UNSIGNED_BYTE,
		textureMat.data);

	return texture;
}

int main()
{
	glfwInit();
	GLFWwindow* window = glfwCreateWindow(1, 1, "LearnOpenGL", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	inititializeGLEW();

	GLuint texture = LoadTexture(SHADER_FOLDER_PATH"texture.jpg",0);

	system("pause");
	return 0;
}