#include <iostream>
#include <GL/glew.h>

#include <glfw/glfw3.h>

#include <opencv2/opencv.hpp>

GLuint shader_programme;
GLuint vs;
GLuint fs;

void loadShader() {
	const char* vertex_shader =
		"#version 400\n"
		"layout(location = 0) in vec3 vp;"
		"void main() {"
		"  gl_Position = vec4(vp.x , vp.y *0.3, vp.z * 0.8, 1.0);"
		"}";

	const char* fragment_shader =
		"#version 400\n"
		"layout(location = 0) out vec4 frag_colour;"
		" out vec4 colorOut;"
		"void main() {"
		"  frag_colour = vec4(1.0f, 0.5f, 0.2f, 1.0f);"
		"  colorOut = vec4(1.0f, 0.0f, 0.0f, 1.0f);"
		"}";

	vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &vertex_shader, NULL);
	glCompileShader(vs);
	fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &fragment_shader, NULL);
	glCompileShader(fs);

	shader_programme = glCreateProgram();
	glAttachShader(shader_programme, fs);
	glAttachShader(shader_programme, vs);
	//glBindFragDataLocation(shader_programme, 0, "colorOut");
	glLinkProgram(shader_programme);

	//打印编译信息，如果编译错误，就可以看见错误信息了  
	auto result = GL_FALSE;
	auto info_length = 0;
	glGetProgramiv(shader_programme, GL_LINK_STATUS, &result);
	if (result == GL_FALSE) {
		glGetProgramiv(shader_programme, GL_INFO_LOG_LENGTH, &info_length);
		std::string program_log((unsigned long)info_length, ' ');
		glGetProgramInfoLog(shader_programme, info_length, NULL, &program_log[0]);
		std::cout << program_log << std::endl;
	}
}

static bool glewInitiatalized = false;

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}

inline void inititializeGLEW() {
	if (!glewInitiatalized) {
		//glewExperimental = GL_TRUE;
		GLenum err = glewInit();
		if (err != GLEW_OK) {
			std::cerr << glewGetErrorString(err) << std::endl;
			throw std::runtime_error("Failed to initialize GLEW.");
		}
		std::cout << "GLEW initialized." << std::endl;
		glewInitiatalized = true;

		int32_t version[2];
		glGetIntegerv(GL_MAJOR_VERSION, &version[0]);
		glGetIntegerv(GL_MINOR_VERSION, &version[1]);
		std::cout << "OpenGL context version: " << version[0] << "." << version[1] << std::endl;
		std::cout << "OpenGL vendor string  : " << glGetString(GL_VENDOR) << std::endl;
		std::cout << "OpenGL renderer string: " << glGetString(GL_RENDERER) << std::endl;

		// consume here any OpenGL error and reset to NO_GL_ERROR:
		glGetError();
	}
}

void snap_shot(int img_w, int img_h)
{
	GLubyte* pPixelData;
	GLint line_width;
	GLint PixelDataLength;

	line_width = img_w * 3; // 得到每一行的像素数据长度 

	PixelDataLength = line_width * img_h;

	// 分配内存和打开文件 
	//pPixelData = (GLubyte*)malloc(PixelDataLength);
	pPixelData = new GLubyte[PixelDataLength];
	if (pPixelData == 0)
		exit(0);


	// 读取像素 
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);

	//glReadPixels(0, 0, img_w, img_h, GL_BGR_EXT, GL_UNSIGNED_BYTE, pPixelData);
	glReadPixels(0, 0, img_w, img_h, GL_RGB, GL_UNSIGNED_BYTE, pPixelData);

	cv::Mat render_result( img_h, img_w, CV_8UC3);

	//different define of the rgb
	//memcpy((render_result.data), pPixelData, line_width * img_h);
	for (size_t i = 0; i < img_h; i++)
	{
		size_t n_line = img_h - i;
		for (size_t j = 0; j < img_w; j++)
		{
			char r = pPixelData[n_line*line_width + j * 3 + 2];
			char g = pPixelData[n_line*line_width + j * 3 + 1];
			char b = pPixelData[n_line*line_width + j * 3 + 0];

			render_result.at<cv::Vec3b>(i, j) = cv::Vec3b(r,g,b);
		}
		
	}

	cv::namedWindow("render_result");

	cv::imshow("render_result",render_result);
}

int main()
{
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
	GLuint FramebufferName;
	glGenFramebuffers(1, &FramebufferName);
	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);

	// The texture we're going to render to
	GLuint renderedTexture;
	glGenTextures(1, &renderedTexture);

	// "Bind" the newly created texture : all future texture functions will modify this texture
	glBindTexture(GL_TEXTURE_2D, renderedTexture);

	// Give an empty image to OpenGL ( the last "0" )
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1024, 768, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);

	// Poor filtering. Needed !
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	// The depth buffer
	GLuint depthrenderbuffer;
	glGenRenderbuffers(1, &depthrenderbuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 1024, 768);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

	// Set "renderedTexture" as our colour attachement #0
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);

	// Set the list of draw buffers.
	GLenum DrawBuffers[1] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers

	// Always check that our framebuffer is ok
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
		return false;

	// Render to our framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);

	float points[] = {
			-0.9f, -0.5f, 0.0f,  // left 
		-0.0f, -0.5f, 0.0f,  // right
		-0.45f, 0.5f, 0.0f,  // top
	};

	loadShader();
	GLuint vbo = 0;
	glGenBuffers(1, &vbo);
	GLuint vao = 0;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);	// Vertex attributes stay the same
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);
	
	glBindVertexArray(vao);
	glUseProgram(shader_programme);
	glDrawArrays(GL_TRIANGLES, 0, 3);	// this call should output a triangle
	glUseProgram(0);
	glBindVertexArray(0);

	//glViewport(0, 0, 1024, 768); // Render on the whole framebuffer, complete from the lower left corner to the upper right
	snap_shot(1024, 768);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);


	//glEnable(GL_DEPTH_TEST);
	while (!glfwWindowShouldClose(window)) {
		// input
		//processInput(window);
	
		//rendering commands ...
		glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		glBindVertexArray(vao);
		glUseProgram(shader_programme);
		
		glDrawArrays(GL_TRIANGLES, 0, 3);	// this call should output a triangle
		glUseProgram(0);
		glBindVertexArray(0);
		//glViewport(0, 0, 1024, 768);
		// check and call events and swap the buffers
		glfwPollEvents();
		glfwSwapBuffers(window);
	}

	
}