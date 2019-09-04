#include <pangolin/pangolin.h>
#include <GL/glew.h>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds

GLuint shader_programme;
GLuint vs;
GLuint fs;

void loadShader() {
	const char* vertex_shader =
		"#version 400\n"
		"in vec3 vp;"
		"void main() {"
		"  gl_Position = vec4(vp, 1.0);"
		"}";

	const char* fragment_shader =
		"#version 400\n"
		"out vec4 frag_colour;"
		"void main() {"
		"  frag_colour = vec4(1.0f, 0.5f, 0.2f, 1.0f);"
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
	glLinkProgram(shader_programme);
}

int main() 
{
	pangolin::CreateWindowAndBind("LearnOpenGL: Map Viewer", 1024, 800);

	// 3D Mouse handler requires depth testing to be enabled  
	glEnable(GL_DEPTH_TEST);

	// Issue specific OpenGl we might need
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// Define Camera Render Object (for view / scene browsing)
	pangolin::OpenGlRenderState s_cam(
		pangolin::ProjectionMatrix(1024, 800, 800, 800, 512, 400, 0.1, 1000),
		pangolin::ModelViewLookAt(50, 50, 50, 0, 0, 0, 0.0, 1.0, 0.0)
	);

	// Add named OpenGL viewport to window and provide 3D Handler
	pangolin::View& d_cam = pangolin::Display("cam")
		.SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 800.0f)
		.SetHandler(new pangolin::Handler3D(s_cam));

	//menu
	pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
	pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
	pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
	pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
	pangolin::Var<bool> menuShowGraph("menu.Show Graph", true, true);
	pangolin::Var<bool> menuLocalizationMode("menu.Localization Mode", false, true);
	pangolin::Var<bool> menuReset("menu.Reset", false, false);


	float points[] = {
			0.0f,  5.f,  0.0f,
			5.f, -5.f,  0.0f,
			-5.f, -5.f,  0.0f
	};
	GLuint vbo = 0;
	glGenBuffers(1, &vbo);
	GLuint vao = 0;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);	// Vertex attributes stay the same
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	


	
	
	//glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);	// Vertex attributes stay the same
	//glEnableVertexAttribArray(0);
	//glBindVertexArray(0);
	//glDeleteBuffers(1, &vbo);
	//glDisableVertexAttribArray(vao);
	loadShader();
	//end

	while (!pangolin::ShouldQuit())
	{
		
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		d_cam.Activate(s_cam);

		//rendering commands ...
		
		glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		pangolin::glDrawAxis(100);
		//pangolin::glDrawColouredCube();
		pangolin::glDraw_y0(5, 20);
		//a triangluate
		// now when we draw the triangle we first use the vertex and orange fragment shader from the first program
		//glUseProgram(shader_programme);
		// draw the first triangle using the data from VAO
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, 3);	// this call should output a triangle
		glBindVertexArray(0);
		glUseProgram(0);

		if (menuReset) {
			s_cam.SetModelViewMatrix(
				pangolin::ModelViewLookAt(50, 50, 50, 0, 0, 0, 0.0, 1.0, 0.0)
			);
			menuReset = false;
		}
		pangolin::FinishFrame();
		std::this_thread::sleep_for(std::chrono::microseconds(10));
	}
}