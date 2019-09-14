#include <pangolin/pangolin.h>
#include <GL/glew.h>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "Config.h"

struct LaserPoint {
    double x;
    double y;
    double z;
    int intense;
    int hour;
    unsigned int ms_top_of_hour;

};


struct LaserFrame {
    std::vector<LaserPoint> framePoints;
    Eigen::Quaterniond q_nl;
    Eigen::Vector3d r_nl;
    void operator=(const LaserFrame& lf)
    {
        q_nl = lf.q_nl;
        r_nl = lf.r_nl;
        framePoints.resize(lf.framePoints.size());
        for (int i = 0; i < framePoints.size(); i++) {
            framePoints[i] = lf.framePoints[i];
        }
    }
};

void loadLaserFrames(std::vector<LaserFrame>& laserFrames, std::string filename)
{
    FILE* file = fopen(filename.c_str(), "rb");
    size_t frameSize;
    fread((char*)&frameSize, sizeof(size_t), 1, file);
    laserFrames.resize(frameSize);
    for (int i = 0; i < frameSize; i++) {
        LaserFrame& tmpFrame = laserFrames[i];
        fread((char*)&(tmpFrame.q_nl), sizeof(Eigen::Quaterniond), 1, file);
        fread((char*)&(tmpFrame.r_nl), sizeof(Eigen::Vector3d), 1, file);
        size_t pointSize;
        fread((char*)&(pointSize), sizeof(size_t), 1, file);
        tmpFrame.framePoints.resize(pointSize);
        for (int j = 0; j < pointSize; j++) {
            fread((char*)&(tmpFrame.framePoints[j]), sizeof(LaserPoint), 1, file);
        }
    }
    fclose(file);
}

int main() 
{
    std::vector<LaserFrame> laserFrames;
    std::string filename = SHADER_FOLDER_PATH"laserFrames.dat";
    loadLaserFrames(laserFrames,filename);

    std::cout<<"we loaded laser frames: "<<laserFrames.size();

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


	float tempFrame = 0;
	while (!pangolin::ShouldQuit())
	{
		
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		d_cam.Activate(s_cam);

		//rendering commands ...
		
		glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		pangolin::glDrawAxis(100);
		//pangolin::glDrawColouredCube();
		pangolin::glDraw_x0(5, 20);


        glPointSize(3.0f);
        glBegin(GL_POINTS);


        for (int i = 0; i < laserFrames[tempFrame].framePoints.size(); ++i) {
            LaserPoint& pt = laserFrames[tempFrame].framePoints[i];
            glColor3f(5*pt.intense/255.0,5*pt.intense/255.0,5*pt.intense/255.0);
            glVertex3f(pt.x,pt.y,pt.z);
        }
        tempFrame = tempFrame+0.2;

        if(tempFrame >= laserFrames.size()){
            tempFrame = 0;
        }

        glEnd();


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