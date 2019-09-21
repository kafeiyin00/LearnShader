//
// Created by 86384 on 2019/9/17.
//

#include <iostream>
#include "laserFramesDefines.h"
#include "Config.h"
#include "cudaFunction.cuh"

#include <opencv2/opencv.hpp>

#include "3dDebuger.h"

void showFeatureMap(float* featureMap, int width, int height, std::string windowname) {
    cv::Mat mat_depth(height, width, CV_32FC1);

    for (int i = 1; i < height-1; i++)
    {
        for (int j = 1; j < width -1 ; j++)
        {
            float depth = featureMap[i*width+j];
            if(depth > 0){
                cv::circle(mat_depth,cv::Point(j,i),2,cv::Scalar(depth),1);
            }

            //std::cout << depth << std::endl;
        }
    }
    cv::Mat grayImage(height, width, CV_8UC1);
    double alpha = 255.0 / (1 - 0);
    mat_depth.convertTo(grayImage, CV_8UC1, alpha, 0  );
    cv::Mat mat_color(height, width, CV_8UC3);
    cv::applyColorMap(grayImage,mat_color,cv::COLORMAP_JET);

    for (int i =0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if(mat_depth.at<float>(i, j) < 10){
                mat_color.at<cv::Vec3b>(i,j) = cv::Vec3b(100,100,100);
            }
        }
    }

    cv::namedWindow(windowname.c_str());
    cv::imshow(windowname.c_str(),mat_color);
    cv::waitKey(100);
}

int main(){

    getCudaState();

    std::vector<LaserFrame> laserFrames;
    std::string filename = SHADER_FOLDER_PATH"laserFrames.dat";
    loadLaserFrames(laserFrames, filename);
    printf("we load => %d <= frames\n",laserFrames.size());

    std::vector<OrganizedFrame> organnizedframes;
    organnizedframes.resize(laserFrames.size());

    for (int i = 0; i < laserFrames.size(); ++i) {
        frame2orgnizedframe(laserFrames[i], organnizedframes[i]);
    }

    ThreeDdebuger debuger;

    std::vector<Eigen::Vector3i> colors;
    colors.resize(16);
    for (int j = 0; j < 16; j++){
        colors[j] = Eigen::Vector3i(rand()%255,rand()%255,rand()%255);
    }



    // simple view input
//    std::function<void ()> func = [&]()->void{
//        glPointSize(3.0f);
//        glBegin(GL_POINTS);
//
//        for (int j = 0; j < 16; j++){
//            for (int i = 0; i < organnizedframes[1].pointschannels[j].size(); ++i) {
//                LaserPoint& pt = organnizedframes[1].pointschannels[j][i];
//                glColor3f(colors[j][0]/255.0,colors[j][1]/255.0,colors[j][2]/255.0);
//                glVertex3f(pt.x,pt.y,pt.z);
//            }
//        }
//
//        glEnd();
//
//    };
//    debuger.RenderLoop(func);
	for (int j = 0; j < organnizedframes.size(); j++)
	{
		for (int i = 0; i < 16; i++)
		{
			segScanLine(organnizedframes[j].pointschannels[i], organnizedframes[j].featurechannels[i]);
		}
	}

	float j = 0;
	while (!pangolin::ShouldQuit())
	{

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		debuger.d_cam->Activate(debuger.s_cam);

		//rendering commands ...

		glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		pangolin::glDrawAxis(100);
		//pangolin::glDrawColouredCube();
		//pangolin::glDraw_z0(5, 20);

		glPointSize(3.0f);
		glBegin(GL_POINTS);

		for (int k = 0; k < 16; k++) {
			for (int i = 0; i < organnizedframes[j].pointschannels[k].size(); ++i) {
				LaserPoint& pt = organnizedframes[j].pointschannels[k][i];
				float feature = organnizedframes[j].featurechannels[k][i];
				glColor3f(feature, feature, 5 * (1 - feature));
				glVertex3f(pt.x, pt.y, pt.z);
			}
		}

		glEnd();




		if (*debuger.menuReset) {
			debuger.s_cam.SetModelViewMatrix(
				pangolin::ModelViewLookAt(50, 50, 50, 0, 0, 0, 0.0, 1.0, 0.0)
			);
			*debuger.menuReset = false;
		}
		pangolin::FinishFrame();
		std::this_thread::sleep_for(std::chrono::microseconds(10));

		j = j + 0.3;
		if (j>= organnizedframes.size())
		{
			j = 0;
		}
	}
	
	

// 	std::function<void()> func = [&]()->void {
// 		glPointSize(3.0f);
// 		glBegin(GL_POINTS);
// 
// 		for (int j = 0; j < 16; j++) {
// 			for (int i = 0; i < organnizedframes[1].pointschannels[j].size(); ++i) {
// 				LaserPoint& pt = organnizedframes[1].pointschannels[j][i];
// 				glColor3f(lineFeatures[j][i], lineFeatures[j][i], 5*(1-lineFeatures[j][i]));
// 				glVertex3f(pt.x, pt.y, pt.z);
// 			}
// 		}
// 
// 		glEnd();
// 
// 	};
// 	debuger.RenderLoop(func);

//        float* depthMap = new float[FRAME_WIDTH*FRAME_HEIGHT];
//        memset(depthMap,0,FRAME_WIDTH*FRAME_HEIGHT);
//        framepoints2depthmap(laserFrames[i].framePoints,depthMap);
//
//        float* planarity = new float[FRAME_WIDTH*FRAME_HEIGHT];
//        memset(planarity,0,FRAME_WIDTH*FRAME_HEIGHT);
//        calculatePlanarity(depthMap, FRAME_WIDTH, FRAME_HEIGHT, planarity);
//        showFeatureMap(planarity,FRAME_WIDTH,FRAME_HEIGHT,"test");


}
