//
// Created by 86384 on 2019/9/15.
//

#include "iostream"
#include "laserFramesDefines.h"
#include "Config.h"

#include <opencv2/opencv.hpp>

void showDepthMap(float* depthMap, int width, int height) {
    cv::Mat mat_depth(height, width, CV_32FC1);

    for (int i = 1; i < height-1; i++)
    {
        for (int j = 1; j < width -1 ; j++)
        {
            float depth = depthMap[i*width+j];
            if(depth > 0){
                cv::circle(mat_depth,cv::Point(j,i),2,cv::Scalar(depth),1);
            }

            //std::cout << depth << std::endl;
        }
    }
    cv::Mat grayImage(height, width, CV_8UC1);
    double alpha = 255.0 / (100 - 40);
    mat_depth.convertTo(grayImage, CV_8UC1, alpha, -alpha * 40  );
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

    cv::namedWindow("test");
    cv::imshow("test",mat_color);
    cv::waitKey(100);
}

int main(){
    printf("test aerial frames\n");

    std::vector<LaserFrame> laserFrames;
    std::string filename = SHADER_FOLDER_PATH"laserFrames.dat";
    loadLaserFrames(laserFrames, filename);
    printf("we load => %d <= frames\n",laserFrames.size());


    for (int i = 0; i < laserFrames.size(); ++i) {
        float* depthMap = new float[FRAME_WIDTH*FRAME_HEIGHT];
        memset(depthMap,0,FRAME_WIDTH*FRAME_HEIGHT);

        framepoints2depthmap(laserFrames[i].framePoints,depthMap);

        showDepthMap(depthMap,FRAME_WIDTH,FRAME_HEIGHT);

        delete [] depthMap;
    }


}