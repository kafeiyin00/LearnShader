//
// Created by 86384 on 2019/9/15.
//

#include "iostream"
#include "laserFramesDefines.h"
#include "Config.h"

#include <opencv2/opencv.hpp>

void showDepthMap(float* depthMap, int width, int height, std::string windowname) {
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

    cv::namedWindow(windowname.c_str());
    cv::imshow(windowname.c_str(),mat_color);
    cv::waitKey(100);
}

void getMapPoints(std::vector<Eigen::Vector3d>& mapPoints,const std::vector<LaserFrame>& laserFrames){
    for (int i = 0; i < laserFrames.size(); ++i) {
        for (int j = 0; j < laserFrames[i].framePoints.size(); ++j) {
            Eigen::Vector3d pt(
                    laserFrames[i].framePoints[j].x,
                    laserFrames[i].framePoints[j].y,
                    laserFrames[i].framePoints[j].z);
            Eigen::Vector3d mapPt = laserFrames[i].q_nl*pt+laserFrames[i].r_nl;
            mapPoints.push_back(mapPt);
        }
    }
}

int main(){
    printf("test aerial frames\n");

    std::vector<LaserFrame> laserFrames;
    std::string filename = SHADER_FOLDER_PATH"laserFrames.dat";
    loadLaserFrames(laserFrames, filename);
    printf("we load => %d <= frames\n",laserFrames.size());

    std::vector<Eigen::Vector3d> mapPoints;
    getMapPoints(mapPoints,laserFrames);
    printf("we load => %d <= mapPoints\n",mapPoints.size());

    for (int i = 0; i < laserFrames.size(); ++i) {
        float* depthMap = new float[FRAME_WIDTH*FRAME_HEIGHT];
        memset(depthMap,0,FRAME_WIDTH*FRAME_HEIGHT);

        framepoints2depthmap(laserFrames[i].framePoints,depthMap);


        showDepthMap(depthMap,FRAME_WIDTH,FRAME_HEIGHT,"one frame");

        float* depthMapMapPoints = new float[FRAME_WIDTH*FRAME_HEIGHT];
        mappoints2depthmap(mapPoints,laserFrames[i].q_nl,laserFrames[i].r_nl,depthMapMapPoints);
        showDepthMap(depthMapMapPoints,FRAME_WIDTH,FRAME_HEIGHT,"map points");

        delete [] depthMap;
        delete [] depthMapMapPoints;
    }

    // projection for map points



}