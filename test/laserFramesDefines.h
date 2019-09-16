//
// Created by 86384 on 2019/9/15.
//

#ifndef LEARNSHADER_LASERFRAMESDEFINES_H
#define LEARNSHADER_LASERFRAMESDEFINES_H

#include <vector>
#include <fstream>
#include <Eigen/Dense>

const int FRAME_WIDTH = 1000;// 180 deg -90-90
const int FRAME_HEIGHT = 500;// 40 deg 70-110

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

void framepoints2depthmap(std::vector<LaserPoint>& framePoints, float* depthMap){
    if(depthMap == nullptr){
        printf("you fogot to allocate the depthMap!!!\n");
        return;
    }
    for (int i = 0; i < framePoints.size(); ++i) {
        float x = framePoints[i].x;
        float y = framePoints[i].y;
        float z = framePoints[i].z;

        //bearing vector to theta phi
        double theta = atan2(y, x) / 3.1415926 * 180.0;
        double phi = atan2(sqrt(x*x + y * y), z) / 3.1415926 * 180.0;
        float depth = sqrt(x * x + y * y + z * z);


        //theta phi to image coordinates
        int u = ((theta-90) / 180.0)*FRAME_WIDTH;
        int v = (phi-70) / 40.0*FRAME_HEIGHT;

        int id = v * FRAME_WIDTH + u;
        if(id <0){
            continue;
        }

        depthMap[id] = depth;
    }
}

void mappoints2depthmap(const std::vector<Eigen::Vector3d>& mapPoints,
        const Eigen::Quaterniond q_nl,
        const Eigen::Vector3d r_nl,
        float* depthMap){
    if(depthMap == nullptr){
        printf("you fogot to allocate the depthMap!!!\n");
        return;
    }
    for (int i = 0; i < mapPoints.size(); ++i) {
        Eigen::Vector3d localPt = q_nl.inverse()*(mapPoints[i]-r_nl);
        float x = localPt[0];
        float y = localPt[1];
        float z = localPt[2];

        //bearing vector to theta phi
        double theta = atan2(y, x) / 3.1415926 * 180.0;
        double phi = atan2(sqrt(x*x + y * y), z) / 3.1415926 * 180.0;
        float depth = sqrt(x * x + y * y + z * z);

        //theta phi to image coordinates
        int u = ((theta-90) / 180.0)*FRAME_WIDTH;
        int v = (phi-70) / 40.0*FRAME_HEIGHT;

        int id = v * FRAME_WIDTH + u;
        if(id <0 || id > FRAME_WIDTH*FRAME_HEIGHT){
            continue;
        }

        depthMap[id] = depth;
    }
}


#endif //LEARNSHADER_LASERFRAMESDEFINES_H
