//
// Created by 86384 on 2019/9/15.
//

#ifndef LEARNSHADER_LASERFRAMESDEFINES_H
#define LEARNSHADER_LASERFRAMESDEFINES_H

#include <vector>
#include <fstream>
#include <Eigen/Dense>

#include <glog/logging.h>

#define FRAME_WIDTH 1200 // 180 deg -90-90
#define FRAME_HEIGHT 300 // 40 deg 70-110
#define OFFSET_THETA 90.0
#define OFFSET_PHI 70.0
#define RANGE_THETA 180.0
#define RANGE_PHI 40.0

struct LaserPoint {
	double x;
	double y;
	double z;
	int intense;
	int hour;
	unsigned int ms_top_of_hour;
	int laser_id;
};

struct OrganizedFrame{
    std::vector<std::vector<LaserPoint>> pointschannels; //16 0-15
	std::vector<std::vector<float>> featurechannels;
    Eigen::Quaterniond q_nl;
    Eigen::Vector3d r_nl;
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
        int u = ((theta-OFFSET_THETA) / RANGE_THETA)*FRAME_WIDTH;
        int v = (phi-OFFSET_PHI) / RANGE_PHI*FRAME_HEIGHT;

        int id = v * FRAME_WIDTH + u;
        if(id <0 || id > FRAME_WIDTH*FRAME_HEIGHT){
            continue;
        }

        depthMap[id] = depth;
    }
}

//should be organized before
#define SEGDURATION 10
void segScanLine(const std::vector<LaserPoint>& framePoints, std::vector<float>& lineFeatures){
	CHECK(framePoints.size() == lineFeatures.size()) << "you forgot to resize the lineFeatures";
	

    for (int i = 0; i < framePoints.size(); ++i) {
        Eigen::Matrix<float, 3, 2*SEGDURATION + 1> raw_data;

        int count = 0;
        for (int j = i-SEGDURATION; j <= i+SEGDURATION; ++j) {
            int tmpIdx = j;
            if(j < 0){
                tmpIdx = j + framePoints.size();
            }
            else if(j >= framePoints.size()){
                tmpIdx = j - framePoints.size();
            }
            raw_data.col(count) = Eigen::Vector3f(framePoints[tmpIdx].x,framePoints[tmpIdx].y,framePoints[tmpIdx].z);
            count++;
        }

        // estimate pca
        Eigen::Vector3f mean = Eigen::Vector3f(
                raw_data.row(0).mean(),
                raw_data.row(1).mean(),
                raw_data.row(2).mean());


        for (int i = 0; i < 2*SEGDURATION + 1; ++i){
            raw_data.col(i) = raw_data.col(i) - mean;
        }
        Eigen::Matrix3f cov_matrix = raw_data*raw_data.transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig; // default constructor
        eig.computeDirect(cov_matrix); // works for 2x2 and 3x3 matrices, does not require loops
        Eigen::Vector3f D = eig.eigenvalues();
        //printf("%f %f %f\n",D(0),D(1),D(2));
        //    Eigen::EigenSolver<Eigen::Matrix3f> es(cov_matrix);
        //    Eigen::Vector3f D = es.pseudoEigenvalueMatrix().diagonal();
		Eigen::Vector3f sD = Eigen::Vector3f(sqrt(abs(D(0))), sqrt(abs(D(1))), sqrt(abs(D(2))));
        float line_feature = (sD(2) - sD(1))/ sD(2);
		//printf("%f\n", line_feature);
		lineFeatures[i] = line_feature;
    }
// 	FILE *file = fopen("debug.txt", "w");
// 	for (int i = 0; i < framePoints.size(); i++)
// 	{
// 		fprintf(file, "%f %f %f %f\n", framePoints[i].x, framePoints[i].y, framePoints[i].z, lineFeatures[i]);
// 	}
// 	fclose(file);
}

void frame2orgnizedframe(const LaserFrame& frame, OrganizedFrame& organizedFrame){
    organizedFrame.q_nl = frame.q_nl;
    organizedFrame.r_nl = frame.r_nl;
    organizedFrame.pointschannels.resize(16);
	organizedFrame.featurechannels.resize(16);

    const std::vector<LaserPoint> &framePoints = frame.framePoints;
    for (int i = 0; i < framePoints.size(); ++i) {
        int laser_id = framePoints[i].laser_id;
        organizedFrame.pointschannels[laser_id].push_back(framePoints[i]);
    }
    for (int i = 0; i < organizedFrame.pointschannels.size(); ++i) {
        std::sort(
                organizedFrame.pointschannels[i].begin(),
                organizedFrame.pointschannels[i].end(),
                [](const LaserPoint& p1,const LaserPoint& p2)->bool{
                    float azium1 = -atan2(p1.y, p1.x)/3.1415926*180 + 180;
                    float azium2 = -atan2(p2.y, p2.x)/3.1415926*180 + 180;
                    return azium1 < azium2;
                });
		organizedFrame.featurechannels[i].resize(organizedFrame.pointschannels[i].size());
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
        int u = ((theta-OFFSET_THETA) / RANGE_THETA)*FRAME_WIDTH;
        int v = (phi-OFFSET_PHI) / RANGE_PHI*FRAME_HEIGHT;

        int id = v * FRAME_WIDTH + u;
        if(id <0 || id > FRAME_WIDTH*FRAME_HEIGHT){
            continue;
        }

        depthMap[id] = depth;
    }
}


#endif //LEARNSHADER_LASERFRAMESDEFINES_H
