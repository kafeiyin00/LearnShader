//
// Created by 86384 on 2019/9/13.
//
#include "Config.h"
#include "src/cudaFunction.cuh"
#include <time.h>
#include <vector>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <time.h>


struct Point{
    float xyz[3];
    int rgb[3];
};

void loadPoints(std::vector<Point>& points){
    std::string filename = SHADER_FOLDER_PATH"sample_points.txt";

    std::ifstream ifs(filename,std::ios_base::in);

    while (!ifs.eof()){
        Point p;
        float x,y,z;
        int r,g,b;
        float useless_f;
        int useless_d;

        char line[256];
        ifs.getline(line, 256);
        sscanf(line, "%f %f %f %d %d %d %d %f\n", &x, &y, &z, &r, &g, &b, &useless_d, &useless_f);


        p.xyz[0] = x;
        p.xyz[1] = y;
        p.xyz[2] = z;

        p.rgb[0] = r;
        p.rgb[1] = g;
        p.rgb[2] = b;

        points.push_back(p);
        //printf("%f, %f, %f\n",x,y,z);
    }
}

void projectCPU(float *h_xs, float *h_ys, float *h_zs, float *h_ds, int *h_pojectionIds, int nPoints) {
    for (int i = 0; i < nPoints; i++)
    {
        float x = h_xs[i];
        float y = h_ys[i];
        float z = h_zs[i];

        //bearing vector to theta phi
        double theta = atan2(y, x) / 3.1415926 * 180.0 + 180.0;
        double phi = atan2(sqrt(x*x + y * y), z) / 3.1415926 * 180.0;
        float depth = sqrt(x * x + y * y + z * z);

        //theta phi to image coordinates
        int u = (theta / 360.0)*2000;
        int v = phi / 180.0*1000;

        h_pojectionIds[i] = v * 2000 + u;
        h_ds[i] = depth;
    }
}

void showNormalMap(float* nx, float* ny,float* nz, int width, int height) {
    cv::Mat mat(height, width, CV_8UC3);

    for (int i =0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            float x = nx[i*width+j];
            float y = ny[i*width+j];
            float z = nz[i*width+j];
            Eigen::Vector3f norm = Eigen::Vector3f(x,y,z).normalized();
            mat.at<cv::Vec3b>(i, j) = cv::Vec3b(abs(norm[0]*255), abs(norm[1]*255), abs(norm[2]*255));
            //std::cout << x << std::endl;
        }

    }

    cv::namedWindow("test");
    cv::imshow("test",mat);
    cv::waitKey(0);
}

int main(){
    std::vector<Point> points;
    loadPoints(points);

    float* h_xs = new float[points.size()];
    float* h_ys = new float[points.size()];
    float* h_zs = new float[points.size()];

    for (int i = 0; i < points.size(); i++)
    {
        h_xs[i] = points[i].xyz[0];
        h_ys[i] = points[i].xyz[1];
        h_zs[i] = points[i].xyz[2];
    }

    int* h_pojectionIds = new int[points.size()];
    float* h_ds = new float[points.size()];


    clock_t start, end;
    double dur;
    start = clock();
    //cpu
    projectCPU(h_xs, h_ys, h_zs, h_ds, h_pojectionIds, points.size());
    end = clock();
    dur = (double)(end - start);
    printf("CPU Use Time:%f\n", (dur * 1000 / CLOCKS_PER_SEC)); // ms

    //z-buffer
    float * depthMap = new float[2000*1000];
    for (size_t i = 0; i < 2000*1000; i++)
    {
        depthMap[i] = 999.9;
    }
    for (size_t i = 0; i < points.size(); i++)
    {
        if (depthMap[h_pojectionIds[i]] < h_ds[i]) {
            continue;
        }
        //assert(h_pojectionIds[i] < 600*300);
        depthMap[h_pojectionIds[i]] = h_ds[i];
    }

    float* h_nx = new float[2000*1000];
    float* h_ny = new float[2000*1000];
    float* h_nz = new float[2000*1000];
    memset((void*)h_nx,0,2000*1000* sizeof(float));
    memset((void*)h_ny,0,2000*1000* sizeof(float));
    memset((void*)h_nz,0,2000*1000* sizeof(float));

    start = clock();
    //gpu
    calculateNormal(depthMap,2000,1000,h_nx,h_ny,h_nz);
    end = clock();

    dur = (double)(end - start);
    printf("GPU Use Time:%f\n", (dur * 1000 / CLOCKS_PER_SEC)); // ms

    showNormalMap(h_nx,h_ny,h_nz,2000,1000);
}
