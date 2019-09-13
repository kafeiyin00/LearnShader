#include <iostream>
// for the IDE detection

//#include "cuda_runtime.h"
//#include "Config.h"
#include "../Config.h"
#include <opencv2/opencv.hpp>

#include <vector>

#include "projection.cuh"

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

void showDepthMap(float* depthMap, int width, int height) {
	cv::Mat mat(height, width, CV_8UC3);

	for (int i =0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int depth = depthMap[i*width+j];
			mat.at<cv::Vec3b>(i, j) = cv::Vec3b(depth, 255-depth, depth/20);
			//std::cout << depth << std::endl;
		}
		
	}

	cv::namedWindow("test");
	cv::imshow("test",mat);
	cv::waitKey(0);
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
		int u = (theta / 360.0)*600;
		int v = phi / 180.0*300;

		h_pojectionIds[i] = v * 600 + u;
		h_ds[i] = depth;
	}
}

int main(){
    std::vector<Point> points;
    loadPoints(points);

	printf("we load points: %d\n", points.size());

    //std::vector<Point> points;

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
	//gpu
	project(h_xs, h_ys, h_zs, h_ds, h_pojectionIds, points.size());
	end = clock();

	dur = (double)(end - start);
	printf("GPU Use Time:%f\n", (dur * 1000 / CLOCKS_PER_SEC)); // ms

	start = clock();
	//cpu
	projectCPU(h_xs, h_ys, h_zs, h_ds, h_pojectionIds, points.size());
	end = clock();
	dur = (double)(end - start);
	printf("CPU Use Time:%f\n", (dur * 1000 / CLOCKS_PER_SEC)); // ms

	//z-buffer
	float * depthMap = new float[600*300];
	for (size_t i = 0; i < 600*300; i++)
	{
		depthMap[i] = 999.9;
	}
	for (size_t i = 0; i < points.size(); i++)
	{
		if (depthMap[h_pojectionIds[i]] < h_ds[i]) {
			continue;
		}
		assert(h_pojectionIds[i] < 600*300);
		depthMap[h_pojectionIds[i]] = h_ds[i];
	}

	showDepthMap(depthMap,600,300);
	system("pause");
    return 0;

}