//
// Created by 86384 on 2019/9/19.
//

#ifndef LEARNSHADER_3DDEBUGER_H
#define LEARNSHADER_3DDEBUGER_H

#include <pangolin/pangolin.h>
#include <functional>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds

class ThreeDdebuger{
public:
    ThreeDdebuger();

    void RenderLoop(std::function<void ()> func);
	void Frame();
    pangolin::OpenGlRenderState s_cam;
    pangolin::View* d_cam;

    //menu
    pangolin::Var<bool>* menuReset;
};

inline ThreeDdebuger::ThreeDdebuger() {

    //menuReset = new pangolin::Var<bool>("menu.Reset", false, false);

    pangolin::CreateWindowAndBind("3D debugger", 1024, 800);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Define Camera Render Object (for view / scene browsing)
    s_cam = pangolin::OpenGlRenderState(
            pangolin::ProjectionMatrix(1024, 800, 800, 800, 512, 400, 0.1, 1000),
            pangolin::ModelViewLookAt(50, 50, 50, 0, 0, 0, 0.0, 1.0, 0.0)
    );

    // Add named OpenGL viewport to window and provide 3D Handler
    pangolin::View& cam = pangolin::Display("cam")
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 800.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
    d_cam = &cam;

    //menu
    pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
    menuReset = new pangolin::Var<bool> ("menu.Reset", false, false);
//    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
//    pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);
//    pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames", true, true);
//    pangolin::Var<bool> menuShowGraph("menu.Show Graph", true, true);



}

inline void ThreeDdebuger::RenderLoop(std::function<void()> func) {


    while (!pangolin::ShouldQuit())
    {

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam->Activate(s_cam);

        //rendering commands ...

        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        pangolin::glDrawAxis(100);
        //pangolin::glDrawColouredCube();
        //pangolin::glDraw_z0(5, 20);

        func();

        if (*menuReset) {
            s_cam.SetModelViewMatrix(
                    pangolin::ModelViewLookAt(50, 50, 50, 0, 0, 0, 0.0, 1.0, 0.0)
            );
            *menuReset = false;
        }
        pangolin::FinishFrame();
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

void PangolinInitialization(){
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


//    while (!pangolin::ShouldQuit())
//    {
//
//        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//        d_cam.Activate(s_cam);
//
//        //rendering commands ...
//
//        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
//        glClear(GL_COLOR_BUFFER_BIT);
//        pangolin::glDrawAxis(100);
//        //pangolin::glDrawColouredCube();
//        pangolin::glDraw_x0(5, 20);
//
//
//        glPointSize(3.0f);
//        glBegin(GL_POINTS);
//
//
//        for (int i = 0; i < laserFrames[tempFrame].framePoints.size(); ++i) {
//            LaserPoint& pt = laserFrames[tempFrame].framePoints[i];
//            glColor3f(5*pt.intense/255.0,5*pt.intense/255.0,5*pt.intense/255.0);
//            glVertex3f(pt.x,pt.y,pt.z);
//        }
//        tempFrame = tempFrame+0.2;
//
//        if(tempFrame >= laserFrames.size()){
//            tempFrame = 0;
//        }
//
//        glEnd();
//
//
//        if (menuReset) {
//            s_cam.SetModelViewMatrix(
//                    pangolin::ModelViewLookAt(50, 50, 50, 0, 0, 0, 0.0, 1.0, 0.0)
//            );
//            menuReset = false;
//        }
//        pangolin::FinishFrame();
//        std::this_thread::sleep_for(std::chrono::microseconds(10));
//    }
}


#endif //LEARNSHADER_3DDEBUGER_H
