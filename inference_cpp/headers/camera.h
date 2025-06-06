#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

#ifndef CAMERA_H 
#define CAMERA_H

class Camera {
    public: 
        cv::Mat return_front_img();
        cv::Mat resize_front_img(); 

    private: 
        cv::Mat front_frame; 
        cv::Mat left_frame; 
        cv::Mat right_frame; 

        cv::VideoCapture front_cam; 
        cv::VideoCapture left_cam; 
        cv::VideoCapture right_cam; 

        Camera(const std::string &front_file_cam, const std::string &left_file_cam, 
            const std::string &right_file_cam); 
}

#endif 