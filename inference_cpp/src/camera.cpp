#include "camera.h"
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream> 

Camera::Camera(const &std::front_file_cam, const &std::left_file_cam, 
    const &std::right_file_cam) {
        
    front_cam.open(front_file_cam); 
    left_file_cam.open(left_file_cam); 
    right_file_cam.open(right_cam)
} 

cv::Mat Camera::return_front_img() {
    return front_cam.read(front_frame); 
}

cv::Mat Camera::resize_front_img(int width, int height) {
    return front_cam.resize(front_cam, (width, height)); 
}