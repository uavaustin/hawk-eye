
#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

int main()
{
    std::string image_path = "/home/alex/Desktop/projects/uav/hawk-eye/hawk_eye/data_generation/assets/competition-2019/image-001001.jpg";
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

    return 0;
}
