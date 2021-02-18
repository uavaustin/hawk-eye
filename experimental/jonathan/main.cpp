
#include <iostream>
#include <sys/stat.h>
#include <ctime>
#include <assert.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

const std::string g_cwd = "/home/jonathan/Documents/uav/hawk-eye/experimental/jonathan/";
inline int file_exists(const std::string &name){
	struct stat buffer;
	if(stat(name.c_str(), &buffer) != 0)return 0;
	else if(buffer.st_mode & S_IFDIR)return 2;
	else return 1;

}
void preprocessing(){
	int check;
	std::string new_directory = g_cwd+"sliced_images";
	//bad practice but too lazy to download libraries to get cwd or pass in argument to main
	if(file_exists(new_directory) != 2){
		char char_array[new_directory.length()+1];
		strcpy(char_array, new_directory.c_str());
		check = mkdir(char_array,0777);
		std::cout<<"made directory";
		assert(check==0);
	}
}
void process_image(const std::string &name, const unsigned num){
	std::vector<cv::Mat>sliced_images;
	const cv::Mat img = cv::imread(name, cv::IMREAD_COLOR);
	int height = img.rows;
	int width = img.cols;
	cv::Size const ROI_SIZE(512,512);
	for(int y = 0; y<=height-ROI_SIZE.height;y+=ROI_SIZE.height){
		for(int x = 0; x<=width-ROI_SIZE.width;x+=ROI_SIZE.width){
			cv::Rect rect = cv::Rect(x,y,ROI_SIZE.width, ROI_SIZE.height);
			sliced_images.push_back(cv::Mat(img,rect));
		}
	}
	for(int i = 0; i < sliced_images.size(); i++){
		std::string output= g_cwd+"sliced_images/image-"+std::to_string(num)+std::to_string(i)+".png";
		cv::imwrite(output,sliced_images[i]);
	}
	std::cout<<"number of sliced images is "<<sliced_images.size()<<"\n";

}
int main()
{
	preprocessing();
    unsigned image_n = 1001;
    std::string initial_path = "/home/jonathan/Documents/uav/hawk-eye/data_generation/assets/competition-2019/image-00";
    std::string image_path = initial_path+std::to_string(image_n)+".jpg";
    //hard coded path since stat doesn't seem to see it
    unsigned long int t = 0;
    while(image_n<=1001){
    	//process image
    	//measuire only this time
    	if(file_exists(image_path)){
    		//lots of hard coding since I'm lazy
    		time_t current_time;
    		time_t end_time;
    		time(&current_time);
    		process_image(image_path,image_n);
    		time(&end_time);
    		long int e = static_cast<long int>(end_time);
    		long int s = static_cast<long int>(current_time);
    		t+=(e-s);
    	}

    	image_path = initial_path+std::to_string(++image_n)+".jpg";
    }
    std::cout<<"total time in seconds spend: "<<t<<"\n";
    //cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

    return 0;
}
