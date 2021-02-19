
/*
*
*
* Example usage:
bazel run //experimental/alex:opencv -- \
    --image_dir /home/alex/Desktop/projects/uav/hawk-eye/hawk_eye/data_generation/assets/competition-2019 \
    --tile_save_dir /tmp/tiles
*
*/

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

#include "absl/strings/substitute.h"
#include "absl/debugging/failure_signal_handler.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

DEFINE_string(image_dir, "", "Path to image folder.");
DEFINE_string(tile_save_dir, "", "Path to save tiles to.");


void tile_image(std::filesystem::path image_path, std::filesystem::path tile_save_dir){
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

    for (int i = 0; i < img.rows; i += 512) {
        if (i + 512 > img.rows) {
            i = img.rows - 512;
        }
        for (int j = 0; j < img.cols; j += 512) {
            if (j + 512 > img.cols) {
                j = img.cols - 512;
            }
            cv::Rect roi(j, i, 512, 512);
            cv::Mat crop = img(roi);
            auto tile_name = tile_save_dir / absl::Substitute(
                "$0_$1_$2$3", image_path.stem().c_str(), i, j, image_path.extension().c_str());
            cv::imwrite(tile_name.c_str(), crop);
        }
    }

}


int main(int argc, char* argv[])
{
    absl::FailureSignalHandlerOptions opts;
    absl::InstallFailureSignalHandler(opts);
    gflags::SetUsageMessage("Tile a directory of images.");
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);

    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    std::filesystem::path tile_save_dir = std::filesystem::path(FLAGS_tile_save_dir);
    std::filesystem::create_directory(FLAGS_tile_save_dir);

    for (const auto& file : std::filesystem::directory_iterator(FLAGS_image_dir)){
        auto start = std::chrono::high_resolution_clock::now();
        tile_image(file, tile_save_dir);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        // To get the value of duration use the count()
        // member function on the duration object
        std::cout << duration.count() << std::endl;
    }

    int tile_count = 0;
    for (const auto& file : std::filesystem::directory_iterator(tile_save_dir)){
        tile_count += 1;
    }
    std::cout << tile_count << std::endl;


    return 0;
}
