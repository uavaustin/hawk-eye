# Description:
#   OpenCV libraries for video/image processing on Linux

load("@rules_cc//cc:defs.bzl", "cc_library")

licenses(["notice"])  # BSD license

cc_library(
    name = "opencv",
    srcs = glob(
        [
            "lib/libopencv_core.so",
            "lib/libopencv_highgui.so",
            "lib/libopencv_imgcodecs.so",
            "lib/libopencv_imgproc.so",
        ],
    ),
    hdrs = glob([
        "include/opencv4/opencv2/**/*.h*",
    ]),
    includes = [
        "include/opencv4/",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
