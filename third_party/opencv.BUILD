load("@rules_foreign_cc//tools/build_defs:cmake.bzl", "cmake_external")

filegroup(
    name = "all",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"]
)

cmake_external(
    name = "opencv",
    cmake_options = [
        "-GNinja",
        "-DBUILD_LIST=core,highgui,imgcodecs,imgproc",
        "-DCMAKE_BUILD_TYPE=RELEASE",
        "-DBUILD_TESTS=OFF",
        "-DBUILD_PERF_TESTS=OFF",
        "-DBUILD_opencv_ts=OFF",
    ],
    lib_source = ":all",
    make_commands = [
        "ninja",
        "ninja install",
    ],
    out_include_dir = "include/opencv4",
    shared_libraries = [
        "libopencv_core.so.4.5",
        "libopencv_highgui.so.4.5",
        "libopencv_imgcodecs.so.4.5",
        "libopencv_imgproc.so.4.5",
    ],
    visibility = ["//visibility:public"],
)
