
cc_binary(
    name = "opencv",
    srcs = ["main.cpp"],
    deps = [
        "@opencv//:opencv",
        "@com_github_gflags_gflags//:gflags",
        "@com_google_absl//absl/debugging:failure_signal_handler",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_logging//:glog",
    ],
    visibility = ["//visibility:public"],
)
