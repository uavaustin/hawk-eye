package(default_visibility = ["//visibility:public"])

py_library(
    name = "classifier",
    srcs = ["classifier.py"],
    deps = [
        ":asset_manager",
        "//third_party",
    ],
)

py_library(
    name = "detector",
    srcs = [
        "detector.py",
        "fpn.py",
    ],
    deps = [
        ":asset_manager",
        "//third_party",
    ],
)

py_library(
    name = "asset_manager",
    srcs = ["asset_manager.py"],
)
