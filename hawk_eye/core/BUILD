package(default_visibility = ["//visibility:public"])

py_library(
    name = "classifier",
    srcs = ["classifier.py"],
    deps = [
        ":asset_manager",
        "//third_party/vovnet:vovnet",
    ],
)

py_library(
    name = "detector",
    srcs = [
        ":asset_manager",
        "detector.py",
        "fpn.py",
    ],
    deps = [
        ":asset_manager",
        "//third_party/vovnet:vovnet",
        "//third_party/rexnet:rexnet",
        "//third_party/detectron2:anchors",
        "//third_party/detectron2:postprocess",
        "//third_party/detectron2:regression",
        "//third_party/detectron2:retinanet_head",
    ],
)

py_library(
    name = "asset_manager",
    srcs = ["asset_manager.py"],
    deps = ["//hawk_eye/data_generation:generate_config"]
)
