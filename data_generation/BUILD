package(default_visibility = ["//visibility:public"])

py_library(
    name = "detection_data_generation",
    srcs = ["create_detection_data.py",],
    deps = [
        ":generate_config",
        "//core:asset_manager"
    ]
)

py_library(
    name = "classification_data_generation",
    srcs = ["create_clf_data.py",],
)

py_library(
    name = "generate_config",
    srcs = ["generate_config.py",],
    data = ["config.yaml",]
)