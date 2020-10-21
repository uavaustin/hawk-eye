package(default_visibility = ["//visibility:public"])

py_library(
    name = "benchmark_inference",
    srcs = ["benchmark_inference.py"],
    deps = [
        "//core:classifier",
        "//core:detector",
        "//data_generation:generate_config"
    ],
)

py_library(
    name = "find_targets",
    srcs = ["find_targets.py"],
    deps = [
        ":types",
        "//core:classifier",
        "//core:detector",
        "//data_generation:generate_config"
    ],
    data = [":requirements.txt"]
)

py_library(
    name = "types",
    srcs = ["types.py"],
)

exports_files(["requirements.txt"])
