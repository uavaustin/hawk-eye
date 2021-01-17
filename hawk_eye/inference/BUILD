package(default_visibility = ["//visibility:public"])


py_library(
    name = "benchmark_inference",
    srcs = ["benchmark_inference.py"],
    deps = [
        "//hawk_eye/core:classifier",
        "//hawk_eye/core:detector",
        "//hawk_eye/data_generation:generate_config"
    ],
)

py_library(
    name = "find_targets",
    srcs = ["find_targets.py"],
    deps = [
        ":types",
        "//hawk_eye/core:classifier",
        "//hawk_eye/core:detector",
        "//hawk_eye/data_generation:generate_config"
    ],
    data = [":requirements.txt"]
)

py_library(
    name = "types",
    srcs = ["types.py"],
)

exports_files([
    "requirements.txt",
    "production_models.py",
])
