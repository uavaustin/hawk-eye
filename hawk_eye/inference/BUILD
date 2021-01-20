load("@rules_python//python:defs.bzl", "py_library")
load("//third_party/bazel_doctest:doctest.bzl", "py_doc_test")

package(default_visibility = ["//visibility:public"])

exports_files(["requirements.txt"])

py_library(
    name = "production_models",
    srcs = ["production_models.py"],
)

py_test(
    name = "production_models_test",
    srcs = ["production_models_test.py"],
    deps = [
        ":production_models",
        "//hawk_eye/core:asset_manager",
    ],
    local = True,  # Used to find GCloud project
)

py_library(
    name = "benchmark_inference",
    srcs = ["benchmark_inference.py"],
    deps = [
        "//hawk_eye/core:classifier",
        "//hawk_eye/core:detector",
        "//hawk_eye/data_generation:generate_config"
    ],
)

py_test(
    name = "benchmark_inference_test",
    srcs = ["benchmark_inference_test.py"],
    deps = [
        "//hawk_eye/inference:production_models",
        "//hawk_eye/inference:benchmark_inference"
    ],
    local = True,  # Used to find GCloud project
)

py_library(
    name = "find_targets",
    srcs = ["find_targets.py"],
    deps = [
        ":inference_types",
        ":production_models",
        "//hawk_eye/core:classifier",
        "//hawk_eye/core:detector",
        "//hawk_eye/data_generation:generate_config"
    ]
)

py_doc_test(
    name = "find_targets_doctest",
    deps = [
        ":find_targets",
    ],
)

py_library(
    name = "inference_types",
    srcs = ["inference_types.py"],
)
