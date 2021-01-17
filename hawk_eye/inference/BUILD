load("@rules_python//python:defs.bzl", "py_library")
load("//third_party/bazel_doctest:doctest.bzl", "py_doc_test")

package(default_visibility = ["//visibility:public"])

exports_files([
    "requirements.txt",
    "production_models.py",
])

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
        ":inference_types",
        "//hawk_eye/core:classifier",
        "//hawk_eye/core:detector",
        "//hawk_eye/data_generation:generate_config"
    ]
)

py_library(
    name = "inference_types",
    srcs = ["inference_types.py"],
)

py_doc_test(
    name = "find_targets_doctest",
    deps = [
        ":find_targets",
    ],
)
