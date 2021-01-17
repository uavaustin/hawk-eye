load("@rules_python//python:defs.bzl", "py_library")
load("//third_party/bazel_doctest:doctest.bzl", "py_doc_test")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "vovnet",
    srcs = ["vovnet.py"],
)

py_doc_test(
    name = "vovnet_doctest",
    deps = [":vovnet"],
)
