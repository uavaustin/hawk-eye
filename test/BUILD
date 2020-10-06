py_test(
    name = "test_data_generation",
    srcs = [
        "test_data_generation.py",
        "test_generate_config.py",
    ],
    data = [
        "@backgrounds//:file",
        "@base_shapes//:file",
        "@fonts//:file",
    ],
    deps = ["//data_generation:detection_data_generation"],
)

py_test(
    name = "test_third_party_docstring",
    srcs = ["test_third_party_docstring.py"],
    deps = ["//third_party"],
)

py_test(
    name = "test_inference",
    srcs = ["test_inference.py"],
    deps = ["//inference:find_targets"],
)

py_test(
    name = "test_benchmark_inference",
    srcs = ["test_benchmark_inference.py"],
    deps = [
        "//inference:benchmark_inference",
        "//inference:find_targets"
    ],
)
