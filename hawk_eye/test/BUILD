py_test(
    name = "test_data_generation",
    srcs = [
        "test_data_generation.py",
        "generate_config_test.py",
    ],
    data = [
        "@backgrounds//:file",
        "@base_shapes//:file",
        "@fonts//:file",
    ],
    deps = [
        "//hawk_eye/data_generation:classification_data_generation",
        "//hawk_eye/data_generation:detection_data_generation",
    ],
)

py_test(
    name = "test_third_party",
    srcs = ["test_third_party.py"],
    deps = ["//third_party"],
)

py_test(
    name = "test_inference",
    srcs = ["test_inference.py"],
    deps = [
        "//hawk_eye/inference:find_targets",
        "//hawk_eye/inference:benchmark_inference"
    ],
    local = True,
)
