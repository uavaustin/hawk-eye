py_test(
    name = "test_data_generation",
    srcs = [
        "test_data_generation.py",
        "test_generate_config.py",
    ],
    deps = ["//data_generation:detection_data_generation",],
    data = [
        "@base_shapes//:file",
        "@backgrounds//:file",
        "@fonts//:file",
    ]
)

py_test(
    name = "test_third_party_docstring",
    srcs = ["test_third_party_docstring.py"],
    deps = ["//third_party"]
)
