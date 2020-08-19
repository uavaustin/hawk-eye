py_test(
    name = "test_data_generation",
    srcs = ["test_data_generation.py"],
    deps = ["//data_generation:detection_data_generation"],
    data = [
        "@backgrounds_test_data//:images",
        "@base_shape_test_data//:images"
    ]
)

py_test(
    name = "test_third_party_docstring",
    srcs = ["test_third_party_docstring.py"],
    deps = ["//third_party"]
)
