load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Download the necessary test data.
http_archive(
    name = "backgrounds_test_data",
    url = "https://dl.bintray.com/uavaustin/target-finder-assets/:backgrounds-v1.tar.gz",
    sha256 = "01c9729257cf77afaba41f5ad09d4be1f24fb409984fe353303c6a969e7f8200",
    build_file_content = """
filegroup(name="images",
          srcs=glob([
              "**/*.png",
            ]),
          visibility = ["//visibility:public"],
)""",
    strip_prefix = "backgrounds-v1",
)

http_archive(
    name = "base_shape_test_data",
    url = "https://dl.bintray.com/uavaustin/target-finder-assets/:base-shapes-v1.tar.gz",
    sha256 = "69f30fa2ba8636110b3752a75a6ffd598c0d094879df378e427324c0023dfbc3",
    build_file_content = """
filegroup(name="images",
          srcs=glob([
              "**/*.png",
            ]),
          visibility = ["//visibility:public"],
)""",
    strip_prefix = "base-shapes-v1",
)