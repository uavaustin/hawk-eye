# Load gcs_archive rule to download using gsutil

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_toolchains",
    sha256 = "7ebb200ed3ca3d1f7505659c7dfed01c4b5cb04c3a6f34140726fe22f5d35e86",
    strip_prefix = "bazel-toolchains-3.4.1",
    urls = [
        "https://github.com/bazelbuild/bazel-toolchains/releases/download/3.4.1/bazel-toolchains-3.4.1.tar.gz",
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-toolchains/releases/download/3.4.1/bazel-toolchains-3.4.1.tar.gz",
    ],
)

load("//third_party:gcs.bzl", "gcs_file")

gcs_file(
    name = "base_shapes",
    bucket = "gs://uav-austin-test",
    file = "assets/base-shapes-v1.tar.gz",
    downloaded_file_path = "base_shapes.tar.gz",
    sha256 = "9266e23087c58ee679903a6891208bb8c636396cdbc0b35f0eeff294411ddbdc",
)

gcs_file(
    name = "backgrounds",
    bucket = "gs://uav-austin-test",
    file = "backgrounds/backgrounds-v1.tar.gz",
    downloaded_file_path = "backgrounds.tar.gz",
    sha256 = "b00778153d14fd158345b9a18e5f79089d420c2cf36eb363a595d439d1b9c089",
)

gcs_file(
    name = "fonts",
    bucket = "gs://uav-austin-test",
    file = "assets/fonts.tar.gz",
    downloaded_file_path = "fonts.tar.gz",
    sha256 = "e67fc398c9e9a55071d2d4edd155c691540bf4951383cfa1bed69aacbee02675",
    strip_prefix = "fonts"
)