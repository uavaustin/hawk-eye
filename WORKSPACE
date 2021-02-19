workspace(name = "hawk_eye")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_python",
    url = "https://github.com/bazelbuild/rules_python/releases/download/0.1.0/rules_python-0.1.0.tar.gz",
    sha256 = "b6d46438523a3ec0f3cead544190ee13223a52f6a6765a29eae7b7cc24cc83a0",
)

http_archive(
    name = "bazel_skylib",
    sha256 = "97e70364e9249702246c0e9444bccdc4b847bed1eb03c5a3ece4f83dfe6abc44",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.0.2/bazel-skylib-1.0.2.tar.gz",
    ],
)

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
    bucket = "gs://uav_austin",
    downloaded_file_path = "base_shapes_test_20210117.tar.gz",
    file = "test-deps/base_shapes_test_20210117.tar.gz",
    sha256 = "79b5a45a6a9855baa7d292890b3b0f512371b9b5df5919581b1b2e28bed53c48",
)

gcs_file(
    name = "backgrounds",
    bucket = "gs://uav_austin",
    downloaded_file_path = "backgrounds_test_2021017.tar.gz",
    file = "test-deps/backgrounds_test_2021017.tar.gz",
    sha256 = "d2118465bc6093ec066506ddf9b286ddacb2a26d67eb7b3a15e39ba242002a80",
)

gcs_file(
    name = "fonts",
    bucket = "gs://uav_austin",
    downloaded_file_path = "fonts_test_20210117.tar.gz",
    file = "test-deps/fonts_test_20210117.tar.gz",
    sha256 = "7c4d5561eac233d95c0f26665c9f29cf810e4637fdabd79a531ce1eb95210c58",
    strip_prefix = "fonts_test",
)

load("//hawk_eye/core:models.bzl", "production_model")

production_model(
    name = "classification_model",
    type = "classifier",
)
production_model(
    name = "detection_model",
    type = "detector",
)

http_archive(
    name = "opencv",
    build_file = "//third_party:opencv.BUILD",
    strip_prefix = "opencv-4.5.0",
    url = "https://github.com/opencv/opencv/archive/4.5.0.zip",
    sha256 = "168f6e61d8462fb3d5a29ba0d19c0375c111125cac753ad01035a359584ccde9",
)

http_archive(
    name = "rules_foreign_cc",
    strip_prefix = "rules_foreign_cc-master",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/master.zip",
)

load("@rules_foreign_cc//:workspace_definitions.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

http_archive(
    name = "com_google_absl",
    sha256 = "f41868f7a938605c92936230081175d1eae87f6ea2c248f41077c8f88316f111",
    strip_prefix = "abseil-cpp-20200225.2",
    urls = ["https://github.com/abseil/abseil-cpp/archive/20200225.2.tar.gz"],
)

http_archive(
    name = "com_google_logging",
    sha256 = "9e1b54eb2782f53cd8af107ecf08d2ab64b8d0dc2b7f5594472f3bd63ca85cdc",
    strip_prefix = "glog-0.4.0",
    urls = ["https://github.com/google/glog/archive/v0.4.0.zip"],
)

http_archive(
    name = "com_github_gflags_gflags",
    sha256 = "34af2f15cf7367513b352bdcd2493ab14ce43692d2dcd9dfc499492966c64dcf",
    strip_prefix = "gflags-2.2.2",
    urls = [
        "https://mirror.bazel.build/github.com/gflags/gflags/archive/v2.2.2.tar.gz",
        "https://github.com/gflags/gflags/archive/v2.2.2.tar.gz",
    ],
)
