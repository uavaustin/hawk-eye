workspace(name = "hawk_eye")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

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

http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "4521794f0fba2e20f3bf15846ab5e01d5332e587e9ce81629c7f96c793bb7036",
    strip_prefix = "rules_docker-0.14.4",
    urls = ["https://github.com/bazelbuild/rules_docker/releases/download/v0.14.4/rules_docker-v0.14.4.tar.gz"],
)

load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)

container_repositories()

load("@io_bazel_rules_docker//repositories:deps.bzl", container_deps = "deps")

container_deps()

load("@io_bazel_rules_docker//repositories:pip_repositories.bzl", "pip_deps")

pip_deps()

load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_pull",
)
load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)

container_repositories()

# Load the macro that allows you to customize the docker toolchain configuration.
load(
    "@io_bazel_rules_docker//toolchains/docker:toolchain.bzl",
    docker_toolchain_configure = "toolchain_configure",
)

# Get the Docker Hub credentials.
docker_toolchain_configure(
    name = "docker_config",
    client_config = "~/.docker",
)

container_pull(
    name = "ubuntu",
    digest = "sha256:e1cafd8b0ea424f0b02ab69c3f63580a1cf643ffc1f30bfd6f4946f6d2302acd",
    registry = "nvcr.io",
    repository = "nvidia/tensorrt:20.07.1-py3",
)

container_pull(
    name = "amd64_base",
    digest = "sha256:547dc36b81eddb7ca8eadd956c61bd96bf432486830701b3dbb019be7f6c9ce2",
    registry = "nvcr.io",
    repository = "nvidia/l4t-base:r32.4.3",
)

load("//third_party:gcs.bzl", "gcs_file")

gcs_file(
    name = "base_shapes",
    bucket = "gs://uav-austin-test",
    downloaded_file_path = "base_shapes.tar.gz",
    file = "assets/base-shapes-v1.tar.gz",
    sha256 = "9266e23087c58ee679903a6891208bb8c636396cdbc0b35f0eeff294411ddbdc",
)

gcs_file(
    name = "backgrounds",
    bucket = "gs://uav-austin-test",
    downloaded_file_path = "backgrounds.tar.gz",
    file = "backgrounds/backgrounds-v1.tar.gz",
    sha256 = "b00778153d14fd158345b9a18e5f79089d420c2cf36eb363a595d439d1b9c089",
)

gcs_file(
    name = "fonts",
    bucket = "gs://uav-austin-test",
    downloaded_file_path = "fonts.tar.gz",
    file = "assets/fonts.tar.gz",
    sha256 = "e67fc398c9e9a55071d2d4edd155c691540bf4951383cfa1bed69aacbee02675",
    strip_prefix = "fonts",
)
