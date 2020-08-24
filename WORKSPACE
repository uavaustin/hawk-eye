workspace(name = "hawk_eye")

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
load("@io_bazel_rules_docker//toolchains/docker:toolchain.bzl",
    docker_toolchain_configure="toolchain_configure"
)

docker_toolchain_configure(
  name = "docker_config",
  # Replace this with an absolute path to a directory which has a custom docker
  # client config.json. Note relative paths are not supported.
  # Docker allows you to specify custom authentication credentials
  # in the client configuration JSON file.
  # See https://docs.docker.com/engine/reference/commandline/cli/#configuration-files
  # for more details.
  client_config="/home/alex/.docker",
)

container_pull(
    name = "ubuntu",
    registry = "nvcr.io",
    repository = "nvidia/tensorrt:20.07.1-py3",
    digest = "sha256:e1cafd8b0ea424f0b02ab69c3f63580a1cf643ffc1f30bfd6f4946f6d2302acd",
)

container_pull(
  name = "amd64_base",
  registry = "nvcr.io",
  repository = "nvidia/l4t-base:r32.4.3",
  digest = "sha256:547dc36b81eddb7ca8eadd956c61bd96bf432486830701b3dbb019be7f6c9ce2",
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
