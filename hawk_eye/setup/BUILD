package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_docker//docker/util:run.bzl", "container_run_and_commit")
load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_image",
    "container_layer",
)
load("@io_bazel_rules_docker//docker/package_managers:download_pkgs.bzl", "download_pkgs")
load("@io_bazel_rules_docker//docker/package_managers:install_pkgs.bzl", "install_pkgs")

exports_files([
    "version.txt",
    "requirements.txt",
    "requirements-cpu.txt",
])

download_pkgs(
    name = "python_pkgs",
    image_tar = "@cuda-ubuntu18//image",
    packages = [
        "python3",
        "python3-pip",
        "python3-dev",
        "cmake",
        "gcc",
        "g++",
        "build-essential",
        "ninja-build",
    ],
)

install_pkgs(
    name = "python_cuda",
    image_tar = "@cuda-ubuntu18//image",
    installables_tar = ":python_pkgs.tar",
    installation_cleanup_commands = "rm -rf /var/lib/apt/lists/*",
    output_image_name = "python_cuda",
)

container_layer(
    name = "hawk_eye_setup",
    data_path = ".",
    directory = "hawk_eye/setup",
    files = [
        "setup_env.sh",
        "requirements.txt",
        "requirements-cpu.txt",
        "setup_bazel.sh"
    ],
)

container_image(
    name = "env_base",
    base = ":python_cuda",
    layers = [
        "//hawk_eye/setup:hawk_eye_setup",
    ],
)

container_run_and_commit(
    name = "python_install",
    commands = [
        "python3 -m pip install -U Cython==0.29.21 numpy==1.17.4 setuptools==53.0.0",
        "python3 -m pip install -U dataclasses==0.6",
        "python3 -m pip install -r hawk_eye/setup/requirements.txt",
        "python3 -m pip install -r hawk_eye/setup/requirements-cpu.txt",
    ],
    image = ":env_base.tar",
)


container_run_and_commit(
    name = "bazel_install",
    commands = [
        "bash hawk_eye/setup/setup_bazel.sh"
    ],
    image = ":python_install_commit.tar",
)

container_image(
    name = "hawk_eye_env",
    base = ":bazel_install_commit.tar",
)
