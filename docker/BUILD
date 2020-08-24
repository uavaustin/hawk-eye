load("@io_bazel_rules_docker//container:container.bzl", "container_image")
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

container_image(
    name = "hawk_eye_arm",
    base = "@ubuntu//image",
    tars = [":hawk_eye_tar"],
    env = { "PYTHONPATH": "/"},
)

# This target packages all the necessary code for inference
# into a tar which will upack itself in the docker container we
# create for the target platform.
pkg_tar(
    name = "hawk_eye_tar",
    srcs = ["//inference:find_targets"],
    package_dir = "hawk_eye"
)
