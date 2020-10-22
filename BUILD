package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_image",
    "container_push",
)

# Start from the LT4 Jetson image and add the hawk_eye
# inference components on top.
container_image(
    name = "hawk_eye_arm_container",
    base = "@amd64_base//image",
    layers = [
        "//hawk_eye/docker:hawk_eye",
        "//hawk_eye/docker:production_models",
    ],
    repository = "uavaustin/hawk-eye-arm",
)

# Push the docker image to Docker Hub tagged based on the
# version.txt at the project root.
container_push(
    name = "push_hawk_eye_arm",
    format = "Docker",
    image = ":hawk_eye_arm_container",
    registry = "index.docker.io",
    repository = "uavaustin/hawk-eye-arm",
    tag_file = "version.txt",
)

exports_files([
    "requirements.txt",
    "version.txt",
])
