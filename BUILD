package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_image",
    "container_layer",
    "container_push"
)

# Start from the LT4 Jetson image and add the hawk_eye
# inference compoenents on top.
container_image(
    name = "hawk_eye_arm_container",
    base = "@amd64_base//image",
    layers = ["//docker:hawk_eye"],
    repository = "uavaustin/hawk-eye-arm",
)

# Push the docker image to Docker Hub tagged based on the
# version.txt at the project root.
container_push(
   name = "push_hawk_eye_arm",
   image = ":hawk_eye_arm_container",
   format = "Docker",
   registry = "index.docker.io",
   repository = "uavaustin/hawk-eye-arm",
   tag_file = "version.txt"
)

exports_files([
    "requirements.txt",
    "version.txt",
])
