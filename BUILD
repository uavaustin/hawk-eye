package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_image",
    "container_layer",
    "container_push"
)


container_image(
    name = "hawk_eye_arm_container",
    base = "@amd64_base//image",
    layers = ["//docker:hawk_eye"],
    env = { "PYTHONPATH": "/"},
    repository = "uavaustin/hawk-eye-arm",
)

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
