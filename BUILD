load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_image",
    "container_layer",
    "container_push"
)


container_image(
    name = "latest",
    base = "@amd64_base//image",
    layers = ["//docker:hawk_eye"],
    env = { "PYTHONPATH": "/"},
    repository = "uavaustin/hawk-eye-arm",
)
