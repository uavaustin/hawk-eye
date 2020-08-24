load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_image",
    "container_layer",
    "container_push"
)


# This target packages all the necessary code for inference
# into a tar which will upack itself in the docker container we
# create for the target platform.
container_layer(
    name = "hawk_eye",
    directory = "hawk_eye",
    files = [
        "//inference:find_targets",
        "//inference:types",
        "//core:classifier",
        "//core:detector",
        "//core:asset_manager",
        "//third_party",
        "//data_generation:generate_config",
        "//data_generation:config.yaml",
    ],
    data_path = ".",
)

container_push(
   name = "push_hawk_eye_arm",
   image = ":latest",
   format = "Docker",
   registry = "index.docker.io",
   repository = "uavaustin/hawk-eye-arm",
   tag = "latest",
)
