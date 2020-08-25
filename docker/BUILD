package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_image",
    "container_layer",
    "container_push"
)

genrule(
    name = "version_file",
    srcs = ["//:version.txt"],
    cmd = 'echo "__version__ ==" `cat $<` > $@',
    outs = ["version.py"]
)

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
        "//:requirements.txt",
        ":version_file",
    ],
    data_path = ".",
)
