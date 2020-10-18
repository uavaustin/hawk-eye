package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_docker//container:container.bzl",
    "container_layer",
)

# This rule takes the version.txt and makes a version.py for
# importing in the python package.
genrule(
    name = "version_file",
    srcs = ["//:version.txt"],
    outs = ["version.py"],
    cmd = 'echo "__version__ ==" `cat $<` > $@',
)

# Build a docker layer of all the needed files for using
# hawk_eye for inference externally.
container_layer(
    name = "hawk_eye",
    data_path = ".",
    directory = "hawk_eye",
    env = {"PYTHONPATH": "/"},
    files = [
        ":version_file",
        "//inference:requirements.txt",
        "//core:asset_manager",
        "//core:classifier",
        "//core:detector",
        "//data_generation:config.yaml",
        "//data_generation:generate_config",
        "//inference:find_targets",
        "//inference:types",
        "//third_party",
        "@production_models",
    ],
)

