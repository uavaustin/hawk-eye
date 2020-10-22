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
        "//hawk_eye/inference:requirements.txt",
        "//hawk_eye/core:asset_manager",
        "//hawk_eye/core:classifier",
        "//hawk_eye/core:detector",
        "//hawk_eye/data_generation:config.yaml",
        "//hawk_eye/data_generation:generate_config",
        "//hawk_eye/inference:find_targets",
        "//hawk_eye/inference:types",
        "//third_party",
    ],
)

container_layer(
    name = "production_models",
    files = [
        "@classification_model//:file",
        "@detection_model//:file",
    ],
    data_path = ".",
    symlinks = {"/home/runs": "/production_models"}
)
