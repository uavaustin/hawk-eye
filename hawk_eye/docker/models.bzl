load(
    "//third_party/remote_config:common.bzl",
    "execute",
    "get_python_bin",
)
load("//third_party:gcs.bzl", "download_gcs_object")

_BUCKET = "gs://uav-austin-test"


def _production_model_impl(repository_ctx):
    """"""
    python_bin = get_python_bin(repository_ctx)
    prod_models_script = repository_ctx.path(Label("//hawk_eye/inference:production_models.py"))
    contents = repository_ctx.read(prod_models_script)

    cmd = (
        "from os import system;" +
        "f = open('script.py', 'w');" +
        "f.write('''%s''');" % contents +
        "f.close();" +
        "system('chmod +x script.py && ./script.py --model_type %s');" % repository_ctx.attr.type
    )
    checked_paths = execute(repository_ctx, [python_bin, "-c", cmd]).stdout
    rm_result = repository_ctx.execute(["rm", 'script.py'])

    models = dict([tuple(x.split(": ")) for x in checked_paths.splitlines()])

    _download_model(
        repository_ctx,
        repository_ctx.attr.type,
        models["timestamp"],
        models["sha256"],
    )


def _download_model(repository_ctx, model_type, model_timestamp, sha256,):

    download_path = "%s/%s.tar.gz" % (model_type, model_timestamp)
    download_path = repository_ctx.path(download_path)

    download_gcs_object(
        repository_ctx,
        _BUCKET,
        str(download_path),
        "%s/%s.tar.gz" % (model_type, model_timestamp),
        sha256,
        "",
        model_timestamp,
        """
package(default_visibility = ["//visibility:public"])
filegroup(
    name = "file",
    srcs = glob(["%s/*"]),
)
""" % model_timestamp
    )


production_model = repository_rule(
    implementation = _production_model_impl,
    environ = [_BUCKET],
    attrs = {
        "type": attr.string(
            doc = "Which model type to download.",
            mandatory = True,
        ),
    },
)
