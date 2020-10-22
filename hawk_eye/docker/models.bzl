load(
    "//third_party/remote_config:common.bzl",
    "execute",
    "get_python_bin",
)
load("//third_party:gcs.bzl", "_gcs_file_impl")

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
        "system('chmod +x script.py && ./script.py');"
    )
    checked_paths = execute(repository_ctx, [python_bin, "-c", cmd]).stdout
    rm_result = repository_ctx.execute(["rm", 'script.py'])

    models = dict([tuple(x.split(": ")) for x in checked_paths.splitlines()])

    for model_type, model_timestamp in models.items():
        _download_model(repository_ctx, model_type, model_timestamp)


def _download_model(repository_ctx, model_type, model_timestamp):
    # Add a top-level BUILD file to export all the downloaded files.
    download_path = "%s.tar.gz" % model_timestamp
    download_path = repository_ctx.path(download_path)
    repository_ctx.attr.bucket = _BUCKET
    repository_ctx.attr.downloaded_file_path = download_path
    repository_ctx.attr.file = "%s/%s.tar.gz" % (model_type, model_timestamp)
    repository_ctx.attr.sha256 = "b00778153d14fd158345a9a18e5f79089d420c2cf36eb363a595d439d1b9c089"

    _gcs_file_impl(repository_ctx)


production_models = repository_rule(
    implementation = _production_model_impl,
    environ = [_BUCKET],
    attrs = {
        "bucket": attr.string(
            doc = "The GCS bucket which contains the file.",
        ),
        "downloaded_file_path": attr.string(
            doc = "Path assigned to the file downloaded.",
        ),
        "file": attr.string(
            doc = "The file which we are downloading.",
        ),
        "sha256": attr.string(
            doc = "The expected SHA-256 of the file downloaded.",
        ),
        "strip_prefix": attr.string(doc = "The contents of the build file for the target"),
    },
)
