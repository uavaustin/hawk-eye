load(
    "//third_party/remote_config:common.bzl",
    "execute",
    "get_python_bin",
)

_BUCKET = "gs://uav-austin-test"
_GCS_FILE_BUILD = """
package(default_visibility = ["//visibility:public"])
filegroup(
    name = "file",
    srcs = glob(["uav-*/**/*"]),
)
"""


def _production_model_impl(repository_ctx):
    """"""
    python_bin = get_python_bin(repository_ctx)
    prod_models_script = repository_ctx.path(Label("//inference:production_models.py"))
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
    repository_ctx.file("BUILD", _GCS_FILE_BUILD)

    # Create a bash script from a template.
    repository_ctx.template(
        "gsutil_cp_and_validate.sh",
        Label("//third_party:gsutil_cp_and_validate.sh.tpl"),
        {
            "%{BUCKET}": _BUCKET,
            "%{DOWNLOAD_PATH}": str(download_path),
            "%{FILE}": "%s/%s.tar.gz" % (model_type, model_timestamp),
            "%{SHA256}": "b00778153d14fd158345a9a18e5f79089d420c2cf36eb363a595d439d1b9c089",
        },
    )
    gsutil_cp_and_validate_result = repository_ctx.execute(["bash", "gsutil_cp_and_validate.sh"])
    if gsutil_cp_and_validate_result.return_code == 255:
        fail("SHA256 of file")
    elif gsutil_cp_and_validate_result.return_code != 0:
        fail("gsutil cp command failed: %s" % (gsutil_cp_and_validate_result.stderr))

    # Extract the downloaded archive.
    repository_ctx.extract(download_path, "uav-%s/%s" %(model_type, model_timestamp))
    rm_result = repository_ctx.execute(["rm", "gsutil_cp_and_validate.sh"])



production_models = repository_rule(
    implementation = _production_model_impl,
    environ = [_BUCKET],
)
