load(
    "//third_party/remote_config:common.bzl",
    "execute",
    "get_python_bin",
)

_BUCKET = "gs://uav-austin-test"


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
        "system('\"%s\" script.py')" % (python_bin) 
    )
    checked_paths = execute(repository_ctx, [python_bin, "-c", cmd]).stdout
    models = dict([tuple(x.split(": ")) for x in checked_paths.splitlines()])


production_models = repository_rule(
    implementation = _production_model_impl,
    environ = [_BUCKET],
)
