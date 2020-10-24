#!/usr/bin/env python3

import distutils.command.build as build
import distutils.command.clean as clean
import pathlib
import platform
import setuptools
import shutil
import subprocess
import sys
import tempfile
from typing import List
import os

import requests

from hawk_eye.inference import production_models

__version__ = pathlib.Path("version.txt").read_text()

_REQUIRED_PACKAGES = [
    "pillow=8.0.0.dev0",
]
_MODELS_DIR = pathlib.Path.cwd() / "hawk_eye/core/production_models"

# if platform.processor() == "x86_64":
#    _REQUIRED_PACKAGES += [pathlib.Path("requirements-gpu.txt").read_text().splitlines()]


class Build(build.build):
    def initialize_options(self):
        build.build.initialize_options(self)

    def finalize_options(self):
        build.build.finalize_options(self)

    def run(self) -> None:
        deps = []
        cpu_arch = platform.processor()
        if False:
            print("Downloading pre-build ARM PyTorch whl.")
            with tempfile.TemporaryDirectory() as d:
                tmp_whl = pathlib.Path(d) / "torch-1.6.0-cp36-cp36m-linux_aarch64.whl"
                r = requests.get(
                    "https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl",
                    stream=True,
                )
                tmp_whl.write_bytes(r.raw.read())
                subprocess.check_call([sys.executable, "-m", "pip", "install", tmp_whl])

        self.run_command("prepare_models")
        build.build.run(self)


class PrepareModels(build.build):
    def run(self):
        _MODELS_DIR.mkdir()
        models = ["classification_model", "detection_model"]
        for model in models:
            self._prepare_model(model)

    def _prepare_model(self, model_target: str):

        bazel_command = ["bazel", "fetch", f"@{model_target}//..."]
        if subprocess.call(bazel_command) != 0:
            sys.exit(-1)

        bazel_external = (
            pathlib.Path(
                subprocess.check_output(["bazel", "info", "output_base"])
                .decode("utf-8")
                .strip()
            )
            / "external"
        )

        if "classification" in model_target:
            shutil.copytree(
                bazel_external
                / model_target
                / production_models._CLASSIFIER["timestamp"],
                _MODELS_DIR / production_models._CLASSIFIER["timestamp"],
            )
        else:
            shutil.copytree(
                bazel_external
                / model_target
                / production_models._DETECTOR["timestamp"],
                _MODELS_DIR / production_models._DETECTOR["timestamp"],
            )


setuptools.setup(
    name="hawk_eye",
    version=__version__,
    description=("Find targets"),
    author="UAV Austin Image Recognition",
    packages=setuptools.find_packages(),
    cmdclass={"build": Build, "prepare_models": PrepareModels,},
    include_package_data=True,
)

shutil.rmtree(_MODELS_DIR)
