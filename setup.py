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

_MODELS_DIR = pathlib.Path.cwd() / "hawk_eye/core/production_models"


def _get_packages() -> List[str]:
    all_deps = pathlib.Path("requirements.txt").read_text().splitlines()
    deps = []
    for dep in all_deps:
        if "numpy" in dep or "pillow" in dep:
            deps.append(dep)

    return deps


class Build(build.build):
    def initialize_options(self):
        build.build.initialize_options(self)

    def finalize_options(self):
        build.build.finalize_options(self)

    def run(self) -> None:
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
    cmdclass={"build": Build, "prepare_models": PrepareModels},
    include_package_data=True,
    install_requires=_get_packages(),
)

shutil.rmtree(_MODELS_DIR)
