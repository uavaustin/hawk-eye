#!/usr/bin/env python3
""" Script to create .whl files for varying target devices.
We support amd64 and arm64. """

import argparse
import pathlib
import platform
from typing import List
import os

import setuptools

import version


def get_requirements(arch: str, device: str) -> List[str]:
    """ The main difference to look for is the target device. The Jetsons have
    special, pre-built pytorch wheels maintained by Nvidia. """
    "https://download.pytorch.org/whl/cpu/torch"

    if arch == "x86_64":
        if device == "gpu":
            deps = pathlib.Path("requirements-gpu.txt").read_text().splitlines()
        else:
            deps = pathlib.Path("requirements-cpu.txt").read_text().splitlines()
    elif arch == "arm":
        deps = [
            "torch @ https://nvidia.box.com/shared/static/9eptse6jyly1ggt9axbja2yrmj6pbarc.whl"
        ]

    return deps + pathlib.Path("requirements.txt").read_text().splitlines()


if __name__ == "__main__":

    device = "gpu" if os.environ.get("GPU", False) else "cpu"
    cpu_arch = "arm" if os.environ.get("ARM", False) else "x86_64"
    print(cpu_arch)
    print(
        f"Building wheel for target device {device} with cpu architecture {cpu_arch}."
    )

    setuptools.setup(
        name="hawk_eye",
        version=version.__version__,
        install_requires=(get_requirements(cpu_arch, device)),
        package_dir={"hawk_eye": "", "inference": "inference"},
        packages=["hawk_eye", "hawk_eye.inference"],
        scripts=[
            "version.py",
            "inference/find_targets.py",
            "inference/types.py",
            "core/classifier.py",
            "core/detector.py",
            "data_generation/generate_config.py",
            "third_party/models/postprocess.py",
        ],
        author="UAVA",
        author_email="awitt2399@utexas.edu",
        description="Library for performing aerial inference.",
        url="https://github.com/uavaustin/hawk-eye",
        python_requires=">=3.6.1",
        license="MIT",
    )
