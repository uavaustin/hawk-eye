# hawk_eye
> Code to find targets in aerial imagery.

![Python3 CI](https://github.com/uavaustin/hawk-eye/workflows/Python3%20CI/badge.svg)

----

`hawk-eye` is an end-to-end pipeline for training models to detect aerial imagery.
Specifically, the goal is to produce models that perform well in the [AUVSI SUAS](https://static1.squarespace.com/static/5d554e14aaa5e300011a4844/t/5fd3780f2662933f59dbedd5/1607694352554/auvsi_suas-2021-rules.pdf)
competition.


## Contents

* [`Setup`](#setup)
* [`Data Generation`](#data-generation)
* [`Model Training`](#model-training)
* [`Inference`](#inference)
* [`Testing`](#testing)
* [`Bazel`](#bazel)
* [`hawk_eye Distibution`](#distribution)
* [`Style`](#style)

## Setup
This project supports Linux, WSL, and Mac systems. To setup one of these environments, run:

```
hawk_eye/setup/setup_env.sh
```

This file also takes an optional argument to a python virtual environment:

```
hawk_eye/setup/setup_env.sh ~/path_to_venv
```

Upon sucessful termination of this script, you need to then recieve access to Google
Cloud Storage. See your lead about gaining permissions, then run

```
hawk_eye/setup/install_google_cloud.sh
```

To ensure you are all setup and ready to code, you can test code by running:

```
bazel test //...
```

## Data Generation

Before we can ever train a model we need data. In our project, we create artificial data
using various python libraries. Inside of `hawk_eye/data_generation` you'll find the
scripts related to data processing.

#### Contents
* `config.yaml`: The file containing many of the tweakable parameters for data
generation.
* `create_clf_data.py`: Creates classification data. This is binary data, in other words,
only two classes: target or background.
* `create_detection_data.py`: Create detection data for the object detection models.
The data will be saved as a COCO formatted archive.
* `create_shape_combinations.py`: **Experimental** script for generating all possible combinations of targets/shapes/colors.
* `generate_config.py`: The config file which references `config.yaml`.

## Model Training

See example model configurations in `models/configs` for inspiration on a model
architecture. Check out `torchvision` for more possibilities. An example
training command is:

```
PYTHONPATH=. hawk_eye/train/train_clf.py \
    --model_config configs/vovnet.yaml
```

Use a detector config with `hawk_eye/train/train_det.py`:

```
PYTHONPATH=. hawk_eye/train/train_det.py \
    --model_config configs/vovnet-det.yaml
```

Most model architectures will be located in `third_party` since we typically implement
other researchers' models. Each model will save itself to `~/runs/uav-{model_type}`.
Inside the timestamped archive, you can find the training log, a tensorboard file, and
the saved weights.

## Inference

Inference on an image will utilize both the classifier and object detector models.
The inference script can be run as follows:

```
PYTHONPATH=$(pwd) inference/find_targets.py \
    --image_path /path/to/image \
    --visualization_dir /path/to/save
```
The command above will visualize and save the models' predictions. See
`inference/find_targets.py` for the full list of available arguments.

One can also specify the model timestamps if you have a certain model to test.

## Testing

Testing is done with `bazel`, but you can alternatively run each test_*.py as a python
executable.

Please look inside the `test` folder for more information. In short, there are python
unit tests and `flake8` style tests. We'll use bazel to run all the test targets:

```
bazel test //...
```

To run the style tests:

```
flake8
```

## Bazel

`Bazel` is an open-sourced Google product used for a variety of build environments. Right
now, we mainly use it for python testing, but if PyTorch eventually supports building
itself as an external third party project, we might start using the C++ PyTorch API.

For now, if you're interested,
[`here`](https://docs.bazel.build/versions/master/user-manual.html)
is some documentation.

## Distribution

Once Image Recognition has an inference pipeline we'd like Infrastructure to use, we need
to create a `pip` package for them to access. We do this using `setuptools` and a
`setup.py` script.

Any time a new release is created on Github, the `.github/workflows/create_release.yaml`
workflow will be kicked off and upload a generic python wheel. This `.whl` file contains
our necessary inference code and the models for inferencing.

To create the wheel locally, run:

```
./setup.py bdist_wheel
```

A `.whl` file will be generated inside of `./dist/` and be named in accordance with the
version inside of `version.txt`.

## Style

All python code will be automatically formatted using `Black` through `pre-commit`.
`flake8` will be employed to correct any other style errors. Please familiarize yourself
with the [`Google python style guide`](https://google.github.io/styleguide/pyguide.html).
