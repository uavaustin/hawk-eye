## Documentation
This file is meant to encompass some tips, tricks, and general documetation for
`hawk_eye`. By no means exhaustive yet, please adapt this file as time progresses.


## Contents

* [`Setup`](#setup)
* [`Data Generation`](#data-generation)
* [`Model Training`](#model-training)
* [`Inference`](#inference)
* [`Testing`](#testing)
* [`Bazel`](#bazel)
* [`hawk_eye Distibution`](#distribution)

## Setup
This projects works best with Linux, WSL, and Mac systems. The `setup_linux.sh` script
in this folder will set up Ubuntu and WSL systems. To run the script, execute:

```
./docs/setup_linux.sh
```

inside of the `hawk_eye` repository. This file also takes an optional argument to a python
virtual environemnt:

```
./docs/setup_linux.sh ~/path_to_venv
```

Upon sucessful termination of this script, you need to then recieve access to Google
Cloud Storage. See your lead about gaining permissions, then run

```
./docs/install_google_cloud.sh
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

## Inference

## Testing

## Bazel

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
