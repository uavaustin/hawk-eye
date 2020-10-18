# hawk_eye
> Code to find targets in aerial imagery.


## Setup

See the `docs` page for installation. The supported opperating systems are:
* Ubuntu20
* Ubuntu18
* WSL1
* WSL2
* MacOS


## Data Generation

To create data to train an object detector on, run:
```
PYTHONPATH=$(pwd) data_generation/create_detection_data.py
```

To create data to train a classifier, run:
```
PYTHONPATH=$(pwd) data_generation/create_clf_data.py
```
Edit the `data_generation/config.yaml` to adjust the amount of data to create.


## Classifier Training

See example model configurations in `models/configs` for inspiration on a model
architecture. Check out `torchvision` for all the possibilities. An example
training command is:
```
PYTHONPATH=$(pwd) train/train_clf.py \
    --model_config configs/resnet18.yaml
```


## Detector Training

See example model configurations in `models/configs` for inspiration on a model
architecture.
```
PYTHONPATH=$(pwd) train/train_det.py \
    --model_config configs/vovnet-det.yaml
```


## Inference on an Image

Inference on an image will utilize both the classifier and object detector model.
The inference script can be run as follows:
```
PYTHONPATH=$(pwd) inference/find_targets.py \
    --image_path /path/to/image \
    --visualization_dir /path/to/save
```
The command above will visualize and save the models' predictions. See
`inference/find_targets.py` for the full list of available arguments.


## Testing

Please look inside the `test` folder for more information. In short, there are python
unit tests and `flake8` style tests. We'll use bazel to run the test targets:

```bazel test //...```

To run the style tests:

```flake8```
