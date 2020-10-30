# Hawk_Eye
> Code to find targets in aerial imagery.


## Setup

Make sure to have Python3 installed. If you would like to use a gpu, install
CUDA and CUDNN. Next, run:
```
python3 -m pip install -U Cython==0.29.21 numpy==1.19.1
python3 -m pip install -r requirements.txt
pre-commit && pre-commit install
```
Then install either `requirements-cpu.txt` or `requirements-gpu.txt` depending
on your target device.

Finally, run:
```
pre-commit install
```


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
    --config configs/vovnet.yaml
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

TODO

# TODO

- [ ] Finish target typer model.

- [ ] Devise a way to be able to create detection, classification, and feature data in
separate independently.

- [ ] Finalize training pipelines.

- [ ] Finish inference pipeline.

- [ ] Weights & Biases integration once project is granted.

- [ ] unittests
