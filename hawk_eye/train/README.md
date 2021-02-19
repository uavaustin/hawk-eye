See example model configurations in `models/configs` for inspiration on a model
architecture. Check out `torchvision` for more possibilities. An example
training command is:

```
PYTHONPATH=. hawk_eye/train/classification/train.py \
    --config hawk_eye/configs/vovnet.yaml
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
