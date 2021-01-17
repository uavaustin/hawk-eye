Training
===========================

There are two types of training currently supported: classifier and detection.
It is an unfortunate fact that most deep learning models need to be trained on
GPUs to be done in a reasonable amount of time.

Luckily our models are pretty light.


Examples
---------------

Classification:
::
    PYTHONPATH=. hawk_eye/train/train_clf.py \
        --config hawk_eye/configs/vovnet.yaml

Detection:
::
    PYTHONPATH=. hawk_eye/train/train_det.py \
        --config hawk_eye/configs/vovnet-det.yaml

Fine Tuning
---------------

If you only have a CPU, you might want to finetune a model. This meaning taking
an model intially trained on lots of data and training it on yours. This can
drastically speed up training.
