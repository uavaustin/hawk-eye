.. role:: hidden
    :class: hidden-section

Real Image Datasets
===========================

Occasionally, we will have a test flight where image rec is able to capture real
pictures of the targets we bring to the airfield. This type of data is ideal for
training our models, but we currently (as of 2021) only have a couple dozen unique
target images.

While an obvious solution is to have more test flights, it's likely each flight
uncovers a new bug, delaying our data collection. Moreover, to robustify our data,
we'd need to keep repainting our targets and making new ones. Over the years, we
might reach a point where we have enough flight test data to train a model.

Since this is not the case (2021), we've devised a scheme for training a model
on completely synthetic data first, then finetuning on the real images we do have.


Creating a Real Image Dataset
------------------------------

.. todo::
    Write the pipeline for transforming a real flight into dataset
