"""

* Metrics generated will differ for detection vs classification.
* Focus on classification (it's easier)

* What are scoring the models on?
    - classifcation: accuracy (TP / (TP + FN))

1 How to load COCO dataset.
    - read *.json file and process the images/labels
    - you can associate labels with images

2 Loading the models
    - take in user timestamp and load the model (on gpu? cpu?)

3 Combine the loaded dataset with model to get predictions

4 Do something with the predictions. Generate the accuracy

"""


def inference_coco_dataset(
    model_timestamp, model_type, dataset,
):
    ...
